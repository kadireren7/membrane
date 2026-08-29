#include <cctype>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "ggml.h"
#include "llama.h"
#include "runtime_core.h"
#include "kv_store_telemetry.h"
#include "decode_loop.h"
#include "product_cli.h"
#include "compat_check.h"
#include "gpu_policy.h"
#include "gpu_device.h"
#include "adaptive_kv_policy.h"
#include "kv_residency_policy.h"
#include "joint_planner.h"
#include "auto_fallback.h"

/*
 * membrane-run: Product Phase 8, the user-facing MEMBRANE entry point.
 *
 * Normal mode (default): model load, ONE llama_context (native or
 * q8), prompt ingestion, generation, print, exit. No native reference
 * pass, no teacher-forced comparison pass, no per-step logit capture
 * -- run_kv_store_pass() is called with capture_logits=false, so
 * gen_run_result_t::logits never grows past empty. This is the single
 * most load-bearing property of this file; see run_normal_mode().
 *
 * Compare mode (--compare-kv, explicit only): reuses the exact Phase 7
 * 3-pass machinery (native reference, q8 canonical, q8 teacher-forced)
 * via the same run_kv_store_pass()/record_kv_store_behavior() shared
 * with membrane-llama-run (tools/membrane-llama-runtime/decode_loop.h)
 * -- one real implementation, two CLIs on top of it.
 */

/* llama.cpp/ggml log to stderr by default at INFO level -- hundreds of
 * lines of tensor/graph internals per run. Section 7: normal output
 * must be readable, internal diagnostics only with --verbose. Installed
 * before model load so it also covers load_tensors/print_info output,
 * not just decode-time messages. Errors always get through regardless
 * of --verbose -- a real failure should never be silently swallowed. */
static void	quiet_log_callback(enum ggml_log_level level, const char *text,
				void *user_data)
{
	(void)user_data;
	if (level == GGML_LOG_LEVEL_ERROR)
		fputs(text, stderr);
}

/* Phase 13.2, Section 17: Linux-only host-memory OBSERVABILITY (never a
 * safety mechanism -- nothing in this file conditions a pass/fail
 * decision on these numbers, only reports them). Isolated in this one
 * function so the /proc/meminfo dependency stays contained rather than
 * spread across call sites -- ok stays false (fields left at 0) on any
 * non-Linux host or unreadable /proc, and every caller already treats
 * that the same way membrane_kv_store_rss_t's own proc_status_ok does:
 * omit the fields rather than report a fabricated zero as real. */
typedef struct s_membrane_host_meminfo
{
	bool		ok;
	uint64_t	total_bytes;
	uint64_t	available_bytes;
	uint64_t	swap_total_bytes;
	uint64_t	swap_free_bytes;
}	membrane_host_meminfo_t;

static void	read_host_meminfo(membrane_host_meminfo_t *out)
{
	FILE		*f;
	char		line[256];
	unsigned long long	kb;
	int			found = 0;

	memset(out, 0, sizeof(*out));
	f = fopen("/proc/meminfo", "r");
	if (f == NULL)
		return ;
	while (fgets(line, sizeof(line), f) != NULL)
	{
		if (sscanf(line, "MemTotal: %llu kB", &kb) == 1)
			out->total_bytes = (uint64_t)kb * 1024, found |= 1;
		else if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1)
			out->available_bytes = (uint64_t)kb * 1024, found |= 2;
		else if (sscanf(line, "SwapTotal: %llu kB", &kb) == 1)
			out->swap_total_bytes = (uint64_t)kb * 1024, found |= 4;
		else if (sscanf(line, "SwapFree: %llu kB", &kb) == 1)
			out->swap_free_bytes = (uint64_t)kb * 1024, found |= 8;
	}
	fclose(f);
	/* Review fix (CodeRabbit, PR #23): a kernel built without swap
	 * support (CONFIG_SWAP=n) omits the SwapTotal/SwapFree lines from
	 * /proc/meminfo entirely rather than printing them as 0 -- requiring
	 * all four fields for `ok` suppressed every host-memory figure on
	 * such a host even though MemTotal/MemAvailable were read
	 * correctly. Require only the two memory fields; the already-
	 * zeroed swap fields (memset above) are a truthful "0 swap" in that
	 * case, not a fabricated value. */
	out->ok = ((found & 3) == 3);
}

static std::string	read_stdin(void)
{
	std::string	s;
	char		buf[4096];
	size_t		n;

	while ((n = fread(buf, 1, sizeof(buf), stdin)) > 0)
		s.append(buf, n);
	return (s);
}

static bool	resolve_prompt(const membrane_run_opts_t &o,
				std::string *prompt_text)
{
	if (o.prompt_mode == MEMBRANE_RUN_PROMPT_TEXT)
	{
		*prompt_text = o.prompt_text;
		return (true);
	}
	if (o.prompt_mode == MEMBRANE_RUN_PROMPT_STDIN)
	{
		*prompt_text = read_stdin();
		return (!prompt_text->empty());
	}
	*prompt_text = read_file(o.prompt_file);
	return (!prompt_text->empty());
}

typedef struct s_model_shape
{
	std::string	arch_name;
	int32_t		n_layer;
	int32_t		n_embd;
	int32_t		n_head;
	int32_t		n_head_kv;
	int64_t		n_embd_gqa;
}	model_shape_t;

static void	read_model_shape(llama_model *model, model_shape_t *s)
{
	char	buf[128];
	int32_t	n;

	n = llama_model_meta_val_str(model, "general.architecture", buf,
			sizeof(buf));
	s->arch_name = (n > 0) ? std::string(buf) : std::string();
	s->n_layer = llama_model_n_layer(model);
	s->n_embd = llama_model_n_embd(model);
	s->n_head = llama_model_n_head(model);
	s->n_head_kv = llama_model_n_head_kv(model);
	s->n_embd_gqa = (s->n_head > 0)
		? (int64_t)(s->n_embd / s->n_head) * s->n_head_kv : 0;
}

static uint64_t	native_kv_bytes(const model_shape_t &s, uint32_t ctx_size)
{
	membrane_kv_store_bytes_t	b;

	b.n_layer = (uint64_t)s.n_layer;
	b.kv_size = ctx_size;
	b.bytes_per_token_k = ggml_row_size(GGML_TYPE_F16, s.n_embd_gqa);
	b.bytes_per_token_v = ggml_row_size(GGML_TYPE_F16, s.n_embd_gqa);
	return (membrane_kv_store_total_bytes(&b));
}

static uint64_t	q8_kv_bytes(const model_shape_t &s, uint32_t ctx_size)
{
	membrane_kv_store_bytes_t	b;

	b.n_layer = (uint64_t)s.n_layer;
	b.kv_size = ctx_size;
	b.bytes_per_token_k = ggml_row_size(GGML_TYPE_Q8_0, s.n_embd_gqa);
	b.bytes_per_token_v = ggml_row_size(GGML_TYPE_Q8_0, s.n_embd_gqa);
	return (membrane_kv_store_total_bytes(&b));
}

static uint64_t	q5_kv_bytes(const model_shape_t &s, uint32_t ctx_size)
{
	membrane_kv_store_bytes_t	b;

	b.n_layer = (uint64_t)s.n_layer;
	b.kv_size = ctx_size;
	b.bytes_per_token_k = ggml_row_size(GGML_TYPE_Q5_1, s.n_embd_gqa);
	b.bytes_per_token_v = ggml_row_size(GGML_TYPE_Q5_1, s.n_embd_gqa);
	return (membrane_kv_store_total_bytes(&b));
}

/* Real ggml_row_size()-based byte accounting for whichever of the
 * three storage modes was requested -- the single place every call
 * site below dispatches through, so native/q8/q5's estimates (used
 * here, and by --gpu-layers auto's policy resolution below) can never
 * drift out of sync with each other the way separately-updated
 * ternaries could. */
static uint64_t	kv_bytes_for_mode(const model_shape_t &s, uint32_t ctx_size,
				int kv_mode)
{
	if (kv_mode == MEMBRANE_KV_STORE_Q8)
		return (q8_kv_bytes(s, ctx_size));
	if (kv_mode == MEMBRANE_KV_STORE_Q5)
		return (q5_kv_bytes(s, ctx_size));
	return (native_kv_bytes(s, ctx_size));
}

static const char	*kv_mode_name(int kv_mode)
{
	if (kv_mode == MEMBRANE_KV_STORE_Q8)
		return ("q8");
	if (kv_mode == MEMBRANE_KV_STORE_Q5)
		return ("q5");
	return ("native");
}

static const char	*kv_type_label(int kv_mode)
{
	if (kv_mode == MEMBRANE_KV_STORE_Q8)
		return ("Q8_0");
	if (kv_mode == MEMBRANE_KV_STORE_Q5)
		return ("Q5_1");
	return ("F16");
}

/* Phase 9B/9B.1: what was requested vs what was actually resolved,
 * kept as separate fields throughout (never collapsed into one) so
 * JSON/human telemetry can honestly distinguish "the user asked for
 * this" from "this is what public ggml_backend_dev_*()/gguf_*() calls
 * found and MEMBRANE explicitly chose" -- membrane-run never leaves
 * this to llama.cpp's own implicit upstream default (see
 * resolve_gpu_config). policy_* fields are only meaningful when
 * policy_used is true (an estimate was actually available and used --
 * see gpu_policy.h's own doc comment on what "estimate" does and does
 * not claim). */
typedef struct s_membrane_gpu_state
{
	bool		requested;				/* o.gpu_layers != 0 */
	bool		backend_gpu_capable;	/* llama_supports_gpu_offload() */
	int32_t		gpu_layers_requested;	/* o.gpu_layers, echoed
									 * (-1=all, -2=auto) */
	int32_t		gpu_layers_selected;	/* the concrete value actually
									 * passed to llama_model_params.
									 * n_gpu_layers -- equals
									 * gpu_layers_requested except for
									 * auto (always a resolved N) */
	std::string	device_requested;		/* o.device, empty if not given */
	std::string	device_selected;		/* empty when CPU-only */
	std::string	backend_selected;		/* "CPU" or e.g. "Vulkan" */
	bool		policy_used;
	uint64_t	device_total_bytes;
	uint64_t	device_free_bytes;
	uint64_t	safety_reserve_bytes;
	uint64_t	estimated_model_bytes;
	uint64_t	estimated_kv_bytes;

	/* Phase 11A: --kv adaptive's resolution, populated exactly once --
	 * inside resolve_gpu_config() when a GPU was requested (using the
	 * same pre-load GGUF estimate/gpu_policy_resolve() calls the
	 * existing --gpu-layers auto guard already makes, so the KV mode
	 * and layer count decisions can never use inconsistent estimates,
	 * Section 6), or by resolve_cpu_adaptive_kv() after model load
	 * otherwise (Section 5). adaptive_used is false whenever o.kv_mode
	 * != MEMBRANE_KV_STORE_ADAPTIVE; every print/telemetry function
	 * below gates on it to decide whether to show requested_kv/
	 * selected_kv/adaptive_reason at all. */
	bool		adaptive_used;
	int			adaptive_selected_mode;	/* MEMBRANE_KV_STORE_Q8/Q5,
											 * meaningful iff
											 * adaptive_used */
	std::string	adaptive_reason;
	uint64_t	adaptive_q8_kv_bytes;
	uint64_t	adaptive_q5_kv_bytes;
	int32_t		adaptive_q8_layers;	/* GPU only, 0 for CPU */
	int32_t		adaptive_q5_layers;	/* GPU only, 0 for CPU */
	int			adaptive_q8_valid;
	int			adaptive_q5_valid;

	/* Phase 12H: --kv-placement's resolved static residency plan,
	 * populated by resolve_kv_placement() in main() AFTER model load
	 * (needs the model's real, post-load layer count and shape) --
	 * kv_placement_resolved stays false, and every other field below
	 * stays at its zero value, whenever o.kv_placement ==
	 * MEMBRANE_KV_PLACEMENT_DEFAULT (the only mode with zero decode-
	 * path behavior change -- Section 4/19). Never consulted by
	 * anything downstream unless kv_placement_resolved is true. */
	bool		kv_placement_resolved;
	membrane_kv_residency_result_t	kv_placement;

	/* Phase 13.1: the single overall "why did the plan end up this way"
	 * reason code (Section 5/10) -- distinct from gpu_policy's own
	 * per-layer-selection reason_code (pr.reason_code, not stored here)
	 * and kv_placement.reason: this one summarizes the TOP-LEVEL
	 * outcome for the plan-summary/JSON output, always set by the time
	 * resolve_gpu_config() returns true. auto_cpu_fallback records
	 * specifically whether --auto's own implicit gpu request gracefully
	 * degraded to CPU-only (Section 1) -- distinct from an ordinary,
	 * always-been-CPU-only run (gpu_layers_requested == 0), which never
	 * sets it. */
	std::string	plan_reason_code;
	bool		auto_cpu_fallback;

	/* Phase 20: set only when the joint planner ran (the plain,
	 * non-compare/bench path with a usable pre-load estimate --
	 * joint_planner_used stays false otherwise, e.g. --compare-kv/
	 * --gpu-bench, or no GGUF estimate at all). candidate_count/
	 * selected_index are diagnostic only -- nothing downstream makes a
	 * decision from them, matching plan_reason_code/reason_trace's own
	 * observability-only role. */
	bool		joint_planner_used;
	int			joint_candidate_count;
	int			joint_selected_index;

	/* Phase 21: the FULL ranked candidate array Phase 20's planner
	 * produced (not just the counts above) -- the ONLY candidate source
	 * the apply-time fallback controller (auto_fallback.h) is ever
	 * allowed to use (Section 4). Meaningful only when
	 * joint_planner_used. Populated verbatim (struct copy) from the
	 * jres local to resolve_gpu_config()'s joint-planner branch. */
	membrane_joint_plan_result_t	joint_result;

	/* Phase 21: populated once the (possibly-retried) run actually
	 * happened -- fallback_trace.n_entries stays 0 for any run that
	 * never engaged the fallback controller at all (compare-kv/
	 * gpu-bench, CPU-only, or no-estimate-available: Section 24/25 --
	 * see print_fallback_json()'s own doc comment for what such a run
	 * reports instead). */
	membrane_fallback_trace_t	fallback_trace;
	bool		fallback_engaged;

	/* Phase 13.2, Section 7: the ordered sequence of semantically
	 * meaningful policy decisions that produced plan_reason_code above
	 * -- e.g. ["AUTO_REQUESTED", "GPU_DEVICE_FOUND", "GPU_FULL_FIT",
	 * "Q8_FULL_RESIDENCY"]. Deliberately short: one entry per real
	 * decision point (device search outcome, layer-selection outcome,
	 * adaptive-precision outcome, kv-placement outcome), never one per
	 * branch condition inside those decisions -- see each push site's
	 * own comment for why that entry is there. */
	std::vector<std::string>	reason_trace;

	/* Phase 13.2, Section 16: informational, non-fatal plan warnings --
	 * built once by build_plan_warnings() after every other field above
	 * is final. Never changes exit_code/ok; purely additive telemetry
	 * for --verbose/--json consumers. */
	std::vector<std::string>	warnings;

	/* Phase 25 (OPT-03): attributes planner_ms's own 8.6-12.0 ms GPU-
	 * requested cost (Phase 24 finding) across the three real sub-steps
	 * resolve_gpu_config() bundles -- device enumeration
	 * (membrane_gpu_list_devices()), GGUF metadata pre-scan
	 * (membrane_gpu_estimate_model()), and the joint planner's own
	 * arithmetic (membrane_joint_plan_resolve()). Each stays 0.0 unless
	 * its own call site actually runs on this invocation (e.g. an
	 * explicit CPU-only request returns before any of the three run;
	 * a device/GGUF error returns before the joint planner call) --
	 * verify-performance-optimization.py checks these three sum to
	 * approximately the already-measured planner_ms, not that all three
	 * are always nonzero. */
	double		device_enumeration_ms;
	double		gguf_prescan_ms;
	double		joint_planner_core_ms;
}	membrane_gpu_state_t;

/* Phase 24: top-level pipeline stage timings that don't belong to any
 * one existing struct -- planner/model-load/tokenization all happen in
 * main() itself, around calls into modules that have no timing
 * awareness of their own (Section 3: one clean boundary per stage,
 * measured by the caller, not threaded into every callee). Context-
 * create/prefill/decode/first-token timings live on
 * membrane_kv_store_telemetry_t instead (decode_loop.cpp already owns
 * that stage), never duplicated here -- print_run_json() reads both
 * this struct and tel together. All fields are milliseconds via
 * seconds_since()/CLOCK_MONOTONIC (decode_loop.h/.cpp's own existing
 * timer, reused rather than a second implementation), left at 0.0 by
 * an aggregate-initialized (`= {}`) instance until their own stage
 * actually runs -- 0.0 is never ambiguous with a real measurement in
 * practice (no real stage completes in exactly zero wall-clock time). */
typedef struct s_membrane_timings
{
	double	planner_ms;
	double	model_load_ms;
	double	tokenization_ms;
	double	total_ms;
}	membrane_timings_t;

static std::string	gpu_layers_label(int32_t gpu_layers)
{
	if (gpu_layers == MEMBRANE_GPU_LAYERS_ALL)
		return ("all");
	if (gpu_layers == MEMBRANE_GPU_LAYERS_AUTO)
		return ("auto");
	return (std::to_string(gpu_layers));
}

/* Resolves --gpu-layers/--device into an explicit llama_model_params
 * device list, using gpu_device.h for all ggml_backend_dev_*()/
 * gguf_*() access (the one place membrane-run touches those APIs) and
 * gpu_policy.h for the llama-free auto/guard arithmetic -- mp->devices
 * is never left NULL for llama.cpp's own upstream default to silently
 * pick a GPU (GPU use must be explicit, not accidental). *device_
 * storage must outlive mp's use in llama_model_load_from_file().
 * ctx_size is REQUIRED (>0) here: product_cli.cpp's parse-time
 * validation rejects ANY nonzero --gpu-layers (all/auto/N) paired
 * with an auto-sized --ctx, precisely so the guard below is always
 * KV-aware and never has to fall back to a weights-only check that
 * could pass here and still fail later once the real (auto-sized)
 * context is known.
 *
 * Returns false (message already printed to stderr) if the request
 * cannot be satisfied -- callers must fail closed
 * (MEMBRANE_EXIT_UNSUPPORTED_KV), never silently fall back to CPU or
 * silently proceed with fewer layers than an EXPLICIT (non-auto)
 * request asked for. */

/* Phase 13.1, Section 3/13: forward-declared so the failure sites
 * inside resolve_gpu_config() below (defined ahead of the rest of this
 * file's print_*_json() helpers) can emit a structured JSON error
 * object on stdout -- defined further down, next to print_json_
 * escaped() which it reuses. Every important runtime failure path
 * (Section 4 of the Phase 13.1 task) calls this exactly once, only
 * when o.want_json is set; --json's stdout otherwise never carries
 * anything but the one success object print_run_json() prints, and
 * plain stderr text is unaffected either way (it is not JSON, so it
 * stays available for humans/log-scraping regardless of --json). */
/* Phase 13.2, Section 10/15: suggestion and available_devices are both
 * optional and additive to the Phase 13.1 error-object shape -- a
 * caller passing neither (the overwhelming majority, unchanged from
 * Phase 13.1) gets the exact same object as before. suggestion must be
 * an ACTIONABLE, factually appropriate remediation (Section 10: "Do
 * not offer impossible fixes") -- only populated at the handful of
 * call sites where one genuinely exists; every other call site passes
 * NULL and the field is omitted entirely, not emitted empty. */
/* Phase 25 (OPT-06): forward-declared here for the same reason as
 * print_error_json() above -- its real definition lives next to
 * print_fallback_json() (which shares its logic), but print_error_json()
 * itself needs to call it. */
static void	print_fallback_trace_json(const membrane_fallback_trace_t &ft,
				bool trailing_comma);

static void	print_error_json(const membrane_run_opts_t &o, int exit_code,
				const char *reason_code, const std::string &message,
				const char *suggestion = NULL,
				const std::vector<std::string> *available_devices = NULL,
				const membrane_fallback_trace_t *fallback_trace = NULL);

/* Review fix (CodeRabbit, PR #22): the two --auto-implied CPU fallback
 * sites inside resolve_gpu_config() below assigned these same six
 * fields, in the same order, verbatim -- a future change to the
 * fallback contract had to be applied twice or silently drift. Always
 * returns true (there is no failure case for falling back to CPU). */
static bool	fall_back_to_cpu(llama_model_params *mp, membrane_gpu_state_t *gs)
{
	mp->n_gpu_layers = 0;
	mp->main_gpu = -1;
	gs->gpu_layers_selected = 0;
	gs->requested = false;
	gs->auto_cpu_fallback = true;
	gs->plan_reason_code = MEMBRANE_GPU_POLICY_REASON_NO_GPU_DEVICE;
	gs->reason_trace.push_back(MEMBRANE_GPU_POLICY_REASON_NO_GPU_DEVICE);
	return (true);
}

/* No usable estimate at all (GGUF metadata unreadable) -- only
 * reachable for explicit all/N (auto and adaptive both already
 * returned earlier, before this point, on missing metadata): proceed
 * unguarded, same as pre-9B.1 behavior. There is no evidence to
 * reject on, so the user's explicit request is honored rather than
 * blocked on a missing measurement. Shared by both the --compare-kv/
 * --gpu-bench path and the Phase 20 joint-planner path below (the
 * exact same fallback pre-Phase-20 code used, factored out here only
 * to avoid duplicating it between the two). */
static void	apply_no_estimate_fallback(const membrane_run_opts_t &o,
				membrane_gpu_state_t *gs)
{
	gs->gpu_layers_selected = o.gpu_layers == MEMBRANE_GPU_LAYERS_ALL
		? -1 : o.gpu_layers;
	gs->plan_reason_code = MEMBRANE_GPU_POLICY_REASON_METADATA_UNAVAILABLE;
}

static bool	resolve_gpu_config(const membrane_run_opts_t &o,
				uint32_t ctx_size,
				std::vector<ggml_backend_dev_t> *device_storage,
				llama_model_params *mp, membrane_gpu_state_t *gs)
{
	std::string	label;
	size_t		n_devices;
	size_t		i;
	size_t		gpu_count;

	/* NOT memset(): membrane_gpu_state_t has std::string members, so a
	 * raw byte-zero would bypass their constructors and corrupt their
	 * internal representation (real -Wclass-memaccess warning caught
	 * this) -- assign the plain-data fields individually and .clear()
	 * the strings instead. */
	gs->device_selected.clear();
	gs->policy_used = false;
	gs->device_total_bytes = 0;
	gs->device_free_bytes = 0;
	gs->safety_reserve_bytes = 0;
	gs->estimated_model_bytes = 0;
	gs->estimated_kv_bytes = 0;
	gs->adaptive_used = false;
	gs->adaptive_selected_mode = 0;
	gs->adaptive_reason.clear();
	gs->adaptive_q8_kv_bytes = 0;
	gs->adaptive_q5_kv_bytes = 0;
	gs->adaptive_q8_layers = 0;
	gs->adaptive_q5_layers = 0;
	gs->adaptive_q8_valid = 0;
	gs->adaptive_q5_valid = 0;
	gs->kv_placement_resolved = false;
	memset(&gs->kv_placement, 0, sizeof(gs->kv_placement));
	gs->plan_reason_code.clear();
	gs->joint_planner_used = false;
	gs->joint_candidate_count = 0;
	gs->joint_selected_index = -1;
	memset(&gs->joint_result, 0, sizeof(gs->joint_result));
	memset(&gs->fallback_trace, 0, sizeof(gs->fallback_trace));
	gs->fallback_engaged = false;
	gs->auto_cpu_fallback = false;
	gs->reason_trace.clear();
	gs->warnings.clear();
	gs->device_enumeration_ms = 0.0;
	gs->gguf_prescan_ms = 0.0;
	gs->joint_planner_core_ms = 0.0;
	if (o.auto_mode)
		gs->reason_trace.push_back("AUTO_REQUESTED");
	gs->requested = (o.gpu_layers != 0);
	gs->backend_gpu_capable = membrane_gpu_backend_available() != 0;
	gs->gpu_layers_requested = o.gpu_layers;
	gs->gpu_layers_selected = o.gpu_layers;
	gs->device_requested = o.want_device ? o.device : "";
	gs->backend_selected = "CPU";
	if (!gs->requested)
	{
		/* explicit CPU-only: also clear main_gpu so a compiled-in GPU
		 * backend cannot be silently selected by llama.cpp's own
		 * default device list -- with mp->devices left NULL here,
		 * llama_prepare_model_devices() takes its auto-detect branch
		 * and would otherwise pick a GPU on main_gpu's own default
		 * (0); n_gpu_layers=0 alone already prevents any layer from
		 * being placed there, but main_gpu=-1 additionally empties
		 * the device list outright, so there is no ambiguity about
		 * why CPU is used. */
		mp->n_gpu_layers = 0;
		mp->main_gpu = -1;
		gs->plan_reason_code = MEMBRANE_GPU_POLICY_REASON_CPU_ONLY_REQUESTED;
		gs->reason_trace.push_back(MEMBRANE_GPU_POLICY_REASON_CPU_ONLY_REQUESTED);
		return (true);
	}
	label = gpu_layers_label(o.gpu_layers);
	/* Phase 13.1, Section 1/15/16: --auto's OWN implicit gpu_layers=auto
	 * request (o.auto_mode && !o.want_gpu_layers -- never an explicit
	 * --gpu-layers auto/all/N, which keeps the pre-existing fail-closed
	 * behavior below unchanged) must not require a GPU backend or
	 * device at all -- gracefully resolve to CPU-only instead of
	 * failing. gs->requested is explicitly reset to false here (not
	 * left true) so main()'s own `!gs->requested` check still routes
	 * --kv adaptive through resolve_cpu_adaptive_kv() below, exactly
	 * as an ordinary --gpu-layers 0 run would. */
	bool	auto_implied_gpu = o.auto_mode && !o.want_gpu_layers;

	if (!gs->backend_gpu_capable)
	{
		if (auto_implied_gpu)
			return (fall_back_to_cpu(mp, gs));
		fprintf(stderr, "membrane-run: --gpu-layers %s requested but this "
			"build has no GPU backend compiled in (rebuild with e.g. "
			"-DGGML_VULKAN=ON)\n", label.c_str());
		print_error_json(o, MEMBRANE_EXIT_UNSUPPORTED_KV,
			MEMBRANE_GPU_POLICY_REASON_NO_GPU_DEVICE,
			"--gpu-layers " + label + " requested but this build has no "
			"GPU backend compiled in",
			"Try --gpu-layers 0 for CPU-only, or rebuild with a GPU "
			"backend enabled (e.g. -DGGML_VULKAN=ON).");
		return (false);
	}

	membrane_gpu_device_info_t	devices[MEMBRANE_GPU_MAX_DEVICES];
	struct timespec				substage_t0;

	clock_gettime(CLOCK_MONOTONIC, &substage_t0);
	n_devices = membrane_gpu_list_devices(devices, MEMBRANE_GPU_MAX_DEVICES);
	gs->device_enumeration_ms = seconds_since(&substage_t0) * 1000.0;
	gpu_count = 0;
	for (i = 0; i < n_devices; ++i)
		if (devices[i].type == MEMBRANE_DEV_TYPE_GPU
			|| devices[i].type == MEMBRANE_DEV_TYPE_IGPU)
			gpu_count++;
	if (gpu_count == 0)
	{
		if (auto_implied_gpu)
			return (fall_back_to_cpu(mp, gs));
		fprintf(stderr, "membrane-run: --gpu-layers %s requested but no "
			"GPU device was found on this host at runtime (driver/"
			"hardware not detected)\n", label.c_str());
		print_error_json(o, MEMBRANE_EXIT_UNSUPPORTED_KV,
			MEMBRANE_GPU_POLICY_REASON_NO_GPU_DEVICE,
			"--gpu-layers " + label + " requested but no GPU device was "
			"found on this host at runtime",
			"Try --gpu-layers 0 for CPU-only, or verify a GPU driver is "
			"installed and visible to this build's backend.");
		return (false);
	}

	size_t	chosen_idx;

	if (o.want_device)
	{
		size_t	idx = 0;
		size_t	matches = membrane_gpu_match_device(devices, n_devices,
				o.device.c_str(), &idx);

		if (matches != 1)
		{
			fprintf(stderr, "membrane-run: --device \"%s\" matched %zu "
				"available GPU device%s%s\n", o.device.c_str(), matches,
				matches == 0 ? "" : "s, expected exactly one",
				matches == 0 ? "" : ":");
			if (matches == 0)
				fprintf(stderr, "Available:\n");
			/* Section 15: the exact same device list the human message
			 * above already prints, kept for --json (available_devices
			 * array) so a scripted caller doesn't have to scrape stderr
			 * to offer the same "pick one of these" UX a human gets. */
			std::vector<std::string>	available_names;

			for (i = 0; i < n_devices; ++i)
				if (devices[i].type == MEMBRANE_DEV_TYPE_GPU
					|| devices[i].type == MEMBRANE_DEV_TYPE_IGPU)
				{
					fprintf(stderr, "  %s (%s)\n", devices[i].name,
						devices[i].description);
					available_names.push_back(devices[i].name);
				}
			print_error_json(o, MEMBRANE_EXIT_UNSUPPORTED_KV,
				MEMBRANE_GPU_POLICY_REASON_DEVICE_NOT_FOUND,
				"--device \"" + o.device + "\" matched "
				+ std::to_string(matches) + " available GPU device(s), "
				"expected exactly one",
				"Pass a --device substring matching exactly one of the "
				"available devices, or omit --device to auto-select.",
				&available_names);
			return (false);
		}
		chosen_idx = idx;
	}
	else
	{
		/* No --device given: prefer a discrete GPU over an integrated
		 * one, matching llama.cpp's own default-selection preference
		 * (llama_prepare_model_devices()) -- device enumeration order
		 * is NOT discrete-first (confirmed by testing on this host:
		 * index 0 was the integrated AMD GPU, a later index the
		 * discrete NVIDIA one), so this must be sought explicitly. */
		chosen_idx = n_devices;
		for (i = 0; i < n_devices; ++i)
			if (devices[i].type == MEMBRANE_DEV_TYPE_GPU)
			{
				chosen_idx = i;
				break;
			}
		if (chosen_idx == n_devices)
			for (i = 0; i < n_devices; ++i)
				if (devices[i].type == MEMBRANE_DEV_TYPE_IGPU)
				{
					chosen_idx = i;
					break;
				}
	}

	const membrane_gpu_device_info_t	&dev = devices[chosen_idx];

	gs->device_selected = dev.name;
	gs->backend_selected = dev.backend;
	gs->device_total_bytes = dev.memory_total;
	gs->device_free_bytes = dev.memory_free;
	/* Section 7: a real, findable GPU device is its own meaningful
	 * decision point, distinct from (and always preceding) whichever
	 * layer-selection/precision outcome follows it below. */
	gs->reason_trace.push_back("GPU_DEVICE_FOUND");

	membrane_gpu_model_estimate_t	est;

	clock_gettime(CLOCK_MONOTONIC, &substage_t0);
	int		have_est = membrane_gpu_estimate_model(o.model_path, &est);
	gs->gguf_prescan_ms = seconds_since(&substage_t0) * 1000.0;
	uint64_t	kv_bytes_estimate = 0;
	uint64_t	kv_bytes_estimate_q5 = 0;	/* adaptive's second
											 * candidate only */
	uint64_t	kv_bytes_real_q8 = 0;	/* adaptive only: Q8's own real
										 * KV bytes, NEVER folded with
										 * native_estimate below --
										 * kv_bytes_estimate itself IS
										 * folded (a real bug: it fed
										 * both the guard call AND, un-
										 * folded-looking but actually
										 * inflated, the reported
										 * candidate telemetry/budget
										 * check) -- CodeRabbit finding,
										 * fixed by keeping this
										 * separate real value for
										 * cand_q8.kv_bytes/the --kv-
										 * budget-mib comparison, while
										 * kv_bytes_estimate itself
										 * stays folded for ONLY the
										 * membrane_gpu_policy_resolve()
										 * guard call below (native and
										 * the candidate context are
										 * never resident at once --
										 * see run_native_vs_compressed_
										 * comparison()'s sequential
										 * create/destroy -- so the
										 * guard must budget for
										 * whichever is larger, but
										 * that is a GUARD sizing
										 * concern only, never what
										 * gets reported as "Q8's KV
										 * bytes"). */
	uint64_t	kv_bytes_real_q5 = 0;	/* same, for Q5 */
	bool	adaptive_requested = (o.kv_mode == MEMBRANE_KV_STORE_ADAPTIVE);

	if (have_est && est.hparams_available && ctx_size > 0)
	{
		model_shape_t	fake_shape;
		uint64_t		native_estimate = 0;

		fake_shape.n_layer = est.n_layer;
		fake_shape.n_embd = est.n_embd;
		fake_shape.n_head = est.n_head;
		fake_shape.n_head_kv = est.n_head_kv;
		fake_shape.n_embd_gqa = (est.n_head > 0)
			? (int64_t)(est.n_embd / est.n_head) * est.n_head_kv : 0;
		if (o.compare_kv || o.gpu_bench)
			native_estimate = native_kv_bytes(fake_shape, ctx_size);
		if (adaptive_requested)
		{
			/* Both candidates computed from the SAME est/ctx_size, via
			 * the exact same kv_bytes_for_mode() every explicit --kv
			 * q8/q5 request already uses -- Section 9: no separate
			 * arithmetic, no drift. kv_bytes_real_q8/q5 are the true,
			 * never-folded per-mode bytes; kv_bytes_estimate/_q5 are
			 * then folded with native_estimate for the guard call
			 * ONLY (see kv_bytes_real_q8's doc comment above). */
			kv_bytes_real_q8 = kv_bytes_for_mode(fake_shape, ctx_size,
				MEMBRANE_KV_STORE_Q8);
			kv_bytes_real_q5 = kv_bytes_for_mode(fake_shape, ctx_size,
				MEMBRANE_KV_STORE_Q5);
			kv_bytes_estimate = native_estimate > kv_bytes_real_q8
				? native_estimate : kv_bytes_real_q8;
			kv_bytes_estimate_q5 = native_estimate > kv_bytes_real_q5
				? native_estimate : kv_bytes_real_q5;
		}
		else
		{
			kv_bytes_estimate = kv_bytes_for_mode(fake_shape, ctx_size,
				o.kv_mode);
			/* --compare-kv/--gpu-bench allocate a NATIVE context in
			 * addition to the candidate one (run_native_vs_compressed_
			 * comparison() below) -- native's F16 KV is always the
			 * larger of the two, so the guard must budget for native's
			 * bytes here too, not just whatever --kv itself asked for,
			 * or auto could pick a layer count the native pass then
			 * can't actually fit. selected_comparison_mode() isn't
			 * visible yet at this point in the file, but the guard
			 * only needs to know "is a native pass also about to run",
			 * not which compressed mode was picked. */
			if (native_estimate > kv_bytes_estimate)
				kv_bytes_estimate = native_estimate;
		}
	}
	if (adaptive_requested && !(have_est && est.hparams_available
			&& est.n_layer > 0))
	{
		fprintf(stderr, "membrane-run: --kv adaptive requires the "
			"model's real hparams and layer structure to choose safely "
			"between q8 and q5, but they could not be read from '%s'\n",
			o.model_path);
		print_error_json(o, MEMBRANE_EXIT_UNSUPPORTED_KV,
			MEMBRANE_GPU_POLICY_REASON_METADATA_UNAVAILABLE,
			std::string("--kv adaptive requires the model's real "
				"hparams and layer structure, but they could not be "
				"read from '") + o.model_path + "'",
			"Verify the model file is a valid GGUF, or use an explicit "
			"--kv q8/q5/native instead of adaptive.");
		return (false);
	}
	if (o.gpu_layers == MEMBRANE_GPU_LAYERS_AUTO && !(have_est
			&& est.n_layer > 0))
	{
		fprintf(stderr, "membrane-run: --gpu-layers auto requested but "
			"the model's layer structure could not be read from '%s' "
			"-- cannot resolve safely\n", o.model_path);
		print_error_json(o, MEMBRANE_EXIT_UNSUPPORTED_KV,
			MEMBRANE_GPU_POLICY_REASON_METADATA_UNAVAILABLE,
			std::string("--gpu-layers auto requested but the model's "
				"layer structure could not be read from '")
				+ o.model_path + "'",
			"Verify the model file is a valid GGUF, or use an explicit "
			"--gpu-layers N instead of auto.");
		return (false);
	}
	/* Phase 20: --compare-kv/--gpu-bench keep the EXACT pre-Phase-20
	 * gpu_policy_resolve()+adaptive_kv_resolve() path, unchanged --
	 * those modes allocate a SECOND, native context in addition to the
	 * candidate one (run_native_vs_compressed_comparison(), further
	 * down this file) and need native_estimate folded into the guard
	 * for both passes, a sizing concern the joint planner below does
	 * not model. This is not a new restriction: product_cli.cpp
	 * already rejects --kv-placement (non-default) together with
	 * --compare-kv/--gpu-bench at parse time (Phase 12H scope), so
	 * these modes were never going to reach the joint planner's own
	 * placement-aware logic regardless. */
	if (o.compare_kv || o.gpu_bench)
	{
		if (adaptive_requested)
		{
			/* have_est && est.hparams_available && est.n_layer > 0
			 * already guaranteed by the fail-closed check above -- both
			 * candidates are always evaluated against the SAME
			 * requested_layers/device_free/device_total/bytes_per_layer
			 * inputs (Section 6: "the policy and GPU planner must
			 * evaluate the same candidate states"), differing only in
			 * kv_bytes_estimate. */
			membrane_gpu_policy_result_t		pr_q8;
			membrane_gpu_policy_result_t		pr_q5;
			membrane_adaptive_kv_candidate_t	cand_q8;
			membrane_adaptive_kv_candidate_t	cand_q5;
			membrane_adaptive_kv_result_t		ar;
			/* Phase 13.1: the weight-layer preflight must not reserve
			 * GPU budget for KV that --kv-placement cpu/auto will never
			 * (cpu) or isn't guaranteed to (auto) actually put on the
			 * GPU -- see kv_residency_policy.h's own doc comment on
			 * this function. */
			int		ok_q8 = membrane_gpu_policy_resolve(o.gpu_layers,
					est.n_layer, dev.memory_free, dev.memory_total,
					est.bytes_per_layer,
					membrane_kv_policy_preflight_reservation(o.kv_placement,
						kv_bytes_estimate), &pr_q8);
			int		ok_q5 = membrane_gpu_policy_resolve(o.gpu_layers,
					est.n_layer, dev.memory_free, dev.memory_total,
					est.bytes_per_layer,
					membrane_kv_policy_preflight_reservation(o.kv_placement,
						kv_bytes_estimate_q5), &pr_q5);

			gs->policy_used = true;
			gs->safety_reserve_bytes = pr_q8.safety_reserve_bytes;
			cand_q8.valid = ok_q8 && (!o.want_kv_budget
				|| kv_bytes_real_q8 <= o.kv_budget_bytes);
			cand_q8.full_residency = cand_q8.valid
				&& pr_q8.selected_layers == est.n_layer;
			cand_q8.selected_layers = ok_q8 ? pr_q8.selected_layers : 0;
			cand_q8.kv_bytes = kv_bytes_real_q8;
			cand_q5.valid = ok_q5 && (!o.want_kv_budget
				|| kv_bytes_real_q5 <= o.kv_budget_bytes);
			cand_q5.full_residency = cand_q5.valid
				&& pr_q5.selected_layers == est.n_layer;
			cand_q5.selected_layers = ok_q5 ? pr_q5.selected_layers : 0;
			cand_q5.kv_bytes = kv_bytes_real_q5;
			gs->adaptive_q8_kv_bytes = cand_q8.kv_bytes;
			gs->adaptive_q5_kv_bytes = cand_q5.kv_bytes;
			gs->adaptive_q8_layers = cand_q8.selected_layers;
			gs->adaptive_q5_layers = cand_q5.selected_layers;
			gs->adaptive_q8_valid = cand_q8.valid;
			gs->adaptive_q5_valid = cand_q5.valid;
			membrane_adaptive_kv_resolve(&cand_q8, &cand_q5, 1, &ar);
			if (!ar.ok)
			{
				fprintf(stderr, "membrane-run: --kv adaptive found no KV "
					"storage mode that fits safely: %s\n", ar.reason);
				print_error_json(o, MEMBRANE_EXIT_UNSUPPORTED_KV, ar.reason,
					"--kv adaptive found no KV storage mode that fits safely",
					"Try --kv-placement cpu, reduce --ctx, or raise "
					"--kv-budget-mib.");
				return (false);
			}
			gs->adaptive_used = true;
			gs->adaptive_selected_mode = ar.selected_mode;
			gs->adaptive_reason = ar.reason;
			gs->estimated_model_bytes = ar.selected_mode == MEMBRANE_KV_STORE_Q8
				? pr_q8.estimated_model_bytes : pr_q5.estimated_model_bytes;
			gs->estimated_kv_bytes = ar.selected_kv_bytes;
			gs->gpu_layers_selected = ar.selected_layers;
			gs->plan_reason_code = ar.selected_mode == MEMBRANE_ADAPTIVE_KV_MODE_Q8
				? pr_q8.reason_code : pr_q5.reason_code;
		}
		else if (have_est && est.n_layer > 0)
		{
			membrane_gpu_policy_result_t	pr;
			/* Phase 13.1: see the adaptive branch's identical comment above
			 * -- same fix, non-adaptive path. */
			int		ok = membrane_gpu_policy_resolve(o.gpu_layers, est.n_layer,
					dev.memory_free, dev.memory_total, est.bytes_per_layer,
					membrane_kv_policy_preflight_reservation(o.kv_placement,
						kv_bytes_estimate), &pr);

			gs->policy_used = true;
			gs->safety_reserve_bytes = pr.safety_reserve_bytes;
			gs->estimated_model_bytes = pr.estimated_model_bytes;
			/* Review fix (CodeRabbit, PR #22): report the real KV estimate,
			 * not pr.estimated_kv_bytes -- that field only echoes back
			 * whatever membrane_kv_policy_preflight_reservation() passed
			 * IN to the preflight guard (0 for cpu/auto placement), which
			 * is a guard input, not the real KV allocation. Reporting 0
			 * here for --kv-placement cpu (which allocates real, nonzero
			 * KV bytes, just not on the GPU) understated telemetry/JSON/
			 * the plan summary and overstated the headroom_bytes
			 * computation below. The adaptive branch above already used
			 * the real, never-folded kv_bytes_real_q8/q5 value for this
			 * same reason -- this branch now matches it. */
			gs->estimated_kv_bytes = kv_bytes_estimate;
			if (!ok)
			{
				fprintf(stderr, "membrane-run: %s\n", pr.reason);
				print_error_json(o, MEMBRANE_EXIT_UNSUPPORTED_KV,
					pr.reason_code, pr.reason,
					"Try --kv-placement cpu, reduce --ctx, or lower "
					"--gpu-layers.");
				return (false);
			}
			gs->gpu_layers_selected = pr.selected_layers;
			gs->plan_reason_code = pr.reason_code;
		}
		else
			apply_no_estimate_fallback(o, gs);
	}
	else if (have_est && est.n_layer > 0)
	{
		/* Phase 20: the joint planner replaces the pre-Phase-20
		 * sequential precision-then-layers decision above with one
		 * bounded, deterministic evaluation across GPU layer count x
		 * KV precision x KV placement together -- see joint_planner.h's
		 * own top comment for the full rationale/citations. Covers
		 * BOTH adaptive_requested and explicit precision internally
		 * (jreq.precision_request selects which). */
		membrane_joint_plan_request_t	jreq;
		membrane_joint_plan_result_t	jres;

		memset(&jreq, 0, sizeof(jreq));
		jreq.n_layer_all = est.n_layer;
		jreq.bytes_per_layer = est.bytes_per_layer;
		jreq.output_role_bytes = est.output_role_bytes;
		jreq.arch_name = est.arch_name;
		jreq.n_embd = est.n_embd;
		jreq.n_head = est.n_head;
		jreq.n_head_kv = est.n_head_kv;
		jreq.ctx_size = ctx_size;
		jreq.device_free_bytes = dev.memory_free;
		jreq.device_total_bytes = dev.memory_total;
		/* Numeric spaces are deliberately identical by construction --
		 * MEMBRANE_JOINT_PLACEMENT_{DEFAULT,GPU,CPU,AUTO} (0-3) mirrors
		 * MEMBRANE_KV_PLACEMENT_* exactly, MEMBRANE_JOINT_GPU_LAYERS_
		 * REQUEST_{ALL,AUTO} mirrors MEMBRANE_GPU_LAYERS_{ALL,AUTO}
		 * exactly -- see joint_planner.h's own header comment on each
		 * constant. */
		jreq.kv_placement_mode = o.kv_placement;
		jreq.gpu_layers_request = o.gpu_layers;
		jreq.want_kv_budget = o.want_kv_budget;
		jreq.kv_budget_bytes = o.kv_budget_bytes;
		if (adaptive_requested)
		{
			jreq.precision_request = MEMBRANE_JOINT_PRECISION_REQUEST_AUTO;
			jreq.kv_bytes_q8 = kv_bytes_real_q8;
			jreq.kv_bytes_q5 = kv_bytes_real_q5;
		}
		else
		{
			jreq.precision_request = o.kv_mode;
			if (o.kv_mode == MEMBRANE_KV_STORE_Q8)
				jreq.kv_bytes_q8 = kv_bytes_estimate;
			else if (o.kv_mode == MEMBRANE_KV_STORE_Q5)
				jreq.kv_bytes_q5 = kv_bytes_estimate;
			else
				jreq.kv_bytes_native = kv_bytes_estimate;
		}
		clock_gettime(CLOCK_MONOTONIC, &substage_t0);
		bool	jplan_ok = membrane_joint_plan_resolve(&jreq, &jres);

		gs->joint_planner_core_ms = seconds_since(&substage_t0) * 1000.0;
		if (!jplan_ok)
		{
			fprintf(stderr, "membrane-run: %s\n", jres.reason);
			print_error_json(o, MEMBRANE_EXIT_UNSUPPORTED_KV,
				jres.reason_code, jres.reason,
				"Try --kv-placement cpu, reduce --ctx, --kv-budget-mib, "
				"or a smaller --gpu-layers.");
			return (false);
		}

		const membrane_joint_candidate_t	&winner
			= jres.candidates[jres.selected_index];

		gs->policy_used = true;
		gs->safety_reserve_bytes = winner.safety_reserve_bytes;
		gs->estimated_model_bytes = winner.estimated_weight_gpu_bytes;
		gs->estimated_kv_bytes = winner.estimated_kv_gpu_bytes
			+ winner.estimated_host_kv_bytes;
		gs->gpu_layers_selected = winner.gpu_layers;
		gs->plan_reason_code = winner.reason_code;
		gs->joint_planner_used = true;
		gs->joint_candidate_count = jres.candidate_count;
		gs->joint_selected_index = jres.selected_index;
		gs->joint_result = jres;
		if (adaptive_requested)
		{
			gs->adaptive_used = true;
			gs->adaptive_selected_mode = winner.kv_precision;
			gs->adaptive_reason = winner.reason_code;
			gs->adaptive_q8_kv_bytes = jres.adaptive_q8_kv_bytes;
			gs->adaptive_q5_kv_bytes = jres.adaptive_q5_kv_bytes;
			gs->adaptive_q8_layers = jres.adaptive_q8_layers;
			gs->adaptive_q5_layers = jres.adaptive_q5_layers;
			gs->adaptive_q8_valid = jres.adaptive_q8_valid;
			gs->adaptive_q5_valid = jres.adaptive_q5_valid;
		}
	}
	else
		apply_no_estimate_fallback(o, gs);
	/* Section 7: the layer-selection/precision outcome for whichever of
	 * the three branches above ran (adaptive, non-adaptive-with-
	 * estimate, or no-estimate) -- always the same value plan_primary_
	 * reason()/print_plan_summary() report as "reason:", pushed here
	 * once so every path that reaches this shared tail is covered
	 * without duplicating the push inside each branch. The adaptive
	 * branch's own precision choice (e.g. "Q8_FULL_RESIDENCY") is a
	 * distinct, later decision than the layer-selection code just
	 * pushed -- both are real, semantically separate outcomes worth
	 * keeping in the trace. */
	if (!gs->plan_reason_code.empty())
		gs->reason_trace.push_back(gs->plan_reason_code);
	/* Phase 20: the joint planner reports ONE unified reason_code per
	 * candidate (covering layer selection AND precision together), so
	 * plan_reason_code/adaptive_reason are now often the identical
	 * string for a joint-planner-resolved run -- skip the redundant
	 * second trace entry in that case; the pre-Phase-20 --compare-kv/
	 * --gpu-bench path still computes them as two genuinely distinct
	 * reasons (layer-selection outcome, then a separate precision
	 * choice) and keeps both. */
	if (gs->adaptive_used && gs->adaptive_reason != gs->plan_reason_code)
		gs->reason_trace.push_back(gs->adaptive_reason);
	device_storage->clear();
	device_storage->push_back((ggml_backend_dev_t)dev.native_handle);
	device_storage->push_back(NULL);
	mp->devices = device_storage->data();
	mp->n_gpu_layers = gs->gpu_layers_selected;
	mp->main_gpu = 0;
	return (true);
}

/* Phase 11A Section 5: --kv adaptive on CPU (o.gpu_layers == 0, i.e.
 * !gs->requested -- resolve_gpu_config() above never runs the GPU
 * adaptive branch in that case). No GPU free-memory query exists here,
 * so there is no full-residency/partial-offload comparison -- CPU
 * adaptive defaults to Q8 unless an explicit --kv-budget-mib rules it
 * out and Q5 fits, matching membrane_adaptive_kv_resolve()'s CPU path.
 * Called AFTER model load (unlike the GPU path), since it uses the
 * real post-load model shape rather than a pre-load GGUF estimate --
 * ctx_size may still have been 0/auto-sized at CLI-parse time, and is
 * only known for certain once the model's tokenizer has run. */
static bool	resolve_cpu_adaptive_kv(const membrane_run_opts_t &o,
				const model_shape_t &shape, uint32_t ctx_size,
				membrane_gpu_state_t *gs)
{
	membrane_adaptive_kv_candidate_t	cand_q8;
	membrane_adaptive_kv_candidate_t	cand_q5;
	membrane_adaptive_kv_result_t		ar;
	uint64_t	bytes_q8 = q8_kv_bytes(shape, ctx_size);
	uint64_t	bytes_q5 = q5_kv_bytes(shape, ctx_size);

	cand_q8.valid = !o.want_kv_budget || bytes_q8 <= o.kv_budget_bytes;
	cand_q8.full_residency = cand_q8.valid;
	cand_q8.selected_layers = 0;
	cand_q8.kv_bytes = bytes_q8;
	cand_q5.valid = !o.want_kv_budget || bytes_q5 <= o.kv_budget_bytes;
	cand_q5.full_residency = cand_q5.valid;
	cand_q5.selected_layers = 0;
	cand_q5.kv_bytes = bytes_q5;
	gs->adaptive_q8_kv_bytes = bytes_q8;
	gs->adaptive_q5_kv_bytes = bytes_q5;
	gs->adaptive_q8_layers = 0;
	gs->adaptive_q5_layers = 0;
	gs->adaptive_q8_valid = cand_q8.valid;
	gs->adaptive_q5_valid = cand_q5.valid;
	membrane_adaptive_kv_resolve(&cand_q8, &cand_q5, 0, &ar);
	if (!ar.ok)
		return (fprintf(stderr, "membrane-run: --kv adaptive found no "
				"KV storage mode that fits safely: %s\n", ar.reason),
			false);
	gs->adaptive_used = true;
	gs->adaptive_selected_mode = ar.selected_mode;
	gs->adaptive_reason = ar.reason;
	gs->estimated_kv_bytes = ar.selected_kv_bytes;
	gs->reason_trace.push_back(ar.reason);
	return (true);
}

/* Phase 12H: --kv-placement's static residency resolution -- called
 * from main() AFTER model load (needs the real, post-load layer
 * count; shape/ctx_size are only final at that point) and BEFORE
 * dispatch to run_normal_mode(), using effective_kv_mode (the already-
 * resolved concrete precision, NEVER MEMBRANE_KV_STORE_ADAPTIVE itself
 * -- Section 27: precision is decided first, placement second,
 * strictly sequential, no joint search). A no-op (returns true,
 * gs->kv_placement_resolved left false) whenever o.kv_placement is
 * MEMBRANE_KV_PLACEMENT_DEFAULT. gs->estimated_model_bytes/
 * device_free_bytes/device_total_bytes are gpu_policy's OWN
 * already-resolved weight-placement numbers, passed through unchanged
 * -- this function never recomputes or influences weight placement
 * (Section 21). Returns false (message already printed) on a fail-
 * closed GPU-mode budget rejection; callers must not silently retry
 * with a different mode. */
static bool	resolve_kv_placement(const membrane_run_opts_t &o,
				const model_shape_t &shape, uint32_t ctx_size,
				int effective_kv_mode, membrane_gpu_state_t *gs)
{
	uint64_t	total_kv_bytes;
	uint64_t	kv_bytes_per_layer;

	if (o.kv_placement == MEMBRANE_KV_PLACEMENT_DEFAULT)
		return (true);
	/* Phase 13.1 review fix (Section 6): gs->auto_cpu_fallback is true
	 * ONLY when --auto's own implicit gpu_layers=auto request (never an
	 * explicit --gpu-layers) gracefully degraded to CPU-only inside
	 * resolve_gpu_config() because no GPU backend/device existed at
	 * runtime -- there is no GPU weight estimate to plan KV placement
	 * against, and there never will be on this host.
	 *
	 * --kv-placement auto's OWN documented contract (product_cli.cpp's
	 * --help text, unchanged by this phase) is unconditional: "Never
	 * fails: an all-CPU-KV plan is a valid auto outcome." That promise
	 * is not scoped to "only when auto was chosen by --auto's own
	 * default" -- an EXPLICIT `--kv-placement auto` makes the identical
	 * promise. So this bypass checks o.kv_placement itself (not
	 * o.want_kv_placement) -- it applies whether AUTO placement came
	 * from --auto's own default OR was typed explicitly, as long as
	 * the underlying reason is the same verified fact (no GPU exists at
	 * all on this host, not "an unverified estimate" -- see the
	 * distinct GGUF-metadata-unreadable case below, which this flag is
	 * never set for and must keep failing closed).
	 *
	 * --kv-placement gpu deliberately does NOT get this bypass, even
	 * though it is reachable with auto_cpu_fallback set too (--auto's
	 * own implicit gpu_layers=auto can still gracefully degrade under
	 * an explicit --kv-placement gpu, since gpu_layers itself was never
	 * explicit) -- gpu's own documented contract is the opposite of
	 * auto's: "every KV layer on the same GPU device ... or fail closed
	 * if it does not safely fit." An explicit request for guaranteed
	 * GPU residency must still fail honestly when there is no GPU,
	 * exactly as it already does for an explicit --gpu-layers request
	 * in the same situation (resolve_gpu_config() above). */
	if (o.kv_placement == MEMBRANE_KV_PLACEMENT_AUTO && gs->auto_cpu_fallback)
		return (true);
	/* Review fix: resolve_gpu_config()'s documented "GGUF metadata
	 * unreadable, explicit numeric --gpu-layers N, proceed unguarded"
	 * fallback (gs->policy_used stays false; see its own comment above)
	 * leaves gs->estimated_model_bytes at 0 -- fine for DEFAULT
	 * (already returned above) and for CPU (no GPU KV budget math at
	 * all, see membrane_kv_residency_resolve's CPU branch), but unsafe
	 * for GPU/AUTO: the planner would then believe the model's real
	 * weight bytes are zero and could overcommit GPU-resident KV.
	 * Fail closed instead of planning against an unverified estimate. */
	if ((o.kv_placement == MEMBRANE_KV_PLACEMENT_GPU
			|| o.kv_placement == MEMBRANE_KV_PLACEMENT_AUTO)
		&& !gs->policy_used)
	{
		fprintf(stderr, "membrane-run: --kv-placement gpu|auto requires "
			"a verified GPU weight-memory estimate, but none is "
			"available for this model/--gpu-layers combination -- try "
			"--kv-placement cpu or --gpu-layers all/auto\n");
		print_error_json(o, MEMBRANE_EXIT_UNSUPPORTED_KV,
			MEMBRANE_KV_PLACEMENT_REASON_INVALID_CONFIG,
			"--kv-placement gpu|auto requires a verified GPU "
			"weight-memory estimate, but none is available",
			"Try --kv-placement cpu, or --gpu-layers all/auto.");
		return (false);
	}
	total_kv_bytes = kv_bytes_for_mode(shape, ctx_size, effective_kv_mode);
	kv_bytes_per_layer = shape.n_layer > 0
		? total_kv_bytes / (uint64_t)shape.n_layer : 0;
	if (!membrane_kv_residency_resolve(o.kv_placement, shape.n_layer,
			kv_bytes_per_layer, gs->device_free_bytes,
			gs->device_total_bytes, gs->estimated_model_bytes,
			/* compute_buffer_estimate_bytes: */ 0, &gs->kv_placement))
	{
		fprintf(stderr, "membrane-run: %s\n", gs->kv_placement.reason);
		/* gs->kv_placement.reason is already "CODE: detail" (kv_
		 * residency_policy.c's own convention) -- extract just the
		 * code, same rule plan_primary_reason() above uses. */
		std::string	reason(gs->kv_placement.reason);
		size_t		colon = reason.find(": ");

		print_error_json(o, MEMBRANE_EXIT_UNSUPPORTED_KV,
			(colon == std::string::npos ? reason
				: reason.substr(0, colon)).c_str(), gs->kv_placement.reason,
			"Try --kv-placement cpu, reduce --ctx, or lower "
			"--gpu-layers.");
		return (false);
	}
	gs->kv_placement_resolved = true;
	/* Section 7: gs.kv_placement.reason is already "CODE" or "CODE:
	 * detail" (kv_residency_policy.c's convention) -- extract just the
	 * code, same rule plan_primary_reason() uses, so the trace only
	 * ever carries stable codes, never free text. */
	{
		std::string	reason(gs->kv_placement.reason);
		size_t		colon = reason.find(": ");

		gs->reason_trace.push_back(colon == std::string::npos ? reason
			: reason.substr(0, colon));
	}
	return (true);
}

/* Phase 13.2, Section 16: a small, deliberately non-exhaustive set of
 * informational plan warnings -- each one only pushed when it is both
 * true AND actionable (Section 16: "Do NOT spam warnings... Do not
 * create warnings without actionable meaning"). Called once, after
 * resolve_gpu_config()/resolve_cpu_adaptive_kv()/resolve_kv_placement()
 * have all already run, so every gs field consulted here is final.
 * Never affects exit_code/ok -- these are additive telemetry only. */
static void	build_plan_warnings(const membrane_run_opts_t &o,
				const membrane_host_meminfo_t &host, membrane_gpu_state_t *gs)
{
	if (gs->auto_cpu_fallback)
		gs->warnings.push_back(MEMBRANE_WARNING_CPU_FALLBACK);
	/* Section 13: a real, valid override -- not an error -- but worth
	 * a machine-readable note so a --json/--verbose consumer can tell
	 * "the user's explicit flag won over --auto's own default" apart
	 * from "every field came from --auto". One entry per overridden
	 * field, not a blanket "something was overridden" note, so a
	 * consumer knows exactly which. */
	if (o.auto_mode)
	{
		if (o.want_gpu_layers)
			gs->warnings.push_back(std::string(MEMBRANE_WARNING_EXPLICIT_OVERRIDE)
				+ ":gpu_layers");
		if (o.want_kv_mode)
			gs->warnings.push_back(std::string(MEMBRANE_WARNING_EXPLICIT_OVERRIDE)
				+ ":kv");
		if (o.want_kv_placement)
			gs->warnings.push_back(std::string(MEMBRANE_WARNING_EXPLICIT_OVERRIDE)
				+ ":kv_placement");
	}
	/* Section 15/9: gs.requested but no gpu_policy estimate was ever
	 * available (GGUF metadata unreadable, explicit numeric --gpu-
	 * layers N proceeding unguarded -- resolve_gpu_config()'s own
	 * documented fallback, see its "no usable estimate at all" branch)
	 * -- the plan below is real device/backend selection, but the
	 * memory numbers a --verbose/--json consumer would otherwise expect
	 * simply do not exist for this run. */
	if (gs->requested && !gs->policy_used)
		gs->warnings.push_back(MEMBRANE_WARNING_METADATA_ESTIMATE_ONLY);
	/* Section 16 example condition: an accepted GPU plan that leaves
	 * very little estimated headroom is worth flagging even though it
	 * was accepted -- 10% of the post-reserve budget, the same
	 * "estimate, not a guarantee" spirit as gpu_policy.h's own reserve.
	 * Never applies when nothing was actually estimated (policy_used)
	 * or the budget itself is 0 (already MEMORY_INSUFFICIENT territory,
	 * which fails closed before reaching here at all). */
	if (gs->policy_used)
	{
		uint64_t	used = gs->estimated_model_bytes + gs->estimated_kv_bytes;
		uint64_t	budget = gs->device_free_bytes > gs->safety_reserve_bytes
			? gs->device_free_bytes - gs->safety_reserve_bytes : 0;
		uint64_t	headroom = budget > used ? budget - used : 0;

		if (budget > 0 && headroom * 10 < budget)
			gs->warnings.push_back(MEMBRANE_WARNING_LOW_GPU_HEADROOM);
	}
	/* Section 17: observability only, and deliberately conservative --
	 * this never influenced any pass/fail decision above, it only
	 * flags a host-wide condition worth a human's attention when KV (or
	 * anything else) is about to compete for system RAM. host.ok is
	 * false (fields all 0) on any host where /proc/meminfo could not
	 * be read -- never warn from a value that was never actually
	 * observed. */
	if (host.ok && host.available_bytes < (uint64_t)512 * 1024 * 1024)
		gs->warnings.push_back(MEMBRANE_WARNING_HOST_MEMORY_PRESSURE);
}

/* Phase 13.1, Section 5/10: the single overall "why did the plan end
 * up this way" code for the plan-summary/JSON output -- gs.plan_reason_
 * code (gpu_policy's own layer-selection reason) when a GPU was
 * requested at all, else gs.kv_placement.reason's own stable-code
 * prefix (already "CODE" or "CODE: detail" -- see kv_residency_
 * policy.h) when placement resolved something, else the fixed
 * default-behavior code for a plain, nothing-automatic run. Never
 * empty. */
static std::string	plan_primary_reason(const membrane_run_opts_t &o,
						const membrane_gpu_state_t &gs)
{
	if (!gs.plan_reason_code.empty())
		return (gs.plan_reason_code);
	if (gs.kv_placement_resolved)
	{
		std::string	reason(gs.kv_placement.reason);
		size_t		colon = reason.find(": ");

		return (colon == std::string::npos ? reason : reason.substr(0, colon));
	}
	if (o.kv_placement != MEMBRANE_KV_PLACEMENT_DEFAULT)
		return (MEMBRANE_KV_PLACEMENT_REASON_DEFAULT_UNCHANGED);
	return (MEMBRANE_REASON_DEFAULT_BEHAVIOR_PRESERVED);
}

/* Phase 13.1, Section 10: one short, product-facing block summarizing
 * the FINAL resolved plan at a glance -- gated on "was any MEMBRANE
 * policy actually active" (--auto, --gpu-layers, --kv-placement, --kv
 * adaptive) so a plain, nothing-requested run prints nothing extra
 * here, matching this file's existing "no behavior change means no
 * new noise" convention. The more detailed lines print_startup_
 * summary() already prints above remain the full record; this block
 * is the one meant to be read at a glance, and is also exactly what
 * print_run_json()'s "plan" object below mirrors for --json. */
static void	print_plan_summary(const membrane_run_opts_t &o,
				int32_t n_layer_total, const membrane_gpu_state_t &gs)
{
	bool	policy_active = o.auto_mode || gs.requested
		|| o.kv_placement != MEMBRANE_KV_PLACEMENT_DEFAULT || gs.adaptive_used;

	if (!policy_active)
		return ;
	fprintf(stderr, "MEMBRANE plan\n");
	fprintf(stderr, "  device: %s\n",
		gs.requested && !gs.device_selected.empty()
			? gs.device_selected.c_str() : "CPU");
	/* Review fix (CodeRabbit, PR #22): print_run_json() already reports
	 * this via "auto":{"cpu_fallback":...} -- the human summary had no
	 * equivalent, so a fallback run printed identically to an ordinary
	 * CPU-only run with only the reason code (NO_GPU_DEVICE) as a clue. */
	if (gs.auto_cpu_fallback)
		fprintf(stderr, "  auto: no GPU available, resolved to CPU-only\n");
	if (gs.requested)
		fprintf(stderr, "  gpu layers: %d/%d\n", gs.gpu_layers_selected,
			n_layer_total);
	fprintf(stderr, "  kv precision: %s\n", kv_type_label(o.kv_mode));
	fprintf(stderr, "  kv placement: %s\n",
		membrane_kv_placement_mode_name(o.kv_placement));
	if (gs.kv_placement_resolved)
		fprintf(stderr, "  kv layers: %d GPU / %d CPU\n",
			gs.kv_placement.gpu_kv_layers, gs.kv_placement.cpu_kv_layers);
	if (gs.policy_used)
		fprintf(stderr, "  estimated weight bytes: %.1f MiB\n",
			(double)gs.estimated_model_bytes / (1024.0 * 1024.0));
	if (gs.kv_placement_resolved)
		fprintf(stderr, "  estimated GPU KV: %.1f MiB, estimated CPU KV: "
			"%.1f MiB\n",
			(double)gs.kv_placement.gpu_kv_bytes / (1024.0 * 1024.0),
			(double)gs.kv_placement.cpu_kv_bytes / (1024.0 * 1024.0));
	else if (gs.policy_used)
		fprintf(stderr, "  estimated KV bytes: %.1f MiB\n",
			(double)gs.estimated_kv_bytes / (1024.0 * 1024.0));
	fprintf(stderr, "  reason: %s\n", plan_primary_reason(o, gs).c_str());
}

/* Phase 13.2, Section 5: the detailed requested/resolved/memory/policy
 * breakdown --verbose adds on top of print_plan_summary()'s concise
 * block above. Same "only print what actually applies" gating as the
 * concise block (Section 4: normal output stays compact; this is the
 * opt-in detail layer, not a replacement for it -- both print when
 * policy_active and --verbose). Every memory figure here is explicitly
 * labelled ESTIMATED/SNAPSHOT (Section 8: never claim measured peak
 * VRAM or a safety guarantee this project cannot make). */
static void	print_verbose_plan(const membrane_run_opts_t &o,
				int32_t n_layer_total, const membrane_gpu_state_t &gs,
				const membrane_host_meminfo_t &host)
{
	bool	policy_active = o.auto_mode || gs.requested
		|| o.kv_placement != MEMBRANE_KV_PLACEMENT_DEFAULT || gs.adaptive_used;
	size_t	i;

	if (!policy_active)
		return ;
	fprintf(stderr, "MEMBRANE verbose plan\n");
	fprintf(stderr, "requested:\n");
	fprintf(stderr, "  gpu layers: %s\n",
		gpu_layers_label(gs.gpu_layers_requested).c_str());
	/* requested_kv_name() (below, print_gpu_json's neighbor) isn't
	 * declared yet at this point in the file -- same logic inlined,
	 * one place fewer to keep in sync would be nice, but a forward
	 * declaration purely to avoid two identical lines is not worth it
	 * here. */
	fprintf(stderr, "  kv precision: %s\n",
		gs.adaptive_used ? "adaptive" : kv_mode_name(o.kv_mode));
	fprintf(stderr, "  kv placement: %s\n",
		membrane_kv_placement_mode_name(o.kv_placement));
	fprintf(stderr, "  device: %s\n",
		gs.device_requested.empty() ? "(any)" : gs.device_requested.c_str());
	fprintf(stderr, "resolved:\n");
	fprintf(stderr, "  backend: %s\n", gs.backend_selected.c_str());
	fprintf(stderr, "  device: %s\n",
		gs.device_selected.empty() ? "(none)" : gs.device_selected.c_str());
	fprintf(stderr, "  model layers: %d\n", n_layer_total);
	if (gs.requested)
	{
		/* Review fix (CodeRabbit, PR #23): resolve_gpu_config()'s "no
		 * usable estimate at all" fallback (GGUF metadata unreadable,
		 * only reachable for an explicit --gpu-layers all/N) leaves
		 * gs.gpu_layers_selected at the raw MEMBRANE_GPU_LAYERS_ALL
		 * sentinel (-1) rather than a resolved count for the "all"
		 * case -- printing that raw would show "gpu layers: -1" and
		 * an arithmetically nonsensical "cpu layers: n_layer_total+1".
		 * -1 means "every layer" by definition, so resolve it to
		 * n_layer_total for display only (matches the true selection,
		 * doesn't change gs itself). An explicit --gpu-layers N always
		 * clamps to a non-negative selected_layers elsewhere, so N is
		 * never negative here. */
		int32_t	gpu_layers_display = gs.gpu_layers_selected < 0
			? n_layer_total : gs.gpu_layers_selected;

		fprintf(stderr, "  gpu layers: %d\n", gpu_layers_display);
		fprintf(stderr, "  cpu layers: %d\n",
			n_layer_total - gpu_layers_display);
	}
	fprintf(stderr, "  kv type: %s\n", kv_type_label(o.kv_mode));
	if (gs.kv_placement_resolved)
	{
		fprintf(stderr, "  gpu kv layers: %d\n", gs.kv_placement.gpu_kv_layers);
		fprintf(stderr, "  cpu kv layers: %d\n", gs.kv_placement.cpu_kv_layers);
	}
	fprintf(stderr, "memory (ESTIMATED unless noted SNAPSHOT):\n");
	if (gs.requested)
		fprintf(stderr, "  gpu free (SNAPSHOT): %.1f MiB\n",
			(double)gs.device_free_bytes / (1024.0 * 1024.0));
	if (gs.policy_used)
	{
		uint64_t	used = gs.estimated_model_bytes + gs.estimated_kv_bytes;
		uint64_t	budget = gs.device_free_bytes > gs.safety_reserve_bytes
			? gs.device_free_bytes - gs.safety_reserve_bytes : 0;
		uint64_t	remaining = budget > used ? budget - used : 0;

		fprintf(stderr, "  safety reserve: %.1f MiB\n",
			(double)gs.safety_reserve_bytes / (1024.0 * 1024.0));
		fprintf(stderr, "  estimated weight bytes: %.1f MiB\n",
			(double)gs.estimated_model_bytes / (1024.0 * 1024.0));
		fprintf(stderr, "  remaining budget: %.1f MiB\n",
			(double)remaining / (1024.0 * 1024.0));
	}
	if (gs.kv_placement_resolved)
		fprintf(stderr, "  estimated gpu kv: %.1f MiB, estimated cpu kv: "
			"%.1f MiB\n",
			(double)gs.kv_placement.gpu_kv_bytes / (1024.0 * 1024.0),
			(double)gs.kv_placement.cpu_kv_bytes / (1024.0 * 1024.0));
	else if (gs.policy_used)
		fprintf(stderr, "  estimated kv bytes: %.1f MiB\n",
			(double)gs.estimated_kv_bytes / (1024.0 * 1024.0));
	if (host.ok)
		fprintf(stderr, "  host available (SNAPSHOT): %.1f MiB / %.1f MiB "
			"total, swap free (SNAPSHOT): %.1f MiB\n",
			(double)host.available_bytes / (1024.0 * 1024.0),
			(double)host.total_bytes / (1024.0 * 1024.0),
			(double)host.swap_free_bytes / (1024.0 * 1024.0));
	fprintf(stderr, "policy:\n");
	fprintf(stderr, "  primary reason: %s\n", plan_primary_reason(o, gs).c_str());
	if (!gs.reason_trace.empty())
	{
		fprintf(stderr, "  reason trace:");
		for (i = 0; i < gs.reason_trace.size(); ++i)
			fprintf(stderr, "%s%s", i == 0 ? " " : " -> ",
				gs.reason_trace[i].c_str());
		fprintf(stderr, "\n");
	}
	fprintf(stderr, "  auto fallback: %s\n",
		gs.auto_cpu_fallback ? "yes" : "no");
	if (gs.adaptive_used)
		fprintf(stderr, "  adaptive reason: %s\n", gs.adaptive_reason.c_str());
	if (!gs.warnings.empty())
	{
		fprintf(stderr, "  warnings:");
		for (i = 0; i < gs.warnings.size(); ++i)
			fprintf(stderr, "%s%s", i == 0 ? " " : ", ",
				gs.warnings[i].c_str());
		fprintf(stderr, "\n");
	}
}

static void	print_startup_summary(const membrane_run_opts_t &o,
				const char *model_label, uint32_t ctx_size,
				uint64_t kv_bytes, int32_t n_layer_total,
				const membrane_gpu_state_t &gs,
				const membrane_host_meminfo_t &host)
{
	fprintf(stderr, "MEMBRANE %s\n", MEMBRANE_VERSION);
	fprintf(stderr, "model      %s\n", model_label);
	fprintf(stderr, "context    %u\n", ctx_size);
	fprintf(stderr, "kv         %s\n",
		o.kv_mode == MEMBRANE_KV_STORE_NATIVE ? "native (F16)"
			: kv_type_label(o.kv_mode));
	/* Section 1: "Human output must clearly show the same decision" --
	 * o here is already resolved (main.cpp's effective_o), so the "kv"
	 * line above always shows what actually runs; this line answers
	 * the separate question of whether that came from an explicit
	 * request or --kv adaptive's own policy, and why. */
	if (gs.adaptive_used)
		fprintf(stderr, "kv adaptive requested=adaptive selected=%s "
			"reason=%s\n", kv_mode_name(gs.adaptive_selected_mode),
			gs.adaptive_reason.c_str());
	/* run_kv_store_pass() (decode_loop.cpp) forces flash attention
	 * ENABLED unconditionally for both native and q8 -- not just q8 --
	 * so a --compare-kv run never mixes "KV cache dtype" with "which
	 * attention kernel ran" as a second, uncontrolled variable. The
	 * summary must report what actually runs, not a mode-dependent
	 * guess. */
	fprintf(stderr, "flash attn enabled\n");
	fprintf(stderr, "kv bytes   %.2f MiB (%s, from real model "
		"hparams -- not measured until context creation)\n",
		(double)kv_bytes / (1024.0 * 1024.0), kv_type_label(o.kv_mode));
	/* Phase 9B/9B.1: backend/device is a REQUEST membrane-run made via
	 * public ggml_backend_dev_*() enumeration, not a confirmed
	 * post-load allocation (no public API exposes that without log-
	 * scraping) -- worded accordingly, never "confirmed"/"placed". */
	if (!gs.requested)
		fprintf(stderr, "backend    CPU (default -- pass --gpu-layers "
			"to use a GPU)\n");
	else
		fprintf(stderr, "backend    %s, device selected: %s "
			"(gpu-layers=%s, selected=%d)\n", gs.backend_selected.c_str(),
			gs.device_selected.c_str(),
			gpu_layers_label(gs.gpu_layers_requested).c_str(),
			gs.gpu_layers_selected);
	/* Phase 13.1, Section 10: replaces the old separate "gpu policy
	 * ..."/"KV placement" blocks (Phase 9B.1/12H) with one consolidated
	 * plan summary -- same information, one place, always ending in a
	 * single stable reason code. */
	print_plan_summary(o, n_layer_total, gs);
	if (o.verbose)
		print_verbose_plan(o, n_layer_total, gs, host);
}

typedef struct s_token_print_ud
{
	bool	first;
}	token_print_ud_t;

static void	stream_token(const char *piece, size_t piece_len, int step,
				void *ud)
{
	(void)step;
	(void)ud;
	fwrite(piece, 1, piece_len, stdout);
	fflush(stdout);
}

/* RFC 8259 requires every control character (< 0x20) inside a JSON
 * string to be escaped -- a raw tab or newline in generated text
 * previously either passed through unescaped (invalid JSON) or was
 * silently dropped with no replacement (corrupting --include-text's
 * output relative to what the model actually generated). Both are
 * real, reachable bugs: models produce tabs and newlines routinely. */
static void	print_json_escaped(const std::string &text)
{
	for (unsigned char c : text)
	{
		if (c == '"' || c == '\\')
		{
			putchar('\\');
			putchar(c);
		}
		else if (c == '\n')
			fputs("\\n", stdout);
		else if (c == '\r')
			fputs("\\r", stdout);
		else if (c == '\t')
			fputs("\\t", stdout);
		else if (c < 0x20)
			printf("\\u%04x", c);
		else
			putchar(c);
	}
}

/* Phase 13.1, Section 3/13: the structured JSON error object --
 * forward-declared above resolve_gpu_config(). Only ever called when
 * o.want_json is set (checked by the caller, not here, so a plain
 * stderr-text run never pays for or risks a stray stdout write); emits
 * exactly one JSON object to stdout and nothing else, matching
 * print_run_json()'s own "stdout is JSON only" contract (Section 3:
 * "In JSON mode: stdout must contain valid JSON only; diagnostic logs
 * must go to stderr" -- the human-readable message the caller ALSO
 * printed to stderr via fprintf is unaffected, unrelated stream). Not
 * every failure path in this file calls this yet (Section 4 scopes
 * this to the important, listed error paths) -- an --json run that
 * fails on an uncovered path still exits with the correct nonzero
 * code and a stderr message, just without a matching stdout object;
 * covered paths are the ones a --json caller is most likely to need
 * to distinguish programmatically (model/metadata/GPU/device
 * failures), not every CLI-parse-time misuse. */
static void	print_error_json(const membrane_run_opts_t &o, int exit_code,
				const char *reason_code, const std::string &message,
				const char *suggestion, const std::vector<std::string> *available_devices,
				const membrane_fallback_trace_t *fallback_trace)
{
	if (!o.want_json)
		return ;
	/* Review fix (CodeRabbit, PR #22): reason_code is escaped too, not
	 * just message -- every current caller passes a bare stable-code
	 * constant (safe by construction: no quotes/backslashes/control
	 * chars), but this function's own JSON-validity contract must not
	 * silently depend on that always staying true for every future
	 * caller. */
	printf("{\"schema_version\":1,\"membrane_version\":\"%s\","
		"\"mode\":\"run\",\"ok\":false,\"exit_code\":%d,\"error\":{"
		"\"reason_code\":\"", MEMBRANE_VERSION, exit_code);
	print_json_escaped(reason_code);
	printf("\",\"message\":\"");
	print_json_escaped(message);
	printf("\"");
	if (suggestion != NULL)
	{
		printf(",\"suggestion\":\"");
		print_json_escaped(suggestion);
		printf("\"");
	}
	if (available_devices != NULL && !available_devices->empty())
	{
		size_t	i;

		printf(",\"available_devices\":[");
		for (i = 0; i < available_devices->size(); ++i)
		{
			if (i > 0)
				printf(",");
			printf("\"");
			print_json_escaped((*available_devices)[i]);
			printf("\"");
		}
		printf("]");
	}
	printf("}");
	/* Phase 25 (OPT-06): additive-only, top-level sibling of "error" --
	 * same key/shape the success schema's print_run_json() already
	 * emits via print_fallback_trace_json(), so a caller does not need
	 * two different parsers for "was there a fallback trace" depending
	 * on ok:true vs ok:false. Only ever passed non-NULL by the one real
	 * AUTO_FALLBACK_EXHAUSTED call site (run_normal_mode(), after
	 * membrane_fallback_run() actually ran) -- every other call site
	 * still passes NULL (unchanged from before this change) and this
	 * field is omitted entirely, not emitted empty. */
	if (fallback_trace != NULL)
	{
		printf(",");
		print_fallback_trace_json(*fallback_trace, false);
	}
	printf("}\n");
}

/* Phase 11A: every print_*() site below receives `o` already resolved
 * to a concrete storage mode (main.cpp's `effective_o` -- see
 * resolve_cpu_adaptive_kv()'s and resolve_gpu_config()'s header
 * comments), never the raw ADAPTIVE request value, so kv_mode_name()/
 * kv_type_label() above are always safe to call on it directly. The
 * ORIGINAL request ("was this --kv adaptive at all?") is instead
 * carried by gs.adaptive_used/adaptive_selected_mode/adaptive_reason
 * -- this helper renders the "requested_kv" field consistently from
 * that, one place, for every print_*() function. */
static const char	*requested_kv_name(const membrane_run_opts_t &o,
					const membrane_gpu_state_t &gs)
{
	if (gs.adaptive_used)
		return ("adaptive");
	return (kv_mode_name(o.kv_mode));
}

/* Phase 19: a SECOND, post-run query of the exact same GPU device
 * gpu_policy's pre-load estimate was checked against -- reusing
 * membrane_gpu_list_devices() (never a new/different API), so this is
 * the same driver-reported free-heap-bytes figure as gs.device_free_
 * bytes, just re-read after run_kv_store_pass() (decode_loop.cpp)
 * returns. CodeRabbit review (PR #29) caught what this call site
 * actually observes: run_kv_store_pass() calls llama_free(ctx) --
 * destroying the KV cache/context -- before it returns, so by the
 * time this runs, the KV cache is already gone; only the model
 * WEIGHTS (freed separately, later, by run_normal_mode()'s caller)
 * are still GPU-resident. This must be compared against
 * estimated_model_bytes ALONE, never + estimated_kv_bytes -- see
 * docs/planner-accuracy.md. Deliberately NOT a peak (nothing here
 * samples memory continuously during decode, and it structurally
 * cannot include the KV cache at all) and deliberately NOT claimed to
 * be exact -- the same driver-reported-budget caveats that apply to
 * the pre-load figure apply here too, and this single post-run read
 * can be diluted by any other process on the same device between the
 * two reads. */
typedef struct s_membrane_gpu_memory_observed
{
	bool		available;
	uint64_t	device_free_bytes_after;
}	membrane_gpu_memory_observed_t;

static void	observe_gpu_memory_after_run(const membrane_gpu_state_t &gs,
				membrane_gpu_memory_observed_t *out)
{
	membrane_gpu_device_info_t	devices[MEMBRANE_GPU_MAX_DEVICES];
	size_t						n;
	size_t						i;

	out->available = false;
	out->device_free_bytes_after = 0;
	if (!gs.requested || gs.device_selected.empty())
		return ;
	n = membrane_gpu_list_devices(devices, MEMBRANE_GPU_MAX_DEVICES);
	for (i = 0; i < n; ++i)
	{
		if (gs.device_selected == devices[i].name)
		{
			out->available = true;
			out->device_free_bytes_after = devices[i].memory_free;
			return ;
		}
	}
}

static void	print_gpu_memory_observed_json(const membrane_gpu_state_t &gs,
				const membrane_gpu_memory_observed_t &obs)
{
	printf("\"gpu_memory_observed\":{\"available\":%s",
		obs.available ? "true" : "false");
	if (obs.available)
	{
		int64_t	delta = (int64_t)gs.device_free_bytes
			- (int64_t)obs.device_free_bytes_after;

		printf(",\"measurement_method\":\"driver-reported free heap "
			"bytes via ggml_backend_dev_get_props(), re-queried after "
			"run_kv_store_pass() returns -- that function already calls "
			"llama_free(ctx) internally before returning, so this "
			"reflects GPU memory attributable to the model WEIGHTS "
			"ONLY; the KV cache/context is already destroyed by this "
			"point. Compare against estimated_model_bytes alone, never "
			"+ estimated_kv_bytes. Not a peak, not a guaranteed live "
			"figure, same caveats as the pre-load estimate this is "
			"compared against\","
			"\"device_free_bytes_before\":%llu,"
			"\"device_free_bytes_after\":%llu,"
			"\"observed_delta_bytes\":%lld",
			(unsigned long long)gs.device_free_bytes,
			(unsigned long long)obs.device_free_bytes_after,
			(long long)delta);
	}
	printf("},");
}

/* Phase 20: additive-only diagnostics for the joint planner --
 * present only when it actually ran (see membrane_gpu_state_t's own
 * joint_planner_used doc comment for the cases where it does not,
 * e.g. --compare-kv/--gpu-bench). Deliberately minimal (Section 20:
 * "Do not dump giant traces") -- the full candidate list stays
 * internal to this run's own membrane_joint_plan_resolve() call, not
 * serialized here. */
static void	print_joint_planner_json(const membrane_gpu_state_t &gs)
{
	printf("\"planner\":{\"used\":%s",
		gs.joint_planner_used ? "true" : "false");
	if (gs.joint_planner_used)
	{
		printf(",\"policy_version\":\"%s\",\"candidate_count\":%d,"
			"\"selected_index\":%d,\"selection_reason\":\"",
			MEMBRANE_JOINT_POLICY_VERSION, gs.joint_candidate_count,
			gs.joint_selected_index);
		print_json_escaped(gs.plan_reason_code);
		printf("\"");
	}
	printf("},");
}

/* Phase 21, Section 14: additive-only JSON for the apply-time fallback
 * controller. Every run emits this object (never omitted -- Section
 * 16's --plan-only rule is the one exception, handled by its own
 * caller passing engaged=false unconditionally): a run that never
 * engaged the controller at all (--compare-kv/--gpu-bench, a CPU-only
 * run, or one where no Phase-20 candidate array exists to iterate --
 * Section 24/25) reports the trivial, zero-behavior-change shape
 * {"attempted":false,"attempt_count":1,"final_status":"success"|
 * "error"} reflecting the ONE real outcome that already happened,
 * never a fabricated candidate iteration. "attempts" (the detailed
 * per-candidate array) is included only when the controller actually
 * ran (engaged=true) -- Section 13: keep the trace bounded and never
 * emit it for a run that has none. */
/* Phase 25 (OPT-06): the actual "attempts" array/attempted/final_status/
 * reason_code emission, factored out of print_fallback_json() below so
 * print_error_json() can emit the SAME real trace on a fully-exhausted
 * run -- previously only the success schema ever called this logic, so
 * an AUTO_FALLBACK_EXHAUSTED failure's JSON carried no "fallback" object
 * at all (a real gap found during Phase 24's own OPT-05 investigation,
 * see docs/performance-profiling.md). No field here is new; this is a
 * pure extraction, byte-identical output to what the success path
 * already printed before this change. */
static void	print_fallback_trace_json(const membrane_fallback_trace_t &ft,
				bool trailing_comma)
{
	int	i;

	printf("\"fallback\":{\"attempted\":%s,\"attempt_count\":%d,"
		"\"initial_candidate_index\":%d,\"final_candidate_index\":%d,"
		"\"final_status\":\"", ft.attempted ? "true" : "false",
		ft.attempt_count, ft.initial_candidate_index,
		ft.final_candidate_index);
	print_json_escaped(ft.final_status);
	printf("\",\"reason_code\":\"");
	print_json_escaped(ft.reason_code);
	printf("\",\"attempts\":[");
	for (i = 0; i < ft.n_entries; ++i)
	{
		const membrane_fallback_attempt_t	&a = ft.entries[i];

		printf("%s{\"candidate_index\":%d,\"gpu_layers\":%d,"
			"\"kv_precision\":%d,\"kv_placement\":%d,\"refreshed\":%s",
			i == 0 ? "" : ",", a.candidate_index, a.gpu_layers,
			a.kv_precision, a.kv_placement, a.refreshed ? "true" : "false");
		if (a.refreshed)
			printf(",\"refreshed_available_gpu_bytes\":%llu",
				(unsigned long long)a.refreshed_available_gpu_bytes);
		printf(",\"fit_after_refresh\":%s,\"apply_started\":%s",
			a.fit_after_refresh ? "true" : "false",
			a.apply_started ? "true" : "false");
		if (a.apply_started)
		{
			printf(",\"apply_ok\":%s", a.apply_ok ? "true" : "false");
			if (!a.apply_ok)
			{
				printf(",\"failure_class\":\"%s\",\"detail\":\"",
					membrane_apply_failure_class_name(a.failure_class));
				print_json_escaped(a.detail);
				printf("\"");
			}
			printf(",\"cleanup_complete\":%s,\"model_reload_required\":%s,"
				"\"apply_wall_ms\":%.3f",
				a.cleanup_complete ? "true" : "false",
				a.model_reload_required ? "true" : "false", a.apply_wall_ms);
		}
		printf("}");
	}
	printf("]}%s", trailing_comma ? "," : "");
}

static void	print_fallback_json(const membrane_gpu_state_t &gs,
				bool engaged, bool ran_at_all, bool overall_ok)
{
	if (!ran_at_all)
	{
		printf("\"fallback\":{\"attempted\":false,\"attempt_count\":0,"
			"\"final_status\":\"not_applicable\"},");
		return ;
	}
	if (!engaged)
	{
		printf("\"fallback\":{\"attempted\":false,\"attempt_count\":1,"
			"\"final_status\":\"%s\"},", overall_ok ? "success" : "error");
		return ;
	}
	print_fallback_trace_json(gs.fallback_trace, true);
}

static void	print_gpu_json(const membrane_gpu_state_t &gs)
{
	printf("\"gpu\":{\"requested\":%s,\"backend_gpu_capable\":%s,",
		gs.requested ? "true" : "false",
		gs.backend_gpu_capable ? "true" : "false");
	printf("\"gpu_layers_requested\":%d,\"gpu_layers_selected\":%d,"
		"\"backend\":\"", gs.gpu_layers_requested, gs.gpu_layers_selected);
	/* Every string field here is escaped, even though backend/
	 * device_selected come from the ggml backend registry rather than
	 * directly from user input (membrane-run doesn't control their
	 * exact contents either) -- device_requested in particular IS raw
	 * --device user input and must not be interpolated unescaped into
	 * JSON (same class of issue print_json_escaped's own "text" field
	 * use already guards against). */
	print_json_escaped(gs.backend_selected);
	printf("\",\"device_requested\":\"");
	print_json_escaped(gs.device_requested);
	printf("\",\"device_selected\":\"");
	print_json_escaped(gs.device_selected);
	printf("\"},");
	if (gs.policy_used)
	{
		/* Section 8: "Record: free memory, reserve, estimated usage,
		 * headroom" -- headroom is the estimated slack left inside the
		 * post-reserve budget after weights+KV, floored at 0 rather
		 * than allowed to go negative when an explicit (non-auto)
		 * request was still accepted right at the edge. */
		uint64_t	used = gs.estimated_model_bytes + gs.estimated_kv_bytes;
		uint64_t	budget = gs.device_free_bytes > gs.safety_reserve_bytes
			? gs.device_free_bytes - gs.safety_reserve_bytes : 0;
		uint64_t	headroom = budget > used ? budget - used : 0;

		printf("\"gpu_policy\":{\"device_total_bytes\":%llu,"
			"\"device_free_bytes\":%llu,\"safety_reserve_bytes\":%llu,"
			"\"estimated_model_bytes\":%llu,\"estimated_kv_bytes\":%llu,"
			"\"headroom_bytes\":%llu},",
			(unsigned long long)gs.device_total_bytes,
			(unsigned long long)gs.device_free_bytes,
			(unsigned long long)gs.safety_reserve_bytes,
			(unsigned long long)gs.estimated_model_bytes,
			(unsigned long long)gs.estimated_kv_bytes,
			(unsigned long long)headroom);
	}
	if (gs.adaptive_used)
	{
		/* Section 4/21: the two already-evaluated candidates behind
		 * the adaptive_reason decision -- selected_layers is 0/unused
		 * for a CPU-resolved decision (see resolve_cpu_adaptive_kv()). */
		printf("\"adaptive\":{\"selected_kv\":\"%s\",\"reason\":\"",
			kv_mode_name(gs.adaptive_selected_mode));
		print_json_escaped(gs.adaptive_reason);
		printf("\",\"candidates\":{"
			"\"q8\":{\"kv_bytes\":%llu,\"valid\":%s,\"selected_layers\":%d},"
			"\"q5\":{\"kv_bytes\":%llu,\"valid\":%s,\"selected_layers\":%d}"
			"}},",
			(unsigned long long)gs.adaptive_q8_kv_bytes,
			gs.adaptive_q8_valid ? "true" : "false", gs.adaptive_q8_layers,
			(unsigned long long)gs.adaptive_q5_kv_bytes,
			gs.adaptive_q5_valid ? "true" : "false", gs.adaptive_q5_layers);
	}
}

/* Phase 12H, Section 17: additive-only -- callers parsing existing
 * JSON output are unaffected (kv_placement_mode is always present and
 * is "default" whenever nothing changed; the rest of the object only
 * appears once a real plan was resolved). */
static void	print_kv_placement_json(const membrane_run_opts_t &o,
				const membrane_gpu_state_t &gs)
{
	printf("\"kv_placement\":{\"kv_placement_mode\":\"%s\"",
		membrane_kv_placement_mode_name(o.kv_placement));
	if (gs.kv_placement_resolved)
	{
		printf(",\"kv_gpu_layers\":%d,\"kv_cpu_layers\":%d,"
			"\"kv_gpu_bytes\":%llu,\"kv_cpu_bytes\":%llu,"
			"\"kv_total_bytes\":%llu,\"placement_reason\":\"",
			gs.kv_placement.gpu_kv_layers, gs.kv_placement.cpu_kv_layers,
			(unsigned long long)gs.kv_placement.gpu_kv_bytes,
			(unsigned long long)gs.kv_placement.cpu_kv_bytes,
			(unsigned long long)gs.kv_placement.total_kv_bytes);
		print_json_escaped(gs.kv_placement.reason);
		printf("\"");
	}
	printf("},");
}

static void	print_json_string_array(const std::vector<std::string> &arr)
{
	size_t	i;

	printf("[");
	for (i = 0; i < arr.size(); ++i)
	{
		if (i > 0)
			printf(",");
		printf("\"");
		print_json_escaped(arr[i]);
		printf("\"");
	}
	printf("]");
}

/* Phase 24, Section 24: additive-only "timings" JSON object, shared by
 * print_run_json() (tel non-NULL: every stage measured) and
 * print_plan_only_json() (tel NULL: plan-only never creates a context
 * or generates, so context_create_ms/prefill_ms/decode_ms/
 * first_token_ms/throughput are genuinely inapplicable -- JSON null,
 * not a fabricated 0, matching this project's established "null +
 * reason" convention: the reason here is simply mode:"plan" itself,
 * already visible one level up). first_token_ms is null (not -1) in
 * JSON specifically when no token was generated -- distinct from a
 * real 0.0 that could never actually happen for a real decode step.
 *
 * Phase 25 (OPT-03): "planner_stages" is always present (never gated on
 * gs.policy_used) -- each of its three sub-costs independently defaults
 * to 0.0 whenever its own resolve_gpu_config() call site didn't run on
 * this invocation (e.g. explicit CPU-only never enumerates a device at
 * all), so a plain, un-conditional numeric 0.0 is the correct value
 * there, not a fabricated placeholder. device_enumeration_ms +
 * gguf_prescan_ms + joint_planner_core_ms is expected to sum to
 * approximately planner_ms (Section 12 of the Phase 25 task) -- this is
 * an attribution/correctness check on the new instrumentation itself,
 * not a performance claim. */
static void	print_timings_json(const membrane_timings_t &timings,
				const membrane_kv_store_telemetry_t *tel,
				const membrane_gpu_state_t &gs)
{
	printf("\"timings\":{\"total_ms\":%.3f,\"planner_ms\":%.3f,"
		"\"planner_stages\":{\"device_enumeration_ms\":%.3f,"
		"\"gguf_prescan_ms\":%.3f,\"joint_planner_core_ms\":%.3f},"
		"\"model_load_ms\":%.3f,\"tokenization_ms\":%.3f,",
		timings.total_ms, timings.planner_ms,
		gs.device_enumeration_ms, gs.gguf_prescan_ms,
		gs.joint_planner_core_ms, timings.model_load_ms,
		timings.tokenization_ms);
	if (tel == NULL)
	{
		printf("\"context_create_ms\":null,\"prefill_ms\":null,"
			"\"decode_ms\":null,\"first_token_ms\":null,"
			"\"throughput\":{\"prefill_tokens_per_second\":null,"
			"\"decode_tokens_per_second\":null}},");
		return ;
	}
	printf("\"context_create_ms\":%.3f,\"prefill_ms\":%.3f,"
		"\"decode_ms\":%.3f,\"first_token_ms\":",
		tel->context_create_ms, tel->prompt_ms, tel->generation_ms);
	if (tel->first_token_ms >= 0.0)
		printf("%.3f", tel->first_token_ms);
	else
		printf("null");
	printf(",\"throughput\":{\"prefill_tokens_per_second\":%.6f,"
		"\"decode_tokens_per_second\":%.6f}},",
		tel->prompt_tok_per_s, tel->generation_tok_per_s);
}

/* Phase 13.2, Section 6/9/17: additive-only JSON, same contract as
 * every other print_*_json() function in this file (Section 18: never
 * rename/remove an existing Phase 13.1 field, only add new ones) --
 * host/success/exit_code/prompt_tokens_count are new inputs purely to
 * populate the new "host_memory"/"execution" objects below; every
 * existing field/argument stays byte-for-byte what it already was. */
static void	print_run_json(const membrane_run_opts_t &o,
				const char *model_label, const membrane_kv_store_telemetry_t &t,
				const std::string &text, const membrane_gpu_state_t &gs,
				const membrane_host_meminfo_t &host, bool success,
				int exit_code, size_t prompt_tokens_count,
				const membrane_gpu_memory_observed_t &gpu_mem_observed,
				const membrane_timings_t &timings)
{
	printf("{\"schema_version\":1,\"membrane_version\":\"%s\","
		"\"mode\":\"run\",\"ok\":true,\"model_label\":\"%s\","
		"\"kv_store\":\"%s\","
		"\"kv_type\":\"%s\",\"requested_kv\":\"%s\",\"selected_kv\":\"%s\","
		"\"adaptive_reason\":\"%s\","
		"\"ctx_size\":%u,\"generated_tokens\":%llu,",
		MEMBRANE_VERSION, model_label, kv_mode_name(o.kv_mode),
		kv_type_label(o.kv_mode), requested_kv_name(o, gs),
		kv_mode_name(o.kv_mode),
		gs.adaptive_used ? gs.adaptive_reason.c_str() : "",
		t.ctx_size, (unsigned long long)t.generated_tokens);
	/* Phase 13.1, Section 3/5: --auto's own resolution state, plus the
	 * single overall plan reason code (same value print_plan_summary()
	 * prints as its own "reason:" line) -- additive-only, existing
	 * fields/order above are untouched. */
	printf("\"auto\":{\"requested\":%s,\"cpu_fallback\":%s},"
		"\"reason_code\":\"", o.auto_mode ? "true" : "false",
		gs.auto_cpu_fallback ? "true" : "false");
	print_json_escaped(plan_primary_reason(o, gs));
	printf("\",");
	printf("\"storage\":{\"kv_allocated_bytes\":%llu},",
		(unsigned long long)t.compressed_kv_allocated_bytes);
	print_gpu_json(gs);
	print_gpu_memory_observed_json(gs, gpu_mem_observed);
	print_joint_planner_json(gs);
	print_fallback_json(gs, gs.fallback_engaged, true, success);
	print_timings_json(timings, &t, gs);
	print_kv_placement_json(o, gs);
	printf("\"memory\":{\"rss_after_model_load_kb\":%llu,"
		"\"rss_after_context_kb\":%llu,\"rss_after_prompt_kb\":%llu,"
		"\"rss_final_kb\":%llu,\"peak_rss_kb\":%llu},",
		(unsigned long long)t.rss_after_model_load.vm_rss_kb,
		(unsigned long long)t.rss_after_context.vm_rss_kb,
		(unsigned long long)t.rss_after_prompt.vm_rss_kb,
		(unsigned long long)t.rss_final.vm_rss_kb,
		(unsigned long long)t.rss_peak.vm_hwm_kb);
	printf("\"performance\":{\"prompt_tok_per_s\":%.6f,"
		"\"generation_tok_per_s\":%.6f},",
		t.prompt_tok_per_s, t.generation_tok_per_s);
	printf("\"no_fallback_occurred\":%s,",
		t.no_fallback_occurred ? "true" : "false");
	/* Phase 13.2, Section 6: a canonical grouping of the requested-side
	 * and resolved-side fields Phase 13.1 already emits scattered
	 * across "auto"/"gpu"/"kv_placement"/top-level requested_kv/
	 * selected_kv above -- same values, one place, so a consumer never
	 * has to reconstruct "what did the user ask for" vs "what did
	 * MEMBRANE pick" from several objects. Existing fields are
	 * unchanged; this is a convenience view only. */
	printf("\"requested\":{\"auto\":%s,\"gpu_layers\":\"%s\",\"kv\":\"%s\","
		"\"kv_placement\":\"%s\",\"device\":\"",
		o.auto_mode ? "true" : "false",
		gpu_layers_label(gs.gpu_layers_requested).c_str(),
		gs.adaptive_used ? "adaptive" : kv_mode_name(o.kv_mode),
		membrane_kv_placement_mode_name(o.kv_placement));
	print_json_escaped(gs.device_requested);
	printf("\"},");
	printf("\"resolved\":{\"backend\":\"");
	print_json_escaped(gs.backend_selected);
	printf("\",\"device\":\"");
	print_json_escaped(gs.device_selected);
	printf("\",\"gpu_layers\":%d,\"kv\":\"%s\",\"kv_placement\":\"%s\"},",
		gs.gpu_layers_selected, kv_mode_name(o.kv_mode),
		membrane_kv_placement_mode_name(o.kv_placement));
	/* Section 7: ordered, semantically-meaningful decision trace behind
	 * reason_code above -- may be empty (e.g. a plain run with no
	 * policy active at all). */
	printf("\"reason_trace\":");
	print_json_string_array(gs.reason_trace);
	printf(",\"warnings\":");
	print_json_string_array(gs.warnings);
	/* Section 9: only fields backed by real measurements -- no
	 * fabricated prompt_eval_time/generation_time/total_time (this
	 * codebase measures throughput, not wall-clock phase durations;
	 * see kv_store_telemetry.h's own performance fields). */
	printf(",\"execution\":{\"success\":%s,\"exit_code\":%d,"
		"\"prompt_tokens\":%zu,\"generated_tokens\":%llu,"
		"\"prompt_tok_per_sec\":%.6f,\"generation_tok_per_sec\":%.6f},",
		success ? "true" : "false", exit_code, prompt_tokens_count,
		(unsigned long long)t.generated_tokens, t.prompt_tok_per_s,
		t.generation_tok_per_s);
	/* Section 17: observability only -- omitted entirely (not zeroed)
	 * when /proc/meminfo could not be read, so a consumer never mistakes
	 * "unavailable" for "zero host memory". Field names say SNAPSHOT
	 * explicitly (Section 8) -- neither a guarantee nor a live figure. */
	printf("\"host_memory\":{\"available\":%s", host.ok ? "true" : "false");
	if (host.ok)
		printf(",\"total_bytes\":%llu,\"available_bytes_snapshot\":%llu,"
			"\"swap_total_bytes\":%llu,\"swap_free_bytes_snapshot\":%llu",
			(unsigned long long)host.total_bytes,
			(unsigned long long)host.available_bytes,
			(unsigned long long)host.swap_total_bytes,
			(unsigned long long)host.swap_free_bytes);
	printf("}");
	if (o.include_text)
	{
		printf(",\"text\":\"");
		print_json_escaped(text);
		printf("\"");
	}
	printf("}\n");
}

static void	print_run_human_stats(const membrane_kv_store_telemetry_t &t)
{
	fprintf(stderr, "\ngenerated  %llu tokens\n",
		(unsigned long long)t.generated_tokens);
	fprintf(stderr, "speed      %.1f tok/s\n", t.generation_tok_per_s);
	fprintf(stderr, "kv memory  %.2f MiB (real allocation)\n",
		(double)t.compressed_kv_allocated_bytes / (1024.0 * 1024.0));
	fprintf(stderr, "rss final  %llu kB (peak %llu kB)\n",
		(unsigned long long)t.rss_final.vm_rss_kb,
		(unsigned long long)t.rss_peak.vm_hwm_kb);
}

/* Phase 13.2, Section 12: --plan-only --json's dedicated schema --
 * deliberately NOT print_run_json() with fields omitted: that object's
 * shape (generated_tokens, execution, performance, ...) claims a
 * completed generation happened, which --plan-only must never imply
 * (Section 11: "NOT run generation"). mode:"plan" instead of "run" so
 * a consumer can tell the two apart unambiguously, and no field here
 * is ever fabricated from incomplete metadata (Section 11) -- every
 * number below comes from the exact same resolved gs the real planner
 * (resolve_gpu_config()/resolve_kv_placement()) already produced. */
static void	print_plan_only_json(const membrane_run_opts_t &o,
				const char *model_label, uint32_t ctx_size,
				int32_t n_layer_total, const membrane_gpu_state_t &gs,
				const membrane_host_meminfo_t &host,
				const membrane_timings_t &timings)
{
	printf("{\"schema_version\":1,\"membrane_version\":\"%s\","
		"\"mode\":\"plan\",\"ok\":true,\"model_label\":\"",
		MEMBRANE_VERSION);
	print_json_escaped(model_label);
	printf("\",\"ctx_size\":%u,\"model_layers\":%d,", ctx_size, n_layer_total);
	printf("\"requested\":{\"auto\":%s,\"gpu_layers\":\"%s\",\"kv\":\"%s\","
		"\"kv_placement\":\"%s\",\"device\":\"",
		o.auto_mode ? "true" : "false",
		gpu_layers_label(gs.gpu_layers_requested).c_str(),
		gs.adaptive_used ? "adaptive" : kv_mode_name(o.kv_mode),
		membrane_kv_placement_mode_name(o.kv_placement));
	print_json_escaped(gs.device_requested);
	printf("\"},");
	printf("\"resolved\":{\"backend\":\"");
	print_json_escaped(gs.backend_selected);
	printf("\",\"device\":\"");
	print_json_escaped(gs.device_selected);
	printf("\",\"gpu_layers\":%d,\"kv\":\"%s\",\"kv_placement\":\"%s\"},",
		gs.gpu_layers_selected, kv_mode_name(o.kv_mode),
		membrane_kv_placement_mode_name(o.kv_placement));
	/* Review fix (CodeRabbit, PR #23): device_total_bytes/device_free_
	 * bytes_snapshot/safety_reserve_bytes/estimated_model_bytes are
	 * only ever populated when a GPU policy estimate actually ran
	 * (gs.policy_used) -- a CPU-only plan (no --gpu-layers request at
	 * all) leaves them at their gs-struct default of 0, which a
	 * consumer could misread as "this device has 0 bytes" rather than
	 * "no GPU policy applies here". print_gpu_json() above already
	 * omits its entire "gpu_policy" object under the same condition
	 * (Phase 13.1) -- this matches that established convention instead
	 * of introducing a new, inconsistent one. estimated_kv_bytes is
	 * unconditional: it is meaningful (and populated) for CPU-only KV
	 * too, via resolve_cpu_adaptive_kv(). */
	printf("\"memory_plan\":{\"estimated_kv_bytes\":%llu",
		(unsigned long long)gs.estimated_kv_bytes);
	if (gs.policy_used)
		printf(",\"device_total_bytes\":%llu,"
			"\"device_free_bytes_snapshot\":%llu,\"safety_reserve_bytes\":%llu,"
			"\"estimated_model_bytes\":%llu",
			(unsigned long long)gs.device_total_bytes,
			(unsigned long long)gs.device_free_bytes,
			(unsigned long long)gs.safety_reserve_bytes,
			(unsigned long long)gs.estimated_model_bytes);
	if (gs.kv_placement_resolved)
		printf(",\"kv_gpu_layers\":%d,\"kv_cpu_layers\":%d,"
			"\"kv_gpu_bytes\":%llu,\"kv_cpu_bytes\":%llu",
			gs.kv_placement.gpu_kv_layers, gs.kv_placement.cpu_kv_layers,
			(unsigned long long)gs.kv_placement.gpu_kv_bytes,
			(unsigned long long)gs.kv_placement.cpu_kv_bytes);
	printf("},");
	print_joint_planner_json(gs);
	/* Section 16: --plan-only never applies a runtime plan, so fallback
	 * NEVER ran -- ran_at_all=false yields the distinct "not_applicable"
	 * final_status, never "success" (which would wrongly imply a real
	 * attempt happened). */
	print_fallback_json(gs, false, false, false);
	print_timings_json(timings, NULL, gs);
	/* Review fix (CodeRabbit, PR #23): host_memory now sits at the same
	 * top level in both JSON schemas (print_run_json emits it
	 * top-level too) -- previously nested inside memory_plan here only,
	 * forcing a consumer to branch on `mode` to find it. */
	printf("\"host_memory\":{\"available\":%s", host.ok ? "true" : "false");
	if (host.ok)
		printf(",\"total_bytes\":%llu,\"available_bytes_snapshot\":%llu,"
			"\"swap_total_bytes\":%llu,\"swap_free_bytes_snapshot\":%llu",
			(unsigned long long)host.total_bytes,
			(unsigned long long)host.available_bytes,
			(unsigned long long)host.swap_total_bytes,
			(unsigned long long)host.swap_free_bytes);
	printf("},");
	printf("\"reason_codes\":{\"primary\":\"");
	print_json_escaped(plan_primary_reason(o, gs));
	printf("\",\"trace\":");
	print_json_string_array(gs.reason_trace);
	printf("},\"warnings\":");
	print_json_string_array(gs.warnings);
	printf("}\n");
}

/* Phase 21: state threaded through membrane_fallback_run()'s apply_fn
 * for the REAL (llama-aware) apply adapter -- the fallback CONTROLLER
 * itself (auto_fallback.h/.c) stays llama-free; this struct/function
 * pair is the one place Phase 21 actually touches llama.h to
 * instantiate a candidate. cb is captured once (matches the pre-Phase-
 * 21 stream/no-stream decision) and reused across every attempt --
 * Section 15: a retry is never silent, and a failed attempt's own
 * partial output (if any -- only reachable for a rare mid-generation
 * decode failure; every context-construction failure happens before
 * any token exists) is not hidden either, matching this project's
 * existing "fail closed, never paper over" convention rather than
 * buffering and discarding it. */
typedef struct s_real_apply_ctx
{
	const membrane_run_opts_t			*o;
	const char							*model_path;
	llama_model_params					mp_template;
	std::vector<ggml_backend_dev_t>	*device_storage;
	llama_model							**model_ptr;
	int32_t								loaded_gpu_layers;
	const std::vector<llama_token>		*prompt_tokens;
	uint32_t							ctx_size;
	membrane_token_cb_t					cb;
	membrane_gpu_state_t				*gs;

	/* Outputs -- meaningful only after a successful call. */
	membrane_kv_store_telemetry_t		tel;
	std::string							text;
	gen_run_result_t					gen_result;
	int									effective_kv_mode;
}	real_apply_ctx_t;

/* Section 3: applies exactly ONE candidate. Reload policy (Section 10):
 * a model reload happens iff this candidate's gpu_layers differs from
 * whichever gpu_layers is currently loaded (attempt #1 always reuses
 * the model main() already loaded for the originally-selected
 * candidate -- loaded_gpu_layers is seeded to that same value). Every
 * failure path sets cleanup_complete=1: model load failure leaves
 * *model_ptr NULL (nothing to free); compat/placement rejection happen
 * before any context exists; run_kv_store_pass() itself already frees
 * its own context/collector on every one of its own failure paths
 * (decode_loop.cpp) -- there is no partial state this function itself
 * ever leaves behind uncleaned. */
static int	real_apply_fn(const membrane_joint_candidate_t *c,
				int candidate_index, void *apply_ctx_v,
				membrane_fallback_apply_result_t *out)
{
	real_apply_ctx_t				*ctx = (real_apply_ctx_t *)apply_ctx_v;
	model_shape_t					shape;
	membrane_kv_placement_map_t	placement_map;
	const membrane_kv_placement_map_t	*placement_arg;
	membrane_kv_residency_result_t	placement_result;
	bool							placement_resolved;
	int								kv_mode;
	uint64_t						kv_bytes;
	int								failure_stage;
	bool							ok;

	memset(out, 0, sizeof(*out));
	out->model_reload_required = (*ctx->model_ptr == NULL
		|| ctx->loaded_gpu_layers != c->gpu_layers);
	if (out->model_reload_required)
	{
		llama_model_params	mp = ctx->mp_template;

		if (*ctx->model_ptr != NULL)
		{
			llama_model_free(*ctx->model_ptr);
			*ctx->model_ptr = NULL;
		}
		if (c->gpu_layers > 0)
		{
			mp.devices = ctx->device_storage->data();
			mp.n_gpu_layers = c->gpu_layers;
			mp.main_gpu = 0;
		}
		else
		{
			mp.devices = NULL;
			mp.n_gpu_layers = 0;
			mp.main_gpu = -1;
		}
		*ctx->model_ptr = llama_model_load_from_file(ctx->model_path, mp);
		if (*ctx->model_ptr == NULL)
		{
			ctx->loaded_gpu_layers = -1;
			out->ok = 0;
			out->failure_class = MEMBRANE_APPLY_MODEL_LOAD_FAILED;
			out->cleanup_complete = 1;
			snprintf(out->detail, sizeof(out->detail), "llama_model_load_"
				"from_file returned NULL for gpu_layers=%d", c->gpu_layers);
			return (0);
		}
		ctx->loaded_gpu_layers = c->gpu_layers;
	}
	read_model_shape(*ctx->model_ptr, &shape);
	kv_mode = c->kv_precision;
	if (kv_mode != MEMBRANE_KV_STORE_NATIVE)
	{
		membrane_compat_result_t	compat;

		if (!membrane_check_kv_compat(shape.arch_name.c_str(), shape.n_embd,
				shape.n_head, shape.n_head_kv, ctx->ctx_size, kv_mode,
				&compat))
		{
			out->ok = 0;
			out->failure_class = MEMBRANE_APPLY_COMPAT_REJECTED;
			out->cleanup_complete = 1;
			snprintf(out->detail, sizeof(out->detail), "%s", compat.reason);
			return (0);
		}
	}
	placement_resolved = false;
	placement_arg = NULL;
	memset(&placement_result, 0, sizeof(placement_result));
	if (ctx->o->kv_placement != MEMBRANE_KV_PLACEMENT_DEFAULT)
	{
		uint64_t	total_kv_bytes = kv_bytes_for_mode(shape, ctx->ctx_size,
				kv_mode);
		uint64_t	kv_bytes_per_layer = shape.n_layer > 0
				? total_kv_bytes / (uint64_t)shape.n_layer : 0;

		if (!membrane_kv_residency_resolve(ctx->o->kv_placement,
				shape.n_layer, kv_bytes_per_layer,
				ctx->gs->device_free_bytes, ctx->gs->device_total_bytes,
				c->estimated_weight_gpu_bytes, 0, &placement_result))
		{
			out->ok = 0;
			out->failure_class = MEMBRANE_APPLY_CONTEXT_CREATE_FAILED;
			out->cleanup_complete = 1;
			snprintf(out->detail, sizeof(out->detail), "%s",
				placement_result.reason);
			return (0);
		}
		placement_resolved = true;
		placement_map.n_layer = placement_result.n_layer;
		placement_map.layer_on_gpu = placement_result.layer_on_gpu;
		placement_arg = &placement_map;
	}
	memset(&ctx->tel, 0, sizeof(ctx->tel));
	ctx->tel.kv_store_mode_name = kv_mode_name(kv_mode);
	ctx->tel.no_fallback_occurred = 1;
	kv_bytes = kv_bytes_for_mode(shape, ctx->ctx_size, kv_mode);
	membrane_kv_store_read_rss(&ctx->tel.rss_after_model_load);
	ctx->text.clear();
	failure_stage = MEMBRANE_KV_PASS_STAGE_NONE;
	ok = run_kv_store_pass(*ctx->model_ptr, *ctx->prompt_tokens,
			ctx->o->gen_tokens, kv_mode, ctx->ctx_size, ctx->o->verbose,
			NULL, false, 0, &ctx->text, &ctx->tel, &ctx->gen_result, ctx->cb,
			NULL, placement_arg, &failure_stage);
	if (!ok)
	{
		out->ok = 0;
		out->cleanup_complete = 1;
		out->failure_class = (failure_stage
				== MEMBRANE_KV_PASS_STAGE_CONTEXT_CREATE)
			? MEMBRANE_APPLY_CONTEXT_CREATE_FAILED : MEMBRANE_APPLY_UNKNOWN_FAILED;
		snprintf(out->detail, sizeof(out->detail),
			"run_kv_store_pass failed (stage=%d)", failure_stage);
		return (0);
	}
	ctx->tel.generated_tokens = ctx->gen_result.tokens.size();
	ctx->tel.ctx_size = ctx->ctx_size;
	ctx->tel.compressed_kv_allocated_bytes = kv_bytes;
	membrane_kv_store_rss_max(&ctx->tel.rss_after_model_load,
		&ctx->tel.rss_after_context, &ctx->tel.rss_peak);
	membrane_kv_store_rss_max(&ctx->tel.rss_peak, &ctx->tel.rss_after_prompt,
		&ctx->tel.rss_peak);
	membrane_kv_store_rss_max(&ctx->tel.rss_peak, &ctx->tel.rss_final,
		&ctx->tel.rss_peak);
	ctx->effective_kv_mode = kv_mode;
	/* CodeRabbit review (PR #31): a fallback candidate may resolve to a
	 * different KV precision than the one adaptive planning originally
	 * picked (the adaptive path's own runner-up precision) -- keep the
	 * adaptive telemetry (print_gpu_json()'s "adaptive" object)
	 * describing what actually ran, never the superseded plan, so it
	 * never contradicts the top-level selected_kv (which already comes
	 * from effective_kv_mode above). */
	if (ctx->gs->adaptive_used)
	{
		ctx->gs->adaptive_selected_mode = kv_mode;
		ctx->gs->adaptive_reason = c->reason_code;
	}
	ctx->gs->kv_placement_resolved = placement_resolved;
	if (placement_resolved)
		ctx->gs->kv_placement = placement_result;
	ctx->gs->gpu_layers_selected = c->gpu_layers;
	ctx->gs->estimated_model_bytes = c->estimated_weight_gpu_bytes;
	ctx->gs->estimated_kv_bytes = c->estimated_kv_gpu_bytes
		+ c->estimated_host_kv_bytes;
	ctx->gs->joint_selected_index = candidate_index;
	ctx->gs->plan_reason_code = c->reason_code;
	out->ok = 1;
	out->cleanup_complete = 1;
	return (1);
}

/* Section 11: re-queries the SAME device gpu_policy's pre-load estimate
 * was checked against (gpu_device.h's membrane_gpu_list_devices(), the
 * identical API observe_gpu_memory_after_run() already uses post-run)
 * -- a fresh free-bytes snapshot for the fallback controller's own
 * memory-refit check, never a second/different query mechanism. Every
 * candidate targets the SAME physical device (Phase 20 never varies
 * WHICH device, only gpu_layers/precision/placement on the one chosen
 * at resolve_gpu_config() time), so gs->device_selected stays valid
 * across every attempt. Returns 0 (no refresh) for a CPU-only run --
 * membrane_fallback_run() then treats every candidate's fit as
 * unchanged from planning time, which is correct: there is no GPU
 * device to query in the first place. */
static int	real_refresh_fn(void *refresh_ctx_v, uint64_t *out_free_bytes)
{
	const membrane_gpu_state_t	*gs = (const membrane_gpu_state_t *)
			refresh_ctx_v;
	membrane_gpu_device_info_t	devices[MEMBRANE_GPU_MAX_DEVICES];
	size_t						n;
	size_t						i;

	if (gs->device_selected.empty())
		return (0);
	n = membrane_gpu_list_devices(devices, MEMBRANE_GPU_MAX_DEVICES);
	i = 0;
	while (i < n)
	{
		if (gs->device_selected == devices[i].name)
		{
			*out_free_bytes = devices[i].memory_free;
			return (1);
		}
		i++;
	}
	return (0);
}

/* Phase 21, Section 15: "a retry is never silent" -- prints a concise
 * transcript of what the fallback controller actually did, always to
 * stderr, gated the same way print_startup_summary() already is
 * (!o.quiet); a run whose primary candidate simply succeeded prints
 * nothing here at all unless --verbose (Section 15: "without --verbose:
 * do not spam"). */
static void	print_fallback_diagnostics(const membrane_fallback_trace_t &ft,
				bool verbose)
{
	int	i;

	if (ft.n_entries == 0)
		return ;
	if (!ft.attempted && strcmp(ft.final_status, "success") == 0)
	{
		if (verbose)
			fprintf(stderr, "MEMBRANE: plan #%d succeeded on the first "
				"attempt\n", ft.initial_candidate_index);
		return ;
	}
	fprintf(stderr, "MEMBRANE: selected plan #%d\n",
		ft.initial_candidate_index);
	i = 0;
	while (i < ft.n_entries)
	{
		const membrane_fallback_attempt_t	&a = ft.entries[i];

		if (!a.apply_started)
			fprintf(stderr, "MEMBRANE: plan #%d skipped: available GPU "
				"memory changed since planning\n", a.candidate_index);
		else if (a.apply_ok)
			fprintf(stderr, "MEMBRANE: plan #%d succeeded\n",
				a.candidate_index);
		else
		{
			fprintf(stderr, "MEMBRANE: plan #%d failed: %s\n",
				a.candidate_index,
				membrane_apply_failure_class_name(a.failure_class));
			if (i + 1 < ft.n_entries)
				fprintf(stderr, "MEMBRANE: retrying with plan #%d\n",
					ft.entries[i + 1].candidate_index);
		}
		i++;
	}
	if (strcmp(ft.final_status, "exhausted") == 0)
		fprintf(stderr, "MEMBRANE: no legal plan could be instantiated "
			"(%s)\n", ft.reason_code);
	else if (strcmp(ft.final_status, "cleanup_blocked") == 0)
		fprintf(stderr, "MEMBRANE: stopped after a resource-cleanup safety "
			"check failed (%s)\n", ft.reason_code);
}

/*
 * Normal single-pass mode. Loads (or, for an auto-managed run whose
 * primary candidate cannot be instantiated, reloads -- Phase 21)
 * exactly the model configuration one candidate needs, creates exactly
 * ONE llama_context (native/q8/q5 per --kv) per attempt, ingests the
 * prompt, generates, prints, exits. capture_logits is ALWAYS false
 * here -- there is no teacher_force pass, no second/third context, and
 * therefore no per-step logit buffer of any size, let alone a
 * full-context one.
 *
 * Phase 21: *model_ptr may be freed and reloaded by this function (via
 * real_apply_fn) when gs.joint_planner_used and the primary candidate
 * needs a fallback retry to a candidate with a different gpu_layers
 * count -- callers must re-read *model_ptr after this returns rather
 * than assuming it is still the model they passed in. Every non-auto-
 * managed run (gs.joint_planner_used false: --compare-kv/--gpu-bench
 * never reach this function at all, and a CPU-only or no-estimate run)
 * takes the EXACT pre-Phase-21 single-attempt path below, unchanged.
 */
static int	run_normal_mode(const membrane_run_opts_t &o,
				llama_model **model_ptr, const char *model_path,
				const llama_model_params &mp_template,
				std::vector<ggml_backend_dev_t> *device_storage,
				const std::vector<llama_token> &prompt_tokens,
				const char *model_label, uint32_t ctx_size,
				const model_shape_t &shape, membrane_gpu_state_t &gs,
				const membrane_host_meminfo_t &host,
				membrane_timings_t timings, const struct timespec &total_t0)
{
	membrane_kv_store_telemetry_t	tel;
	gen_run_result_t				result;
	std::string						text;
	bool							stream;
	uint64_t						kv_bytes;
	membrane_kv_placement_map_t	placement_map;
	const membrane_kv_placement_map_t	*placement_arg;
	membrane_run_opts_t				effective_o = o;
	bool							ok_overall;

	stream = !o.want_json && !o.quiet;
	if (gs.joint_planner_used && gs.joint_result.candidate_count > 0
		&& gs.joint_result.selected_index >= 0)
	{
		real_apply_ctx_t	actx = {};

		if (!o.quiet)
			print_startup_summary(o, model_label, ctx_size,
				kv_bytes_for_mode(shape, ctx_size, o.kv_mode), shape.n_layer,
				gs, host);
		actx.o = &o;
		actx.model_path = model_path;
		actx.mp_template = mp_template;
		actx.device_storage = device_storage;
		actx.model_ptr = model_ptr;
		actx.loaded_gpu_layers = gs.gpu_layers_selected;
		actx.prompt_tokens = &prompt_tokens;
		actx.ctx_size = ctx_size;
		actx.cb = stream ? stream_token : NULL;
		actx.gs = &gs;
		gs.fallback_engaged = true;
		ok_overall = membrane_fallback_run(gs.joint_result.candidates,
				gs.joint_result.candidate_count,
				gs.joint_result.selected_index, gs.device_total_bytes,
				gs.safety_reserve_bytes, real_apply_fn, &actx,
				real_refresh_fn, &gs, &gs.fallback_trace) != 0;
		if (!o.quiet)
			print_fallback_diagnostics(gs.fallback_trace, o.verbose);
		if (!ok_overall)
		{
			fprintf(stderr, "membrane-run: every legal auto-managed plan "
				"failed to instantiate (%s)\n", gs.fallback_trace.reason_code);
			print_error_json(o, MEMBRANE_EXIT_RUNTIME_ERROR,
				gs.fallback_trace.reason_code,
				"every legal auto-managed plan failed to instantiate",
				"Try a smaller --ctx, explicit --gpu-layers/--kv/"
				"--kv-placement, or --gpu-layers 0 for CPU-only.",
				NULL, &gs.fallback_trace);
			return (MEMBRANE_EXIT_RUNTIME_ERROR);
		}
		tel = actx.tel;
		text = actx.text;
		result = actx.gen_result;
		effective_o.kv_mode = actx.effective_kv_mode;
	}
	else
	{
		memset(&tel, 0, sizeof(tel));
		tel.kv_store_mode_name = kv_mode_name(o.kv_mode);
		/* This path never falls back: run_kv_store_pass() either
		 * succeeds with the requested storage type or fails the whole
		 * run -- there is no retry-with-native path for a failed q8/q5
		 * context (gs.joint_planner_used is false here by construction:
		 * see this function's own header comment). */
		tel.no_fallback_occurred = 1;
		kv_bytes = kv_bytes_for_mode(shape, ctx_size, o.kv_mode);
		membrane_kv_store_read_rss(&tel.rss_after_model_load);
		if (!o.quiet)
			print_startup_summary(o, model_label, ctx_size, kv_bytes,
				shape.n_layer, gs, host);
		/* Phase 12H: gs.kv_placement_resolved is only true when
		 * o.kv_placement != MEMBRANE_KV_PLACEMENT_DEFAULT AND
		 * resolve_kv_placement() succeeded (main() already returned
		 * MEMBRANE_EXIT_UNSUPPORTED_KV otherwise, before reaching here)
		 * -- placement_arg stays NULL (kv_dev_override left untouched)
		 * for plain default-mode runs. */
		placement_arg = NULL;
		if (gs.kv_placement_resolved)
		{
			placement_map.n_layer = gs.kv_placement.n_layer;
			placement_map.layer_on_gpu = gs.kv_placement.layer_on_gpu;
			placement_arg = &placement_map;
		}
		if (!run_kv_store_pass(*model_ptr, prompt_tokens, o.gen_tokens,
				o.kv_mode, ctx_size, o.verbose, NULL, false, 0, &text, &tel,
				&result, stream ? stream_token : NULL, NULL, placement_arg))
		{
			fprintf(stderr, "membrane-run: generation failed\n");
			print_error_json(o, MEMBRANE_EXIT_RUNTIME_ERROR,
				MEMBRANE_REASON_GENERATION_FAILED, "generation failed");
			return (MEMBRANE_EXIT_RUNTIME_ERROR);
		}
		tel.generated_tokens = result.tokens.size();
		tel.ctx_size = ctx_size;
		/* kv_bytes is the real allocation size (real ggml_row_size()
		 * times real per-model constants, same arithmetic llama.cpp's
		 * own allocator uses -- see Phase 7) -- not re-derived from
		 * anything run_kv_store_pass() itself measured, since that
		 * function has no reason to know about byte accounting; it
		 * only manages the context/decode loop. */
		tel.compressed_kv_allocated_bytes = kv_bytes;
		membrane_kv_store_rss_max(&tel.rss_after_model_load,
			&tel.rss_after_context, &tel.rss_peak);
		membrane_kv_store_rss_max(&tel.rss_peak, &tel.rss_after_prompt,
			&tel.rss_peak);
		membrane_kv_store_rss_max(&tel.rss_peak, &tel.rss_final,
			&tel.rss_peak);
	}
	timings.total_ms = seconds_since(&total_t0) * 1000.0;
	if (o.want_json)
	{
		membrane_gpu_memory_observed_t	gpu_mem_observed;

		observe_gpu_memory_after_run(gs, &gpu_mem_observed);
		print_run_json(effective_o, model_label, tel, text, gs, host,
			result.ok, result.ok ? MEMBRANE_EXIT_SUCCESS
				: MEMBRANE_EXIT_RUNTIME_ERROR, prompt_tokens.size(),
			gpu_mem_observed, timings);
	}
	else
	{
		if (!stream)
			fwrite(text.data(), 1, text.size(), stdout);
		if (!o.quiet)
			print_run_human_stats(tel);
		else
			putchar('\n');
	}
	return (result.ok ? MEMBRANE_EXIT_SUCCESS : MEMBRANE_EXIT_RUNTIME_ERROR);
}

static void	print_compare_json(const char *model_label, uint32_t ctx_size,
				uint64_t native_bytes,
				const membrane_kv_store_telemetry_t &tel_candidate,
				const membrane_gpu_state_t &gs)
{
	/* Backward compatibility: scripts/assemble-v0.2-artifact.py (and
	 * potentially other external tooling) reads storage.
	 * q8_kv_allocated_bytes/memory_q8 from --compare-kv's JSON output
	 * -- Section 14's own rule is that existing behavior stays
	 * unchanged unless q5 is explicitly requested, so when the
	 * compared mode actually IS q8 (the default), keep emitting those
	 * exact legacy keys too, purely additive, alongside the new
	 * generic ones q5 needs. Never emitted (and never misleadingly
	 * populated with q5 bytes) when the compared mode is q5. */
	bool	is_legacy_q8;

	is_legacy_q8 = strcmp(tel_candidate.kv_store_mode_name, "q8") == 0;
	/* Section 10: "adaptive resolves once before comparison ... JSON
	 * should include requested_kv: adaptive, selected_kv: q8|q5,
	 * compressed_kv: q8|q5, adaptive_reason". When --kv adaptive was
	 * NOT given, requested_kv mirrors compressed_kv (the pre-existing
	 * q8-default/--kv-q5 rule already fully determines it) -- purely
	 * additive, no existing field's meaning changes. */
	printf("{\"schema_version\":1,\"membrane_version\":\"%s\","
		"\"mode\":\"compare\",\"model_label\":\"%s\",\"ctx_size\":%u,"
		"\"compressed_kv\":\"%s\",\"requested_kv\":\"%s\","
		"\"selected_kv\":\"%s\",\"adaptive_reason\":\"%s\",",
		MEMBRANE_VERSION, model_label, ctx_size,
		tel_candidate.kv_store_mode_name,
		gs.adaptive_used ? "adaptive" : tel_candidate.kv_store_mode_name,
		tel_candidate.kv_store_mode_name,
		gs.adaptive_used ? gs.adaptive_reason.c_str() : "");
	printf("\"storage\":{\"native_kv_allocated_bytes\":%llu,"
		"\"compressed_kv_allocated_bytes\":%llu",
		(unsigned long long)native_bytes,
		(unsigned long long)tel_candidate.compressed_kv_allocated_bytes);

	if (is_legacy_q8)
		printf(",\"q8_kv_allocated_bytes\":%llu",
			(unsigned long long)tel_candidate.compressed_kv_allocated_bytes);
	printf("},");
	print_gpu_json(gs);
	printf("\"memory_compressed\":{\"rss_after_context_kb\":%llu,"
		"\"rss_final_kb\":%llu,\"peak_rss_kb\":%llu},",
		(unsigned long long)tel_candidate.rss_after_context.vm_rss_kb,
		(unsigned long long)tel_candidate.rss_final.vm_rss_kb,
		(unsigned long long)tel_candidate.rss_peak.vm_hwm_kb);
	if (is_legacy_q8)
		printf("\"memory_q8\":{\"rss_after_context_kb\":%llu,"
			"\"rss_final_kb\":%llu,\"peak_rss_kb\":%llu},",
			(unsigned long long)tel_candidate.rss_after_context.vm_rss_kb,
			(unsigned long long)tel_candidate.rss_final.vm_rss_kb,
			(unsigned long long)tel_candidate.rss_peak.vm_hwm_kb);
	printf("\"performance\":{\"prompt_tok_per_s\":%.6f,"
		"\"generation_tok_per_s\":%.6f},",
		tel_candidate.prompt_tok_per_s, tel_candidate.generation_tok_per_s);
	printf("\"quality\":{\"available\":%s,\"token_identity\":%s,"
		"\"first_divergence\":%d,\"logit_rel_l2\":%.6f,"
		"\"top1_preservation\":%.6f,\"delta_nll\":%.6f},",
		tel_candidate.quality_available ? "true" : "false",
		tel_candidate.token_identity ? "true" : "false",
		tel_candidate.first_divergence, tel_candidate.logit_rel_l2,
		tel_candidate.top1_preservation, tel_candidate.delta_nll);
	printf("\"no_fallback_occurred\":%s}\n",
		tel_candidate.no_fallback_occurred ? "true" : "false");
}

static void	print_compare_human(const char *model_label, uint32_t ctx_size,
				uint64_t native_bytes,
				const membrane_kv_store_telemetry_t &tel_candidate,
				const membrane_gpu_state_t &gs)
{
	const char	*mode_name = tel_candidate.kv_store_mode_name;

	fprintf(stderr, "MEMBRANE %s -- compare mode (native vs %s)\n",
		MEMBRANE_VERSION, mode_name);
	fprintf(stderr, "model            %s\n", model_label);
	fprintf(stderr, "context          %u\n", ctx_size);
	if (gs.adaptive_used)
		fprintf(stderr, "kv adaptive      requested=adaptive selected=%s "
			"reason=%s\n", mode_name, gs.adaptive_reason.c_str());
	if (!gs.requested)
		fprintf(stderr, "backend          CPU (default)\n");
	else
		fprintf(stderr, "backend          %s, device: %s "
			"(gpu-layers=%s)\n", gs.backend_selected.c_str(),
			gs.device_selected.c_str(),
			gpu_layers_label(gs.gpu_layers_requested).c_str());
	fprintf(stderr, "native kv bytes  %.2f MiB\n",
		(double)native_bytes / (1024.0 * 1024.0));
	fprintf(stderr, "%s kv bytes      %.2f MiB (%.3fx smaller)\n", mode_name,
		(double)tel_candidate.compressed_kv_allocated_bytes
			/ (1024.0 * 1024.0),
		(double)native_bytes
			/ (double)tel_candidate.compressed_kv_allocated_bytes);
	fprintf(stderr, "%s rss (ctx)     %llu kB\n", mode_name,
		(unsigned long long)tel_candidate.rss_after_context.vm_rss_kb);
	fprintf(stderr, "%s tok/s (gen)   %.1f\n", mode_name,
		tel_candidate.generation_tok_per_s);
	if (tel_candidate.quality_available)
		fprintf(stderr, "quality          token_identity=%s "
			"first_divergence=%d top1=%.4f logit_rel_l2=%.6f "
			"delta_nll=%.6f\n",
			tel_candidate.token_identity ? "identical" : "diverged",
			tel_candidate.first_divergence, tel_candidate.top1_preservation,
			tel_candidate.logit_rel_l2, tel_candidate.delta_nll);
	else
		fprintf(stderr, "quality          unavailable (aligned "
			"comparison pass did not complete)\n");
}

/*
 * Explicit benchmark/compare mode (Section 6): reuses the exact Phase
 * 7 3-pass design (native reference, candidate canonical, candidate
 * teacher-forced) via the SAME run_kv_store_pass()/
 * record_kv_store_behavior() as membrane-llama-run's --kv-store q8 --
 * including the pass-ordering fix from that phase's own review cycle
 * (the canonical, memory-reported pass runs FIRST, immediately after
 * model load, so its peak-RSS reading is never contaminated by an
 * earlier pass in the same process). Never runs during normal mode.
 *
 * Phase 10C: generalized from a hardcoded q8-only comparison to take
 * candidate_mode explicitly -- MEMBRANE_KV_STORE_Q8 (the backward-
 * compatible default for both --compare-kv and --gpu-bench) or
 * MEMBRANE_KV_STORE_Q5 (only if --kv q5 was also given). Always
 * exactly native vs ONE selected compressed mode, never both at once
 * -- see run_compare_mode()/run_gpu_bench_mode() below, which are the
 * only callers and the only place candidate_mode is decided. Returns
 * false (message already printed) only on a hard pass failure; a
 * failed teacher-forced pass alone still returns true with
 * tel_candidate.quality_available left 0 (memory/throughput results
 * are still meaningful without it). */
static bool	run_native_vs_compressed_comparison(const membrane_run_opts_t &o,
				llama_model *model,
				const std::vector<llama_token> &prompt_tokens,
				uint32_t ctx_size, const model_shape_t &shape,
				int candidate_mode,
				membrane_kv_store_telemetry_t *tel_candidate,
				membrane_kv_store_telemetry_t *tel_native,
				uint64_t *native_bytes, gen_run_result_t *result_native,
				gen_run_result_t *result_candidate)
{
	membrane_kv_store_telemetry_t	tel_scratch;
	gen_run_result_t				result_candidate_tf;
	membrane_runtime_divergence_t	divergence;
	int32_t							n_vocab;
	const char						*candidate_name;

	candidate_name = kv_mode_name(candidate_mode);
	memset(tel_candidate, 0, sizeof(*tel_candidate));
	tel_candidate->kv_store_mode_name = candidate_name;
	tel_candidate->no_fallback_occurred = 1;
	tel_candidate->ctx_size = ctx_size;
	tel_candidate->compressed_kv_allocated_bytes =
		kv_bytes_for_mode(shape, ctx_size, candidate_mode);
	*native_bytes = native_kv_bytes(shape, ctx_size);
	n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
	membrane_kv_store_read_rss(&tel_candidate->rss_after_model_load);
	if (!run_kv_store_pass(model, prompt_tokens, o.gen_tokens,
			candidate_mode, ctx_size, o.verbose, NULL, false, 0,
			NULL, tel_candidate, result_candidate))
		return (fprintf(stderr, "membrane-run: %s canonical pass failed\n",
				candidate_name), false);
	tel_candidate->generated_tokens = result_candidate->tokens.size();
	memset(tel_native, 0, sizeof(*tel_native));
	tel_native->kv_store_mode_name = "native";
	tel_native->no_fallback_occurred = 1;
	tel_native->ctx_size = ctx_size;
	tel_native->compressed_kv_allocated_bytes = *native_bytes;
	membrane_kv_store_read_rss(&tel_native->rss_after_model_load);
	if (!run_kv_store_pass(model, prompt_tokens, o.gen_tokens,
			MEMBRANE_KV_STORE_NATIVE, ctx_size, o.verbose, NULL, true,
			n_vocab, NULL, tel_native, result_native))
		return (fprintf(stderr,
				"membrane-run: native reference pass failed\n"), false);
	tel_native->generated_tokens = result_native->tokens.size();
	membrane_kv_store_rss_max(&tel_native->rss_after_model_load,
		&tel_native->rss_after_context, &tel_native->rss_peak);
	membrane_kv_store_rss_max(&tel_native->rss_peak,
		&tel_native->rss_after_prompt, &tel_native->rss_peak);
	membrane_kv_store_rss_max(&tel_native->rss_peak, &tel_native->rss_final,
		&tel_native->rss_peak);
	membrane_runtime_detect_divergence(result_native->tokens.data(),
		result_native->tokens.size(), result_candidate->tokens.data(),
		result_candidate->tokens.size(), &divergence);
	tel_candidate->token_identity = divergence.identical;
	tel_candidate->first_divergence = (int32_t)divergence.first_divergence_step;
	if (!result_native->tokens.empty())
	{
		memset(&tel_scratch, 0, sizeof(tel_scratch));
		if (!run_kv_store_pass(model, prompt_tokens, o.gen_tokens,
				candidate_mode, ctx_size, o.verbose,
				&result_native->tokens, true, n_vocab, NULL, &tel_scratch,
				&result_candidate_tf))
			fprintf(stderr, "membrane-run: aligned teacher-forced pass "
				"failed -- logit/NLL comparison unavailable, memory/"
				"throughput results otherwise still reported\n");
		else if (record_kv_store_behavior(*result_native, result_candidate_tf,
				n_vocab, tel_candidate))
			tel_candidate->quality_available = 1;
	}
	membrane_kv_store_rss_max(&tel_candidate->rss_after_model_load,
		&tel_candidate->rss_after_context, &tel_candidate->rss_peak);
	membrane_kv_store_rss_max(&tel_candidate->rss_peak,
		&tel_candidate->rss_after_prompt, &tel_candidate->rss_peak);
	membrane_kv_store_rss_max(&tel_candidate->rss_peak,
		&tel_candidate->rss_final, &tel_candidate->rss_peak);
	return (true);
}

/* The one selected compressed mode --compare-kv/--gpu-bench compare
 * native against: q5 if the user explicitly asked for it via --kv q5,
 * otherwise q8 (the long-standing default, kept for backward
 * compatibility so existing --compare-kv/--gpu-bench invocations
 * without --kv never change behavior). Never a 4-way/3-way
 * comparison -- exactly native vs exactly one candidate, always. */
static int	selected_comparison_mode(const membrane_run_opts_t &o)
{
	return (o.kv_mode == MEMBRANE_KV_STORE_Q5
		? MEMBRANE_KV_STORE_Q5 : MEMBRANE_KV_STORE_Q8);
}

static int	run_compare_mode(const membrane_run_opts_t &o, llama_model *model,
				const std::vector<llama_token> &prompt_tokens,
				const char *model_label, uint32_t ctx_size,
				const model_shape_t &shape, const membrane_gpu_state_t &gs)
{
	membrane_kv_store_telemetry_t	tel_candidate;
	membrane_kv_store_telemetry_t	tel_native;
	uint64_t						native_bytes;
	gen_run_result_t				result_native;
	gen_run_result_t				result_candidate;
	int								candidate_mode;

	candidate_mode = selected_comparison_mode(o);
	if (!run_native_vs_compressed_comparison(o, model, prompt_tokens,
			ctx_size, shape, candidate_mode, &tel_candidate, &tel_native,
			&native_bytes, &result_native, &result_candidate))
		return (MEMBRANE_EXIT_RUNTIME_ERROR);
	if (o.want_json)
		print_compare_json(model_label, ctx_size, native_bytes,
			tel_candidate, gs);
	else
		print_compare_human(model_label, ctx_size, native_bytes,
			tel_candidate, gs);
	return (result_candidate.ok && result_native.ok
		? MEMBRANE_EXIT_SUCCESS : MEMBRANE_EXIT_RUNTIME_ERROR);
}

/* Phase 9B.1 Section 8's schema. No VRAM figure is emitted here even
 * though nvidia-smi-based numbers exist in this project's own
 * scratch validation runs -- Section 7 is explicit that real process
 * VRAM can't be measured portably from inside MEMBRANE without
 * shelling out, and external nvidia-smi measurements stay validation
 * artifacts (results/v0.3/gpu-vulkan-validation.json), never product
 * telemetry a user relies on at runtime. */
static void	print_gpu_bench_json(const char *model_label, uint32_t ctx_size,
				uint64_t native_bytes,
				const membrane_kv_store_telemetry_t &tel_native,
				const membrane_kv_store_telemetry_t &tel_candidate,
				const membrane_gpu_state_t &gs)
{
	double	kv_reduction_ratio =
		tel_candidate.compressed_kv_allocated_bytes > 0
		? (double)native_bytes
			/ (double)tel_candidate.compressed_kv_allocated_bytes
		: 0.0;
	double	throughput_delta_pct = tel_native.generation_tok_per_s > 0.0
		? 100.0 * (tel_candidate.generation_tok_per_s
			- tel_native.generation_tok_per_s)
			/ tel_native.generation_tok_per_s : 0.0;

	printf("{\"schema_version\":1,\"membrane_version\":\"%s\","
		"\"mode\":\"gpu_bench\",\"compressed_kv\":\"%s\","
		"\"requested_kv\":\"%s\",\"selected_kv\":\"%s\","
		"\"adaptive_reason\":\"%s\",\"model_label\":\"",
		MEMBRANE_VERSION, tel_candidate.kv_store_mode_name,
		gs.adaptive_used ? "adaptive" : tel_candidate.kv_store_mode_name,
		tel_candidate.kv_store_mode_name,
		gs.adaptive_used ? gs.adaptive_reason.c_str() : "");
	print_json_escaped(model_label);
	printf("\",\"ctx_size\":%u,", ctx_size);
	print_gpu_json(gs);
	printf("\"native\":{\"kv_allocated_bytes\":%llu,"
		"\"rss_after_context_kb\":%llu,\"prompt_tok_per_s\":%.6f,"
		"\"generation_tok_per_s\":%.6f},",
		(unsigned long long)native_bytes,
		(unsigned long long)tel_native.rss_after_context.vm_rss_kb,
		tel_native.prompt_tok_per_s, tel_native.generation_tok_per_s);
	printf("\"compressed\":{\"kv_allocated_bytes\":%llu,"
		"\"rss_after_context_kb\":%llu,\"prompt_tok_per_s\":%.6f,"
		"\"generation_tok_per_s\":%.6f},",
		(unsigned long long)tel_candidate.compressed_kv_allocated_bytes,
		(unsigned long long)tel_candidate.rss_after_context.vm_rss_kb,
		tel_candidate.prompt_tok_per_s, tel_candidate.generation_tok_per_s);
	printf("\"comparison\":{\"kv_reduction_ratio\":%.6f,"
		"\"generation_throughput_delta_pct\":%.4f},",
		kv_reduction_ratio, throughput_delta_pct);
	printf("\"quality\":{\"available\":%s,\"token_identity\":%s,"
		"\"first_divergence\":%d,\"logit_rel_l2\":%.6f,"
		"\"top1_preservation\":%.6f,\"delta_nll\":%.6f},",
		tel_candidate.quality_available ? "true" : "false",
		tel_candidate.token_identity ? "true" : "false",
		tel_candidate.first_divergence, tel_candidate.logit_rel_l2,
		tel_candidate.top1_preservation, tel_candidate.delta_nll);
	printf("\"no_fallback_occurred\":%s}\n",
		(tel_native.no_fallback_occurred
				&& tel_candidate.no_fallback_occurred)
			? "true" : "false");
}

static void	print_gpu_bench_human(const char *model_label, uint32_t ctx_size,
				uint64_t native_bytes,
				const membrane_kv_store_telemetry_t &tel_native,
				const membrane_kv_store_telemetry_t &tel_candidate,
				const membrane_gpu_state_t &gs)
{
	const char	*mode_name = tel_candidate.kv_store_mode_name;
	double	throughput_delta_pct = tel_native.generation_tok_per_s > 0.0
		? 100.0 * (tel_candidate.generation_tok_per_s
			- tel_native.generation_tok_per_s)
			/ tel_native.generation_tok_per_s : 0.0;

	fprintf(stderr, "MEMBRANE %s -- gpu-bench (native vs %s)\n",
		MEMBRANE_VERSION, mode_name);
	fprintf(stderr, "model              %s\n", model_label);
	fprintf(stderr, "context            %u\n", ctx_size);
	if (gs.adaptive_used)
		fprintf(stderr, "kv adaptive        requested=adaptive "
			"selected=%s reason=%s\n", mode_name,
			gs.adaptive_reason.c_str());
	fprintf(stderr, "backend            %s, device: %s "
		"(gpu-layers=%s, selected=%d)\n", gs.backend_selected.c_str(),
		gs.device_selected.c_str(),
		gpu_layers_label(gs.gpu_layers_requested).c_str(),
		gs.gpu_layers_selected);
	if (gs.policy_used)
		fprintf(stderr, "gpu policy         device_free=%.1f MiB "
			"reserve=%.1f MiB est_weights=%.1f MiB est_kv=%.1f MiB\n",
			(double)gs.device_free_bytes / (1024.0 * 1024.0),
			(double)gs.safety_reserve_bytes / (1024.0 * 1024.0),
			(double)gs.estimated_model_bytes / (1024.0 * 1024.0),
			(double)gs.estimated_kv_bytes / (1024.0 * 1024.0));
	fprintf(stderr, "native kv bytes    %.2f MiB, %.1f tok/s (gen)\n",
		(double)native_bytes / (1024.0 * 1024.0),
		tel_native.generation_tok_per_s);
	double	kv_ratio = tel_candidate.compressed_kv_allocated_bytes > 0
		? (double)native_bytes
			/ (double)tel_candidate.compressed_kv_allocated_bytes
		: 0.0;

	fprintf(stderr, "%s kv bytes        %.2f MiB (%.3fx smaller), "
		"%.1f tok/s (gen, %+.1f%% vs native)\n", mode_name,
		(double)tel_candidate.compressed_kv_allocated_bytes
			/ (1024.0 * 1024.0),
		kv_ratio, tel_candidate.generation_tok_per_s, throughput_delta_pct);
	if (tel_candidate.quality_available)
		fprintf(stderr, "quality            token_identity=%s "
			"first_divergence=%d top1=%.4f logit_rel_l2=%.6f "
			"delta_nll=%.6f\n",
			tel_candidate.token_identity ? "identical" : "diverged",
			tel_candidate.first_divergence, tel_candidate.top1_preservation,
			tel_candidate.logit_rel_l2, tel_candidate.delta_nll);
	else
		fprintf(stderr, "quality            unavailable (aligned "
			"comparison pass did not complete)\n");
}

/* --gpu-bench: like --compare-kv but always under an explicit GPU
 * configuration (product_cli.cpp's parse-time validation already
 * requires --gpu-layers != 0 for this mode), with gpu_policy telemetry
 * and a native/candidate/comparison-shaped JSON schema instead of
 * compare-kv's. Reuses the exact same 3-pass machinery via
 * run_native_vs_compressed_comparison(); the selected candidate mode
 * is the same q8-default/q5-if-requested rule as --compare-kv (see
 * selected_comparison_mode()). */
static int	run_gpu_bench_mode(const membrane_run_opts_t &o,
				llama_model *model,
				const std::vector<llama_token> &prompt_tokens,
				const char *model_label, uint32_t ctx_size,
				const model_shape_t &shape, const membrane_gpu_state_t &gs)
{
	membrane_kv_store_telemetry_t	tel_candidate;
	membrane_kv_store_telemetry_t	tel_native;
	uint64_t						native_bytes;
	gen_run_result_t				result_native;
	gen_run_result_t				result_candidate;
	int								candidate_mode;

	candidate_mode = selected_comparison_mode(o);
	if (!run_native_vs_compressed_comparison(o, model, prompt_tokens,
			ctx_size, shape, candidate_mode, &tel_candidate, &tel_native,
			&native_bytes, &result_native, &result_candidate))
		return (MEMBRANE_EXIT_RUNTIME_ERROR);
	if (o.want_json)
		print_gpu_bench_json(model_label, ctx_size, native_bytes,
			tel_native, tel_candidate, gs);
	else
		print_gpu_bench_human(model_label, ctx_size, native_bytes,
			tel_native, tel_candidate, gs);
	return (result_candidate.ok && result_native.ok
		? MEMBRANE_EXIT_SUCCESS : MEMBRANE_EXIT_RUNTIME_ERROR);
}

int	main(int argc, char **argv)
{
	membrane_run_opts_t			o;
	int								rc;
	llama_model						*model;
	const llama_vocab				*vocab;
	std::string						prompt_text;
	std::vector<llama_token>		prompt_tokens;
	model_shape_t					shape;
	uint32_t						ctx_size;
	membrane_compat_result_t		compat;
	const char						*model_label;
	membrane_gpu_state_t			gs;
	llama_model_params				mp;
	std::vector<ggml_backend_dev_t>	device_storage;
	membrane_host_meminfo_t			host;
	membrane_timings_t				timings = {};
	struct timespec					total_t0;
	struct timespec					stage_t0;

	rc = membrane_run_parse_opts(argc, argv, &o);
	if (o.want_help)
		return (membrane_run_usage(stdout), MEMBRANE_EXIT_SUCCESS);
	if (o.want_version)
		return (membrane_run_print_version(stdout), MEMBRANE_EXIT_SUCCESS);
	if (rc != MEMBRANE_EXIT_SUCCESS)
		return (membrane_run_usage(stderr), rc);
	if (!resolve_prompt(o, &prompt_text))
	{
		fprintf(stderr, "membrane-run: empty or unreadable prompt\n");
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	if (!o.verbose)
		llama_log_set(quiet_log_callback, NULL);
	/* Phase 24: total_ms's own start point -- deliberately AFTER CLI
	 * parse/prompt resolution (negligible, and not itself one of the
	 * named stages Section 2 of the Phase 24 task lists), not before --
	 * process exec/dynamic-linker overhead is not this program's own
	 * cost to measure or claim. */
	clock_gettime(CLOCK_MONOTONIC, &total_t0);
	/* Section 17: read once, before any of the potentially-slow model/
	 * GPU work below -- a snapshot taken here is no less valid than one
	 * taken later, and reading it once (not per print site) keeps every
	 * consumer of `host` this run looking at the exact same numbers. */
	read_host_meminfo(&host);
	llama_backend_init();
	/* GPU/device resolution needs no loaded model -- fail fast, before
	 * spending time on model load, if the request can't be satisfied.
	 * o.ctx is the only context-size information available pre-load
	 * (0 if auto-sizing from the prompt, which needs a loaded model's
	 * tokenizer) -- product_cli.cpp's parse-time validation rejects
	 * any nonzero --gpu-layers (all/auto/N) paired with an auto-sized
	 * --ctx, so o.ctx is never 0 here in the one case that would need
	 * it to be KV-aware. */
	mp = llama_model_default_params();
	clock_gettime(CLOCK_MONOTONIC, &stage_t0);
	if (!resolve_gpu_config(o, o.ctx, &device_storage, &mp, &gs))
		return (llama_backend_free(), MEMBRANE_EXIT_UNSUPPORTED_KV);
	timings.planner_ms = seconds_since(&stage_t0) * 1000.0;
	clock_gettime(CLOCK_MONOTONIC, &stage_t0);
	model = llama_model_load_from_file(o.model_path, mp);
	timings.model_load_ms = seconds_since(&stage_t0) * 1000.0;
	if (model == NULL)
	{
		fprintf(stderr, "membrane-run: failed to load model '%s'\n",
			o.model_path);
		print_error_json(o, MEMBRANE_EXIT_MODEL_ERROR,
			MEMBRANE_REASON_MODEL_LOAD_FAILED,
			std::string("failed to load model '") + o.model_path + "'");
		return (llama_backend_free(), MEMBRANE_EXIT_MODEL_ERROR);
	}
	model_label = membrane_runtime_safe_basename(o.model_path);
	vocab = llama_model_get_vocab(model);
	prompt_tokens.resize(prompt_text.size() + 8);
	clock_gettime(CLOCK_MONOTONIC, &stage_t0);
	rc = llama_tokenize(vocab, prompt_text.c_str(),
			(int32_t)prompt_text.size(), prompt_tokens.data(),
			(int32_t)prompt_tokens.size(), true, false);
	timings.tokenization_ms = seconds_since(&stage_t0) * 1000.0;
	if (rc < 0)
	{
		fprintf(stderr, "membrane-run: tokenization failed\n");
		print_error_json(o, MEMBRANE_EXIT_RUNTIME_ERROR,
			MEMBRANE_REASON_TOKENIZATION_FAILED, "tokenization failed");
		llama_model_free(model);
		return (llama_backend_free(), MEMBRANE_EXIT_RUNTIME_ERROR);
	}
	prompt_tokens.resize(rc);
	ctx_size = o.ctx > 0 ? o.ctx
		: (uint32_t)prompt_tokens.size() + (uint32_t)o.gen_tokens + 8;
	/* An explicit --ctx too small to even hold the prompt previously
	 * surfaced as an opaque "kv-store context creation failed" or
	 * "kv-store prompt decode failed" from deep inside decode_loop.cpp
	 * -- this is the user-facing product CLI, so name the real cause
	 * (the requested --ctx value) instead of an unrelated-sounding
	 * internal failure. */
	if (o.ctx > 0 && (uint64_t)o.ctx <= prompt_tokens.size())
	{
		fprintf(stderr, "membrane-run: --ctx %u is too small for this "
			"prompt (%zu tokens) -- need at least %zu\n", o.ctx,
			prompt_tokens.size(), prompt_tokens.size() + 1);
		print_error_json(o, MEMBRANE_EXIT_CLI_ERROR,
			MEMBRANE_REASON_CTX_TOO_SMALL,
			"--ctx " + std::to_string(o.ctx) + " is too small for this "
			"prompt (" + std::to_string(prompt_tokens.size()) + " tokens)",
			("Increase --ctx to at least "
				+ std::to_string(prompt_tokens.size() + 1) + ".").c_str());
		llama_model_free(model);
		return (llama_backend_free(), MEMBRANE_EXIT_CLI_ERROR);
	}
	read_model_shape(model, &shape);
	/* Phase 11A: resolve --kv adaptive into a concrete Q8/Q5 exactly
	 * once, into a separate `effective_o` copy -- `o.kv_mode` itself is
	 * left untouched (still ADAPTIVE when requested) so requested_kv_
	 * name(o.kv_mode) below and every print_*() site (via gs.adaptive_
	 * used/adaptive_selected_mode/adaptive_reason) can still report the
	 * ORIGINAL request alongside the resolution. GPU adaptive was
	 * already resolved inside resolve_gpu_config() above (before model
	 * load, from the pre-load GGUF estimate) -- gs.adaptive_used is
	 * already true there; only the CPU case (no GPU requested) still
	 * needs resolving here, now that the real post-load model shape
	 * and final ctx_size are both known. */
	membrane_run_opts_t	effective_o = o;

	if (o.kv_mode == MEMBRANE_KV_STORE_ADAPTIVE)
	{
		if (!gs.requested && !resolve_cpu_adaptive_kv(o, shape, ctx_size,
				&gs))
		{
			llama_model_free(model);
			return (llama_backend_free(), MEMBRANE_EXIT_UNSUPPORTED_KV);
		}
		effective_o.kv_mode = gs.adaptive_selected_mode;
	}
	/* --compare-kv/--gpu-bench precheck whichever mode they'll actually
	 * compare against (selected_comparison_mode() -- q8 by default, q5
	 * if --kv q5 was given, or adaptive's own resolved mode via
	 * effective_o if --kv adaptive was given); a plain run precheck's
	 * whatever effective_o.kv_mode resolved to. Either way this never
	 * checks more than the one mode about to be used. */
	if (o.compare_kv || o.gpu_bench)
	{
		if (!membrane_check_kv_compat(shape.arch_name.c_str(),
				shape.n_embd, shape.n_head, shape.n_head_kv, ctx_size,
				selected_comparison_mode(effective_o), &compat))
		{
			fprintf(stderr, "MEMBRANE: KV storage unsupported for "
				"this model: %s.\n", compat.reason);
			print_error_json(o, MEMBRANE_EXIT_UNSUPPORTED_KV,
				MEMBRANE_REASON_KV_COMPAT_UNSUPPORTED, compat.reason);
			llama_model_free(model);
			return (llama_backend_free(), MEMBRANE_EXIT_UNSUPPORTED_KV);
		}
	}
	else if (effective_o.kv_mode != MEMBRANE_KV_STORE_NATIVE)
	{
		if (!membrane_check_kv_compat(shape.arch_name.c_str(),
				shape.n_embd, shape.n_head, shape.n_head_kv, ctx_size,
				effective_o.kv_mode, &compat))
		{
			fprintf(stderr, "MEMBRANE: KV storage unsupported for "
				"this model: %s.\n", compat.reason);
			print_error_json(o, MEMBRANE_EXIT_UNSUPPORTED_KV,
				MEMBRANE_REASON_KV_COMPAT_UNSUPPORTED, compat.reason);
			llama_model_free(model);
			return (llama_backend_free(), MEMBRANE_EXIT_UNSUPPORTED_KV);
		}
	}
	/* Phase 12H: product_cli.cpp's parse-time validation already
	 * rejects --kv-placement combined with --compare-kv/--gpu-bench, so
	 * this only ever runs on the plain single-pass path -- see
	 * resolve_kv_placement()'s own doc comment for why AFTER model
	 * load, using effective_o.kv_mode. */
	if (!o.gpu_bench && !o.compare_kv
		&& !resolve_kv_placement(o, shape, ctx_size, effective_o.kv_mode,
			&gs))
	{
		llama_model_free(model);
		return (llama_backend_free(), MEMBRANE_EXIT_UNSUPPORTED_KV);
	}
	/* Phase 13.2, Section 11: --plan-only stops exactly here -- every
	 * resolution step above (GPU/adaptive/placement) has already run,
	 * using the real planner, no separate/lighter-weight estimate path
	 * (product_cli.cpp's parse-time validation already rejected
	 * --plan-only with --compare-kv/--gpu-bench, so this is never
	 * reached for those modes). Nothing below this point (run_kv_store_
	 * pass and everything downstream of it) ever executes. */
	if (o.plan_only)
	{
		build_plan_warnings(o, host, &gs);
		/* Review fix (CodeRabbit, PR #23): --quiet on a NORMAL run
		 * suppresses the startup summary/stats while the actual
		 * product of the command (generated text) still prints -- a
		 * real fallback exists. --plan-only has no such fallback: the
		 * plan IS the product, so gating it on !o.quiet made
		 * `--plan-only --quiet` a silent, indistinguishable-from-broken
		 * no-op (exit 0, zero output). Always print it regardless of
		 * --quiet.
		 *
		 * Self-review follow-up (same PR): this also must not be
		 * gated on !o.want_json -- run_normal_mode() above already
		 * establishes the project's own convention that stderr
		 * diagnostics print unconditionally on --json (only --quiet
		 * gates them; --json only changes what's on STDOUT). The
		 * previous version of this fix put print_startup_summary()
		 * behind an `else` on o.want_json, which silently dropped
		 * --verbose's own diagnostic block for `--plan-only --verbose
		 * --json` despite --help's "(both) always go to stderr"
		 * promise for --verbose. Printing it here unconditionally
		 * matches run_normal_mode()'s own convention and is harmless
		 * for a --json caller, which is expected to ignore stderr. */
		print_startup_summary(effective_o, model_label, ctx_size,
			kv_bytes_for_mode(shape, ctx_size, effective_o.kv_mode),
			shape.n_layer, gs, host);
		timings.total_ms = seconds_since(&total_t0) * 1000.0;
		if (o.want_json)
			print_plan_only_json(effective_o, model_label, ctx_size,
				shape.n_layer, gs, host, timings);
		else
			fprintf(stderr,
				"membrane-run: plan resolved (no generation performed)\n");
		llama_model_free(model);
		return (llama_backend_free(), MEMBRANE_EXIT_SUCCESS);
	}
	if (o.gpu_bench)
		rc = run_gpu_bench_mode(effective_o, model, prompt_tokens,
				model_label, ctx_size, shape, gs);
	else if (o.compare_kv)
		rc = run_compare_mode(effective_o, model, prompt_tokens,
				model_label, ctx_size, shape, gs);
	else
	{
		build_plan_warnings(o, host, &gs);
		rc = run_normal_mode(effective_o, &model, o.model_path, mp,
				&device_storage, prompt_tokens, model_label, ctx_size, shape,
				gs, host, timings, total_t0);
	}
	llama_model_free(model);
	llama_backend_free();
	return (rc);
}
