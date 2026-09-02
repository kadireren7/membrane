#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

/* Phase 28, Section 6: memfd_create()/dup2()/lseek() for the --json
 * parse-error stderr capture (parse_opts_capture_stderr() below) --
 * Linux-only, matching this project's existing Linux-only scope
 * (docs/install.md). */
#include <sys/mman.h>
#include <unistd.h>

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
#include "context_recommender.h"
#include "host_memory_guard.h"
#include "context_auto_cli.h"
#include "runtime_session.h"

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


/* Section 19/20 of the Phase 35 task: the concise, always-stderr
 * "Context recommendation" block -- printed once, right before
 * print_startup_summary()'s own existing plan block, for both the
 * normal-run and --plan-only paths. Never dumps every candidate by
 * default (Section 19: "Keep concise") -- that detail lives behind
 * --verbose, in print_ctxauto_verbose() below. Safety wording (Section
 * 20) is fixed, never "guaranteed"/"OOM-proof". Defined this early
 * (well before membrane_resolve_ctx_auto() itself, further down) because run_
 * normal_mode() (below) needs to call it at its own two internal
 * print_startup_summary() call sites. */
static void	print_ctxauto_summary(const membrane_run_opts_t &o,
				const membrane_ctxauto_outcome_t &ctxauto, bool plan_matches,
				const std::string &mismatch_detail)
{
	if (o.ctx_mode != MEMBRANE_RUN_CTX_AUTO || !ctxauto.rec.ok)
		return ;
	fprintf(stderr, "Context recommendation\n");
	fprintf(stderr, "  model max: %llu\n",
		(unsigned long long)ctxauto.rec.model_max_context);
	fprintf(stderr, "  minimum required: %llu\n",
		(unsigned long long)ctxauto.rec.minimum_required_context);
	fprintf(stderr, "  hardware fit: %llu\n",
		(unsigned long long)ctxauto.rec.hardware_fit_context);
	fprintf(stderr, "  recommended: %llu\n",
		(unsigned long long)ctxauto.rec.recommended_context);
	fprintf(stderr, "  policy: %s\n", ctxauto.rec.recommendation_policy);
	if (ctxauto.rec.host_memory_checked)
		fprintf(stderr, "  host memory: %s (required %.1f MiB, reserve "
			"%.1f MiB)\n", ctxauto.rec.host_memory_fit ? "fits"
				: "does not fit",
			(double)ctxauto.rec.host_required_bytes / (1024.0 * 1024.0),
			(double)ctxauto.rec.host_reserve_bytes / (1024.0 * 1024.0));
	fprintf(stderr, "  estimated to fit under the current memory snapshot "
		"and MEMBRANE's safety reserve -- not a guaranteed-no-OOM "
		"promise\n");
	if (!plan_matches)
		fprintf(stderr, "  NOTE: the applied plan differs from the "
			"recommendation (%s) -- memory conditions changed between "
			"recommendation and apply; the existing fallback mechanism "
			"resolved a different, still-legal plan. Context itself "
			"never changes once recommended.\n", mismatch_detail.c_str());
}

/* --verbose extension (Section 21): every evaluated candidate, its
 * feasibility, and reason code -- opt-in detail, never printed by
 * default. */
static void	print_ctxauto_verbose(const membrane_run_opts_t &o,
				const membrane_ctxauto_outcome_t &ctxauto)
{
	size_t	i;

	if (o.ctx_mode != MEMBRANE_RUN_CTX_AUTO || !o.verbose
		|| !ctxauto.rec.ok)
		return ;
	fprintf(stderr, "Context recommendation candidates (prompt tokens: "
		"%zu, %s path)\n", ctxauto.prompt_token_count,
		ctxauto.used_cpu_only_adaptive_path ? "CPU-only adaptive"
			: "joint planner");
	for (i = 0; i < ctxauto.rec.evaluated_count; ++i)
	{
		const membrane_ctxrec_evaluated_t	&ev = ctxauto.rec.evaluated[i];

		fprintf(stderr, "  ctx=%llu %s reason=%s",
			(unsigned long long)ev.ctx, ev.feasible ? "FEASIBLE"
				: "rejected", ev.reason_code);
		if (ev.feasible)
			fprintf(stderr, " gpu_layers=%d kv_precision=%d "
				"host_required=%.1f MiB", ev.selected_gpu_layers,
				ev.selected_kv_precision,
				(double)ev.host_required_bytes / (1024.0 * 1024.0));
		fprintf(stderr, "%s\n",
			(int)i == ctxauto.rec.hardware_fit_index ? "  <- selected" : "");
	}
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
 * inside membrane_resolve_gpu_config() below (defined ahead of the rest of this
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


/* Phase 13.2, Section 16: a small, deliberately non-exhaustive set of
 * informational plan warnings -- each one only pushed when it is both
 * true AND actionable (Section 16: "Do NOT spam warnings... Do not
 * create warnings without actionable meaning"). Called once, after
 * membrane_resolve_gpu_config()/membrane_resolve_cpu_adaptive_kv()/membrane_resolve_kv_placement()
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
	 * layers N proceeding unguarded -- membrane_resolve_gpu_config()'s own
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
	fprintf(stderr, "  kv precision: %s\n", membrane_kv_type_label(o.kv_mode));
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
		membrane_gpu_layers_label(gs.gpu_layers_requested).c_str());
	/* requested_kv_name() (below, print_gpu_json's neighbor) isn't
	 * declared yet at this point in the file -- same logic inlined,
	 * one place fewer to keep in sync would be nice, but a forward
	 * declaration purely to avoid two identical lines is not worth it
	 * here. */
	fprintf(stderr, "  kv precision: %s\n",
		gs.adaptive_used ? "adaptive" : membrane_kv_mode_name(o.kv_mode));
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
		/* Review fix (CodeRabbit, PR #23): membrane_resolve_gpu_config()'s "no
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
	fprintf(stderr, "  kv type: %s\n", membrane_kv_type_label(o.kv_mode));
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
			: membrane_kv_type_label(o.kv_mode));
	/* Section 1: "Human output must clearly show the same decision" --
	 * o here is already resolved (main.cpp's effective_o), so the "kv"
	 * line above always shows what actually runs; this line answers
	 * the separate question of whether that came from an explicit
	 * request or --kv adaptive's own policy, and why. */
	if (gs.adaptive_used)
		fprintf(stderr, "kv adaptive requested=adaptive selected=%s "
			"reason=%s\n", membrane_kv_mode_name(gs.adaptive_selected_mode),
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
		(double)kv_bytes / (1024.0 * 1024.0), membrane_kv_type_label(o.kv_mode));
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
			membrane_gpu_layers_label(gs.gpu_layers_requested).c_str(),
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
 * forward-declared above membrane_resolve_gpu_config(). Only ever called when
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
				const char *suggestion = NULL,
				const std::vector<std::string> *available_devices = NULL,
				const membrane_fallback_trace_t *fallback_trace = NULL)
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
 * membrane_resolve_cpu_adaptive_kv()'s and membrane_resolve_gpu_config()'s header
 * comments), never the raw ADAPTIVE request value, so membrane_kv_mode_name()/
 * membrane_kv_type_label() above are always safe to call on it directly. The
 * ORIGINAL request ("was this --kv adaptive at all?") is instead
 * carried by gs.adaptive_used/adaptive_selected_mode/adaptive_reason
 * -- this helper renders the "requested_kv" field consistently from
 * that, one place, for every print_*() function. */
static const char	*requested_kv_name(const membrane_run_opts_t &o,
					const membrane_gpu_state_t &gs)
{
	if (gs.adaptive_used)
		return ("adaptive");
	return (membrane_kv_mode_name(o.kv_mode));
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
		 * for a CPU-resolved decision (see membrane_resolve_cpu_adaptive_kv()). */
		printf("\"adaptive\":{\"selected_kv\":\"%s\",\"reason\":\"",
			membrane_kv_mode_name(gs.adaptive_selected_mode));
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
 * to 0.0 whenever its own membrane_resolve_gpu_config() call site didn't run on
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
/* Section 22 of the Phase 35 task: additive-only -- schema_version
 * stays 1. Printed (as its own top-level "context_recommendation"
 * object, with its own leading comma) only when --ctx auto actually
 * ran; a plain --ctx N run's JSON is byte-for-byte unaffected (this
 * function prints nothing at all for that case). */
static void	print_context_recommendation_json(const membrane_run_opts_t &o,
				const membrane_ctxauto_outcome_t &ctxauto)
{
	if (o.ctx_mode != MEMBRANE_RUN_CTX_AUTO)
		return ;
	printf(",");
	printf("\"context_recommendation\":{\"requested\":\"auto\","
		"\"model_max_context\":%llu,\"minimum_required_context\":%llu,"
		"\"hardware_fit_context\":%llu,\"recommended_context\":%llu,"
		"\"policy\":\"%s\",\"host_memory_checked\":%s,"
		"\"host_memory_fit\":%s,\"host_required_bytes\":%llu,"
		"\"host_available_bytes\":%llu,\"host_reserve_bytes\":%llu}",
		(unsigned long long)ctxauto.rec.model_max_context,
		(unsigned long long)ctxauto.rec.minimum_required_context,
		(unsigned long long)ctxauto.rec.hardware_fit_context,
		(unsigned long long)ctxauto.rec.recommended_context,
		ctxauto.rec.recommendation_policy,
		ctxauto.rec.host_memory_checked ? "true" : "false",
		ctxauto.rec.host_memory_fit ? "true" : "false",
		(unsigned long long)ctxauto.rec.host_required_bytes,
		(unsigned long long)ctxauto.rec.host_available_bytes,
		(unsigned long long)ctxauto.rec.host_reserve_bytes);
}

static void	print_run_json(const membrane_run_opts_t &o,
				const char *model_label, const membrane_kv_store_telemetry_t &t,
				const std::string &text, const membrane_gpu_state_t &gs,
				const membrane_host_meminfo_t &host, bool success,
				int exit_code, size_t prompt_tokens_count,
				const membrane_gpu_memory_observed_t &gpu_mem_observed,
				const membrane_timings_t &timings,
				const membrane_ctxauto_outcome_t &ctxauto)
{
	printf("{\"schema_version\":1,\"membrane_version\":\"%s\","
		"\"mode\":\"run\",\"ok\":true,\"model_label\":\"%s\","
		"\"kv_store\":\"%s\","
		"\"kv_type\":\"%s\",\"requested_kv\":\"%s\",\"selected_kv\":\"%s\","
		"\"adaptive_reason\":\"%s\","
		"\"ctx_size\":%u,\"generated_tokens\":%llu,",
		MEMBRANE_VERSION, model_label, membrane_kv_mode_name(o.kv_mode),
		membrane_kv_type_label(o.kv_mode), requested_kv_name(o, gs),
		membrane_kv_mode_name(o.kv_mode),
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
		membrane_gpu_layers_label(gs.gpu_layers_requested).c_str(),
		gs.adaptive_used ? "adaptive" : membrane_kv_mode_name(o.kv_mode),
		membrane_kv_placement_mode_name(o.kv_placement));
	print_json_escaped(gs.device_requested);
	printf("\"},");
	printf("\"resolved\":{\"backend\":\"");
	print_json_escaped(gs.backend_selected);
	printf("\",\"device\":\"");
	print_json_escaped(gs.device_selected);
	printf("\",\"gpu_layers\":%d,\"kv\":\"%s\",\"kv_placement\":\"%s\"},",
		gs.gpu_layers_selected, membrane_kv_mode_name(o.kv_mode),
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
	print_context_recommendation_json(o, ctxauto);
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
 * (membrane_resolve_gpu_config()/membrane_resolve_kv_placement()) already produced. */
static void	print_plan_only_json(const membrane_run_opts_t &o,
				const char *model_label, uint32_t ctx_size,
				int32_t n_layer_total, const membrane_gpu_state_t &gs,
				const membrane_host_meminfo_t &host,
				const membrane_timings_t &timings,
				const membrane_ctxauto_outcome_t &ctxauto)
{
	printf("{\"schema_version\":1,\"membrane_version\":\"%s\","
		"\"mode\":\"plan\",\"ok\":true,\"model_label\":\"",
		MEMBRANE_VERSION);
	print_json_escaped(model_label);
	printf("\",\"ctx_size\":%u,\"model_layers\":%d,", ctx_size, n_layer_total);
	printf("\"requested\":{\"auto\":%s,\"gpu_layers\":\"%s\",\"kv\":\"%s\","
		"\"kv_placement\":\"%s\",\"device\":\"",
		o.auto_mode ? "true" : "false",
		membrane_gpu_layers_label(gs.gpu_layers_requested).c_str(),
		gs.adaptive_used ? "adaptive" : membrane_kv_mode_name(o.kv_mode),
		membrane_kv_placement_mode_name(o.kv_placement));
	print_json_escaped(gs.device_requested);
	printf("\"},");
	printf("\"resolved\":{\"backend\":\"");
	print_json_escaped(gs.backend_selected);
	printf("\",\"device\":\"");
	print_json_escaped(gs.device_selected);
	printf("\",\"gpu_layers\":%d,\"kv\":\"%s\",\"kv_placement\":\"%s\"},",
		gs.gpu_layers_selected, membrane_kv_mode_name(o.kv_mode),
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
	 * too, via membrane_resolve_cpu_adaptive_kv(). */
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
	printf("}");
	print_context_recommendation_json(o, ctxauto);
	printf(",\"reason_codes\":{\"primary\":\"");
	print_json_escaped(plan_primary_reason(o, gs));
	printf("\",\"trace\":");
	print_json_string_array(gs.reason_trace);
	printf("},\"warnings\":");
	print_json_string_array(gs.warnings);
	printf("}\n");
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
/*
 * Mega Phase A, PR A1: delegates the entire post-model-load pipeline
 * (tokenize, ctx_size, adaptive-KV-CPU, compat, KV placement, generate --
 * with the same bounded apply-time fallback, Phase 21, when the session's
 * plan came from the joint planner) to the reusable runtime-session core
 * (runtime_session.h/.cpp) via membrane_session_generate(), then prints the
 * EXACT same output this function always did -- every print_*() call below
 * is unchanged from before this extraction, only now fed from the session
 * API's plain-data result instead of local variables computed inline.
 *
 * `o`/`ctx_size`/`shape` are main()'s own already-resolved copies (used
 * only for the pre-generation "Context recommendation"/startup-summary
 * print, which must happen before generation starts, matching the
 * original ordering exactly) -- membrane_session_generate() re-derives its
 * own copies of these from `prompt_text`/the session's model internally
 * (deterministic, so always identical); this is deliberate, small,
 * disclosed redundancy in exchange for a single shared implementation,
 * not a correctness risk (see runtime_session.h's own top comment).
 */
static int	run_normal_mode(const membrane_run_opts_t &o,
				membrane_model_session_t *session,
				const std::string &prompt_text, const char *model_label,
				uint32_t ctx_size, const model_shape_t &shape,
				membrane_gpu_state_t &gs, const membrane_host_meminfo_t &host,
				membrane_timings_t timings, const struct timespec &total_t0,
				const membrane_ctxauto_outcome_t &ctxauto,
				bool ctxauto_plan_matches,
				const std::string &ctxauto_plan_mismatch_detail)
{
	membrane_generation_request_t	req = {};
	membrane_generation_result_t	res;
	bool							stream;

	stream = !o.want_json && !o.quiet;
	if (!o.quiet)
	{
		print_ctxauto_summary(o, ctxauto, ctxauto_plan_matches,
			ctxauto_plan_mismatch_detail);
		print_ctxauto_verbose(o, ctxauto);
		print_startup_summary(o, model_label, ctx_size,
			membrane_kv_bytes_for_mode(shape, ctx_size, o.kv_mode),
			shape.n_layer, gs, host);
	}
	req.o = &o;
	req.prompt_text = prompt_text;
	req.ctx_size = ctx_size;
	req.token_cb = stream ? stream_token : NULL;
	membrane_session_generate(session, req, &res);
	gs = session->gs;
	if (res.fallback_engaged && !o.quiet)
		print_fallback_diagnostics(res.fallback_trace, o.verbose);
	if (!res.ok && res.err.set)
	{
		fprintf(stderr, "%s\n", res.err.human.c_str());
		print_error_json(o, res.exit_code, res.err.reason_code,
			res.err.message,
			res.err.suggestion.empty() ? NULL : res.err.suggestion.c_str(),
			res.err.available_devices.empty() ? NULL
				: &res.err.available_devices,
			res.fallback_engaged ? &res.fallback_trace : NULL);
		return (res.exit_code);
	}
	timings.total_ms = seconds_since(&total_t0) * 1000.0;
	if (o.want_json)
	{
		membrane_gpu_memory_observed_t	gpu_mem_observed;
		membrane_run_opts_t				effective_o = o;

		effective_o.kv_mode = res.effective_kv_mode;
		observe_gpu_memory_after_run(gs, &gpu_mem_observed);
		print_run_json(effective_o, model_label, res.tel, res.text, gs, host,
			res.gen_result.ok, res.gen_result.ok ? MEMBRANE_EXIT_SUCCESS
				: MEMBRANE_EXIT_RUNTIME_ERROR, res.prompt_tokens.size(),
			gpu_mem_observed, timings, ctxauto);
	}
	else
	{
		if (!stream)
			fwrite(res.text.data(), 1, res.text.size(), stdout);
		if (!o.quiet)
			print_run_human_stats(res.tel);
		else
			putchar('\n');
	}
	return (res.gen_result.ok ? MEMBRANE_EXIT_SUCCESS
		: MEMBRANE_EXIT_RUNTIME_ERROR);
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
			membrane_gpu_layers_label(gs.gpu_layers_requested).c_str());
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

	candidate_name = membrane_kv_mode_name(candidate_mode);
	memset(tel_candidate, 0, sizeof(*tel_candidate));
	tel_candidate->kv_store_mode_name = candidate_name;
	tel_candidate->no_fallback_occurred = 1;
	tel_candidate->ctx_size = ctx_size;
	tel_candidate->compressed_kv_allocated_bytes =
		membrane_kv_bytes_for_mode(shape, ctx_size, candidate_mode);
	*native_bytes = membrane_native_kv_bytes(shape, ctx_size);
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
		membrane_gpu_layers_label(gs.gpu_layers_requested).c_str(),
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

/* Phase 28, Section 6: pure argv scan, independent of membrane_run_
 * parse_opts()'s own internal state -- the whole point is to know
 * whether --json was requested even when parsing fails on an EARLIER
 * argument before the parser's own loop would ever reach --json (e.g.
 * `membrane-run --bogus --json`: the parser returns on --bogus, so
 * o.want_json is never set even though the user did ask for JSON). */
static bool	argv_has_json_flag(int argc, char **argv)
{
	int	i;

	i = 1;
	while (i < argc)
	{
		if (strcmp(argv[i], "--json") == 0)
			return (true);
		++i;
	}
	return (false);
}

/* Phase 28, Section 6: membrane_run_parse_opts() already prints one
 * clear "membrane-run: ...\n" line (occasionally a couple more, e.g.
 * --device's "Available:" listing) to stderr on every failure --
 * captured here (fd 2 redirected to an anonymous in-memory file for
 * the duration of the call only, via memfd_create -- Linux-only, same
 * scope as the rest of this project) so a --json caller gets that
 * exact text inside a machine-readable object on stdout, without a
 * second hand-maintained copy of the parser's own error strings. The
 * real stderr still receives the message afterward (re-emitted
 * verbatim once fd 2 is restored) -- a --json run's stderr diagnostics
 * are unaffected, matching this file's existing "stderr always gets
 * diagnostics regardless of --json" convention. Falls back to an
 * uncaptured call (captured left empty) if either fd operation fails,
 * rather than losing the parse result itself. */
static int	parse_opts_capture_stderr(int argc, char **argv,
				membrane_run_opts_t *o, std::string *captured)
{
	int		saved_fd;
	int		mem_fd;
	int		rc;
	off_t	len;

	captured->clear();
	fflush(stderr);
	saved_fd = dup(STDERR_FILENO);
	mem_fd = (int)memfd_create("membrane-run-parse-stderr", 0);
	if (saved_fd < 0 || mem_fd < 0)
	{
		if (saved_fd >= 0)
			close(saved_fd);
		if (mem_fd >= 0)
			close(mem_fd);
		return (membrane_run_parse_opts(argc, argv, o));
	}
	dup2(mem_fd, STDERR_FILENO);
	rc = membrane_run_parse_opts(argc, argv, o);
	fflush(stderr);
	dup2(saved_fd, STDERR_FILENO);
	close(saved_fd);
	len = lseek(mem_fd, 0, SEEK_END);
	if (len > 0)
	{
		std::vector<char>	buf((size_t)len);
		ssize_t				n;

		lseek(mem_fd, 0, SEEK_SET);
		n = read(mem_fd, buf.data(), (size_t)len);
		if (n > 0)
			captured->assign(buf.data(), (size_t)n);
	}
	close(mem_fd);
	if (!captured->empty())
		fputs(captured->c_str(), stderr);
	return (rc);
}

/* Phase 28, Section 6: the CLI-parse-time JSON error contract -- a
 * SEPARATE, smaller object shape from print_error_json()'s runtime
 * error object above (that one requires a fully-parsed `o` and a
 * chosen reason_code per failure class; a parse failure has neither).
 * Purely additive: this path previously emitted no JSON at all on
 * stdout (a pre-existing gap -- Phase 27 found parse-time failures did
 * not respect --json), so there is no prior shape to stay compatible
 * with here. suggestions is always a non-empty array -- Section 10:
 * never an empty/absent suggestion, and never one that doesn't apply
 * to the actual failure. */
static void	print_cli_parse_error_json(const membrane_run_opts_t &o,
				int exit_code, const std::string &raw_message)
{
	std::string			message = raw_message;
	const std::string	prefix = "membrane-run: ";

	if (message.compare(0, prefix.size(), prefix) == 0)
		message.erase(0, prefix.size());
	while (!message.empty() && (message.back() == '\n' || message.back() == '\r'))
		message.pop_back();
	if (message.empty())
		message = "invalid command line arguments";
	printf("{\"schema_version\":1,\"membrane_version\":\"%s\","
		"\"ok\":false,\"exit_code\":%d,\"reason_code\":\"%s\","
		"\"message\":\"", MEMBRANE_VERSION, exit_code,
		MEMBRANE_REASON_CLI_PARSE_ERROR);
	print_json_escaped(message);
	printf("\",\"suggestions\":[");
	/* Section 4/7: the one parse-time failure with a genuinely specific
	 * fix known at this point -- every other parse failure gets the
	 * always-true, always-actionable fallback (Section 7: never a
	 * generic "try random flags" suggestion, but "see --help" for the
	 * full legal option set is always both true and actionable). */
	if (o.auto_mode && o.ctx == 0)
		printf("\"--ctx 2048\"");
	else
		printf("\"Run 'membrane-run --help' for the full option list.\"");
	printf("]}\n");
	fflush(stdout);
}

/* Phase 28, Section 10: --list-devices needs no model, no context size,
 * no policy resolution -- just the same device enumeration resolve_gpu_
 * config() already uses for --device's own error listing, printed
 * directly. Always succeeds (MEMBRANE_EXIT_SUCCESS): an empty GPU list
 * is a truthful, non-error diagnostic outcome (Section 10: "This should
 * be lightweight. No model required."). */
static int	run_list_devices_mode(const membrane_run_opts_t &o)
{
	membrane_gpu_device_info_t	devices[MEMBRANE_GPU_MAX_DEVICES];
	size_t						n_devices;
	size_t						i;

	n_devices = membrane_gpu_list_devices(devices, MEMBRANE_GPU_MAX_DEVICES);
	if (o.want_json)
	{
		printf("{\"schema_version\":1,\"membrane_version\":\"%s\","
			"\"mode\":\"list-devices\",\"ok\":true,\"devices\":[",
			MEMBRANE_VERSION);
		for (i = 0; i < n_devices; ++i)
		{
			printf("%s{\"index\":%zu,\"backend\":\"", i == 0 ? "" : ",", i);
			print_json_escaped(devices[i].backend);
			printf("\",\"name\":\"");
			print_json_escaped(devices[i].name);
			printf("\",\"description\":\"");
			print_json_escaped(devices[i].description);
			printf("\",\"type\":\"%s\",\"memory_total_bytes\":%llu,"
				"\"memory_free_bytes\":%llu}",
				devices[i].type == MEMBRANE_DEV_TYPE_CPU ? "cpu"
					: devices[i].type == MEMBRANE_DEV_TYPE_GPU ? "gpu"
					: devices[i].type == MEMBRANE_DEV_TYPE_IGPU ? "igpu"
					: "unknown",
				(unsigned long long)devices[i].memory_total,
				(unsigned long long)devices[i].memory_free);
		}
		printf("]}\n");
		fflush(stdout);
		return (MEMBRANE_EXIT_SUCCESS);
	}
	printf("INDEX  BACKEND    TYPE  TOTAL MiB   FREE MiB  NAME\n");
	for (i = 0; i < n_devices; ++i)
		printf("%-6zu %-10s %-5s %10.1f %10.1f  %s (%s)\n", i,
			devices[i].backend, devices[i].type == MEMBRANE_DEV_TYPE_CPU
				? "cpu" : devices[i].type == MEMBRANE_DEV_TYPE_GPU ? "gpu"
				: devices[i].type == MEMBRANE_DEV_TYPE_IGPU ? "igpu"
				: "unk",
			(double)devices[i].memory_total / (1024.0 * 1024.0),
			(double)devices[i].memory_free / (1024.0 * 1024.0),
			devices[i].name, devices[i].description);
	if (n_devices == 0)
		printf("(no backend devices enumerated)\n");
	return (MEMBRANE_EXIT_SUCCESS);
}

/* Phase 28, Section 15: a handful of cheap, non-destructive checks --
 * deliberately NOT a subsystem (Section 15: "If implementing --doctor
 * would become a subsystem: do not add it"). Every check here reuses
 * an already-existing, already-tested query (gpu_device.h's device
 * enumeration, membrane_read_host_meminfo(), a plain fopen() readability probe,
 * and -- Phase 30 -- membrane_gpu_estimate_model()/membrane_check_kv_
 * compat() when --model is given, the SAME GGUF-metadata-only read and
 * compatibility check --inspect-model uses below, never a second
 * implementation). Always exits 0: a [WARN] line is informational
 * (Section 15's own example shows a [WARN] for "CUDA not supported"
 * without failing the command), never a failure. */
static int	run_doctor_mode(const membrane_run_opts_t &o)
{
	membrane_gpu_device_info_t	devices[MEMBRANE_GPU_MAX_DEVICES];
	size_t						n_devices;
	size_t						i;
	size_t						gpu_count = 0;
	const char					*cpu_description = NULL;
	membrane_host_meminfo_t		host;
	bool						model_readable = false;
	bool						model_checked = false;
	membrane_gpu_model_estimate_t	est;
	bool						model_hparams_ok = false;
	membrane_compat_result_t		compat_q8 = {};
	membrane_compat_result_t		compat_q5 = {};

	n_devices = membrane_gpu_list_devices(devices, MEMBRANE_GPU_MAX_DEVICES);
	for (i = 0; i < n_devices; ++i)
	{
		if (devices[i].type == MEMBRANE_DEV_TYPE_GPU
			|| devices[i].type == MEMBRANE_DEV_TYPE_IGPU)
			++gpu_count;
		/* Phase 30, Section 9: the enumeration already includes a real
		 * CPU entry (ggml's own CPU backend description, e.g. "AMD
		 * Ryzen 5 5600H...") -- --list-devices already shows it; doctor
		 * previously didn't, even though it's free (already fetched
		 * above for the GPU count) and directly answers Section 9's
		 * "detected CPU" ask with no new detection logic. */
		else if (devices[i].type == MEMBRANE_DEV_TYPE_CPU
			&& cpu_description == NULL)
			cpu_description = devices[i].description;
	}
	membrane_read_host_meminfo(&host);
	if (o.model_path != NULL)
	{
		FILE	*f;

		model_checked = true;
		f = fopen(o.model_path, "rb");
		if (f != NULL)
			model_readable = true, fclose(f);
		/* Cheap (GGUF-metadata-only, no model load) -- same call
		 * --inspect-model uses. Only attempted if the file opened
		 * above; membrane_gpu_estimate_model() itself also re-verifies
		 * readability/parseability, so a model_readable=true/hparams
		 * read failure IS possible (e.g. a non-GGUF file with a
		 * readable header) and handled below via model_hparams_ok. */
		if (model_readable
			&& membrane_gpu_estimate_model(o.model_path, &est)
			&& est.hparams_available)
		{
			model_hparams_ok = true;
			membrane_check_kv_compat(est.arch_name, est.n_embd, est.n_head,
				est.n_head_kv, 1, MEMBRANE_KV_STORE_Q8, &compat_q8);
			membrane_check_kv_compat(est.arch_name, est.n_embd, est.n_head,
				est.n_head_kv, 1, MEMBRANE_KV_STORE_Q5, &compat_q5);
		}
	}
	if (o.want_json)
	{
		printf("{\"schema_version\":1,\"membrane_version\":\"%s\","
			"\"mode\":\"doctor\",\"ok\":true,"
			"\"executable\":true,"
			"\"gpu_backend_available\":%s,"
			"\"gpu_devices_found\":%zu,"
			"\"host_meminfo_available\":%s",
			MEMBRANE_VERSION,
			membrane_gpu_backend_available() ? "true" : "false", gpu_count,
			host.ok ? "true" : "false");
		if (cpu_description != NULL)
		{
			printf(",\"cpu\":\"");
			print_json_escaped(cpu_description);
			printf("\"");
		}
		if (host.ok)
			printf(",\"host_total_bytes\":%llu,\"host_available_bytes\":%llu",
				(unsigned long long)host.total_bytes,
				(unsigned long long)host.available_bytes);
		if (model_checked)
		{
			printf(",\"model_readable\":%s",
				model_readable ? "true" : "false");
			if (model_hparams_ok)
			{
				printf(",\"model_architecture\":\"");
				print_json_escaped(est.arch_name);
				printf("\",\"model_compatibility\":{\"q8\":%s,\"q5\":%s,"
					"\"adaptive\":%s}",
					compat_q8.ok ? "true" : "false",
					compat_q5.ok ? "true" : "false",
					(compat_q8.ok || compat_q5.ok) ? "true" : "false");
			}
		}
		printf("}\n");
		fflush(stdout);
		return (MEMBRANE_EXIT_SUCCESS);
	}
	printf("MEMBRANE diagnostics\n\n");
	printf("[OK] MEMBRANE %s\n", MEMBRANE_VERSION);
	printf("[OK] executable\n");
	if (cpu_description != NULL)
		printf("[OK] CPU: %s\n", cpu_description);
	/* Phase 29: llama_supports_gpu_offload() (which membrane_gpu_
	 * backend_available() wraps) checks for an actually-ENUMERATED GPU/
	 * IGPU device (ggml_backend_dev_by_type() != NULL), not "was a GPU
	 * backend compiled into this binary" -- confirmed directly against
	 * third_party/llama.cpp/src/llama.cpp. A Vulkan-enabled build with
	 * zero visible devices (a real, common case: a container with no
	 * /dev/dri passthrough, discovered via this phase's own real
	 * install/remove container validation) therefore makes
	 * membrane_gpu_backend_available() return false for the SAME reason
	 * gpu_count is 0 -- a prior version of this branch asserted "no GPU
	 * backend compiled in" (with rebuild advice) in exactly that case,
	 * which is false and actively misleading on an already-Vulkan-
	 * enabled build. gpu_count alone -- not membrane_gpu_backend_
	 * available() -- is the only ground truth this CLI can honestly
	 * report; one combined message covers both real causes without
	 * asserting which one it is. */
	if (gpu_count > 0)
	{
		printf("[OK] GPU backend compiled in\n");
		for (i = 0; i < n_devices; ++i)
			if (devices[i].type == MEMBRANE_DEV_TYPE_GPU
				|| devices[i].type == MEMBRANE_DEV_TYPE_IGPU)
				printf("[OK] GPU device: %s (%s)\n", devices[i].name,
					devices[i].description);
	}
	else
		printf("[WARN] no GPU device visible to this build -- either no "
			"GPU backend is compiled in (rebuild with e.g. "
			"-DGGML_VULKAN=ON), or a backend is compiled in but found no "
			"usable device on this host at runtime (driver/hardware not "
			"detected, or none passed through if this is a container)\n");
	if (host.ok)
		printf("[OK] host RAM detected: %.1f MiB total, %.1f MiB "
			"available\n", (double)host.total_bytes / (1024.0 * 1024.0),
			(double)host.available_bytes / (1024.0 * 1024.0));
	else
		printf("[WARN] host RAM could not be read (non-Linux host, or "
			"/proc/meminfo unavailable)\n");
	if (model_checked)
	{
		if (!model_readable)
			printf("[WARN] model file not readable: %s\n",
				membrane_runtime_safe_basename(o.model_path));
		else if (!model_hparams_ok)
			printf("[WARN] model file readable but its GGUF metadata "
				"could not be read: %s\n",
				membrane_runtime_safe_basename(o.model_path));
		else
		{
			printf("[OK] model file readable: %s\n",
				membrane_runtime_safe_basename(o.model_path));
			printf("[OK] model architecture: %s\n", est.arch_name);
			if (compat_q8.ok || compat_q5.ok)
				printf("[OK] compressed KV supported: %s%s%s\n",
					compat_q8.ok ? "q8" : "",
					(compat_q8.ok && compat_q5.ok) ? ", " : "",
					compat_q5.ok ? "q5" : "");
			else
				printf("[WARN] compressed KV (q8/q5/adaptive) not "
					"supported for architecture '%s' -- native KV still "
					"works\n", est.arch_name);
		}
	}
	printf("[OK] LLM_ARCH_LLAMA and LLM_ARCH_QWEN2 compressed-KV "
		"compatibility available (see docs/compatibility.md)\n");
	/* Phase 30, Section 10: end with ONE concrete, relevant next
	 * command -- never irrelevant advice (Section 10: "Do not print
	 * irrelevant advice"). Model-aware when --model was given (reuses
	 * the real basename, not a placeholder); GPU-aware otherwise
	 * (--auto --plan-only only makes sense to suggest when a GPU is
	 * actually visible to offload to -- Section 10's own two examples). */
	printf("\n");
	if (model_checked && model_readable)
	{
		const char	*label = membrane_runtime_safe_basename(o.model_path);

		if (gpu_count > 0)
			printf("Next: membrane-run --model %s --ctx 2048 --auto "
				"--plan-only\n", label);
		else
			printf("Next: membrane-run --model %s --prompt \"Hello\"\n",
				label);
	}
	else if (gpu_count > 0)
		printf("Next: membrane-run --model model.gguf --ctx 2048 --auto "
			"--plan-only\n");
	else
		printf("Next: membrane-run --model model.gguf --prompt \"Hello\"\n");
	return (MEMBRANE_EXIT_SUCCESS);
}

/* Phase 30, Section 5-8: --inspect-model -- reads ONLY GGUF metadata
 * (membrane_gpu_estimate_model(), gpu_device.h -- no llama_model_load_
 * from_file(), no llama_backend_init() even) and reuses the exact same
 * membrane_check_kv_compat() a real run's precheck calls (compat_
 * check.c) -- never a second compatibility implementation (Section 8).
 * Scope deliberately narrow (Section 6: "Do NOT turn this into a GGUF
 * explorer"): basename, architecture, layer/embedding/head shape,
 * native/q8/q5/adaptive support, and -- only if the caller also gave
 * --ctx -- an estimated KV size per mode at that context. No --ctx
 * means no memory-estimate section at all, never a silently invented
 * default (Section 6). */
static void	print_inspect_model_json(const char *model_label,
				const membrane_gpu_model_estimate_t &est,
				const membrane_compat_result_t &compat_q8,
				const membrane_compat_result_t &compat_q5,
				bool have_ctx, uint32_t ctx_size,
				const model_shape_t &shape)
{
	printf("{\"schema_version\":1,\"membrane_version\":\"%s\","
		"\"mode\":\"inspect-model\",\"ok\":true,\"model\":{"
		"\"file\":\"", MEMBRANE_VERSION);
	print_json_escaped(model_label);
	printf("\",\"architecture\":\"");
	print_json_escaped(est.arch_name);
	printf("\",\"layers\":%d,\"embedding\":%d,\"heads\":%d,"
		"\"kv_heads\":%d", est.n_layer, est.n_embd, est.n_head,
		est.n_head_kv);
	/* Phase 35, Section 26: additive -- the model's own real GGUF
	 * context-length ceiling (gpu_device.h's model_max_context, Phase
	 * 33), independent of whether --ctx was given at all. */
	if (est.model_max_context_available)
		printf(",\"model_max_context\":%llu",
			(unsigned long long)est.model_max_context);
	printf("},\"compatibility\":{\"native\":true,"
		"\"q8\":%s,\"q5\":%s,\"adaptive\":%s",
		compat_q8.ok ? "true" : "false", compat_q5.ok ? "true" : "false",
		(compat_q8.ok || compat_q5.ok) ? "true" : "false");
	if (!compat_q8.ok)
	{
		printf(",\"q8_reason\":\"");
		print_json_escaped(compat_q8.reason);
		printf("\"");
	}
	if (!compat_q5.ok)
	{
		printf(",\"q5_reason\":\"");
		print_json_escaped(compat_q5.reason);
		printf("\"");
	}
	printf("}");
	if (have_ctx)
	{
		printf(",\"kv_estimate_bytes\":{\"ctx\":%u,\"native\":%llu,"
			"\"q8\":%llu,\"q5\":%llu}", ctx_size,
			(unsigned long long)membrane_kv_bytes_for_mode(shape, ctx_size,
				MEMBRANE_KV_STORE_NATIVE),
			(unsigned long long)membrane_kv_bytes_for_mode(shape, ctx_size,
				MEMBRANE_KV_STORE_Q8),
			(unsigned long long)membrane_kv_bytes_for_mode(shape, ctx_size,
				MEMBRANE_KV_STORE_Q5));
	}
	printf("}\n");
	fflush(stdout);
}

static int	run_inspect_model_mode(const membrane_run_opts_t &o)
{
	membrane_gpu_model_estimate_t	est;
	const char						*model_label;

	model_label = membrane_runtime_safe_basename(o.model_path);
	if (!membrane_gpu_estimate_model(o.model_path, &est)
		|| !est.hparams_available)
	{
		/* Two genuinely different real causes -- confirmed directly
		 * (models/stories15M.gguf, a real fixture in this repo, is a
		 * valid GGUF with real "blk.N." tensors that membrane_gpu_
		 * estimate_model() reads fine, but lacks the arch-prefixed
		 * hparam keys this tool needs, so hparams_available is 0 with
		 * no parse failure at all -- a prior version of this message
		 * called that "not a valid GGUF file", which is false and
		 * confusing for exactly this case). */
		bool	parse_failed = !est.hparams_available
			&& est.n_layer == 0 && est.bytes_per_layer == 0;
		std::string	detail = parse_failed
			? (std::string("could not read '") + model_label
				+ "' as a GGUF file -- verify the path and file format")
			: (std::string("'") + model_label + "' is a readable GGUF "
				"file, but its architecture/hparams metadata could not "
				"be read -- compatibility can't be determined for it");

		fprintf(stderr, "membrane-run: %s\n", detail.c_str());
		if (o.want_json)
		{
			printf("{\"schema_version\":1,\"membrane_version\":\"%s\","
				"\"mode\":\"inspect-model\",\"ok\":false,\"exit_code\":%d,"
				"\"error\":{\"reason_code\":\"%s\",\"message\":\"",
				MEMBRANE_VERSION, MEMBRANE_EXIT_MODEL_ERROR,
				MEMBRANE_GPU_POLICY_REASON_METADATA_UNAVAILABLE);
			print_json_escaped(detail);
			printf("\"}}\n");
			fflush(stdout);
		}
		return (MEMBRANE_EXIT_MODEL_ERROR);
	}

	model_shape_t	shape;

	shape.arch_name = est.arch_name;
	shape.n_layer = est.n_layer;
	shape.n_embd = est.n_embd;
	shape.n_head = est.n_head;
	shape.n_head_kv = est.n_head_kv;
	shape.n_embd_gqa = (shape.n_head > 0)
		? (int64_t)(shape.n_embd / shape.n_head) * shape.n_head_kv : 0;

	membrane_compat_result_t	compat_q8;
	membrane_compat_result_t	compat_q5;

	/* ctx_size argument here is a compat-check precondition only (must
	 * be >=1 -- compat_check.c never uses it for divisibility math or
	 * any other decision), NOT a memory estimate -- that only happens
	 * below, and only if the caller actually gave --ctx. */
	membrane_check_kv_compat(est.arch_name, est.n_embd, est.n_head,
		est.n_head_kv, 1, MEMBRANE_KV_STORE_Q8, &compat_q8);
	membrane_check_kv_compat(est.arch_name, est.n_embd, est.n_head,
		est.n_head_kv, 1, MEMBRANE_KV_STORE_Q5, &compat_q5);

	bool	have_ctx = o.ctx > 0;

	if (o.want_json)
	{
		print_inspect_model_json(model_label, est, compat_q8, compat_q5,
			have_ctx, o.ctx, shape);
		return (MEMBRANE_EXIT_SUCCESS);
	}
	printf("Model\n");
	printf("  file: %s\n", model_label);
	printf("  architecture: %s\n", est.arch_name);
	printf("  layers: %d\n", est.n_layer);
	printf("  embedding: %d\n", est.n_embd);
	printf("  heads: %d\n", est.n_head);
	printf("  KV heads: %d\n", est.n_head_kv);
	if (est.model_max_context_available)
		printf("  context maximum: %llu\n",
			(unsigned long long)est.model_max_context);
	printf("\nMEMBRANE compatibility\n");
	printf("  native KV: supported\n");
	printf("  q8: %s\n", compat_q8.ok ? "supported" : compat_q8.reason);
	printf("  q5: %s\n", compat_q5.ok ? "supported" : compat_q5.reason);
	printf("  adaptive: %s\n", (compat_q8.ok || compat_q5.ok)
		? "supported" : "not supported (neither q8 nor q5 is)");
	if (have_ctx)
	{
		printf("\nEstimated KV @ ctx %u\n", o.ctx);
		printf("  native: %.2f MiB\n",
			(double)membrane_kv_bytes_for_mode(shape, o.ctx, MEMBRANE_KV_STORE_NATIVE)
				/ (1024.0 * 1024.0));
		printf("  q8: %.2f MiB\n",
			(double)membrane_kv_bytes_for_mode(shape, o.ctx, MEMBRANE_KV_STORE_Q8)
				/ (1024.0 * 1024.0));
		printf("  q5: %.2f MiB\n",
			(double)membrane_kv_bytes_for_mode(shape, o.ctx, MEMBRANE_KV_STORE_Q5)
				/ (1024.0 * 1024.0));
	}
	/* Phase 35, Section 26: --inspect-model stays deliberately
	 * lightweight -- it never runs the hardware-aware recommendation
	 * itself (that needs a real prompt, which this read-only mode
	 * never requires). --ctx auto is accepted here (no parse error)
	 * but has no effect on this mode's own output; this note is the
	 * only acknowledgment it was given at all. */
	if (o.ctx_mode == MEMBRANE_RUN_CTX_AUTO && !o.want_json)
		printf("\nTo get a hardware-aware context recommendation for a "
			"real prompt:\n  use --ctx auto with a prompt (without "
			"--inspect-model)\n");
	return (MEMBRANE_EXIT_SUCCESS);
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
	membrane_runtime_t				rt = {};
	membrane_model_session_t		session;
	membrane_host_meminfo_t			host;
	membrane_timings_t				timings = {};
	struct timespec					total_t0;
	struct timespec					stage_t0;

	{
		bool			want_json_argv;
		std::string		parse_stderr;

		/* Phase 28, Section 6: only pay for the stderr-capture
		 * indirection when --json is actually in play -- the
		 * overwhelming common case (no --json) calls the parser
		 * exactly as before this phase. */
		want_json_argv = argv_has_json_flag(argc, argv);
		if (want_json_argv)
			rc = parse_opts_capture_stderr(argc, argv, &o, &parse_stderr);
		else
			rc = membrane_run_parse_opts(argc, argv, &o);
		if (o.want_help)
			return (membrane_run_usage(stdout), MEMBRANE_EXIT_SUCCESS);
		if (o.want_version)
			return (membrane_run_print_version(stdout), MEMBRANE_EXIT_SUCCESS);
		if (rc != MEMBRANE_EXIT_SUCCESS)
		{
			if (want_json_argv)
				print_cli_parse_error_json(o, rc, parse_stderr);
			return (membrane_run_usage(stderr), rc);
		}
	}
	/* Phase 28: --list-devices/--doctor enumerate real backend devices,
	 * which triggers each GPU backend's own INFO-level startup logging
	 * (e.g. ggml_vulkan's "Found N Vulkan devices" block) -- suppressed
	 * here up front, same as the normal-run path below, so neither
	 * command is noisy by default without --verbose. */
	if (!o.verbose)
		llama_log_set(quiet_log_callback, NULL);
	if (o.want_list_devices || o.want_doctor)
	{
		llama_backend_init();
		rc = o.want_list_devices ? run_list_devices_mode(o)
			: run_doctor_mode(o);
		return (llama_backend_free(), rc);
	}
	/* Phase 30: --inspect-model reads only GGUF metadata -- no
	 * llama_backend_init() needed at all, unlike --list-devices/
	 * --doctor above (both enumerate real backend devices). */
	if (o.want_inspect_model)
		return (run_inspect_model_mode(o));
	if (!resolve_prompt(o, &prompt_text))
	{
		fprintf(stderr, "membrane-run: empty or unreadable prompt\n");
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
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
	membrane_read_host_meminfo(&host);
	llama_backend_init();
	rt.initialized = true;
	/* Phase 35: --ctx auto resolves o.ctx to a concrete, real,
	 * hardware-aware value HERE -- before membrane_resolve_gpu_config() runs,
	 * exactly like an explicit --ctx N always has, so every existing
	 * ctx==0/ctx>0 check below (and membrane_resolve_gpu_config()'s own
	 * ctx_size-required contract) is completely unaffected. */
	membrane_ctxauto_outcome_t	ctxauto = {};

	if (o.ctx_mode == MEMBRANE_RUN_CTX_AUTO)
	{
		if (!membrane_resolve_ctx_auto(o, prompt_text, host, &ctxauto))
		{
			fprintf(stderr, "membrane-run: --ctx auto could not tokenize "
				"the prompt (cheap vocab-only load of '%s' failed)\n",
				o.model_path);
			print_error_json(o, MEMBRANE_EXIT_MODEL_ERROR,
				MEMBRANE_REASON_CTX_AUTO_TOKENIZE_FAILED,
				std::string("--ctx auto could not tokenize the prompt "
					"(vocab-only load of '") + o.model_path + "' failed)");
			return (llama_backend_free(), MEMBRANE_EXIT_MODEL_ERROR);
		}
		if (!ctxauto.rec.ok)
		{
			membrane_ctxauto_suggestions_t	suggestions;
			const char						*rep_reason = NULL;

			if (ctxauto.rec.evaluated_count > 0)
				rep_reason = ctxauto.rec.evaluated[0].reason_code;
			membrane_ctxauto_suggest(ctxauto.rec.status, rep_reason,
				o.gen_tokens, o.want_kv_mode, o.want_kv_placement,
				&suggestions);
			fprintf(stderr, "membrane-run: --ctx auto could not "
				"recommend a safe context: %s\n", ctxauto.rec.reason);
			for (size_t si = 0; si < suggestions.count; ++si)
				fprintf(stderr, "  - %s\n", suggestions.text[si]);
			if (o.want_json)
			{
				printf("{\"schema_version\":1,\"membrane_version\":\"%s\","
					"\"mode\":\"run\",\"ok\":false,\"exit_code\":%d,"
					"\"error\":{\"reason_code\":\"",
					MEMBRANE_VERSION, MEMBRANE_EXIT_UNSUPPORTED_KV);
				print_json_escaped(ctxauto.rec.status);
				printf("\",\"message\":\"");
				print_json_escaped(ctxauto.rec.reason);
				printf("\",\"suggestions\":[");
				for (size_t si = 0; si < suggestions.count; ++si)
				{
					if (si > 0)
						printf(",");
					printf("\"");
					print_json_escaped(suggestions.text[si]);
					printf("\"");
				}
				printf("]},\"context_recommendation\":{\"requested\":"
					"\"auto\",\"model_max_context\":%llu,"
					"\"minimum_required_context\":%llu}}\n",
					(unsigned long long)ctxauto.rec.model_max_context,
					(unsigned long long)ctxauto.rec.minimum_required_context);
				fflush(stdout);
			}
			return (llama_backend_free(), MEMBRANE_EXIT_UNSUPPORTED_KV);
		}
		o.ctx = (uint32_t)ctxauto.rec.recommended_context;
	}
	/* GPU/device resolution needs no loaded model -- fail fast, before
	 * spending time on model load, if the request can't be satisfied.
	 * o.ctx is the only context-size information available pre-load
	 * (0 if auto-sizing from the prompt, which needs a loaded model's
	 * tokenizer) -- product_cli.cpp's parse-time validation rejects
	 * any nonzero --gpu-layers (all/auto/N) paired with an auto-sized
	 * --ctx, so o.ctx is never 0 here in the one case that would need
	 * it to be KV-aware (--ctx auto already resolved it above, exactly
	 * like an explicit --ctx N always has). */
	/* Section 18 of the Phase 35 task: host/GPU memory can change
	 * between the recommendation snapshot above and this, the actually
	 * expensive step -- GPU staleness is already handled by the
	 * existing Phase 21 apply-time fallback; host memory has no such
	 * retry mechanism, so this re-reads it fresh, immediately before
	 * model load, and fails clearly rather than proceeding under a
	 * stale recommendation. This is a RECHECK of the SAME already-
	 * selected context's own host requirement -- never a second
	 * context recommendation (ctx itself never changes here). Only
	 * runs when --ctx auto actually required host memory in the first
	 * place (host_memory_checked && required_bytes > 0) -- an explicit
	 * --ctx N run, or a fully GPU-resident --ctx auto plan, is
	 * unaffected. */
	if (o.ctx_mode == MEMBRANE_RUN_CTX_AUTO && ctxauto.rec.host_memory_checked
		&& ctxauto.rec.host_required_bytes > 0)
	{
		membrane_host_meminfo_t		fresh_host;
		membrane_host_guard_request_t	hreq;
		membrane_host_guard_result_t	hres;

		membrane_read_host_meminfo(&fresh_host);
		memset(&hreq, 0, sizeof(hreq));
		hreq.host_total_bytes = fresh_host.total_bytes;
		hreq.host_available_bytes = fresh_host.available_bytes;
		hreq.host_available_known = fresh_host.ok;
		hreq.host_weight_bytes = ctxauto.rec.host_required_bytes;
		hreq.host_kv_bytes = 0;	/* already combined into host_weight_bytes
								 * above -- see membrane_host_guard_
								 * request_t's own contract (either
								 * split arbitrarily between the two
								 * fields, since the guard only ever
								 * sums them). */
		membrane_host_memory_guard_resolve(&hreq, &hres);
		if (!hres.ok)
		{
			fprintf(stderr, "membrane-run: the --ctx auto plan (ctx=%u) no "
				"longer fits real host memory -- it changed since "
				"recommendation: %s\n", o.ctx, hres.reason);
			print_error_json(o, MEMBRANE_EXIT_UNSUPPORTED_KV,
				MEMBRANE_REASON_CTX_AUTO_HOST_MEMORY_STALE, hres.reason,
				"Try again (host memory availability may recover), or use "
				"an explicit smaller --ctx N.");
			return (llama_backend_free(), MEMBRANE_EXIT_UNSUPPORTED_KV);
		}
	}
	/* Mega Phase A, PR A1: resolves GPU layers/KV precision/KV
	 * placement's device-residency half (membrane_resolve_gpu_config(),
	 * unchanged) and loads the model exactly once, via the reusable
	 * runtime-session core (runtime_session.h) -- the same core a
	 * future long-lived server calls into, never a second "CLI runtime"
	 * implementation. Every failure path prints the EXACT same text
	 * membrane_resolve_gpu_config()/model load always did, reconstructed
	 * here from the structured membrane_run_error_t the module returns
	 * instead of printing directly (see runtime_session.h's own top
	 * comment for why). */
	{
		membrane_run_error_t	open_err;

		if (!membrane_model_open(&rt, o.model_path, o, o.ctx, &session,
				&open_err, &timings.planner_ms, &timings.model_load_ms))
		{
			fprintf(stderr, "%s\n", open_err.human.c_str());
			print_error_json(o, open_err.exit_code, open_err.reason_code,
				open_err.message,
				open_err.suggestion.empty() ? NULL
					: open_err.suggestion.c_str(),
				open_err.available_devices.empty() ? NULL
					: &open_err.available_devices);
			return (llama_backend_free(), open_err.exit_code);
		}
	}
	model = session.model;
	gs = session.gs;
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
	membrane_read_model_shape(model, &shape);
	/* Phase 11A: resolve --kv adaptive into a concrete Q8/Q5 exactly
	 * once, into a separate `effective_o` copy -- `o.kv_mode` itself is
	 * left untouched (still ADAPTIVE when requested) so requested_kv_
	 * name(o.kv_mode) below and every print_*() site (via gs.adaptive_
	 * used/adaptive_selected_mode/adaptive_reason) can still report the
	 * ORIGINAL request alongside the resolution. GPU adaptive was
	 * already resolved inside membrane_resolve_gpu_config() above (before model
	 * load, from the pre-load GGUF estimate) -- gs.adaptive_used is
	 * already true there; only the CPU case (no GPU requested) still
	 * needs resolving here, now that the real post-load model shape
	 * and final ctx_size are both known. */
	membrane_run_opts_t	effective_o = o;

	if (o.kv_mode == MEMBRANE_KV_STORE_ADAPTIVE)
	{
		membrane_run_error_t	adaptive_err;

		if (!gs.requested && !membrane_resolve_cpu_adaptive_kv(o, shape,
				ctx_size, &gs, &adaptive_err))
		{
			/* Original (pre-extraction) behavior: this ONE failure path
			 * never called print_error_json() even under --json -- an
			 * existing, unrelated asymmetry with membrane_resolve_gpu_
			 * config()'s own equivalent adaptive-failure message (which
			 * does), preserved exactly as-is (not this phase's scope to
			 * fix). */
			fprintf(stderr, "%s\n", adaptive_err.human.c_str());
			llama_model_free(model);
			return (llama_backend_free(), MEMBRANE_EXIT_UNSUPPORTED_KV);
		}
		effective_o.kv_mode = gs.adaptive_selected_mode;
	}
	/* Section 15/16 of the Phase 35 task: plan identity -- proves (does
	 * not merely assume) that the plan actually being applied is the
	 * SAME plan the recommendation step selected (gpu_layers + KV
	 * precision, the two dimensions both paths finalize pre-decode).
	 * KV PLACEMENT is deliberately excluded from this comparison: its
	 * own authoritative resolution is a separate, later, post-load step
	 * (membrane_resolve_kv_placement() below) that was already documented, before
	 * this phase, as using a pre-load ESTIMATE for ranking purposes only
	 * (joint_planner.h's own top comment) -- the recommendation's own
	 * placement suggestion inherits that SAME pre-existing, disclosed
	 * limitation, not a new one Phase 35 introduces. A real mismatch
	 * here (only possible if GPU/host facts genuinely changed between
	 * the recommendation snapshot and this point) is reported honestly,
	 * never silently claimed as a match. */
	bool		ctxauto_plan_matches = true;
	std::string	ctxauto_plan_mismatch_detail;

	if (o.ctx_mode == MEMBRANE_RUN_CTX_AUTO && ctxauto.rec.ok)
	{
		const membrane_joint_candidate_t	&rec_win = ctxauto.rec.selected_plan
				.candidates[ctxauto.rec.selected_plan.selected_index];

		if (rec_win.gpu_layers != gs.gpu_layers_selected)
		{
			ctxauto_plan_matches = false;
			ctxauto_plan_mismatch_detail += "gpu_layers: recommended="
				+ std::to_string(rec_win.gpu_layers) + " applied="
				+ std::to_string(gs.gpu_layers_selected) + "; ";
		}
		if (rec_win.kv_precision != effective_o.kv_mode)
		{
			ctxauto_plan_matches = false;
			ctxauto_plan_mismatch_detail += "kv_precision: recommended="
				+ std::to_string(rec_win.kv_precision) + " applied="
				+ std::to_string(effective_o.kv_mode) + "; ";
		}
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
				MEMBRANE_REASON_KV_COMPAT_UNSUPPORTED, compat.reason,
				"Use --kv native (no architecture restriction), or see "
				"docs/compatibility.md for the current list of "
				"architectures q8/q5/adaptive support.");
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
				MEMBRANE_REASON_KV_COMPAT_UNSUPPORTED, compat.reason,
				"Use --kv native (no architecture restriction), or see "
				"docs/compatibility.md for the current list of "
				"architectures q8/q5/adaptive support.");
			llama_model_free(model);
			return (llama_backend_free(), MEMBRANE_EXIT_UNSUPPORTED_KV);
		}
	}
	/* Phase 12H: product_cli.cpp's parse-time validation already
	 * rejects --kv-placement combined with --compare-kv/--gpu-bench, so
	 * this only ever runs on the plain single-pass path -- see
	 * membrane_resolve_kv_placement()'s own doc comment for why AFTER model
	 * load, using effective_o.kv_mode. */
	if (!o.gpu_bench && !o.compare_kv)
	{
		membrane_run_error_t	placement_err;

		if (!membrane_resolve_kv_placement(o, shape, ctx_size,
				effective_o.kv_mode, &gs, &placement_err))
		{
			fprintf(stderr, "%s\n", placement_err.human.c_str());
			print_error_json(o, MEMBRANE_EXIT_UNSUPPORTED_KV,
				placement_err.reason_code, placement_err.message,
				placement_err.suggestion.empty() ? NULL
					: placement_err.suggestion.c_str());
			llama_model_free(model);
			return (llama_backend_free(), MEMBRANE_EXIT_UNSUPPORTED_KV);
		}
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
		print_ctxauto_summary(o, ctxauto, ctxauto_plan_matches,
			ctxauto_plan_mismatch_detail);
		print_ctxauto_verbose(o, ctxauto);
		print_startup_summary(effective_o, model_label, ctx_size,
			membrane_kv_bytes_for_mode(shape, ctx_size, effective_o.kv_mode),
			shape.n_layer, gs, host);
		timings.total_ms = seconds_since(&total_t0) * 1000.0;
		if (o.want_json)
			print_plan_only_json(effective_o, model_label, ctx_size,
				shape.n_layer, gs, host, timings, ctxauto);
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
		rc = run_normal_mode(effective_o, &session, prompt_text, model_label,
				ctx_size, shape, gs, host, timings, total_t0, ctxauto,
				ctxauto_plan_matches, ctxauto_plan_mismatch_detail);
		model = session.model;
	}
	llama_model_free(model);
	llama_backend_free();
	return (rc);
}
