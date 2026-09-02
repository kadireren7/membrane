/*
 * tools/membrane-run/runtime_session.cpp -- Mega Phase A, PR A1's reusable
 * runtime-session core. See runtime_session.h's own top comment for the full
 * contract. Extracted, behavior-preserving, from main.cpp -- every doc
 * comment below that predates this extraction is kept verbatim from there
 * (phase/section numbers refer to main.cpp's original history, not this
 * file's own).
 *
 * PR A1's one real behavior change (not just a move): membrane_resolve_gpu_
 * config()/membrane_resolve_cpu_adaptive_kv()/membrane_resolve_kv_placement()
 * no longer print directly (fprintf(stderr, ...) + print_error_json()) --
 * every failure path now populates a membrane_run_error_t via the
 * membrane_set_err() helper just below instead, and the CALLER (main.cpp's
 * single call site each, unchanged) does the exact same printing it always
 * did, from that struct's fields. This is what makes the module actually
 * reusable by a future non-CLI caller (Section 3 of the Mega Phase A task) --
 * confirmed by direct audit as the one genuine "CLI leaking into runtime
 * core" issue in the pre-extraction code.
 */

#include "runtime_session.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include "gpu_device.h"
#include "gpu_policy.h"
#include "adaptive_kv_policy.h"
#include "kv_residency_policy.h"
#include "compat_check.h"
#include "host_memory_guard.h"
#include "context_auto_cli.h"

/* Builds a membrane_run_error_t from the same pieces every pre-extraction
 * print site already had on hand -- `human` is the complete text a CLI
 * caller fprintf(stderr, "%s\n", ...)'s verbatim (no further formatting);
 * `message`/`suggestion`/`available_devices` mirror print_error_json()'s own
 * parameter list so a CLI caller forwards them unchanged. */
static void	membrane_set_err(membrane_run_error_t *err, int exit_code,
				const char *reason_code, const std::string &human,
				const std::string &message, const std::string &suggestion = "",
				const std::vector<std::string> *available_devices = NULL)
{
	err->set = true;
	err->exit_code = exit_code;
	snprintf(err->reason_code, sizeof(err->reason_code), "%s", reason_code);
	err->human = human;
	err->message = message;
	err->suggestion = suggestion;
	err->available_devices.clear();
	if (available_devices != NULL)
		err->available_devices = *available_devices;
}

/* Phase 13.2, Section 17: Linux-only host-memory OBSERVABILITY (never a
 * safety mechanism -- nothing in this file conditions a pass/fail
 * decision on these numbers, only reports them). Isolated in this one
 * function so the /proc/meminfo dependency stays contained rather than
 * spread across call sites -- ok stays false (fields left at 0) on any
 * non-Linux host or unreadable /proc, and every caller already treats
 * that the same way membrane_kv_store_rss_t's own proc_status_ok does:
 * omit the fields rather than report a fabricated zero as real. */

void	membrane_read_host_meminfo(membrane_host_meminfo_t *out)
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



void	membrane_read_model_shape(llama_model *model, model_shape_t *s)
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

uint64_t	membrane_native_kv_bytes(const model_shape_t &s, uint32_t ctx_size)
{
	membrane_kv_store_bytes_t	b;

	b.n_layer = (uint64_t)s.n_layer;
	b.kv_size = ctx_size;
	b.bytes_per_token_k = ggml_row_size(GGML_TYPE_F16, s.n_embd_gqa);
	b.bytes_per_token_v = ggml_row_size(GGML_TYPE_F16, s.n_embd_gqa);
	return (membrane_kv_store_total_bytes(&b));
}

uint64_t	membrane_q8_kv_bytes(const model_shape_t &s, uint32_t ctx_size)
{
	membrane_kv_store_bytes_t	b;

	b.n_layer = (uint64_t)s.n_layer;
	b.kv_size = ctx_size;
	b.bytes_per_token_k = ggml_row_size(GGML_TYPE_Q8_0, s.n_embd_gqa);
	b.bytes_per_token_v = ggml_row_size(GGML_TYPE_Q8_0, s.n_embd_gqa);
	return (membrane_kv_store_total_bytes(&b));
}

uint64_t	membrane_q5_kv_bytes(const model_shape_t &s, uint32_t ctx_size)
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
uint64_t	membrane_kv_bytes_for_mode(const model_shape_t &s, uint32_t ctx_size,
				int kv_mode)
{
	if (kv_mode == MEMBRANE_KV_STORE_Q8)
		return (membrane_q8_kv_bytes(s, ctx_size));
	if (kv_mode == MEMBRANE_KV_STORE_Q5)
		return (membrane_q5_kv_bytes(s, ctx_size));
	return (membrane_native_kv_bytes(s, ctx_size));
}

const char	*membrane_kv_mode_name(int kv_mode)
{
	if (kv_mode == MEMBRANE_KV_STORE_Q8)
		return ("q8");
	if (kv_mode == MEMBRANE_KV_STORE_Q5)
		return ("q5");
	return ("native");
}

const char	*membrane_kv_type_label(int kv_mode)
{
	if (kv_mode == MEMBRANE_KV_STORE_Q8)
		return ("Q8_0");
	if (kv_mode == MEMBRANE_KV_STORE_Q5)
		return ("Q5_1");
	return ("F16");
}

std::string	membrane_gpu_layers_label(int32_t gpu_layers)
{
	if (gpu_layers == MEMBRANE_GPU_LAYERS_ALL)
		return ("all");
	if (gpu_layers == MEMBRANE_GPU_LAYERS_AUTO)
		return ("auto");
	return (std::to_string(gpu_layers));
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

/* Phase 35: --ctx auto's own uniform result -- populated by whichever
 * of resolve_ctx_auto_normal()/resolve_ctx_auto_cpu_adaptive() applies
 * (both defined much further down, near membrane_resolve_ctx_auto() itself,
 * which needs no forward reference since it's a function, not a type)
 * -- declared here, early, because run_normal_mode() (below) needs it
 * in its own signature. See membrane_resolve_ctx_auto()'s own doc comment,
 * further down, for the full contract. */

/* Review fix (CodeRabbit, PR #22): the two --auto-implied CPU fallback
 * sites inside membrane_resolve_gpu_config() below assigned these same six
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

/* Phase 35: pure device-selection logic factored out of resolve_gpu_
 * config() below (behavior-preserving extraction only -- resolve_gpu_
 * config() calls this exactly where its own inline logic used to live,
 * with identical results) so --ctx auto's own pre-load recommendation
 * step can reuse the EXACT same selection (Section 11 of the Phase 35
 * task: "Use existing... device selection... Capture them ONCE"),
 * never a second, potentially-drifting copy.
 *
 * Returns true and sets *out_chosen_idx when a device was selected;
 * false otherwise, with *out_match_count set to membrane_gpu_match_
 * device()'s own return value when --device was given ((size_t)-1,
 * an impossible match count, when it was not -- callers distinguish
 * "explicit --device matched != 1 device" from "no --device given, no
 * GPU/IGPU device exists at all" this way). */
bool	membrane_select_gpu_device(const membrane_run_opts_t &o,
				const membrane_gpu_device_info_t *devices, size_t n_devices,
				size_t *out_chosen_idx, size_t *out_match_count)
{
	size_t	i;

	*out_match_count = (size_t)-1;
	if (o.want_device)
	{
		size_t	idx = 0;
		size_t	matches = membrane_gpu_match_device(devices, n_devices,
				o.device.c_str(), &idx);

		*out_match_count = matches;
		if (matches != 1)
			return (false);
		*out_chosen_idx = idx;
		return (true);
	}
	/* No --device given: prefer a discrete GPU over an integrated one,
	 * matching llama.cpp's own default-selection preference
	 * (llama_prepare_model_devices()) -- device enumeration order is
	 * NOT discrete-first (confirmed by testing on this host: index 0
	 * was the integrated AMD GPU, a later index the discrete NVIDIA
	 * one), so this must be sought explicitly. */
	for (i = 0; i < n_devices; ++i)
		if (devices[i].type == MEMBRANE_DEV_TYPE_GPU)
		{
			*out_chosen_idx = i;
			return (true);
		}
	for (i = 0; i < n_devices; ++i)
		if (devices[i].type == MEMBRANE_DEV_TYPE_IGPU)
		{
			*out_chosen_idx = i;
			return (true);
		}
	return (false);
}

bool	membrane_resolve_gpu_config(const membrane_run_opts_t &o,
				uint32_t ctx_size,
				std::vector<ggml_backend_dev_t> *device_storage,
				llama_model_params *mp, membrane_gpu_state_t *gs,
				membrane_run_error_t *err)
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
	label = membrane_gpu_layers_label(o.gpu_layers);
	/* Phase 13.1, Section 1/15/16: --auto's OWN implicit gpu_layers=auto
	 * request (o.auto_mode && !o.want_gpu_layers -- never an explicit
	 * --gpu-layers auto/all/N, which keeps the pre-existing fail-closed
	 * behavior below unchanged) must not require a GPU backend or
	 * device at all -- gracefully resolve to CPU-only instead of
	 * failing. gs->requested is explicitly reset to false here (not
	 * left true) so main()'s own `!gs->requested` check still routes
	 * --kv adaptive through membrane_resolve_cpu_adaptive_kv() below, exactly
	 * as an ordinary --gpu-layers 0 run would. */
	bool	auto_implied_gpu = o.auto_mode && !o.want_gpu_layers;

	if (!gs->backend_gpu_capable)
	{
		if (auto_implied_gpu)
			return (fall_back_to_cpu(mp, gs));
		membrane_set_err(err, MEMBRANE_EXIT_UNSUPPORTED_KV,
			MEMBRANE_GPU_POLICY_REASON_NO_GPU_DEVICE,
			"membrane-run: --gpu-layers " + label + " requested but this "
			"build has no GPU backend compiled in (rebuild with e.g. "
			"-DGGML_VULKAN=ON)",
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
		membrane_set_err(err, MEMBRANE_EXIT_UNSUPPORTED_KV,
			MEMBRANE_GPU_POLICY_REASON_NO_GPU_DEVICE,
			"membrane-run: --gpu-layers " + label + " requested but no "
			"GPU device was found on this host at runtime (driver/"
			"hardware not detected)",
			"--gpu-layers " + label + " requested but no GPU device was "
			"found on this host at runtime",
			"Try --gpu-layers 0 for CPU-only, or verify a GPU driver is "
			"installed and visible to this build's backend.");
		return (false);
	}

	size_t	chosen_idx;
	size_t	match_count;

	if (!membrane_select_gpu_device(o, devices, n_devices, &chosen_idx, &match_count))
	{
		/* membrane_select_gpu_device() only fails for two real reasons: an
		 * explicit --device matched != 1 device (match_count != SIZE_MAX
		 * in that case), or no --device was given and no GPU/IGPU device
		 * exists at all among a nonzero n_devices (match_count stays
		 * SIZE_MAX) -- the latter can't actually happen here (gpu_count
		 * == 0 already returned above), kept only as a defensive branch
		 * matching the pre-extraction code's own shape exactly. */
		if (match_count != (size_t)-1)
		{
			/* Section 15: the exact same device list the human message
			 * below carries, kept for --json (available_devices array) so
			 * a scripted caller doesn't have to scrape stderr to offer the
			 * same "pick one of these" UX a human gets. */
			std::vector<std::string>	available_names;
			std::string	human = "membrane-run: --device \""
				+ o.device + "\" matched " + std::to_string(match_count)
				+ " available GPU device"
				+ (match_count == 0 ? "" : "s, expected exactly one")
				+ (match_count == 0 ? "" : ":");

			if (match_count == 0)
				human += "\nAvailable:";
			for (i = 0; i < n_devices; ++i)
				if (devices[i].type == MEMBRANE_DEV_TYPE_GPU
					|| devices[i].type == MEMBRANE_DEV_TYPE_IGPU)
				{
					human += "\n  " + std::string(devices[i].name) + " ("
						+ devices[i].description + ")";
					available_names.push_back(devices[i].name);
				}
			membrane_set_err(err, MEMBRANE_EXIT_UNSUPPORTED_KV,
				MEMBRANE_GPU_POLICY_REASON_DEVICE_NOT_FOUND, human,
				"--device \"" + o.device + "\" matched "
				+ std::to_string(match_count) + " available GPU device(s), "
				"expected exactly one",
				"Pass a --device substring matching exactly one of the "
				"available devices, or omit --device to auto-select.",
				&available_names);
		}
		return (false);
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
			native_estimate = membrane_native_kv_bytes(fake_shape, ctx_size);
		if (adaptive_requested)
		{
			/* Both candidates computed from the SAME est/ctx_size, via
			 * the exact same membrane_kv_bytes_for_mode() every explicit --kv
			 * q8/q5 request already uses -- Section 9: no separate
			 * arithmetic, no drift. kv_bytes_real_q8/q5 are the true,
			 * never-folded per-mode bytes; kv_bytes_estimate/_q5 are
			 * then folded with native_estimate for the guard call
			 * ONLY (see kv_bytes_real_q8's doc comment above). */
			kv_bytes_real_q8 = membrane_kv_bytes_for_mode(fake_shape, ctx_size,
				MEMBRANE_KV_STORE_Q8);
			kv_bytes_real_q5 = membrane_kv_bytes_for_mode(fake_shape, ctx_size,
				MEMBRANE_KV_STORE_Q5);
			kv_bytes_estimate = native_estimate > kv_bytes_real_q8
				? native_estimate : kv_bytes_real_q8;
			kv_bytes_estimate_q5 = native_estimate > kv_bytes_real_q5
				? native_estimate : kv_bytes_real_q5;
		}
		else
		{
			kv_bytes_estimate = membrane_kv_bytes_for_mode(fake_shape, ctx_size,
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
		membrane_set_err(err, MEMBRANE_EXIT_UNSUPPORTED_KV,
			MEMBRANE_GPU_POLICY_REASON_METADATA_UNAVAILABLE,
			std::string("membrane-run: --kv adaptive requires the "
				"model's real hparams and layer structure to choose safely "
				"between q8 and q5, but they could not be read from '")
				+ o.model_path + "'",
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
		membrane_set_err(err, MEMBRANE_EXIT_UNSUPPORTED_KV,
			MEMBRANE_GPU_POLICY_REASON_METADATA_UNAVAILABLE,
			std::string("membrane-run: --gpu-layers auto requested but "
				"the model's layer structure could not be read from '")
				+ o.model_path + "' -- cannot resolve safely",
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
				membrane_set_err(err, MEMBRANE_EXIT_UNSUPPORTED_KV,
					ar.reason,
					std::string("membrane-run: --kv adaptive found no KV "
						"storage mode that fits safely: ") + ar.reason,
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
				membrane_set_err(err, MEMBRANE_EXIT_UNSUPPORTED_KV,
					pr.reason_code,
					std::string("membrane-run: ") + pr.reason, pr.reason,
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
			membrane_set_err(err, MEMBRANE_EXIT_UNSUPPORTED_KV,
				jres.reason_code,
				std::string("membrane-run: ") + jres.reason, jres.reason,
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
 * !gs->requested -- membrane_resolve_gpu_config() above never runs the GPU
 * adaptive branch in that case). No GPU free-memory query exists here,
 * so there is no full-residency/partial-offload comparison -- CPU
 * adaptive defaults to Q8 unless an explicit --kv-budget-mib rules it
 * out and Q5 fits, matching membrane_adaptive_kv_resolve()'s CPU path.
 * Called AFTER model load (unlike the GPU path), since it uses the
 * real post-load model shape rather than a pre-load GGUF estimate --
 * ctx_size may still have been 0/auto-sized at CLI-parse time, and is
 * only known for certain once the model's tokenizer has run. */
bool	membrane_resolve_cpu_adaptive_kv(const membrane_run_opts_t &o,
				const model_shape_t &shape, uint32_t ctx_size,
				membrane_gpu_state_t *gs, membrane_run_error_t *err)
{
	membrane_adaptive_kv_candidate_t	cand_q8;
	membrane_adaptive_kv_candidate_t	cand_q5;
	membrane_adaptive_kv_result_t		ar;
	uint64_t	bytes_q8 = membrane_q8_kv_bytes(shape, ctx_size);
	uint64_t	bytes_q5 = membrane_q5_kv_bytes(shape, ctx_size);

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
	{
		membrane_set_err(err, MEMBRANE_EXIT_UNSUPPORTED_KV, ar.reason,
			std::string("membrane-run: --kv adaptive found no KV storage "
				"mode that fits safely: ") + ar.reason,
			"--kv adaptive found no KV storage mode that fits safely");
		return (false);
	}
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
bool	membrane_resolve_kv_placement(const membrane_run_opts_t &o,
				const model_shape_t &shape, uint32_t ctx_size,
				int effective_kv_mode, membrane_gpu_state_t *gs,
				membrane_run_error_t *err)
{
	uint64_t	total_kv_bytes;
	uint64_t	kv_bytes_per_layer;

	if (o.kv_placement == MEMBRANE_KV_PLACEMENT_DEFAULT)
		return (true);
	/* Phase 13.1 review fix (Section 6): gs->auto_cpu_fallback is true
	 * ONLY when --auto's own implicit gpu_layers=auto request (never an
	 * explicit --gpu-layers) gracefully degraded to CPU-only inside
	 * membrane_resolve_gpu_config() because no GPU backend/device existed at
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
	 * in the same situation (membrane_resolve_gpu_config() above). */
	if (o.kv_placement == MEMBRANE_KV_PLACEMENT_AUTO && gs->auto_cpu_fallback)
		return (true);
	/* Review fix: membrane_resolve_gpu_config()'s documented "GGUF metadata
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
		membrane_set_err(err, MEMBRANE_EXIT_UNSUPPORTED_KV,
			MEMBRANE_KV_PLACEMENT_REASON_INVALID_CONFIG,
			"membrane-run: --kv-placement gpu|auto requires "
			"a verified GPU weight-memory estimate, but none is "
			"available for this model/--gpu-layers combination -- try "
			"--kv-placement cpu or --gpu-layers all/auto",
			"--kv-placement gpu|auto requires a verified GPU "
			"weight-memory estimate, but none is available",
			"Try --kv-placement cpu, or --gpu-layers all/auto.");
		return (false);
	}
	total_kv_bytes = membrane_kv_bytes_for_mode(shape, ctx_size, effective_kv_mode);
	kv_bytes_per_layer = shape.n_layer > 0
		? total_kv_bytes / (uint64_t)shape.n_layer : 0;
	if (!membrane_kv_residency_resolve(o.kv_placement, shape.n_layer,
			kv_bytes_per_layer, gs->device_free_bytes,
			gs->device_total_bytes, gs->estimated_model_bytes,
			/* compute_buffer_estimate_bytes: */ 0, &gs->kv_placement))
	{
		/* gs->kv_placement.reason is already "CODE: detail" (kv_
		 * residency_policy.c's own convention) -- extract just the
		 * code, same rule plan_primary_reason() above uses. */
		std::string	reason(gs->kv_placement.reason);
		size_t		colon = reason.find(": ");

		membrane_set_err(err, MEMBRANE_EXIT_UNSUPPORTED_KV,
			(colon == std::string::npos ? reason
				: reason.substr(0, colon)).c_str(),
			std::string("membrane-run: ") + gs->kv_placement.reason,
			gs->kv_placement.reason,
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
	membrane_read_model_shape(*ctx->model_ptr, &shape);
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
		uint64_t	total_kv_bytes = membrane_kv_bytes_for_mode(shape, ctx->ctx_size,
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
	ctx->tel.kv_store_mode_name = membrane_kv_mode_name(kv_mode);
	ctx->tel.no_fallback_occurred = 1;
	kv_bytes = membrane_kv_bytes_for_mode(shape, ctx->ctx_size, kv_mode);
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
 * at membrane_resolve_gpu_config() time), so gs->device_selected stays valid
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

/* Section 6/7 of the Phase 35 task: a cheap vocab-only model load
 * (llama_model_params::vocab_only -- loads only the tokenizer, never
 * the weight tensors) so --ctx auto can learn a real prompt-token
 * count BEFORE the real, full model load (which itself can't safely
 * happen until GPU-layer resolution -- membrane_resolve_gpu_config() --
 * knows a concrete ctx). Tokenizes with the EXACT SAME llama_tokenize()
 * call/parameters (add_special=true, parse_special=false)
 * membrane_session_generate()'s own tokenize step uses -- deterministic
 * by construction (same GGUF file's vocab data both times), so the two
 * token counts can never disagree. Returns false if the vocab-only
 * load or the tokenize call itself fails (message not printed here --
 * the caller decides how to report it, JSON vs stderr). */
static bool	tokenize_prompt_vocab_only(const char *model_path,
				const std::string &prompt_text,
				std::vector<llama_token> *out_tokens)
{
	llama_model_params	mp;
	llama_model			*model;
	const llama_vocab	*vocab;
	int					rc;

	mp = llama_model_default_params();
	mp.vocab_only = true;
	model = llama_model_load_from_file(model_path, mp);
	if (model == NULL)
		return (false);
	vocab = llama_model_get_vocab(model);
	out_tokens->resize(prompt_text.size() + 8);
	rc = llama_tokenize(vocab, prompt_text.c_str(),
			(int32_t)prompt_text.size(), out_tokens->data(),
			(int32_t)out_tokens->size(), true, false);
	llama_model_free(model);
	if (rc < 0)
		return (false);
	out_tokens->resize(rc);
	return (true);
}

/* Section 13/14 of the Phase 35 task: the ONE scenario Phase 34 found
 * genuinely broken in the joint planner (adaptive precision, zero GPU
 * budget -- joint_planner.c's resolve_adaptive_precision() always
 * calls membrane_adaptive_kv_resolve() with is_gpu_backend hardcoded
 * to 1, so it has no CPU-only fallback candidate). This function does
 * NOT touch joint_planner.c at all, and does NOT introduce a second
 * planner (Section 12) -- it composes EXISTING, unchanged pure
 * primitives (membrane_adaptive_kv_resolve() with is_gpu_backend=0,
 * the EXACT same call this file's own membrane_resolve_cpu_adaptive_kv()
 * already makes for the real, shipped CPU-only `--kv adaptive` path;
 * membrane_host_memory_guard_resolve(), Phase 34's own guard) in a NEW
 * OUTER loop over membrane_ctxrec_generate_candidates()'s own candidate
 * list -- never a new ranking decision, just the CPU-only-adaptive
 * equivalent of what context_recommender.c already does for the
 * GPU-capable case. Fills *out in the exact same shape membrane_ctxrec_
 * resolve() itself produces, so every downstream consumer (printing,
 * JSON, plan-identity comparison) handles both paths uniformly. */
static void	resolve_ctx_auto_cpu_adaptive(
				const membrane_gpu_model_estimate_t &est,
				const membrane_host_meminfo_t &host,
				uint64_t minimum_required_context, int want_kv_budget,
				uint64_t kv_budget_bytes, membrane_ctxrec_result_t *out)
{
	uint64_t	candidate_ctxs[MEMBRANE_CTXREC_MAX_CANDIDATES];
	size_t		n;
	size_t		i;
	int			best = -1;
	model_shape_t	fake_shape;

	memset(out, 0, sizeof(*out));
	out->hardware_fit_index = -1;
	out->model_max_context = est.model_max_context;
	out->minimum_required_context = minimum_required_context;
	if (!est.model_max_context_available)
	{
		snprintf(out->status, sizeof(out->status), "%s",
			MEMBRANE_CTXREC_STATUS_MODEL_MAX_CONTEXT_UNKNOWN);
		snprintf(out->reason, sizeof(out->reason), "%s",
			"model metadata did not expose a context-length ceiling");
		return ;
	}
	if (est.model_max_context == 0)
	{
		snprintf(out->status, sizeof(out->status), "%s",
			MEMBRANE_CTXREC_STATUS_INVALID_MODEL_MAX_CONTEXT);
		snprintf(out->reason, sizeof(out->reason), "%s",
			"model_max_context is 0");
		return ;
	}
	if (minimum_required_context > est.model_max_context)
	{
		snprintf(out->status, sizeof(out->status), "%s",
			MEMBRANE_CTXREC_STATUS_MINIMUM_EXCEEDS_MODEL_MAX);
		snprintf(out->reason, sizeof(out->reason), "%s",
			"minimum_required_context exceeds model_max_context");
		return ;
	}
	fake_shape.n_layer = est.n_layer;
	fake_shape.n_embd = est.n_embd;
	fake_shape.n_head = est.n_head;
	fake_shape.n_head_kv = est.n_head_kv;
	fake_shape.n_embd_gqa = (est.n_head > 0)
		? (int64_t)(est.n_embd / est.n_head) * est.n_head_kv : 0;
	n = membrane_ctxrec_generate_candidates(est.model_max_context,
			minimum_required_context, candidate_ctxs,
			MEMBRANE_CTXREC_MAX_CANDIDATES);
	out->candidate_count = n;
	out->evaluated_count = n;
	if (n == 0)
	{
		snprintf(out->status, sizeof(out->status), "%s",
			MEMBRANE_CTXREC_STATUS_NO_FEASIBLE_CONTEXT);
		snprintf(out->reason, sizeof(out->reason),
			"no candidates to evaluate (effective floor exceeds "
			"model_max_context)");
		return ;
	}
	for (i = 0; i < n; ++i)
	{
		membrane_ctxrec_evaluated_t		*ev = &out->evaluated[i];
		membrane_compat_result_t			compat_q8;
		membrane_compat_result_t			compat_q5;
		membrane_adaptive_kv_candidate_t	cand_q8;
		membrane_adaptive_kv_candidate_t	cand_q5;
		membrane_adaptive_kv_result_t		ar;
		membrane_host_guard_request_t		hreq;
		membrane_host_guard_result_t		hres;

		memset(ev, 0, sizeof(*ev));
		ev->ctx = candidate_ctxs[i];
		if (candidate_ctxs[i] > UINT32_MAX)
		{
			snprintf(ev->reason_code, sizeof(ev->reason_code), "%s",
				MEMBRANE_CTXREC_REASON_CTX_EXCEEDS_UINT32);
			continue ;
		}
		membrane_check_kv_compat(est.arch_name, est.n_embd, est.n_head,
			est.n_head_kv, (uint32_t)candidate_ctxs[i],
			MEMBRANE_KV_STORE_Q8, &compat_q8);
		membrane_check_kv_compat(est.arch_name, est.n_embd, est.n_head,
			est.n_head_kv, (uint32_t)candidate_ctxs[i],
			MEMBRANE_KV_STORE_Q5, &compat_q5);
		memset(&cand_q8, 0, sizeof(cand_q8));
		cand_q8.kv_bytes = membrane_kv_bytes_for_mode(fake_shape,
				(uint32_t)candidate_ctxs[i], MEMBRANE_KV_STORE_Q8);
		cand_q8.valid = compat_q8.ok && (!want_kv_budget
				|| cand_q8.kv_bytes <= kv_budget_bytes);
		cand_q8.full_residency = cand_q8.valid;
		cand_q8.selected_layers = 0;
		memset(&cand_q5, 0, sizeof(cand_q5));
		cand_q5.kv_bytes = membrane_kv_bytes_for_mode(fake_shape,
				(uint32_t)candidate_ctxs[i], MEMBRANE_KV_STORE_Q5);
		cand_q5.valid = compat_q5.ok && (!want_kv_budget
				|| cand_q5.kv_bytes <= kv_budget_bytes);
		cand_q5.full_residency = cand_q5.valid;
		cand_q5.selected_layers = 0;
		/* is_gpu_backend=0 -- exactly what this file's own
		 * membrane_resolve_cpu_adaptive_kv() already passes for the real,
		 * shipped CPU-only `--kv adaptive` path. */
		membrane_adaptive_kv_resolve(&cand_q8, &cand_q5, 0, &ar);
		if (!ar.ok)
		{
			snprintf(ev->reason_code, sizeof(ev->reason_code), "%s",
				ar.reason);
			continue ;
		}
		memset(&hreq, 0, sizeof(hreq));
		hreq.host_total_bytes = host.total_bytes;
		hreq.host_available_bytes = host.available_bytes;
		hreq.host_available_known = host.ok;
		hreq.host_weight_bytes = est.total_bytes;
		hreq.host_kv_bytes = ar.selected_kv_bytes;
		membrane_host_memory_guard_resolve(&hreq, &hres);
		ev->host_memory_checked = 1;
		ev->host_memory_fit = hres.ok;
		ev->host_required_bytes = hres.required_bytes;
		ev->host_reserve_bytes = hres.reserve_bytes;
		ev->selected_gpu_layers = 0;
		ev->selected_kv_precision = ar.selected_mode;
		ev->selected_kv_placement = MEMBRANE_JOINT_PLACEMENT_DEFAULT;
		ev->host_resident = 1;
		if (!hres.ok)
		{
			snprintf(ev->reason_code, sizeof(ev->reason_code), "%s",
				hres.reason_code);
			continue ;
		}
		ev->feasible = 1;
		snprintf(ev->reason_code, sizeof(ev->reason_code), "%s", ar.reason);
		if (best == -1 || ev->ctx > out->evaluated[best].ctx)
			best = (int)i;
	}
	if (best == -1)
	{
		snprintf(out->status, sizeof(out->status), "%s",
			MEMBRANE_CTXREC_STATUS_PLANNER_REJECTED_ALL);
		snprintf(out->reason, sizeof(out->reason),
			"every candidate was rejected (CPU-only adaptive path)");
		return ;
	}
	out->hardware_fit_index = best;
	out->hardware_fit_context = out->evaluated[best].ctx;
	out->recommended_context = out->hardware_fit_context;
	snprintf(out->recommendation_policy, sizeof(out->recommendation_policy),
		"%s", MEMBRANE_CTXREC_POLICY_MAX_ESTIMATED_FIT);
	out->host_memory_unvalidated = 1;
	out->host_memory_checked = out->evaluated[best].host_memory_checked;
	out->host_memory_fit = out->evaluated[best].host_memory_fit;
	out->host_required_bytes = out->evaluated[best].host_required_bytes;
	out->host_available_bytes = host.available_bytes;
	out->host_reserve_bytes = out->evaluated[best].host_reserve_bytes;
	out->ok = 1;
	snprintf(out->status, sizeof(out->status), "%s", MEMBRANE_CTXREC_STATUS_OK);
	snprintf(out->explanation, sizeof(out->explanation),
		"Recommended ctx=%llu (policy: %s, CPU-only adaptive path) -- "
		"the largest of %zu evaluated candidates with a legal plan under "
		"current host-memory estimates.",
		(unsigned long long)out->recommended_context,
		out->recommendation_policy, out->evaluated_count);
	{
		membrane_joint_candidate_t	*win;

		memset(&out->selected_plan, 0, sizeof(out->selected_plan));
		out->selected_plan.ok = 1;
		out->selected_plan.selected_index = 0;
		out->selected_plan.candidate_count = 1;
		win = &out->selected_plan.candidates[0];
		win->gpu_layers = 0;
		win->kv_precision = out->evaluated[best].selected_kv_precision;
		win->kv_placement = MEMBRANE_JOINT_PLACEMENT_DEFAULT;
		win->compatible = 1;
		win->fits_gpu = 1;
		win->fits_host = out->evaluated[best].host_memory_fit;
		win->eligible = 1;
		snprintf(win->reason_code, sizeof(win->reason_code), "%s",
			out->evaluated[best].reason_code);
		snprintf(out->selected_plan.reason_code,
			sizeof(out->selected_plan.reason_code), "%s", win->reason_code);
	}
}

/* Section 11/12 of the Phase 35 task: the normal (GPU-capable, or
 * CPU-only with an explicit non-adaptive precision -- Phase 34 already
 * proved this case works via the joint planner's own AUTO-gpu-layers
 * fallback candidate) recommendation path -- a bounded outer loop
 * around the EXISTING, unchanged membrane_ctxrec_resolve() (Phase 33/
 * 34), never a second planner. Builds each candidate's real KV bytes
 * via the exact same membrane_kv_bytes_for_mode()/ggml_row_size() formula every
 * other call site in this file already uses. */
static void	resolve_ctx_auto_normal(const membrane_run_opts_t &o,
				const membrane_gpu_model_estimate_t &est,
				const membrane_host_meminfo_t &host,
				uint64_t minimum_required_context,
				uint64_t device_free_bytes, uint64_t device_total_bytes,
				bool adaptive_requested, membrane_ctxrec_result_t *out)
{
	membrane_ctxrec_request_t	req;
	model_shape_t				fake_shape;
	size_t						n;
	size_t						i;

	memset(&req, 0, sizeof(req));
	req.n_layer_all = est.n_layer;
	req.bytes_per_layer = est.bytes_per_layer;
	req.output_role_bytes = est.output_role_bytes;
	req.arch_name = est.arch_name;
	req.n_embd = est.n_embd;
	req.n_head = est.n_head;
	req.n_head_kv = est.n_head_kv;
	req.total_weight_bytes = est.total_bytes;
	req.device_free_bytes = device_free_bytes;
	req.device_total_bytes = device_total_bytes;
	req.model_max_context = est.model_max_context;
	req.model_max_context_known = est.model_max_context_available;
	req.minimum_required_context = minimum_required_context;
	req.host_total_bytes = host.total_bytes;
	req.host_available_bytes = host.available_bytes;
	req.host_available_known = host.ok;
	/* Numeric spaces identical by construction (joint_planner.h's own
	 * doc comment) -- same mapping membrane_resolve_gpu_config()'s own joint-
	 * planner call uses, Section 9 of joint_planner.h's top comment. */
	req.kv_placement_mode = o.kv_placement;
	req.gpu_layers_request = o.gpu_layers;
	req.want_kv_budget = o.want_kv_budget;
	req.kv_budget_bytes = o.kv_budget_bytes;
	req.precision_request = adaptive_requested
			? MEMBRANE_JOINT_PRECISION_REQUEST_AUTO : o.kv_mode;
	if (!(est.model_max_context_available && est.model_max_context > 0))
	{
		/* Let membrane_ctxrec_resolve() itself report the precise
		 * UNKNOWN/INVALID status -- candidate_count stays 0, which is
		 * always a safe, well-defined input to it (Section 33 of
		 * context_recommender.c's own contract). */
		membrane_ctxrec_resolve(&req, out);
		return ;
	}
	fake_shape.n_layer = est.n_layer;
	fake_shape.n_embd = est.n_embd;
	fake_shape.n_head = est.n_head;
	fake_shape.n_head_kv = est.n_head_kv;
	fake_shape.n_embd_gqa = (est.n_head > 0)
		? (int64_t)(est.n_embd / est.n_head) * est.n_head_kv : 0;
	{
		uint64_t	candidate_ctxs[MEMBRANE_CTXREC_MAX_CANDIDATES];

		n = membrane_ctxrec_generate_candidates(est.model_max_context,
				minimum_required_context, candidate_ctxs,
				MEMBRANE_CTXREC_MAX_CANDIDATES);
		for (i = 0; i < n; ++i)
		{
			uint32_t	c32 = (candidate_ctxs[i] <= UINT32_MAX)
					? (uint32_t)candidate_ctxs[i] : UINT32_MAX;

			req.candidates[i].ctx = candidate_ctxs[i];
			/* A candidate this large already fails UINT32_MAX-overflow
			 * inside membrane_ctxrec_resolve() itself (Section 18) --
			 * the KV-byte values computed here for it are simply
			 * unused in that case, never a wrapped/wrong number. */
			req.candidates[i].kv_bytes_native = membrane_kv_bytes_for_mode(
					fake_shape, c32, MEMBRANE_KV_STORE_NATIVE);
			req.candidates[i].kv_bytes_q8 = membrane_kv_bytes_for_mode(fake_shape,
					c32, MEMBRANE_KV_STORE_Q8);
			req.candidates[i].kv_bytes_q5 = membrane_kv_bytes_for_mode(fake_shape,
					c32, MEMBRANE_KV_STORE_Q5);
		}
		req.candidate_count = n;
	}
	membrane_ctxrec_resolve(&req, out);
}

/* Section 14 of the Phase 35 task: the single --ctx auto integration
 * adapter -- gathers real facts (GGUF metadata, host memory, GPU
 * device, a vocab-only prompt tokenization) ONCE (Section 11: "Capture
 * them ONCE per recommendation attempt"), then dispatches to whichever
 * of the two pure resolvers above applies. Mirrors membrane_resolve_gpu_config()'s
 * OWN backend-availability/device-enumeration/auto-implied-CPU-fallback
 * logic exactly (via the shared membrane_select_gpu_device() helper) so the
 * SAME real facts membrane_resolve_gpu_config() will use moments later are what
 * this recommendation was based on -- Section 15/16: "the actual run
 * should correspond to the plan the recommender selected". Returns
 * false only for a CLI-adapter-level failure this module owns
 * (tokenization failure) -- a recommendation-core failure (no feasible
 * context, host memory insufficient, etc.) still returns true with
 * out->rec.ok == false; the caller inspects out->rec.status for that.
 * (membrane_ctxauto_outcome_t itself is defined near membrane_timings_t,
 * above -- run_normal_mode() needs it in its own signature, well before
 * this point in the file.) */
bool	membrane_resolve_ctx_auto(const membrane_run_opts_t &o,
				const std::string &prompt_text,
				const membrane_host_meminfo_t &host,
				membrane_ctxauto_outcome_t *out)
{
	std::vector<llama_token>		tokens;
	membrane_gpu_model_estimate_t	est;
	bool							have_est;
	uint64_t						min_ctx = 0;
	bool							adaptive_requested;
	bool							requested_gpu;
	bool							backend_available;
	bool							auto_implied_gpu;
	bool							effective_cpu_only;
	uint64_t						device_free_bytes = 0;
	uint64_t						device_total_bytes = 0;

	memset(out, 0, sizeof(*out));
	out->ok = false;
	if (!tokenize_prompt_vocab_only(o.model_path, prompt_text, &tokens))
		return (false);
	out->prompt_token_count = tokens.size();
	membrane_ctxauto_minimum_required_context(tokens.size(),
		(uint64_t)o.gen_tokens, &min_ctx);
	have_est = membrane_gpu_estimate_model(o.model_path, &est);
	if (!have_est)
		memset(&est, 0, sizeof(est));
	adaptive_requested = (o.kv_mode == MEMBRANE_KV_STORE_ADAPTIVE);
	requested_gpu = (o.gpu_layers != 0);
	backend_available = membrane_gpu_backend_available() != 0;
	auto_implied_gpu = o.auto_mode && !o.want_gpu_layers;
	effective_cpu_only = !requested_gpu;
	if (requested_gpu)
	{
		if (!backend_available)
			effective_cpu_only = auto_implied_gpu;
		else
		{
			membrane_gpu_device_info_t	devices[MEMBRANE_GPU_MAX_DEVICES];
			size_t						n_devices = membrane_gpu_list_devices(
					devices, MEMBRANE_GPU_MAX_DEVICES);
			size_t						gpu_count = 0;
			size_t						i;

			for (i = 0; i < n_devices; ++i)
				if (devices[i].type == MEMBRANE_DEV_TYPE_GPU
					|| devices[i].type == MEMBRANE_DEV_TYPE_IGPU)
					gpu_count++;
			if (gpu_count == 0)
				effective_cpu_only = auto_implied_gpu;
			else
			{
				size_t	chosen_idx;
				size_t	match_count;

				if (membrane_select_gpu_device(o, devices, n_devices, &chosen_idx,
						&match_count))
				{
					device_free_bytes = devices[chosen_idx].memory_free;
					device_total_bytes = devices[chosen_idx].memory_total;
				}
				else if (auto_implied_gpu)
					effective_cpu_only = true;
				/* else: explicit GPU request that matched no device --
				 * device_free/total stay 0, effective_cpu_only stays
				 * false; membrane_resolve_gpu_config() (called moments later by
				 * main()) independently re-checks the exact same
				 * condition and hard-fails with its own real, existing,
				 * correctly-worded error -- this function does not
				 * duplicate that error surface. */
			}
		}
	}
	if (!have_est || !est.hparams_available)
	{
		snprintf(out->rec.status, sizeof(out->rec.status), "%s",
			MEMBRANE_CTXREC_STATUS_INVALID_INPUT);
		snprintf(out->rec.reason, sizeof(out->rec.reason),
			"%s: model metadata could not be read for recommendation",
			MEMBRANE_CTXREC_STATUS_INVALID_INPUT);
		out->ok = true;
		return (true);
	}
	if (effective_cpu_only && adaptive_requested)
	{
		out->used_cpu_only_adaptive_path = true;
		resolve_ctx_auto_cpu_adaptive(est, host, min_ctx, o.want_kv_budget,
			o.kv_budget_bytes, &out->rec);
	}
	else
		resolve_ctx_auto_normal(o, est, host, min_ctx, device_free_bytes,
			device_total_bytes, adaptive_requested, &out->rec);
	out->ok = true;
	return (true);
}

/* ------------------------------------------------------------------------
 * Session API (runtime_session.h's own top comment has the full contract).
 * ------------------------------------------------------------------------
 */

void	membrane_runtime_init(membrane_runtime_t *rt)
{
	llama_backend_init();
	rt->initialized = true;
}

void	membrane_runtime_shutdown(membrane_runtime_t *rt)
{
	if (rt->initialized)
		llama_backend_free();
	rt->initialized = false;
}

bool	membrane_model_open(membrane_runtime_t *rt, const char *model_path,
				const membrane_run_opts_t &o, uint32_t ctx_size,
				membrane_model_session_t *session, membrane_run_error_t *err,
				double *out_planner_ms, double *out_model_load_ms)
{
	struct timespec	stage_t0;

	(void)rt;
	*err = membrane_run_error_t();
	session->model = NULL;
	session->model_path = model_path;
	session->device_storage.clear();
	session->plan_ctx_size = ctx_size;
	session->mp_template = llama_model_default_params();
	clock_gettime(CLOCK_MONOTONIC, &stage_t0);
	if (!membrane_resolve_gpu_config(o, ctx_size, &session->device_storage,
			&session->mp_template, &session->gs, err))
	{
		if (out_planner_ms != NULL)
			*out_planner_ms = seconds_since(&stage_t0) * 1000.0;
		return (false);
	}
	if (out_planner_ms != NULL)
		*out_planner_ms = seconds_since(&stage_t0) * 1000.0;
	clock_gettime(CLOCK_MONOTONIC, &stage_t0);
	session->model = llama_model_load_from_file(model_path,
			session->mp_template);
	if (out_model_load_ms != NULL)
		*out_model_load_ms = seconds_since(&stage_t0) * 1000.0;
	if (session->model == NULL)
	{
		membrane_set_err(err, MEMBRANE_EXIT_MODEL_ERROR,
			MEMBRANE_REASON_MODEL_LOAD_FAILED,
			std::string("membrane-run: failed to load model '") + model_path
				+ "'",
			std::string("failed to load model '") + model_path + "'");
		return (false);
	}
	session->loaded_gpu_layers = session->gs.gpu_layers_selected;
	return (true);
}

void	membrane_model_close(membrane_model_session_t *session)
{
	if (session->model != NULL)
		llama_model_free(session->model);
	session->model = NULL;
}

/* Section 4/7 of the Mega Phase A task: tokenizes req.prompt_text against
 * the session's already-loaded model, resolves ctx_size/CPU-adaptive-KV/
 * compatibility/KV-placement exactly as main.cpp's main()/run_normal_mode()
 * always did post-load (moved, not redesigned), then generates -- via the
 * existing bounded apply-time fallback controller (Phase 21) when the
 * session's plan came from the joint planner, or a direct single-attempt
 * run_kv_store_pass() otherwise, matching run_normal_mode()'s own two
 * branches exactly. Never prints -- see runtime_session.h's own top
 * comment. */
bool	membrane_session_generate(membrane_model_session_t *session,
				const membrane_generation_request_t &req,
				membrane_generation_result_t *out)
{
	const membrane_run_opts_t	&o = *req.o;
	const llama_vocab			*vocab;
	std::vector<llama_token>	prompt_tokens;
	model_shape_t				shape;
	membrane_run_opts_t			effective_o = o;
	uint32_t					ctx_size;
	int							rc;

	out->err = membrane_run_error_t();
	out->ok = false;
	out->fallback_engaged = false;
	memset(&out->fallback_trace, 0, sizeof(out->fallback_trace));
	out->text.clear();
	vocab = llama_model_get_vocab(session->model);
	prompt_tokens.resize(req.prompt_text.size() + 8);
	rc = llama_tokenize(vocab, req.prompt_text.c_str(),
			(int32_t)req.prompt_text.size(), prompt_tokens.data(),
			(int32_t)prompt_tokens.size(), true, false);
	if (rc < 0)
	{
		membrane_set_err(&out->err, MEMBRANE_EXIT_RUNTIME_ERROR,
			MEMBRANE_REASON_TOKENIZATION_FAILED,
			"membrane-run: tokenization failed", "tokenization failed");
		out->exit_code = MEMBRANE_EXIT_RUNTIME_ERROR;
		return (false);
	}
	prompt_tokens.resize(rc);
	ctx_size = req.ctx_size > 0 ? req.ctx_size
		: (uint32_t)prompt_tokens.size() + (uint32_t)o.gen_tokens + 8;
	if (req.ctx_size > 0 && (uint64_t)req.ctx_size <= prompt_tokens.size())
	{
		membrane_set_err(&out->err, MEMBRANE_EXIT_CLI_ERROR,
			MEMBRANE_REASON_CTX_TOO_SMALL,
			"membrane-run: --ctx " + std::to_string(req.ctx_size)
				+ " is too small for this prompt ("
				+ std::to_string(prompt_tokens.size()) + " tokens) -- need "
				"at least " + std::to_string(prompt_tokens.size() + 1),
			"--ctx " + std::to_string(req.ctx_size) + " is too small for "
				"this prompt (" + std::to_string(prompt_tokens.size())
				+ " tokens)",
			"Increase --ctx to at least "
				+ std::to_string(prompt_tokens.size() + 1) + ".");
		out->exit_code = MEMBRANE_EXIT_CLI_ERROR;
		return (false);
	}
	membrane_read_model_shape(session->model, &shape);
	if (o.kv_mode == MEMBRANE_KV_STORE_ADAPTIVE)
	{
		if (!session->gs.requested
			&& !membrane_resolve_cpu_adaptive_kv(o, shape, ctx_size,
				&session->gs, &out->err))
		{
			out->exit_code = MEMBRANE_EXIT_UNSUPPORTED_KV;
			return (false);
		}
		effective_o.kv_mode = session->gs.adaptive_selected_mode;
	}
	if (effective_o.kv_mode != MEMBRANE_KV_STORE_NATIVE)
	{
		membrane_compat_result_t	compat;

		if (!membrane_check_kv_compat(shape.arch_name.c_str(), shape.n_embd,
				shape.n_head, shape.n_head_kv, ctx_size, effective_o.kv_mode,
				&compat))
		{
			membrane_set_err(&out->err, MEMBRANE_EXIT_UNSUPPORTED_KV,
				MEMBRANE_REASON_KV_COMPAT_UNSUPPORTED,
				std::string("MEMBRANE: KV storage unsupported for this "
					"model: ") + compat.reason + ".",
				compat.reason,
				"Use --kv native (no architecture restriction), or see "
				"docs/compatibility.md for the current list of "
				"architectures q8/q5/adaptive support.");
			out->exit_code = MEMBRANE_EXIT_UNSUPPORTED_KV;
			return (false);
		}
	}
	if (!membrane_resolve_kv_placement(o, shape, ctx_size, effective_o.kv_mode,
			&session->gs, &out->err))
	{
		out->exit_code = MEMBRANE_EXIT_UNSUPPORTED_KV;
		return (false);
	}
	out->ctx_size = ctx_size;
	out->prompt_tokens = prompt_tokens;
	if (session->gs.joint_planner_used
		&& session->gs.joint_result.candidate_count > 0
		&& session->gs.joint_result.selected_index >= 0)
	{
		real_apply_ctx_t	actx = {};
		bool				ok_overall;

		actx.o = &effective_o;
		actx.model_path = session->model_path.c_str();
		actx.mp_template = session->mp_template;
		actx.device_storage = &session->device_storage;
		actx.model_ptr = &session->model;
		actx.loaded_gpu_layers = session->loaded_gpu_layers;
		actx.prompt_tokens = &prompt_tokens;
		actx.ctx_size = ctx_size;
		actx.cb = req.token_cb;
		actx.gs = &session->gs;
		session->gs.fallback_engaged = true;
		out->fallback_engaged = true;
		ok_overall = membrane_fallback_run(session->gs.joint_result.candidates,
				session->gs.joint_result.candidate_count,
				session->gs.joint_result.selected_index,
				session->gs.device_total_bytes,
				session->gs.safety_reserve_bytes, real_apply_fn, &actx,
				real_refresh_fn, &session->gs, &session->gs.fallback_trace)
			!= 0;
		out->fallback_trace = session->gs.fallback_trace;
		session->loaded_gpu_layers = actx.loaded_gpu_layers;
		if (!ok_overall)
		{
			membrane_set_err(&out->err, MEMBRANE_EXIT_RUNTIME_ERROR,
				session->gs.fallback_trace.reason_code,
				"membrane-run: every legal auto-managed plan failed to "
					"instantiate",
				"every legal auto-managed plan failed to instantiate",
				"Try a smaller --ctx, explicit --gpu-layers/--kv/"
				"--kv-placement, or --gpu-layers 0 for CPU-only.");
			out->exit_code = MEMBRANE_EXIT_RUNTIME_ERROR;
			return (false);
		}
		out->tel = actx.tel;
		out->text = actx.text;
		out->gen_result = actx.gen_result;
		out->effective_kv_mode = actx.effective_kv_mode;
	}
	else
	{
		membrane_kv_placement_map_t		placement_map;
		const membrane_kv_placement_map_t	*placement_arg;
		uint64_t							kv_bytes;

		memset(&out->tel, 0, sizeof(out->tel));
		out->tel.kv_store_mode_name = membrane_kv_mode_name(
				effective_o.kv_mode);
		out->tel.no_fallback_occurred = 1;
		kv_bytes = membrane_kv_bytes_for_mode(shape, ctx_size,
				effective_o.kv_mode);
		membrane_kv_store_read_rss(&out->tel.rss_after_model_load);
		placement_arg = NULL;
		if (session->gs.kv_placement_resolved)
		{
			placement_map.n_layer = session->gs.kv_placement.n_layer;
			placement_map.layer_on_gpu = session->gs.kv_placement.layer_on_gpu;
			placement_arg = &placement_map;
		}
		if (!run_kv_store_pass(session->model, prompt_tokens, o.gen_tokens,
				effective_o.kv_mode, ctx_size, o.verbose, NULL, false, 0,
				&out->text, &out->tel, &out->gen_result, req.token_cb, NULL,
				placement_arg))
		{
			membrane_set_err(&out->err, MEMBRANE_EXIT_RUNTIME_ERROR,
				MEMBRANE_REASON_GENERATION_FAILED,
				"membrane-run: generation failed", "generation failed");
			out->exit_code = MEMBRANE_EXIT_RUNTIME_ERROR;
			return (false);
		}
		out->effective_kv_mode = effective_o.kv_mode;
		out->tel.generated_tokens = out->gen_result.tokens.size();
		out->tel.ctx_size = ctx_size;
		out->tel.compressed_kv_allocated_bytes = kv_bytes;
		membrane_kv_store_rss_max(&out->tel.rss_after_model_load,
			&out->tel.rss_after_context, &out->tel.rss_peak);
		membrane_kv_store_rss_max(&out->tel.rss_peak,
			&out->tel.rss_after_prompt, &out->tel.rss_peak);
		membrane_kv_store_rss_max(&out->tel.rss_peak, &out->tel.rss_final,
			&out->tel.rss_peak);
	}
	out->ok = out->gen_result.ok;
	out->exit_code = out->gen_result.ok ? MEMBRANE_EXIT_SUCCESS
		: MEMBRANE_EXIT_RUNTIME_ERROR;
	return (out->ok);
}
