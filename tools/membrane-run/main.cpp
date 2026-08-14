#include <cctype>
#include <cstdio>
#include <cstring>
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
}	membrane_gpu_state_t;

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
		return (true);
	}
	label = gpu_layers_label(o.gpu_layers);
	if (!gs->backend_gpu_capable)
		return (fprintf(stderr, "membrane-run: --gpu-layers %s requested "
				"but this build has no GPU backend compiled in "
				"(rebuild with e.g. -DGGML_VULKAN=ON)\n", label.c_str()),
			false);

	membrane_gpu_device_info_t	devices[MEMBRANE_GPU_MAX_DEVICES];

	n_devices = membrane_gpu_list_devices(devices, MEMBRANE_GPU_MAX_DEVICES);
	gpu_count = 0;
	for (i = 0; i < n_devices; ++i)
		if (devices[i].type == MEMBRANE_DEV_TYPE_GPU
			|| devices[i].type == MEMBRANE_DEV_TYPE_IGPU)
			gpu_count++;
	if (gpu_count == 0)
		return (fprintf(stderr, "membrane-run: --gpu-layers %s requested "
				"but no GPU device was found on this host at runtime "
				"(driver/hardware not detected)\n", label.c_str()),
			false);

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
			for (i = 0; i < n_devices; ++i)
				if (devices[i].type == MEMBRANE_DEV_TYPE_GPU
					|| devices[i].type == MEMBRANE_DEV_TYPE_IGPU)
					fprintf(stderr, "  %s (%s)\n", devices[i].name,
						devices[i].description);
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

	membrane_gpu_model_estimate_t	est;
	int		have_est = membrane_gpu_estimate_model(o.model_path, &est);
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
		return (fprintf(stderr, "membrane-run: --kv adaptive requires "
				"the model's real hparams and layer structure to "
				"choose safely between q8 and q5, but they could not "
				"be read from '%s'\n", o.model_path),
			false);
	if (o.gpu_layers == MEMBRANE_GPU_LAYERS_AUTO && !(have_est
			&& est.n_layer > 0))
		return (fprintf(stderr, "membrane-run: --gpu-layers auto "
				"requested but the model's layer structure could not "
				"be read from '%s' -- cannot resolve safely\n",
				o.model_path),
			false);
	if (adaptive_requested)
	{
		/* have_est && est.hparams_available && est.n_layer > 0 already
		 * guaranteed by the fail-closed check above -- both candidates
		 * are always evaluated against the SAME requested_layers/
		 * device_free/device_total/bytes_per_layer inputs (Section 6:
		 * "the policy and GPU planner must evaluate the same candidate
		 * states"), differing only in kv_bytes_estimate. */
		membrane_gpu_policy_result_t		pr_q8;
		membrane_gpu_policy_result_t		pr_q5;
		membrane_adaptive_kv_candidate_t	cand_q8;
		membrane_adaptive_kv_candidate_t	cand_q5;
		membrane_adaptive_kv_result_t		ar;
		int		ok_q8 = membrane_gpu_policy_resolve(o.gpu_layers,
				est.n_layer, dev.memory_free, dev.memory_total,
				est.bytes_per_layer, kv_bytes_estimate, &pr_q8);
		int		ok_q5 = membrane_gpu_policy_resolve(o.gpu_layers,
				est.n_layer, dev.memory_free, dev.memory_total,
				est.bytes_per_layer, kv_bytes_estimate_q5, &pr_q5);

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
			return (fprintf(stderr, "membrane-run: --kv adaptive found "
					"no KV storage mode that fits safely: %s\n",
					ar.reason),
				false);
		gs->adaptive_used = true;
		gs->adaptive_selected_mode = ar.selected_mode;
		gs->adaptive_reason = ar.reason;
		gs->estimated_model_bytes = ar.selected_mode == MEMBRANE_KV_STORE_Q8
			? pr_q8.estimated_model_bytes : pr_q5.estimated_model_bytes;
		gs->estimated_kv_bytes = ar.selected_kv_bytes;
		gs->gpu_layers_selected = ar.selected_layers;
	}
	else if (have_est && est.n_layer > 0)
	{
		membrane_gpu_policy_result_t	pr;
		int		ok = membrane_gpu_policy_resolve(o.gpu_layers, est.n_layer,
				dev.memory_free, dev.memory_total, est.bytes_per_layer,
				kv_bytes_estimate, &pr);

		gs->policy_used = true;
		gs->safety_reserve_bytes = pr.safety_reserve_bytes;
		gs->estimated_model_bytes = pr.estimated_model_bytes;
		gs->estimated_kv_bytes = pr.estimated_kv_bytes;
		if (!ok)
			return (fprintf(stderr, "membrane-run: %s\n", pr.reason), false);
		gs->gpu_layers_selected = pr.selected_layers;
	}
	else
	{
		/* No usable estimate at all (GGUF metadata unreadable) --
		 * only reachable for explicit all/N (auto and adaptive both
		 * already returned above): proceed unguarded, same as
		 * pre-9B.1 behavior. There is no evidence to reject on, so
		 * the user's explicit request is honored rather than blocked
		 * on a missing measurement. */
		gs->gpu_layers_selected = o.gpu_layers == MEMBRANE_GPU_LAYERS_ALL
			? -1 : o.gpu_layers;
	}
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
	return (true);
}

static void	print_startup_summary(const membrane_run_opts_t &o,
				const char *model_label, uint32_t ctx_size,
				uint64_t kv_bytes, const membrane_gpu_state_t &gs)
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
	{
		fprintf(stderr, "backend    %s, device selected: %s "
			"(gpu-layers=%s, selected=%d)\n", gs.backend_selected.c_str(),
			gs.device_selected.c_str(),
			gpu_layers_label(gs.gpu_layers_requested).c_str(),
			gs.gpu_layers_selected);
		if (gs.policy_used)
			fprintf(stderr, "gpu policy device_free=%.1f MiB "
				"reserve=%.1f MiB est_weights=%.1f MiB est_kv=%.1f MiB "
				"(estimates, not measured)\n",
				(double)gs.device_free_bytes / (1024.0 * 1024.0),
				(double)gs.safety_reserve_bytes / (1024.0 * 1024.0),
				(double)gs.estimated_model_bytes / (1024.0 * 1024.0),
				(double)gs.estimated_kv_bytes / (1024.0 * 1024.0));
	}
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

static void	print_run_json(const membrane_run_opts_t &o,
				const char *model_label, const membrane_kv_store_telemetry_t &t,
				const std::string &text, const membrane_gpu_state_t &gs)
{
	printf("{\"schema_version\":1,\"membrane_version\":\"%s\","
		"\"mode\":\"run\",\"model_label\":\"%s\",\"kv_store\":\"%s\","
		"\"kv_type\":\"%s\",\"requested_kv\":\"%s\",\"selected_kv\":\"%s\","
		"\"adaptive_reason\":\"%s\","
		"\"ctx_size\":%u,\"generated_tokens\":%llu,",
		MEMBRANE_VERSION, model_label, kv_mode_name(o.kv_mode),
		kv_type_label(o.kv_mode), requested_kv_name(o, gs),
		kv_mode_name(o.kv_mode),
		gs.adaptive_used ? gs.adaptive_reason.c_str() : "",
		t.ctx_size, (unsigned long long)t.generated_tokens);
	printf("\"storage\":{\"kv_allocated_bytes\":%llu},",
		(unsigned long long)t.compressed_kv_allocated_bytes);
	print_gpu_json(gs);
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
	printf("\"no_fallback_occurred\":%s",
		t.no_fallback_occurred ? "true" : "false");
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

/*
 * Normal single-pass mode. Loads the model, creates exactly ONE
 * llama_context (native or q8 per --kv), ingests the prompt, generates,
 * prints, exits. capture_logits is ALWAYS false here -- there is no
 * teacher_force pass, no second/third context, and therefore no
 * per-step logit buffer of any size, let alone a full-context one.
 */
static int	run_normal_mode(const membrane_run_opts_t &o, llama_model *model,
				const std::vector<llama_token> &prompt_tokens,
				const char *model_label, uint32_t ctx_size,
				const model_shape_t &shape, const membrane_gpu_state_t &gs)
{
	membrane_kv_store_telemetry_t	tel;
	gen_run_result_t				result;
	std::string						text;
	membrane_token_cb_t				cb;
	bool							stream;
	uint64_t						kv_bytes;

	memset(&tel, 0, sizeof(tel));
	tel.kv_store_mode_name = kv_mode_name(o.kv_mode);
	/* This implementation never falls back: run_kv_store_pass() either
	 * succeeds with the requested storage type or fails the whole run
	 * (see the "generation failed" branch below) -- there is no retry-
	 * with-native path for a failed q8/q5 context. */
	tel.no_fallback_occurred = 1;
	kv_bytes = kv_bytes_for_mode(shape, ctx_size, o.kv_mode);
	membrane_kv_store_read_rss(&tel.rss_after_model_load);
	if (!o.quiet)
		print_startup_summary(o, model_label, ctx_size, kv_bytes, gs);
	stream = !o.want_json && !o.quiet;
	cb = stream ? stream_token : NULL;
	if (!run_kv_store_pass(model, prompt_tokens, o.gen_tokens, o.kv_mode,
			ctx_size, o.verbose, NULL, false, 0, &text, &tel, &result, cb,
			NULL))
	{
		fprintf(stderr, "membrane-run: generation failed\n");
		return (MEMBRANE_EXIT_RUNTIME_ERROR);
	}
	tel.generated_tokens = result.tokens.size();
	tel.ctx_size = ctx_size;
	/* kv_bytes is the real allocation size (real ggml_row_size() times
	 * real per-model constants, same arithmetic llama.cpp's own
	 * allocator uses -- see Phase 7) -- not re-derived from anything
	 * run_kv_store_pass() itself measured, since that function has no
	 * reason to know about byte accounting; it only manages the
	 * context/decode loop. */
	tel.compressed_kv_allocated_bytes = kv_bytes;
	membrane_kv_store_rss_max(&tel.rss_after_model_load,
		&tel.rss_after_context, &tel.rss_peak);
	membrane_kv_store_rss_max(&tel.rss_peak, &tel.rss_after_prompt,
		&tel.rss_peak);
	membrane_kv_store_rss_max(&tel.rss_peak, &tel.rss_final, &tel.rss_peak);
	if (o.want_json)
		print_run_json(o, model_label, tel, text, gs);
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
	if (!resolve_gpu_config(o, o.ctx, &device_storage, &mp, &gs))
		return (llama_backend_free(), MEMBRANE_EXIT_UNSUPPORTED_KV);
	model = llama_model_load_from_file(o.model_path, mp);
	if (model == NULL)
	{
		fprintf(stderr, "membrane-run: failed to load model '%s'\n",
			o.model_path);
		return (llama_backend_free(), MEMBRANE_EXIT_MODEL_ERROR);
	}
	model_label = membrane_runtime_safe_basename(o.model_path);
	vocab = llama_model_get_vocab(model);
	prompt_tokens.resize(prompt_text.size() + 8);
	rc = llama_tokenize(vocab, prompt_text.c_str(),
			(int32_t)prompt_text.size(), prompt_tokens.data(),
			(int32_t)prompt_tokens.size(), true, false);
	if (rc < 0)
	{
		fprintf(stderr, "membrane-run: tokenization failed\n");
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
			llama_model_free(model);
			return (llama_backend_free(), MEMBRANE_EXIT_UNSUPPORTED_KV);
		}
	}
	if (o.gpu_bench)
		rc = run_gpu_bench_mode(effective_o, model, prompt_tokens,
				model_label, ctx_size, shape, gs);
	else if (o.compare_kv)
		rc = run_compare_mode(effective_o, model, prompt_tokens,
				model_label, ctx_size, shape, gs);
	else
		rc = run_normal_mode(effective_o, model, prompt_tokens, model_label,
				ctx_size, shape, gs);
	llama_model_free(model);
	llama_backend_free();
	return (rc);
}
