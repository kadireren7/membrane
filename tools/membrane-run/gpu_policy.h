#ifndef MEMBRANE_RUN_GPU_POLICY_H
# define MEMBRANE_RUN_GPU_POLICY_H

# include <stddef.h>
# include <stdint.h>

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Phase 9B.1: llama-free (no ggml/llama header) pure arithmetic for
 * resolving --gpu-layers {all|auto|N} into an actual layer count
 * against an estimated memory budget. No I/O, no device access, no
 * model load -- every input is a plain integer the caller already
 * has (or estimated via gpu_device.h/gguf_init_from_file elsewhere).
 * Testable without a GPU or a model file (test_gpu_policy.c), same
 * pattern as compat_check.h's Q8 pre-flight check.
 *
 * This module intentionally does NOT claim to predict llama.cpp's
 * actual allocator behavior exactly -- ggml's allocator has its own
 * alignment/scratch/fragmentation overhead this can't see without
 * loading the model. It is a conservative, documented estimate meant
 * to reject configurations that are clearly unsafe and otherwise
 * choose a defensible layer count -- never a guarantee against OOM.
 */

/* requested_layers sentinel values, matching product_cli.h's
 * membrane_run_opts_t::gpu_layers convention (0 = CPU-only is never
 * passed here -- callers only invoke this module when GPU use was
 * actually requested). */
# define MEMBRANE_GPU_LAYERS_ALL	(-1)
# define MEMBRANE_GPU_LAYERS_AUTO	(-2)

/* Safety reserve policy: whichever is LARGER of a fixed floor and a
 * percentage of total device memory. The fixed floor absorbs
 * non-MEMBRANE GPU usage on an ordinary desktop host (compositor,
 * browser, other applications) that a headless/dedicated inference
 * box wouldn't have; the percentage floor scales the reserve up for
 * larger cards rather than leaving a fixed-MiB reserve that becomes
 * proportionally tiny (and therefore not meaningfully safer) as
 * device memory grows. Neither number claims to be exact -- both are
 * a documented, conservative policy choice, not a measured guarantee.
 */
# define MEMBRANE_GPU_RESERVE_FIXED_BYTES	((uint64_t)512 * 1024 * 1024)
# define MEMBRANE_GPU_RESERVE_PCT			15

/* Phase 13.1, Section 5: stable, machine-readable reason codes -- never
 * change the MEANING of an already-shipped code, only add new ones.
 * Same convention as kv_residency_policy.h's MEMBRANE_KV_PLACEMENT_
 * REASON_ set and adaptive_kv_policy.h's MEMBRANE_ADAPTIVE_REASON_
 * set: reason below is always at least the bare code on success; on
 * failure it is the code followed by ": " and a free-text detail
 * (never the reverse, so a strncmp(reason, CODE, strlen(CODE)) call
 * reliably extracts the code either way). */
# define MEMBRANE_GPU_POLICY_REASON_GPU_FULL_FIT		"GPU_FULL_FIT"
# define MEMBRANE_GPU_POLICY_REASON_GPU_LAYERS_CLAMPED	"GPU_LAYERS_CLAMPED"
# define MEMBRANE_GPU_POLICY_REASON_CPU_ONLY_REQUESTED	"CPU_ONLY_REQUESTED"
# define MEMBRANE_GPU_POLICY_REASON_MEMORY_INSUFFICIENT	"GPU_MEMORY_INSUFFICIENT"
# define MEMBRANE_GPU_POLICY_REASON_INVALID_CONFIG		"GPU_POLICY_INVALID_CONFIG"
# define MEMBRANE_GPU_POLICY_REASON_METADATA_UNAVAILABLE	"MODEL_METADATA_UNAVAILABLE"
/* Phase 13.1, Section 15/16: emitted ONLY when --auto's own implicit
 * gpu_layers=auto request (never an explicit --gpu-layers auto/all/N)
 * gracefully resolves to CPU-only because no GPU backend is compiled
 * in, or none was found at runtime -- main.cpp's resolve_gpu_config(),
 * not this module (this module has no concept of "was this implicit"
 * -- it only ever sees a resolved kv_bytes_estimate/layer count). An
 * explicit GPU request in the same situation still fails closed with
 * the pre-existing free-text error, unchanged. */
# define MEMBRANE_GPU_POLICY_REASON_NO_GPU_DEVICE			"NO_GPU_DEVICE"
/* Phase 13.1, Section 4/13: --device NAME matched zero or more than
 * one device -- always an explicit user request, never something
 * --auto's own graceful degradation applies to (--auto does not pass
 * --device). */
# define MEMBRANE_GPU_POLICY_REASON_DEVICE_NOT_FOUND		"GPU_DEVICE_NOT_FOUND"

typedef struct s_membrane_gpu_policy_result
{
	int			ok;
	int32_t		selected_layers;		/* meaningful only if ok */
	uint64_t	safety_reserve_bytes;
	uint64_t	budget_bytes;			/* device_free - reserve, 0 if
										 * reserve exceeds free */
	uint64_t	estimated_model_bytes;	/* for selected_layers (or the
										 * rejected request, if !ok) */
	uint64_t	estimated_kv_bytes;		/* echoed input, for telemetry */
	char		reason_code[40];		/* one of the MEMBRANE_GPU_POLICY_
										 * REASON_* strings above, always
										 * set (both on success and
										 * failure) -- Section 5 */
	char		reason[256];			/* code, or "CODE: details" when
										 * !ok -- see reason_code's own
										 * comment */
}	membrane_gpu_policy_result_t;

/*
 * requested_layers: MEMBRANE_GPU_LAYERS_ALL, MEMBRANE_GPU_LAYERS_AUTO,
 * or an explicit N >= 0.
 * n_layer_all: the model's real total layer count (llama_model_n_layer).
 * device_free_bytes/device_total_bytes: from ggml_backend_dev_memory().
 * bytes_per_layer_estimate: real per-layer weight size estimated from
 * GGUF tensor metadata (gguf_get_tensor_size()), NOT a hardcoded
 * per-GPU or per-model-family number.
 * kv_bytes_estimate: real KV allocation formula for the requested
 * ctx/kv-type (same arithmetic membrane-run's existing native_kv_bytes/
 * q8_kv_bytes() helpers already use).
 *
 * Returns 1 (out->ok = 1, out->selected_layers set) if the request can
 * be satisfied within the estimated budget, 0 otherwise (out->reason
 * filled in) -- callers must fail closed, never silently retry with a
 * smaller/zero layer count on their own.
 */
int	membrane_gpu_policy_resolve(int32_t requested_layers,
		int32_t n_layer_all, uint64_t device_free_bytes,
		uint64_t device_total_bytes, uint64_t bytes_per_layer_estimate,
		uint64_t kv_bytes_estimate, membrane_gpu_policy_result_t *out);

# ifdef __cplusplus
}
# endif

#endif
