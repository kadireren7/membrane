#ifndef MEMBRANE_LLAMA_HOOK_H
# define MEMBRANE_LLAMA_HOOK_H

# include <stdbool.h>
# include <stdint.h>

# include "runtime_core.h"

struct ggml_tensor;

# ifdef __cplusplus
extern "C" {
# endif

/*
 * llama-facing adapter: the only file in this tool that includes a
 * ggml/llama.cpp header or touches a ggml_tensor. MEMBRANE's own
 * policy/quantization/telemetry logic (runtime_core.h/.c) never sees a
 * ggml or llama type -- this file extracts live tensor data into a
 * plain uint16_t buffer and calls straight into
 * membrane_runtime_observe_tensor().
 *
 * Opaque; created once per run and passed to both the eval callback
 * (as cb_eval_user_data) and membrane_llama_hook_flush_step().
 * NULL collector: the hook is inert -- ask always returns false, zero
 * overhead. (The true baseline path additionally never installs
 * cb_eval at all -- see main.cpp -- so this is defense in depth, not
 * how baseline mode is actually achieved.)
 */
typedef struct s_membrane_llama_hook_ctx	membrane_llama_hook_ctx_t;

membrane_llama_hook_ctx_t	*membrane_llama_hook_create(
								membrane_runtime_collector_t *collector,
								membrane_simd_backend_t backend, int debug,
								uint32_t max_layers);
void	membrane_llama_hook_destroy(membrane_llama_hook_ctx_t *ctx);

/*
 * Matches ggml_backend_sched_eval_callback exactly (ggml/include/
 * ggml-backend.h) -- install as params.cb_eval = membrane_llama_eval_
 * callback, params.cb_eval_user_data = ctx. Observes only tensors
 * named "Kcur-%d"/"Vcur-%d" (see docs/live-runtime.md for why: these
 * are the per-layer K/V projection tensors llm_graph_context::build_
 * qkv() tags via ggml_format_name, and they are what flows into cpy_k/
 * cpy_v, the actual KV-cache write) -- every other graph node returns
 * false on `ask` and is never materialized or copied.
 *
 * IMPORTANT: "Kcur-%d" is tagged TWICE per layer on the pinned commit
 * (once before RoPE, inside build_qkv, and again after RoPE by the
 * caller -- two distinct tensor objects, since RoPE produces a new
 * node) -- only the post-RoPE value is what cpy_k actually writes to
 * cache. This callback does NOT process a tensor's data immediately:
 * it extracts and buffers it, keyed by (layer, is_v), overwriting any
 * earlier value for the same key observed this step. Actual MEMBRANE
 * processing (select/encode/decode/validate) happens in
 * membrane_llama_hook_flush_step(), called after llama_decode()
 * returns -- by then, dependency-ordered execution guarantees the
 * buffered value for each key is the LAST one computed, i.e. the one
 * that actually reached the cache write. "Vcur-%d" is tagged twice
 * too, but both tags reference the same tensor object (V is never
 * RoPE'd) -- one real graph node, one real observation either way.
 *
 * Read-only and side-effect-free on the graph: never writes to
 * t->data, never returns false on the ask==false (materialized) call
 * (ggml-backend.h documents that returning false there cancels the
 * whole graph compute -- this must never happen for a tensor MEMBRANE
 * itself requested). This is what makes SHADOW mode shadow: llama's
 * own computation, and therefore its generated tokens, are completely
 * unaffected by whether this callback is installed.
 *
 * Extracted data is copied out via ggml_backend_tensor_get() (backend-
 * agnostic, safe regardless of whether the tensor lives on a CPU or
 * device buffer) into memory owned entirely by `ctx` -- no pointer
 * into ggml/llama-managed memory is ever retained past this single
 * call.
 */
bool	membrane_llama_eval_callback(struct ggml_tensor *t, bool ask,
			void *user_data);

/*
 * Processes every (layer, is_v) key buffered since the last flush --
 * exactly once each, using its last-written value -- via
 * membrane_runtime_observe_tensor(), then clears the buffer for the
 * next step. Call once after every llama_decode() returns. A NULL ctx
 * or a ctx whose collector is NULL is a no-op (baseline mode: nothing
 * was ever buffered, since the eval callback was never installed).
 */
void	membrane_llama_hook_flush_step(membrane_llama_hook_ctx_t *ctx);

# ifdef __cplusplus
}
# endif

#endif
