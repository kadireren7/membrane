#ifndef MEMBRANE_LLAMA_RUNTIME_DECODE_LOOP_H
# define MEMBRANE_LLAMA_RUNTIME_DECODE_LOOP_H

#include <cstdint>
#include <string>
#include <vector>

#include "llama.h"
#include "llama_hook.h"
#include "runtime_core.h"
#include "kv_store_telemetry.h"

/*
 * Shared llama-enabled decode-loop primitives, factored out of
 * membrane-llama-run's main.cpp (Phase 5/6/7) so the product entry
 * point (tools/membrane-run) reuses the exact same, already-tested
 * code instead of a second hand-copied implementation -- one real
 * decode loop, two CLIs on top of it.
 */

std::string	read_file(const char *path);
double		seconds_since(const struct timespec *t0);
int			argmax(const float *v, int n);

/* Times llama_decode() together with this step's MEMBRANE flush; see
 * decode_loop.cpp for the full timing-semantics comment. hook_ctx may
 * be NULL (no eval callback installed -- the plain-storage path). */
bool	timed_decode(llama_context *ctx, llama_batch batch,
			uint64_t abs_pos_start, uint64_t n_tok,
			membrane_runtime_collector_t *collector,
			membrane_llama_hook_ctx_t *hook_ctx, int debug,
			const char *step_label);

bool	decode_prompt(llama_context *ctx,
			const std::vector<llama_token> &prompt_tokens,
			int32_t n_batch, membrane_runtime_collector_t *collector,
			membrane_llama_hook_ctx_t *hook_ctx, int debug,
			uint64_t *abs_pos);

typedef struct s_gen_run_result
{
	std::vector<int32_t>				tokens;
	std::vector<std::vector<float>>	logits;	/* captured iff requested */
	bool								ok;
}	gen_run_result_t;

/* See decode_loop.cpp for the full free-running/teacher-forced
 * contract. token_cb, if non-NULL, is called once per newly decoded
 * token (its piece text and index), immediately after that token's
 * decode succeeds -- for streaming output; never called for a
 * teacher-forced pass's replayed tokens (those aren't "new" output). */
typedef void (*membrane_token_cb_t)(const char *piece, size_t piece_len,
	int step, void *user_data);

void	run_generation(llama_context *ctx, const llama_vocab *vocab,
			int32_t n_vocab, int gen_tokens,
			membrane_runtime_collector_t *collector,
			membrane_llama_hook_ctx_t *hook_ctx, int debug,
			bool capture_logits, const std::vector<int32_t> *teacher_force,
			uint64_t *abs_pos, std::string *text_out, gen_run_result_t *out,
			membrane_token_cb_t token_cb = NULL, void *token_cb_ud = NULL);

/* Phase 12H: optional static per-layer KV device residency map,
 * threaded through to llama_context_params.kv_dev_override at
 * construction time ONLY -- never consulted again afterward, no
 * runtime movement. layer_on_gpu[i] (length n_layer) is 1 for
 * GPU-resident KV, 0 for CPU-resident KV; only meaningful for indices
 * < n_layer. A NULL membrane_kv_placement_map_t* (the default on every
 * existing call site, including every membrane-llama-run caller)
 * leaves llama_context_params.kv_dev_override NULL -- byte-identical
 * to pre-Phase-12H behavior (Section 4/19). */
typedef struct s_membrane_kv_placement_map
{
	int32_t			n_layer;
	const uint8_t	*layer_on_gpu;
}	membrane_kv_placement_map_t;

/*
 * Product Phase 7: creates ONE llama_context whose KV cache tensors
 * are allocated at kv_store_mode's ggml type, decodes the prompt and
 * generates, and destroys the context before returning. See
 * decode_loop.cpp for the full contract (flash-attention forcing,
 * RSS-checkpoint capture, scratch-byte accounting). ctx_size must
 * already be resolved (never 0).
 */
bool	run_kv_store_pass(llama_model *model,
			const std::vector<llama_token> &prompt_tokens, int gen_tokens,
			int kv_store_mode, uint32_t ctx_size, int debug,
			const std::vector<int32_t> *teacher_force, bool capture_logits,
			int32_t n_vocab_for_scratch, std::string *text_out,
			membrane_kv_store_telemetry_t *tel, gen_run_result_t *out,
			membrane_token_cb_t token_cb = NULL, void *token_cb_ud = NULL,
			const membrane_kv_placement_map_t *kv_placement = NULL);

/* Phase 7 analogue of the Phase 6 aligned-behavior comparison, for the
 * kv-store telemetry struct. `a` is the native-storage reference pass,
 * `c` is the q8-storage pass teacher-forced on `a`'s own tokens.
 * Returns false (tel left untouched) only on allocation failure. */
bool	record_kv_store_behavior(const gen_run_result_t &a,
			const gen_run_result_t &c, int32_t n_vocab,
			membrane_kv_store_telemetry_t *tel);

#endif
