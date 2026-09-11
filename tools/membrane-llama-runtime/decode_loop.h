#ifndef MEMBRANE_LLAMA_RUNTIME_DECODE_LOOP_H
# define MEMBRANE_LLAMA_RUNTIME_DECODE_LOOP_H

#include <atomic>
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
	bool								stopped_eog;	/* Post-v1 product-
											 * polish: true iff the free-
											 * running loop stopped because
											 * llama_vocab_is_eog() matched
											 * the model's OWN chosen token
											 * -- false for every other exit
											 * (step limit reached,
											 * cancelled, decode failure,
											 * or a teacher-forced replay,
											 * which has no such concept).
											 * The one authoritative signal
											 * a caller needs to report
											 * finish_reason="stop" for a
											 * real EOG rather than
											 * inferring it from a token-
											 * count comparison (a real,
											 * fixed v1.0.0 correctness
											 * gap -- see chat_finish_
											 * reason.h, tools/membrane). */
}	gen_run_result_t;

/* See decode_loop.cpp for the full free-running/teacher-forced
 * contract. token_cb, if non-NULL, is called once per newly decoded
 * token (its piece text and index), immediately after that token's
 * decode succeeds -- for streaming output; never called for a
 * teacher-forced pass's replayed tokens (those aren't "new" output). */
typedef void (*membrane_token_cb_t)(const char *piece, size_t piece_len,
	int step, void *user_data);

/* Phase 24: out_first_token_ms, iff non-NULL, is always written on
 * return -- the wall-clock duration (milliseconds, CLOCK_MONOTONIC via
 * this file's own seconds_since()) of the single decode step that
 * produced the FIRST generated token, or -1.0 if none was (immediate
 * EOG, a failed decode, or a teacher-forced pass -- which replays
 * tokens rather than generating them, so "first token" has no
 * meaning there). One stage-boundary timestamp pair, not per-token
 * instrumentation (Section 25 of the Phase 24 task) -- every other
 * step's own duration is not captured here. */
/* Mega Phase B, PR B2: cancel_flag, iff non-NULL, is polled once per
 * free-running step (never mid-teacher-force, never mid-decode -- the
 * in-flight decode of the CURRENT token always completes; only the
 * NEXT step is skipped) via a plain relaxed atomic load -- this is the
 * ENTIRE cancellation contract the runtime core understands: "caller
 * requested cancellation," never anything server-specific (no socket,
 * no HTTP awareness anywhere in this file). A caller with no
 * cancellation concept (every pre-B2 call site) passes NULL and sees
 * byte-identical behavior. out_cancelled, iff non-NULL, is always
 * written on return (true iff the loop actually stopped early because
 * of cancel_flag, false otherwise -- including on ordinary EOG/limit/
 * decode-failure exits) -- same "always write an optional out-param"
 * convention as out_first_token_ms above. */
/* render_special_tokens: forwarded verbatim as llama_token_to_piece()'s
 * own `special` argument (llama.h) -- true (the default, byte-identical
 * to every pre-existing caller) renders a control/special token's own
 * literal text into text_out/token_cb; false suppresses it entirely
 * (llama.h's own canonical LLAMA_TOKEN_ATTR_CONTROL/UNKNOWN classifi-
 * cation, not a hardcoded string list -- see llama-vocab.cpp's own
 * token_to_piece()). Server chat completions (server.cpp) is the one
 * caller that passes false, so a chat-control token that survives past
 * llama_vocab_is_eog()'s own check (e.g. a mid-generation "<|im_start|>"
 * that is not itself an end-of-generation token) never leaks into an
 * API response's assistant content -- a real, fixed v1.0.0 bug. Every
 * other existing caller (membrane-run's own CLI/comparison tooling)
 * keeps its prior, unfiltered debug-visible behavior unchanged. */
void	run_generation(llama_context *ctx, const llama_vocab *vocab,
			int32_t n_vocab, int gen_tokens,
			membrane_runtime_collector_t *collector,
			membrane_llama_hook_ctx_t *hook_ctx, int debug,
			bool capture_logits, const std::vector<int32_t> *teacher_force,
			uint64_t *abs_pos, std::string *text_out, gen_run_result_t *out,
			membrane_token_cb_t token_cb = NULL, void *token_cb_ud = NULL,
			double *out_first_token_ms = NULL,
			const std::atomic<bool> *cancel_flag = NULL,
			bool *out_cancelled = NULL,
			bool render_special_tokens = true);

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

/* Exposed (not `static`) ONLY for test_decode_loop.cpp -- not part of
 * run_kv_store_pass's public contract, never called directly by any
 * product call site (it is installed as llama_context_params.
 * kv_dev_override, never invoked by MEMBRANE code itself). */
ggml_backend_dev_t	kv_placement_dev_override_cb(int32_t il,
			ggml_backend_dev_t default_dev, void *user_data);

/* Phase 21: which internal stage a run_kv_store_pass() failure
 * happened at, for a caller that needs to classify the failure (the
 * apply-time fallback controller, tools/membrane-run/auto_fallback.h)
 * rather than just knowing pass/fail -- llama.h itself exposes no
 * error code or reason string from llama_init_from_model() (a bare
 * NULL is the only signal), so this is the most specific information
 * this project can honestly report. Never set for a successful pass
 * (left at MEMBRANE_KV_PASS_STAGE_NONE, the caller's own responsibility
 * to pre-initialize if it cares). */
# define MEMBRANE_KV_PASS_STAGE_NONE				0
# define MEMBRANE_KV_PASS_STAGE_CONTEXT_CREATE		1	/* llama_init_from_
													 * model() returned NULL
													 * -- construction/
													 * allocation failure,
													 * no further detail
													 * available */
# define MEMBRANE_KV_PASS_STAGE_DECODE				2	/* context existed;
													 * prompt or generation
													 * decode itself failed
													 * (decode_prompt()/
													 * run_generation()) */

/*
 * Product Phase 7: creates ONE llama_context whose KV cache tensors
 * are allocated at kv_store_mode's ggml type, decodes the prompt and
 * generates, and destroys the context before returning. See
 * decode_loop.cpp for the full contract (flash-attention forcing,
 * RSS-checkpoint capture, scratch-byte accounting). ctx_size must
 * already be resolved (never 0). out_failure_stage, iff non-NULL, is
 * always written on return (MEMBRANE_KV_PASS_STAGE_NONE on success) --
 * optional and additive, every pre-Phase-21 call site passes NULL and
 * is unaffected.
 */
/* Mega Phase B, PR B2: cancel_flag/out_cancelled -- see run_generation()'s
 * own doc comment above, threaded straight through to the one
 * run_generation() call this function makes; NULL (every pre-B2 call
 * site) is byte-identical to before. */
/* render_special_tokens: forwarded verbatim to the one run_generation()
 * call this function makes -- see that function's own doc comment. */
bool	run_kv_store_pass(llama_model *model,
			const std::vector<llama_token> &prompt_tokens, int gen_tokens,
			int kv_store_mode, uint32_t ctx_size, int debug,
			const std::vector<int32_t> *teacher_force, bool capture_logits,
			int32_t n_vocab_for_scratch, std::string *text_out,
			membrane_kv_store_telemetry_t *tel, gen_run_result_t *out,
			membrane_token_cb_t token_cb = NULL, void *token_cb_ud = NULL,
			const membrane_kv_placement_map_t *kv_placement = NULL,
			int *out_failure_stage = NULL,
			const std::atomic<bool> *cancel_flag = NULL,
			bool *out_cancelled = NULL,
			bool render_special_tokens = true);

/* Phase 7 analogue of the Phase 6 aligned-behavior comparison, for the
 * kv-store telemetry struct. `a` is the native-storage reference pass,
 * `c` is the q8-storage pass teacher-forced on `a`'s own tokens.
 * Returns false (tel left untouched) only on allocation failure. */
bool	record_kv_store_behavior(const gen_run_result_t &a,
			const gen_run_result_t &c, int32_t n_vocab,
			membrane_kv_store_telemetry_t *tel);

#endif
