#ifndef MEMBRANE_CHAT_FINISH_REASON_H
# define MEMBRANE_CHAT_FINISH_REASON_H

/*
 * Post-v1 product-polish: the ONE authoritative "why did this chat
 * completion stop" decision, shared verbatim by server.cpp's streaming
 * (stream_worker_fn()) and non-streaming (handle_chat_completions())
 * chat-completion paths -- never two independently-drifting copies of
 * the same three-way choice.
 *
 * Real, fixed v1.0.0 correctness gap this closes: finish_reason used to
 * be INFERRED from a token-count comparison (gen_result.tokens.size()
 * >= max_tokens) rather than from the real, tracked reason generation
 * actually stopped -- coincidentally correct as long as gen_tokens ==
 * max_tokens exactly and every early exit funnels through that one
 * count invariant, but that is a property of the surrounding code, not
 * something this decision itself verified. gen_run_result_t::
 * stopped_eog (decode_loop.h) now threads the real cause through
 * explicitly instead.
 *
 * Precedence, matching OpenAI's own finish_reason semantics:
 *   1. A user-supplied "stop" sequence matched -- "stop" (the client
 *      asked for exactly this, Section 12 of the D7 task).
 *   2. The model itself emitted a real end-of-generation token
 *      (llama_vocab_is_eog(), decode_loop.cpp's run_generation()) --
 *      "stop".
 *   3. Neither of the above -- the configured token limit was reached
 *      first -- "length". Never reported merely because termination
 *      state was lost.
 * Pure, deterministic, no I/O -- directly unit-testable.
 */
const char	*membrane_chat_finish_reason(bool stop_matched, bool stopped_eog);

#endif
