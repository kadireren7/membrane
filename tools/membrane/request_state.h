#ifndef MEMBRANE_REQUEST_STATE_H
# define MEMBRANE_REQUEST_STATE_H

/*
 * Mega Phase E, PR E1, Section 4 of the task: an explicit per-REQUEST
 * lifecycle state -- distinct from s_membrane_model_state (server.cpp's
 * own pre-existing membrane_model_state_name()/MEMBRANE_MODEL_STATE_*),
 * which describes the single shared model resource (EMPTY/LOADING/READY/
 * GENERATING/UNLOADING/ERROR), not any one request. "No hidden implicit
 * lifecycle": every in-flight chat completion request (streaming or not)
 * is always in exactly one of these states, and every real transition in
 * server.cpp is a plain, traceable store to a variable of this type --
 * never inferred after the fact from other bookkeeping (admission ticket
 * held/not held, thread joinable/not joinable, etc).
 *
 *   QUEUED     -- admitted (passed request_admission_gate_t) but not yet
 *                 holding a decode_concurrency_gate_t ticket; waiting its
 *                 bounded turn (Section 5: "bounded queue... backpressure").
 *   ACTIVE     -- holding a decode ticket, membrane_session_generate() is
 *                 actually running for this request right now.
 *   CANCELLING -- a cancellation was requested (client disconnect or a
 *                 stop-sequence match) but generation has not yet
 *                 observed/honored it and returned.
 *   CANCELLED  -- generation stopped early because of a real cancellation
 *                 (gen_res.cancelled == true, no stop-sequence match --
 *                 see server.cpp's own stream_worker_fn() comment on
 *                 telling the two apart).
 *   DONE       -- generation completed normally (including a stop-
 *                 sequence match, which is a NORMAL completion from the
 *                 client's point of view, not CANCELLED).
 *   ERROR      -- generation failed (gen_res.ok == false), or the request
 *                 never reached generation at all (e.g. model load
 *                 failure, chat template failure).
 *
 * This is bookkeeping/observability, not a scheduler policy of its own --
 * decode_concurrency_gate_t/request_admission_gate_t remain the real
 * admission/concurrency mechanisms; this enum only names, explicitly,
 * which of their states a given request is currently in.
 */

enum class membrane_request_state_t
{
	QUEUED,
	ACTIVE,
	CANCELLING,
	CANCELLED,
	DONE,
	ERROR,
};

static inline const char	*membrane_request_state_name(
			membrane_request_state_t state)
{
	switch (state)
	{
		case membrane_request_state_t::QUEUED: return ("queued");
		case membrane_request_state_t::ACTIVE: return ("active");
		case membrane_request_state_t::CANCELLING: return ("cancelling");
		case membrane_request_state_t::CANCELLED: return ("cancelled");
		case membrane_request_state_t::DONE: return ("done");
		case membrane_request_state_t::ERROR: return ("error");
		default: return ("unknown");
	}
}

#endif
