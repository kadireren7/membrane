#ifndef MEMBRANE_STREAM_QUEUE_H
# define MEMBRANE_STREAM_QUEUE_H

# include <atomic>
# include <chrono>
# include <condition_variable>
# include <cstddef>
# include <deque>
# include <mutex>
# include <string>
# include <utility>

/*
 * Mega Phase B, PR B2: the ONE synchronization point between a streaming
 * chat completion's generation worker thread (pushes, from inside the
 * push-based decode-loop token callback) and the HTTP connection
 * thread's own pull-based content-provider callback (pops) -- pulled out
 * of server.cpp into its own header-only, llama/httplib-free module
 * specifically so this project's own real concurrency (mutex + condition
 * variable + a cross-thread atomic cancel flag) is unit-testable under
 * ASan/TSan WITHOUT a real model or a real HTTP round trip, matching the
 * lesson already learned once this project's own history (PR A3's
 * server-thread-sanitizer job, added after discovering the existing
 * thread-sanitizer CI job provided zero real coverage of new
 * multi-threaded server code -- see docs/server.md's known_limitations).
 */

# define MEMBRANE_STREAM_QUEUE_CAPACITY	64

enum e_stream_event_type
{
	MEMBRANE_STREAM_EVENT_TOKEN = 0,
	MEMBRANE_STREAM_EVENT_DONE,
	MEMBRANE_STREAM_EVENT_ERROR,
	MEMBRANE_STREAM_EVENT_CANCELLED,
};

struct s_stream_event
{
	e_stream_event_type	type = MEMBRANE_STREAM_EVENT_TOKEN;
	std::string			text;			/* TOKEN: the text delta */
	std::string			error_code;		/* ERROR only */
	std::string			error_message;	/* ERROR only */
	std::string			finish_reason;	/* DONE only: "stop"/"length" */
	size_t				prompt_tokens = 0;		/* DONE only */
	size_t				completion_tokens = 0;	/* DONE only */
	bool				include_usage = false;	/* DONE only -- Section
											 * 19: usage is only emitted
											 * if the request explicitly
											 * asked via stream_options.
											 * include_usage, matching
											 * real OpenAI streaming
											 * behavior -- never sent
											 * unconditionally, which
											 * would NOT match the API
											 * this project claims
											 * compatibility with */
};

/* Bounded, thread-safe TOKEN/DONE/ERROR/CANCELLED queue.
 * push_blocking() blocks while the queue is at capacity (real
 * backpressure, Section 24: "never grow memory unboundedly if the
 * client consumes slowly"), waking early and returning false the moment
 * *cancel_flag becomes true -- a disconnected client's own dead queue
 * can never wedge the worker thread forever. */
class stream_queue_t
{
	public:
		explicit stream_queue_t(const std::atomic<bool> *cancel_flag)
			: cancel_flag_(cancel_flag) {}

		/* Real bug found and fixed by test_stream_queue.cpp's own
		 * test_cancel_flag_wakes_blocked_push(): cancel_flag is a plain
		 * std::atomic<bool> set from OUTSIDE this class entirely (by
		 * server.cpp's stream_provide()/stream_release(), or by this
		 * test) -- nothing about setting it ever calls notify_all() on
		 * cv_, so an unbounded cv_.wait() blocked here would never wake
		 * up on cancellation alone; only a future, unrelated
		 * push/pop happening to also notify would ever free it (which,
		 * for a genuinely disconnected client, never happens -- a real
		 * deadlocked worker thread). Fixed by polling with a bounded
		 * wait_for() instead of relying on every future cancel_flag
		 * setter to remember an explicit notify -- a whole class of
		 * "forgot to notify" bugs removed at the cost of at most 50ms of
		 * extra cancellation latency, fully acceptable (Section 24 of
		 * the task: "correctness over throughput"). */
		bool	push_blocking(s_stream_event ev)
		{
			std::unique_lock<std::mutex>	lock(mtx_);

			while (q_.size() >= MEMBRANE_STREAM_QUEUE_CAPACITY
				&& !cancel_flag_->load(std::memory_order_relaxed))
				cv_.wait_for(lock, std::chrono::milliseconds(50));
			if (cancel_flag_->load(std::memory_order_relaxed))
				return (false);
			q_.push_back(std::move(ev));
			cv_.notify_all();
			return (true);
		}

		/* Returns false on a bounded wait timeout (queue stayed empty) --
		 * the caller (the content-provider callback) uses this gap to
		 * check DataSink::is_writable() without blocking forever on a
		 * worker that has stalled or a client that never sends anything
		 * back (an HTTP response body is one-directional, but a dead TCP
		 * peer is only reliably detected by actually attempting/probing
		 * the connection -- httplib's own is_writable() does exactly
		 * that). */
		bool	pop_wait(std::chrono::milliseconds timeout, s_stream_event *out)
		{
			std::unique_lock<std::mutex>	lock(mtx_);

			if (!cv_.wait_for(lock, timeout, [&]{ return !q_.empty(); }))
				return (false);
			*out = std::move(q_.front());
			q_.pop_front();
			cv_.notify_all();
			return (true);
		}

		/* Test-only introspection (also handy for assertions elsewhere):
		 * never used by server.cpp's own real request path. */
		size_t	size_for_test(void)
		{
			std::lock_guard<std::mutex>	lock(mtx_);

			return (q_.size());
		}

	private:
		std::deque<s_stream_event>	q_;
		std::mutex					mtx_;
		std::condition_variable	cv_;
		const std::atomic<bool>	*cancel_flag_;
};

#endif
