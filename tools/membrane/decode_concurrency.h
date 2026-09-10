#ifndef MEMBRANE_DECODE_CONCURRENCY_H
# define MEMBRANE_DECODE_CONCURRENCY_H

# include <condition_variable>
# include <mutex>

/*
 * Mega Phase E, PR E1, Section 5/9 of the task: a BOUNDED counting
 * semaphore for how many POST /v1/chat/completions requests may be
 * inside an actual llama_decode() call AT THE SAME TIME.
 *
 * This is deliberately separate from request_admission_gate_t
 * (request_admission.h), which only bounds how many requests may be
 * ADMITTED (queued + decoding) before an immediate 503 -- that gate
 * never granted real concurrent decode; every admitted request still
 * fully serialized on s_membrane_server_model_state::mtx before this
 * phase. This gate is what actually replaces that serialization: it
 * blocks (never rejects -- a request that reached this point has
 * already been admitted and is expected to wait its bounded turn,
 * Section 5's own "bounded queue... backpressure" requirement) until
 * fewer than `capacity` decodes are already running, then lets exactly
 * that many run truly concurrently.
 *
 * Capacity is deliberately small and real-memory-motivated, not
 * performance-motivated: this project's per-request architecture opens
 * a brand-new llama_context (and therefore a brand-new, independent KV
 * cache) for every request (runtime_session.h's own "persistent model,
 * new context per request" contract, unchanged by this phase -- see
 * decode_concurrency.h's sibling comment in server.cpp for why E1 does
 * NOT adopt upstream llama.cpp's shared-context/seq_id slot design).
 * Two decodes in flight means two independent KV caches resident at
 * once; on a memory-constrained host (this project's own real dev/CI
 * hosts included, see dev-machine-memory-constraints in the project's
 * own operating history) an unbounded or even moderately large
 * capacity would risk real OOM. Default 2 is intentionally
 * conservative -- see MEMBRANE_DEFAULT_MAX_CONCURRENT_DECODE below.
 *
 * FIFO fairness is NOT guaranteed by this implementation (a plain
 * std::condition_variable::notify_all() wakes every waiter, and the OS
 * scheduler decides which one actually re-acquires the mutex first) --
 * this is disclosed honestly in docs/release-v1.0.0.md and the E1
 * fairness evidence rather than overclaimed as strict ordering.
 */

# define MEMBRANE_DEFAULT_MAX_CONCURRENT_DECODE	2

class decode_concurrency_gate_t
{
	public:
		explicit decode_concurrency_gate_t(int capacity)
			: capacity_(capacity), in_use_(0) {}

		/* Blocks until a slot is free, then reserves it. Capacity <= 0
		 * degenerates to "always block forever" only if capacity_ is
		 * exactly 0 forever (never a real product configuration --
		 * membrane_max_concurrent_decode() in server.cpp floors it at 1). */
		void	acquire(void)
		{
			std::unique_lock<std::mutex>	lock(mtx_);

			cv_.wait(lock, [this] { return (in_use_ < capacity_); });
			++in_use_;
		}

		void	release(void)
		{
			{
				std::lock_guard<std::mutex>	lock(mtx_);

				--in_use_;
			}
			cv_.notify_all();
		}

		int	capacity(void) const
		{
			return (capacity_);
		}

		/* Test-only introspection, same convention as
		 * request_admission_gate_t::count_for_test(). */
		int	in_use_for_test(void)
		{
			std::lock_guard<std::mutex>	lock(mtx_);

			return (in_use_);
		}

	private:
		int						capacity_;
		int						in_use_;
		std::mutex				mtx_;
		std::condition_variable	cv_;
};

/* RAII ticket -- acquire()s in the constructor (blocking), release()s in
 * the destructor. Move-only, same ownership-transfer contract as
 * request_admission_ticket_t (needed for the streaming path, whose real
 * decode happens on a worker thread outliving the function that admits
 * the request). */
class decode_concurrency_ticket_t
{
	public:
		decode_concurrency_ticket_t(void)
			: gate_(NULL) {}

		explicit decode_concurrency_ticket_t(decode_concurrency_gate_t *gate)
			: gate_(gate)
		{
			gate_->acquire();
		}

		decode_concurrency_ticket_t(const decode_concurrency_ticket_t &) = delete;
		decode_concurrency_ticket_t	&operator=(
			const decode_concurrency_ticket_t &) = delete;

		decode_concurrency_ticket_t(decode_concurrency_ticket_t &&other) noexcept
			: gate_(other.gate_)
		{
			other.gate_ = NULL;
		}

		decode_concurrency_ticket_t	&operator=(
			decode_concurrency_ticket_t &&other) noexcept
		{
			if (this != &other)
			{
				release_if_held();
				gate_ = other.gate_;
				other.gate_ = NULL;
			}
			return (*this);
		}

		~decode_concurrency_ticket_t(void)
		{
			release_if_held();
		}

	private:
		void	release_if_held(void)
		{
			if (gate_ != NULL)
				gate_->release();
			gate_ = NULL;
		}

		decode_concurrency_gate_t	*gate_;
};

#endif
