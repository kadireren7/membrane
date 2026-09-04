#ifndef MEMBRANE_REQUEST_ADMISSION_H
# define MEMBRANE_REQUEST_ADMISSION_H

# include <atomic>

/*
 * Mega Phase B, PR B3, Section 29 of the task: a BOUNDED pending-request
 * admission gate for POST /v1/chat/completions -- this project has no
 * continuous batching and fully serializes actual generation (one
 * std::mutex, unchanged since PR A3/B2), so an unbounded pile-up of
 * concurrent requests would otherwise just grow an unbounded queue of
 * threads blocked on that mutex forever. try_admit() gives a request an
 * immediate, deterministic yes/no instead -- callers that fail admission
 * return 503 SERVICE_UNAVAILABLE with Retry-After, never silently queue
 * past the bound and never allocate anything unbounded to track it (a
 * single atomic counter).
 *
 * Pulled into its own header-only, httplib/llama-free module specifically
 * so it is unit-testable with real concurrent std::thread producers
 * (test_request_admission.cpp), matching stream_queue.h's own precedent
 * from PR B2.
 */

class request_admission_gate_t
{
	public:
		explicit request_admission_gate_t(int capacity)
			: capacity_(capacity), count_(0) {}

		/* Non-blocking. Returns true (and reserves one slot -- the caller
		 * MUST eventually call release() exactly once) iff fewer than
		 * `capacity` requests are currently admitted; false otherwise,
		 * with no side effect. */
		bool	try_admit(void)
		{
			int	cur = count_.load(std::memory_order_relaxed);

			while (cur < capacity_)
			{
				if (count_.compare_exchange_weak(cur, cur + 1,
						std::memory_order_acq_rel, std::memory_order_relaxed))
					return (true);
			}
			return (false);
		}

		void	release(void)
		{
			count_.fetch_sub(1, std::memory_order_acq_rel);
		}

		int	capacity(void) const
		{
			return (capacity_);
		}

		/* Test-only introspection -- never used by server.cpp's own real
		 * request path (which only ever needs try_admit()/release()). */
		int	count_for_test(void) const
		{
			return (count_.load(std::memory_order_relaxed));
		}

	private:
		int					capacity_;
		std::atomic<int>	count_;
};

/* RAII ticket: release()s automatically on destruction iff admitted() was
 * true -- a caller that fails admission still constructs one (admitted()
 * == false), which is always safe to destroy (a no-op release). Move-only
 * (mirrors std::unique_lock's own ownership-transfer contract) so it can
 * be threaded through PR B2's own streaming path (server.cpp's
 * s_stream_request_state), whose real "occupied" duration spans an async
 * worker thread outliving the function that created the ticket. */
class request_admission_ticket_t
{
	public:
		request_admission_ticket_t(void)
			: gate_(NULL), admitted_(false) {}

		request_admission_ticket_t(request_admission_gate_t *gate, bool admitted)
			: gate_(gate), admitted_(admitted) {}

		request_admission_ticket_t(const request_admission_ticket_t &) = delete;
		request_admission_ticket_t	&operator=(
			const request_admission_ticket_t &) = delete;

		request_admission_ticket_t(request_admission_ticket_t &&other) noexcept
			: gate_(other.gate_), admitted_(other.admitted_)
		{
			other.gate_ = NULL;
			other.admitted_ = false;
		}

		request_admission_ticket_t	&operator=(
			request_admission_ticket_t &&other) noexcept
		{
			if (this != &other)
			{
				release_if_admitted();
				gate_ = other.gate_;
				admitted_ = other.admitted_;
				other.gate_ = NULL;
				other.admitted_ = false;
			}
			return (*this);
		}

		~request_admission_ticket_t(void)
		{
			release_if_admitted();
		}

		bool	admitted(void) const
		{
			return (admitted_);
		}

	private:
		void	release_if_admitted(void)
		{
			if (admitted_ && gate_ != NULL)
				gate_->release();
			admitted_ = false;
		}

		request_admission_gate_t	*gate_;
		bool						admitted_;
};

static inline request_admission_ticket_t	membrane_try_admit(
			request_admission_gate_t *gate)
{
	return (request_admission_ticket_t(gate, gate->try_admit()));
}

#endif
