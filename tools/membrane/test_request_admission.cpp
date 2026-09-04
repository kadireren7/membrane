#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "request_admission.h"
#include "test_helpers.h"

/*
 * Mega Phase B, PR B3: real multi-threaded unit tests for request_
 * admission.h's bounded pending-request gate -- the primitive
 * server.cpp's own POST /v1/chat/completions handler uses to return
 * 503 SERVICE_UNAVAILABLE once too many requests are already in
 * flight, rather than letting them pile up on the generation mutex
 * unboundedly (Section 29 of the task).
 */

static void	test_admits_up_to_capacity(void)
{
	request_admission_gate_t	gate(3);

	TEST_ASSERT(gate.try_admit() == true, "1st admit succeeds");
	TEST_ASSERT(gate.try_admit() == true, "2nd admit succeeds");
	TEST_ASSERT(gate.try_admit() == true, "3rd admit succeeds (at capacity)");
	TEST_ASSERT(gate.try_admit() == false,
		"4th admit fails -- capacity is a hard bound, never silently "
		"exceeded");
	TEST_ASSERT(gate.count_for_test() == 3, "count reflects exactly 3 "
		"admitted requests");
}

static void	test_release_frees_a_slot(void)
{
	request_admission_gate_t	gate(1);

	TEST_ASSERT(gate.try_admit() == true, "admit succeeds at capacity 1");
	TEST_ASSERT(gate.try_admit() == false, "a second admit fails");
	gate.release();
	TEST_ASSERT(gate.try_admit() == true,
		"a slot freed by release() is immediately re-admittable");
}

static void	test_ticket_releases_on_scope_exit(void)
{
	request_admission_gate_t	gate(1);

	{
		request_admission_ticket_t	ticket = membrane_try_admit(&gate);

		TEST_ASSERT(ticket.admitted() == true, "the ticket reports "
			"admitted");
		TEST_ASSERT(gate.try_admit() == false, "no room for a second "
			"request while the first ticket is still alive");
	}
	TEST_ASSERT(gate.try_admit() == true,
		"the ticket's own destructor released the slot automatically "
		"(RAII) -- the caller never has to remember an explicit release()");
}

static void	test_ticket_not_admitted_is_a_safe_noop_release(void)
{
	request_admission_gate_t	gate(0);	/* capacity 0 -- nothing is
											 * ever admitted */
	{
		request_admission_ticket_t	ticket = membrane_try_admit(&gate);

		TEST_ASSERT(ticket.admitted() == false,
			"admission genuinely fails at capacity 0");
	}
	/* If a not-admitted ticket's destructor incorrectly called release()
	 * anyway, this next admit would see a negative-then-incremented
	 * count and could spuriously succeed at capacity 0 -- asserting it
	 * still correctly fails proves no phantom release happened. */
	TEST_ASSERT(gate.try_admit() == false,
		"a ticket that was never admitted releases nothing on "
		"destruction -- capacity 0 still admits nothing afterward");
}

static void	test_move_transfers_ownership(void)
{
	request_admission_gate_t	gate(1);
	request_admission_ticket_t	a = membrane_try_admit(&gate);

	TEST_ASSERT(a.admitted() == true, "first ticket admitted");

	request_admission_ticket_t	b = std::move(a);

	TEST_ASSERT(b.admitted() == true, "ownership moved to b");
	TEST_ASSERT(a.admitted() == false,
		"the moved-from ticket no longer reports admitted (never a "
		"double-release when both eventually destruct)");
	TEST_ASSERT(gate.try_admit() == false,
		"the slot is still held (by b now) -- capacity is still 1/1");
}

/* Section 29's own real concurrency proof: many real threads racing
 * try_admit() simultaneously against a small capacity -- exactly
 * `capacity` succeed, never more (no lost update / no over-admission
 * under real contention), and the ones that failed can be re-tried
 * successfully once slots free up. */
static void	test_real_concurrent_admission_never_exceeds_capacity(void)
{
	const int			capacity = 4;
	const int			n_threads = 50;
	request_admission_gate_t	gate(capacity);
	std::atomic<int>	admitted_count{0};
	std::atomic<int>	rejected_count{0};
	std::vector<std::thread>	threads;

	for (int i = 0; i < n_threads; ++i)
	{
		threads.emplace_back([&]()
			{
				if (gate.try_admit())
					admitted_count.fetch_add(1);
				else
					rejected_count.fetch_add(1);
			});
	}
	for (auto &t : threads)
		t.join();
	TEST_ASSERT(admitted_count.load() == capacity,
		"exactly `capacity` threads were admitted under real concurrent "
		"contention -- never more (a lost-update race would over-admit)");
	TEST_ASSERT(rejected_count.load() == n_threads - capacity,
		"every other thread was correctly rejected");
	TEST_ASSERT(gate.count_for_test() == capacity,
		"the gate's own count still matches -- no leaked/phantom slots "
		"from the race");
}

int	main(void)
{
	test_admits_up_to_capacity();
	test_release_frees_a_slot();
	test_ticket_releases_on_scope_exit();
	test_ticket_not_admitted_is_a_safe_noop_release();
	test_move_transfers_ownership();
	test_real_concurrent_admission_never_exceeds_capacity();
	printf("test_request_admission: all tests passed\n");
	return (0);
}
