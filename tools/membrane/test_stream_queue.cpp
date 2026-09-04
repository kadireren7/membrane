#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "stream_queue.h"
#include "test_helpers.h"

/*
 * Mega Phase B, PR B2: real multi-threaded unit tests for stream_queue.h's
 * bounded blocking queue -- the ENTIRE synchronization surface a
 * streaming request's own generation worker thread and HTTP connection
 * thread share (server.cpp's own top comment on stream_queue.h's usage
 * has the full architecture). Real std::thread producers/consumers, run
 * under ASan/TSan in CI (service-lifecycle-tests-style target-scoped
 * job -- no llama/ggml link needed, this header has none), exactly the
 * kind of coverage this project's own history shows is easy to
 * accidentally skip (see docs/server.md's known_limitations on PR A3's
 * own server-thread-sanitizer gap).
 */

static void	test_single_push_pop(void)
{
	std::atomic<bool>	cancel{false};
	stream_queue_t		q(&cancel);
	s_stream_event		ev;

	ev.type = MEMBRANE_STREAM_EVENT_TOKEN;
	ev.text = "hello";
	TEST_ASSERT(q.push_blocking(ev) == true, "push succeeds when not full "
		"and not cancelled");

	s_stream_event	popped;

	TEST_ASSERT(q.pop_wait(std::chrono::milliseconds(500), &popped) == true,
		"pop_wait finds the already-pushed event immediately");
	TEST_ASSERT(popped.type == MEMBRANE_STREAM_EVENT_TOKEN, "type round-trips");
	TEST_ASSERT(popped.text == "hello", "text round-trips");
}

static void	test_pop_wait_times_out_on_empty_queue(void)
{
	std::atomic<bool>	cancel{false};
	stream_queue_t		q(&cancel);
	s_stream_event		out;
	auto				t0 = std::chrono::steady_clock::now();
	bool				got = q.pop_wait(std::chrono::milliseconds(100), &out);
	auto				elapsed = std::chrono::steady_clock::now() - t0;

	TEST_ASSERT(got == false, "pop_wait returns false on a real timeout "
		"against an empty queue");
	TEST_ASSERT(elapsed >= std::chrono::milliseconds(90),
		"the timeout was actually honored (not an immediate spurious "
		"return)");
}

static void	test_fifo_order_preserved(void)
{
	std::atomic<bool>	cancel{false};
	stream_queue_t		q(&cancel);

	for (int i = 0; i < 5; ++i)
	{
		s_stream_event	ev;

		ev.type = MEMBRANE_STREAM_EVENT_TOKEN;
		ev.text = std::to_string(i);
		TEST_ASSERT(q.push_blocking(ev) == true, "push succeeds");
	}
	for (int i = 0; i < 5; ++i)
	{
		s_stream_event	out;

		TEST_ASSERT(q.pop_wait(std::chrono::milliseconds(500), &out) == true,
			"pop succeeds");
		TEST_ASSERT(out.text == std::to_string(i),
			"events are popped in the exact order they were pushed "
			"(FIFO, never reordered)");
	}
}

/* Section 24 of the task: real backpressure -- a push beyond capacity
 * genuinely BLOCKS (never silently drops, never grows unboundedly) until
 * a consumer thread makes room. Proven here with a real producer thread
 * that fills the queue to capacity and attempts one more push; the main
 * thread (acting as the slow consumer) only starts popping after
 * confirming the producer is still blocked. */
static void	test_backpressure_blocks_until_consumed(void)
{
	std::atomic<bool>	cancel{false};
	stream_queue_t		q(&cancel);
	std::atomic<bool>	overflow_push_returned{false};
	std::atomic<bool>	overflow_push_result{false};

	for (int i = 0; i < MEMBRANE_STREAM_QUEUE_CAPACITY; ++i)
	{
		s_stream_event	ev;

		ev.type = MEMBRANE_STREAM_EVENT_TOKEN;
		TEST_ASSERT(q.push_blocking(ev) == true,
			"filling the queue to exactly capacity succeeds without "
			"blocking");
	}
	std::thread	producer([&]()
		{
			s_stream_event	ev;

			ev.type = MEMBRANE_STREAM_EVENT_TOKEN;
			bool	ok = q.push_blocking(ev);

			overflow_push_result.store(ok);
			overflow_push_returned.store(true);
		});

	std::this_thread::sleep_for(std::chrono::milliseconds(150));
	TEST_ASSERT(overflow_push_returned.load() == false,
		"a push beyond capacity is still blocked after 150ms -- real "
		"backpressure, not a silent drop or an unbounded queue");

	s_stream_event	out;

	TEST_ASSERT(q.pop_wait(std::chrono::milliseconds(500), &out) == true,
		"popping one event frees exactly one slot");

	producer.join();
	TEST_ASSERT(overflow_push_returned.load() == true,
		"the blocked push unblocked once a slot freed up");
	TEST_ASSERT(overflow_push_result.load() == true,
		"the previously-blocked push ultimately succeeded (never "
		"silently cancelled just because it once blocked)");
}

/* Section 20 of the task: setting cancel_flag wakes a BLOCKED push
 * immediately -- a disconnected client's own full, un-drained queue can
 * never wedge the generation worker thread forever. */
static void	test_cancel_flag_wakes_blocked_push(void)
{
	std::atomic<bool>	cancel{false};
	stream_queue_t		q(&cancel);

	for (int i = 0; i < MEMBRANE_STREAM_QUEUE_CAPACITY; ++i)
	{
		s_stream_event	ev;

		ev.type = MEMBRANE_STREAM_EVENT_TOKEN;
		q.push_blocking(ev);
	}
	std::atomic<bool>	push_returned{false};
	std::atomic<bool>	push_result{true};
	std::thread	producer([&]()
		{
			s_stream_event	ev;

			ev.type = MEMBRANE_STREAM_EVENT_TOKEN;
			bool	ok = q.push_blocking(ev);	/* the queue is already full
											 * and NOBODY ever pops here --
											 * only cancel_flag can free
											 * this thread */

			push_result.store(ok);
			push_returned.store(true);
		});

	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	TEST_ASSERT(push_returned.load() == false,
		"still blocked before cancellation, exactly as expected");
	cancel.store(true);
	producer.join();
	TEST_ASSERT(push_returned.load() == true,
		"setting cancel_flag wakes the blocked push -- no dependency on "
		"a consumer that will never come");
	TEST_ASSERT(push_result.load() == false,
		"a push that only unblocked because of cancellation reports "
		"failure -- the caller (the token callback) knows to stop, "
		"never mistakes this for a real successful enqueue");
}

/* A real producer/consumer pair on separate threads, generating enough
 * volume that a real race (if one existed) would be likely to surface
 * under ASan/TSan -- not just a single-threaded call sequence dressed up
 * as a "concurrency" test. */
static void	test_real_producer_consumer_threads(void)
{
	std::atomic<bool>	cancel{false};
	stream_queue_t		q(&cancel);
	const int			n = 500;
	std::vector<int>	consumed;

	std::thread	producer([&]()
		{
			for (int i = 0; i < n; ++i)
			{
				s_stream_event	ev;

				ev.type = MEMBRANE_STREAM_EVENT_TOKEN;
				ev.text = std::to_string(i);
				q.push_blocking(ev);
			}
		});
	std::thread	consumer([&]()
		{
			for (int i = 0; i < n; ++i)
			{
				s_stream_event	out;

				while (!q.pop_wait(std::chrono::milliseconds(1000), &out))
					;
				consumed.push_back(std::stoi(out.text));
			}
		});

	producer.join();
	consumer.join();
	TEST_ASSERT((int)consumed.size() == n,
		"the consumer thread received exactly every event the producer "
		"thread sent -- no lost, no duplicated events");
	bool	in_order = true;

	for (int i = 0; i < n; ++i)
		if (consumed[(size_t)i] != i)
			in_order = false;
	TEST_ASSERT(in_order, "500 events across real producer/consumer "
		"threads arrive in exact FIFO order");
}

int	main(void)
{
	test_single_push_pop();
	test_pop_wait_times_out_on_empty_queue();
	test_fifo_order_preserved();
	test_backpressure_blocks_until_consumed();
	test_cancel_flag_wakes_blocked_push();
	test_real_producer_consumer_threads();
	printf("test_stream_queue: all tests passed\n");
	return (0);
}
