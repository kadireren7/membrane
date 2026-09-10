#include <cstdio>

#include "residency_planner.h"
#include "test_helpers.h"

/*
 * Mega Phase E, PR E2: pure unit tests for residency_planner.h --
 * synthetic slot states, no real llama_model/mutex/disk I/O at all (the
 * same testable-without-real-hardware pattern test_variant_selector.cpp
 * already established for hardware-aware quant selection). Covers every
 * policy requirement from Section 12 of the task directly: LRU eviction,
 * "never evict actively generating", "never evict pinned", "fail safely
 * when nothing is evictable", and hot-switch/fresh-slot selection.
 */

static membrane_residency_slot_view_t	slot(const char *name,
				bool model_loaded, bool pinned, bool generating,
				int64_t last_used_seq)
{
	membrane_residency_slot_view_t	s;

	s.name = name;
	s.model_loaded = model_loaded;
	s.pinned = pinned;
	s.generating = generating;
	s.last_used_seq = last_used_seq;
	return (s);
}

static void	test_empty_slots_use_first_free(void)
{
	std::vector<membrane_residency_slot_view_t>	slots;

	slots.push_back(slot("", false, false, false, 0));
	slots.push_back(slot("", false, false, false, 0));

	membrane_residency_plan_t	plan = membrane_plan_residency(slots, "A");

	TEST_ASSERT(plan.decision == membrane_residency_decision_t::USE_EMPTY,
		"two free slots -- USE_EMPTY");
	TEST_ASSERT(plan.slot_index == 0, "the first free slot is chosen");
}

static void	test_hot_hit_already_resident(void)
{
	std::vector<membrane_residency_slot_view_t>	slots;

	slots.push_back(slot("A", true, false, false, 5));
	slots.push_back(slot("B", true, false, false, 9));

	membrane_residency_plan_t	plan = membrane_plan_residency(slots, "B");

	TEST_ASSERT(plan.decision == membrane_residency_decision_t::HOT_HIT,
		"a name already resident in some slot is a HOT_HIT");
	TEST_ASSERT(plan.slot_index == 1, "the correct slot is identified");
}

static void	test_hot_hit_wins_over_free_slot(void)
{
	std::vector<membrane_residency_slot_view_t>	slots;

	slots.push_back(slot("A", true, false, false, 5));
	slots.push_back(slot("", false, false, false, 0));

	membrane_residency_plan_t	plan = membrane_plan_residency(slots, "A");

	TEST_ASSERT(plan.decision == membrane_residency_decision_t::HOT_HIT,
		"an already-resident name is a hot hit even when another slot "
		"is free -- never a spurious second load into the free slot");
	TEST_ASSERT(plan.slot_index == 0, "the resident slot, not the free "
		"one, is returned");
}

static void	test_hot_hit_on_in_progress_load(void)
{
	/* model_loaded == false but name is already claimed -- another
	 * request is mid-load for this exact name; a second request for the
	 * SAME name must be pointed at that same slot (to wait on its real
	 * mtx), never treated as free. */
	std::vector<membrane_residency_slot_view_t>	slots;

	slots.push_back(slot("A", false, false, false, 0));

	membrane_residency_plan_t	plan = membrane_plan_residency(slots, "A");

	TEST_ASSERT(plan.decision == membrane_residency_decision_t::HOT_HIT,
		"a slot mid-load for this exact name is a hot hit, not free");
	TEST_ASSERT(plan.slot_index == 0, "the correct (in-progress) slot");
}

static void	test_lru_eviction_picks_oldest(void)
{
	std::vector<membrane_residency_slot_view_t>	slots;

	slots.push_back(slot("A", true, false, false, 10));
	slots.push_back(slot("B", true, false, false, 3));

	membrane_residency_plan_t	plan = membrane_plan_residency(slots, "C");

	TEST_ASSERT(plan.decision == membrane_residency_decision_t::EVICT,
		"both slots full, both evictable -- EVICT");
	TEST_ASSERT(plan.slot_index == 1, "the LEAST recently used slot "
		"(lower last_used_seq) is evicted, never the more recently "
		"used one");
}

static void	test_never_evicts_pinned(void)
{
	std::vector<membrane_residency_slot_view_t>	slots;

	/* The only unpinned candidate is MORE recently used than the pinned
	 * one -- a naive "always evict the oldest" would incorrectly pick
	 * the pinned slot; pinning must win regardless of recency. */
	slots.push_back(slot("A", true, true, false, 1));
	slots.push_back(slot("B", true, false, false, 99));

	membrane_residency_plan_t	plan = membrane_plan_residency(slots, "C");

	TEST_ASSERT(plan.decision == membrane_residency_decision_t::EVICT,
		"one unpinned candidate still exists -- EVICT");
	TEST_ASSERT(plan.slot_index == 1, "the pinned slot (index 0) is "
		"never chosen, even though it is the oldest");
}

static void	test_never_evicts_generating(void)
{
	std::vector<membrane_residency_slot_view_t>	slots;

	slots.push_back(slot("A", true, false, true, 1));
	slots.push_back(slot("B", true, false, false, 99));

	membrane_residency_plan_t	plan = membrane_plan_residency(slots, "C");

	TEST_ASSERT(plan.decision == membrane_residency_decision_t::EVICT,
		"one non-generating candidate still exists -- EVICT");
	TEST_ASSERT(plan.slot_index == 1, "the actively-generating slot "
		"(index 0) is never chosen, even though it is the oldest");
}

static void	test_exhausted_when_all_pinned(void)
{
	std::vector<membrane_residency_slot_view_t>	slots;

	slots.push_back(slot("A", true, true, false, 1));
	slots.push_back(slot("B", true, true, false, 2));

	membrane_residency_plan_t	plan = membrane_plan_residency(slots, "C");

	TEST_ASSERT(plan.decision == membrane_residency_decision_t::EXHAUSTED,
		"every slot pinned, none free -- EXHAUSTED (fail safe, never "
		"violate pinning)");
}

static void	test_exhausted_when_all_generating(void)
{
	std::vector<membrane_residency_slot_view_t>	slots;

	slots.push_back(slot("A", true, false, true, 1));
	slots.push_back(slot("B", true, false, true, 2));

	membrane_residency_plan_t	plan = membrane_plan_residency(slots, "C");

	TEST_ASSERT(plan.decision == membrane_residency_decision_t::EXHAUSTED,
		"every slot actively generating, none free -- EXHAUSTED (fail "
		"safe, never evict a live generation)");
}

static void	test_exhausted_mixed_pinned_and_generating(void)
{
	std::vector<membrane_residency_slot_view_t>	slots;

	slots.push_back(slot("A", true, true, false, 1));
	slots.push_back(slot("B", true, false, true, 2));

	membrane_residency_plan_t	plan = membrane_plan_residency(slots, "C");

	TEST_ASSERT(plan.decision == membrane_residency_decision_t::EXHAUSTED,
		"one slot pinned, the other generating -- neither evictable, "
		"EXHAUSTED");
}

static void	test_tie_break_prefers_lowest_index(void)
{
	std::vector<membrane_residency_slot_view_t>	slots;

	slots.push_back(slot("A", true, false, false, 7));
	slots.push_back(slot("B", true, false, false, 7));

	membrane_residency_plan_t	plan = membrane_plan_residency(slots, "C");

	TEST_ASSERT(plan.decision == membrane_residency_decision_t::EVICT,
		"both full, both evictable -- EVICT");
	TEST_ASSERT(plan.slot_index == 0, "an exact last_used_seq tie "
		"deterministically prefers the lowest slot index");
}

static void	test_deterministic_repeat(void)
{
	std::vector<membrane_residency_slot_view_t>	slots;

	slots.push_back(slot("A", true, false, false, 4));
	slots.push_back(slot("B", true, false, false, 1));

	membrane_residency_plan_t	p1 = membrane_plan_residency(slots, "C");
	membrane_residency_plan_t	p2 = membrane_plan_residency(slots, "C");

	TEST_ASSERT(p1.decision == p2.decision && p1.slot_index == p2.slot_index,
		"identical inputs always produce an identical decision");
}

int	main(void)
{
	test_empty_slots_use_first_free();
	test_hot_hit_already_resident();
	test_hot_hit_wins_over_free_slot();
	test_hot_hit_on_in_progress_load();
	test_lru_eviction_picks_oldest();
	test_never_evicts_pinned();
	test_never_evicts_generating();
	test_exhausted_when_all_pinned();
	test_exhausted_when_all_generating();
	test_exhausted_mixed_pinned_and_generating();
	test_tie_break_prefers_lowest_index();
	test_deterministic_repeat();
	printf("test_residency_planner: all tests passed\n");
	return (0);
}
