#ifndef MEMBRANE_RESIDENCY_PLANNER_H
# define MEMBRANE_RESIDENCY_PLANNER_H

# include <cstdint>
# include <string>
# include <vector>

/*
 * Mega Phase E, PR E2, Section 10-12 of the task: the PURE slot-selection
 * decision for multi-model residency -- deliberately separated from
 * server.cpp's own real mutexes/llama_model_session_t/disk I/O, same
 * "pure decision logic, no real hardware access" split this project
 * already uses for variant_selector.h (hardware-aware quant selection
 * against synthetic host facts) and decode_concurrency.h/request_
 * admission.h (real concurrency primitives, but no model-specific
 * knowledge). This means the actual eviction/pinning/hot-switch POLICY
 * is unit-testable with synthetic slot states and no real GGUF model at
 * all (test_residency_planner.cpp) -- server.cpp's own acquire_model_
 * slot() (see its top comment) only ever snapshots real slot state INTO
 * a vector of these views, calls this function once, and acts on the
 * result; it never re-implements the decision itself.
 *
 * `name` mirrors server.cpp's own s_model_slot::claimed_name -- the name
 * this slot currently represents to residency selection, empty iff the
 * slot is genuinely free. It is set (by the caller, under its own
 * residency_mtx) BOTH for a fully resident, settled model AND for one a
 * concurrent request is still in the middle of loading/evicting into --
 * membrane_plan_residency() itself does not need to tell those two
 * cases apart: either way, a second request for that SAME name should
 * be pointed at the SAME slot (HOT_HIT -- the real settle/fail outcome
 * is then determined by that slot's own mtx, which the in-progress
 * loader still holds), and a request for a DIFFERENT name must never
 * treat this slot as free.
 */

struct membrane_residency_slot_view_t
{
	std::string	name;			/* "" = free */
	bool		model_loaded = false;	/* true only once a load has
									 * actually settled -- never true
									 * while a load/evict for `name`
									 * is still in progress */
	bool		pinned = false;
	bool		generating = false;	/* decode_inflight > 0 */
	int64_t		last_used_seq = 0;	/* higher = more recently used;
									 * only meaningful when model_loaded */
};

enum class membrane_residency_decision_t
{
	HOT_HIT,	/* slot_index already represents `name` (settled or
				 * in-progress) -- never reload/evict */
	USE_EMPTY,	/* slot_index is genuinely free */
	EVICT,		/* slot_index holds a different, evictable model --
				 * caller must close it before loading `name` */
	EXHAUSTED,	/* no slot is available -- every slot is either
				 * claimed by a different in-progress load, pinned, or
				 * actively generating; slot_index is meaningless */
};

struct membrane_residency_plan_t
{
	membrane_residency_decision_t	decision;
	int								slot_index = -1;
};

/*
 * Section 12 of the task: deterministic LRU eviction. Never evicts an
 * actively-generating slot (`generating`) or a pinned one (`pinned`) --
 * if the ENTIRE set of occupied slots is unavailable for one of those
 * two reasons and no slot is free, this returns EXHAUSTED (Section 12:
 * "if all residents are non-evictable, fail safely") rather than
 * violating either rule. No hidden background thrashing: this function
 * is pure and only ever called in direct response to one real incoming
 * request naming `target_name` -- it never runs on its own, on a timer,
 * or speculatively.
 *
 * Tie-breaking (equal last_used_seq -- Section 27/artifact of two slots
 * loaded in the same instant, e.g. server startup) deterministically
 * prefers the LOWEST slot index, so this function's own output is fully
 * reproducible for a given input, never dependent on std::vector's own
 * iteration order in a way a caller could not predict.
 */
inline membrane_residency_plan_t	membrane_plan_residency(
			const std::vector<membrane_residency_slot_view_t> &slots,
			const std::string &target_name)
{
	membrane_residency_plan_t	plan;

	for (size_t i = 0; i < slots.size(); ++i)
	{
		if (!slots[i].name.empty() && slots[i].name == target_name)
		{
			plan.decision = membrane_residency_decision_t::HOT_HIT;
			plan.slot_index = (int)i;
			return (plan);
		}
	}
	for (size_t i = 0; i < slots.size(); ++i)
	{
		if (slots[i].name.empty())
		{
			plan.decision = membrane_residency_decision_t::USE_EMPTY;
			plan.slot_index = (int)i;
			return (plan);
		}
	}

	int	evict_index = -1;

	for (size_t i = 0; i < slots.size(); ++i)
	{
		const membrane_residency_slot_view_t	&s = slots[i];

		if (!s.model_loaded || s.pinned || s.generating)
			continue ;
		if (evict_index == -1
			|| s.last_used_seq < slots[evict_index].last_used_seq)
			evict_index = (int)i;
	}
	if (evict_index == -1)
	{
		plan.decision = membrane_residency_decision_t::EXHAUSTED;
		return (plan);
	}
	plan.decision = membrane_residency_decision_t::EVICT;
	plan.slot_index = evict_index;
	return (plan);
}

#endif
