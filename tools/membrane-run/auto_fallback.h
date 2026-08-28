#ifndef MEMBRANE_RUN_AUTO_FALLBACK_H
# define MEMBRANE_RUN_AUTO_FALLBACK_H

# include <stdint.h>

# include "joint_planner.h"

# ifdef __cplusplus
extern "C" {
# endif

/*
 * Phase 21: llama-free (no ggml/llama header) bounded, deterministic
 * APPLY-TIME fallback controller -- if the primary Phase-20 candidate
 * cannot actually be instantiated at runtime, retry another already-
 * legal, already-ranked Phase-20 candidate. This is complete-plan
 * re-instantiation ("static plan A fails -> clean up -> static plan B
 * -> retry"), never live KV migration, promotion/demotion, or a
 * runtime scheduler -- there is no concept here of moving a live
 * context's state between plans.
 *
 * This module does NOT generate candidates -- it consumes EXACTLY the
 * ranked candidate array membrane_joint_plan_resolve() (joint_planner.h)
 * already produced for this run (Phase 20's own candidates[]/
 * candidate_count/selected_index), iterated in that array's own order
 * (Phase 20's own ranking policy -- see joint_planner.h's top comment),
 * starting from selected_index. It never invents a candidate Phase 20
 * did not already rank as legal and eligible, so every explicit user
 * constraint Phase 20 already enforced (fixed precision, fixed
 * placement, fixed/all gpu-layers) stays enforced here automatically --
 * this module has no path to violate one.
 *
 * Real instantiation (model load, context creation, decode) is
 * injected via apply_fn -- this module never touches llama.h/ggml.h
 * itself, so it is fully unit-testable with a synthetic apply_fn
 * (Section 17/18: a pure, llama-free fallback CONTROLLER, separate
 * from the llama-aware apply ADAPTER main.cpp provides in production).
 */

/* Apply-time failure classification (Section 2). Only classifications
 * the underlying API can actually prove are ever set by this project's
 * real apply adapter (tools/membrane-run/main.cpp) -- GPU_OOM_CONFIRMED/
 * HOST_OOM_CONFIRMED/DEVICE_LOST/BACKEND_ALLOCATION_FAILED are reserved
 * taxonomy slots for a future llama.cpp error-introspection capability
 * this project does not have today (llama.h exposes no error code or
 * reason string from llama_model_load_from_file()/llama_init_from_model()
 * -- both return a bare NULL on any failure) -- see docs/auto-fallback.md
 * for exactly which of these the real adapter can and cannot produce.
 * Never fabricate a more specific class than the evidence supports. */
# define MEMBRANE_APPLY_OK							0
# define MEMBRANE_APPLY_MODEL_LOAD_FAILED			1
# define MEMBRANE_APPLY_CONTEXT_CREATE_FAILED		2
# define MEMBRANE_APPLY_BACKEND_ALLOCATION_FAILED	3
# define MEMBRANE_APPLY_GPU_OOM_CONFIRMED			4
# define MEMBRANE_APPLY_HOST_OOM_CONFIRMED			5
# define MEMBRANE_APPLY_DEVICE_LOST					6
# define MEMBRANE_APPLY_COMPAT_REJECTED				7
# define MEMBRANE_APPLY_UNKNOWN_FAILED				8

/* Stable, machine-readable name for a MEMBRANE_APPLY_* value (Section
 * 33) -- never renamed once shipped. Returns "UNKNOWN_APPLY_FAILURE"
 * for any value outside the known set (defensive, never crashes). */
const char	*membrane_apply_failure_class_name(int failure_class);

/* Section 7: only a failure class where a DIFFERENT already-legal
 * candidate (different gpu_layers/precision/placement) could plausibly
 * avoid the same failure is retryable. COMPAT_REJECTED is never
 * retryable here -- Phase 20's own candidate generation already
 * filtered every candidate through compat_check.h before it could ever
 * reach this module's eligible[] set (Section 26: Qwen2.5's
 * compatibility boundary stays authoritative, this module cannot
 * un-reject it), so a genuine COMPAT_REJECTED at apply time would mean
 * something changed out from under a deterministic check -- fail
 * closed, do not retry. UNKNOWN_APPLY_FAILED is conservatively
 * non-retryable too (Section 2: prefer a less-specific classification
 * over guessing; an unclassifiable failure is not assumed to be
 * memory-related just because a memory-related fallback exists). */
int			membrane_apply_failure_is_retryable(int failure_class);

/* Section 8: derived directly from the current candidate-generation
 * architecture, not an arbitrary number -- membrane_joint_plan_resolve()
 * (joint_planner.c) never produces more than 3 candidates for a single
 * request shape (resolve_adaptive_precision()'s winner + runner-up +
 * same-precision 0-layer fallback is the largest case; resolve_single_
 * precision() produces at most 2). 3 is therefore the natural bound:
 * large enough to exhaust every candidate a real request can ever
 * generate, never larger than necessary. */
# define MEMBRANE_MAX_AUTO_ATTEMPTS	3

/* Stable top-level reason codes (Section 33) -- never rename once
 * shipped, same convention as every other *_policy.h module. */
# define MEMBRANE_FALLBACK_REASON_PRIMARY_SUCCEEDED	"AUTO_PRIMARY_SUCCEEDED"
# define MEMBRANE_FALLBACK_REASON_FALLBACK_SUCCEEDED	"AUTO_FALLBACK_SUCCEEDED"
# define MEMBRANE_FALLBACK_REASON_EXHAUSTED			"AUTO_FALLBACK_EXHAUSTED"
# define MEMBRANE_FALLBACK_REASON_NO_RETRYABLE_FAILURE	"AUTO_NO_RETRYABLE_FAILURE"
# define MEMBRANE_FALLBACK_REASON_CLEANUP_FAILED		"AUTO_CLEANUP_FAILED"

typedef struct s_membrane_fallback_apply_result
{
	int		ok;						/* 1 iff this candidate is now live and
									 * ready to generate/serve */
	int		failure_class;			/* MEMBRANE_APPLY_*, meaningful iff !ok */
	char	detail[256];			/* free-text detail, always NUL-terminated */
	int		cleanup_complete;		/* 1 iff every resource this attempt
									 * allocated is already fully freed --
									 * the controller STOPS (never retries)
									 * when this is 0, even for an
									 * otherwise-retryable failure_class
									 * (Section 9: cleanup safety is a
									 * blocker, not a policy choice) */
	int		model_reload_required;	/* diagnostic only -- echoed into the
									 * trace, never consulted by the
									 * controller itself */
}	membrane_fallback_apply_result_t;

/* Attempts to instantiate exactly ONE candidate (Section 3: this
 * function applies one candidate only, no retry policy inside it).
 * candidate_index is candidate's own index into the original
 * candidates[] array (for trace/logging). Returns out->ok (also the
 * function's own return value). Must leave the system in a known-clean
 * state before returning on failure (out->cleanup_complete=1) or STOP
 * is the only safe outcome the controller can choose. */
typedef int	(*membrane_fallback_apply_fn)(
				const membrane_joint_candidate_t *candidate,
				int candidate_index, void *apply_ctx,
				membrane_fallback_apply_result_t *out);

/* Section 11: re-snapshot available GPU memory before each attempt.
 * Returns 1 and sets *out_free_bytes on success; 0 if no refresh is
 * possible (e.g. no live device query available in this context) --
 * membrane_fallback_run() then treats the candidate's fit as unchanged
 * from planning time, never blocking an attempt just because refresh
 * itself was unavailable. May be NULL (skip refreshing entirely). */
typedef int	(*membrane_fallback_refresh_fn)(void *refresh_ctx,
				uint64_t *out_free_bytes);

typedef struct s_membrane_fallback_attempt
{
	int			candidate_index;
	int32_t		gpu_layers;
	int			kv_precision;
	int			kv_placement;

	int			refreshed;						/* 1 iff refresh_fn ran and
												 * succeeded for this entry */
	uint64_t	refreshed_available_gpu_bytes;
	int			fit_after_refresh;				/* 1 = still fits (or n/a --
												 * no refresh, or a CPU-only
												 * candidate with nothing to
												 * refit), 0 = SKIPPED_
												 * UPDATED_MEMORY */

	int			apply_started;					/* 1 iff apply_fn was
												 * actually called for this
												 * entry (0 for a memory-
												 * refit skip) */
	int			apply_ok;
	int			failure_class;					/* MEMBRANE_APPLY_*,
												 * meaningful iff
												 * apply_started && !apply_ok */
	char		detail[256];
	int			cleanup_complete;
	int			model_reload_required;
	double		apply_wall_ms;					/* Phase 24: wall-clock
												 * duration of THIS one
												 * apply_fn() call (whatever
												 * it did internally -- model
												 * reload, compat/placement
												 * checks, context creation,
												 * decode -- this module has
												 * no visibility into that
												 * breakdown, only the total;
												 * see the real adapter's own
												 * finer-grained timings on
												 * membrane_kv_store_
												 * telemetry_t for a
												 * SUCCESSFUL attempt's
												 * breakdown). 0.0 for a
												 * skipped entry
												 * (apply_started == 0). */
}	membrane_fallback_attempt_t;

typedef struct s_membrane_fallback_trace
{
	int			attempted;					/* n_entries > 1 -- the
											 * controller moved past
											 * evaluating just the
											 * originally-selected
											 * candidate, whether because
											 * it was skipped (Section 11
											 * memory refit) or because it
											 * failed to apply. A skip is
											 * exactly as real a fallback
											 * event as an apply failure
											 * is -- the eventual result
											 * is not the plan that was
											 * originally selected either
											 * way. */
	int			attempt_count;				/* number of entries with
											 * apply_started == 1 */
	int			initial_candidate_index;	/* == the request's own
											 * selected_index, echoed */
	int			final_candidate_index;		/* the candidate that
											 * ultimately succeeded, -1 if
											 * none did */
	char		final_status[24];			/* "success" | "exhausted" |
											 * "cleanup_blocked" */
	char		reason_code[40];			/* MEMBRANE_FALLBACK_REASON_* */

	int			n_entries;
	membrane_fallback_attempt_t	entries[MEMBRANE_JOINT_MAX_CANDIDATES];
}	membrane_fallback_trace_t;

/*
 * Bounded, deterministic fallback controller (Section 3/4/8/9/11/18).
 *
 * candidates/candidate_count/selected_index: EXACTLY Phase 20's own
 * membrane_joint_plan_result_t fields, unmodified -- this is the ONLY
 * candidate source (Section 4); this function never generates, reorders
 * (beyond starting from selected_index), or invents a candidate.
 *
 * device_total_bytes/reserve_bytes: for recomputing a GPU candidate's
 * fit after a memory refresh (Section 11) -- reserve_bytes is assumed
 * stable across attempts (gpu_policy.h's own reserve is a function of
 * device_total_bytes alone, which does not change mid-run); only the
 * free-bytes figure is re-queried, via refresh_fn.
 *
 * apply_fn/apply_ctx: required -- instantiates exactly one candidate
 * per call (Section 3).
 * refresh_fn/refresh_ctx: optional (NULL skips the memory-refresh step
 * entirely -- every eligible candidate is attempted in order with no
 * re-fit check, e.g. for a caller with no live device-query capability).
 *
 * Iteration order (Section 4): [selected_index] then every other
 * ELIGIBLE candidate index in the array's own ascending order -- never
 * an ineligible candidate (Phase 20 already rejected those), never the
 * same index twice (Section 8, test 6), bounded at
 * MEMBRANE_MAX_AUTO_ATTEMPTS real apply_fn calls (Section 8) --
 * memory-refit skips do not themselves consume this bound (Section 11:
 * a skip is not a "try").
 *
 * Stops immediately (no further candidates attempted) on: success: the
 * winning candidate is left instantiated and live; a NON-retryable
 * failure_class (membrane_apply_failure_is_retryable()); or
 * cleanup_complete == 0 (Section 9 -- a cleanup-safety blocker always
 * overrides an otherwise-retryable failure_class).
 *
 * Deterministic: identical inputs and apply_fn/refresh_fn behavior
 * always produce identical output, no randomness, no I/O of its own.
 * Returns 1 iff a candidate succeeded (out->final_status == "success"),
 * 0 for "exhausted" or "cleanup_blocked".
 */
int	membrane_fallback_run(const membrane_joint_candidate_t *candidates,
		int candidate_count, int selected_index,
		uint64_t device_total_bytes, uint64_t reserve_bytes,
		membrane_fallback_apply_fn apply_fn, void *apply_ctx,
		membrane_fallback_refresh_fn refresh_fn, void *refresh_ctx,
		membrane_fallback_trace_t *out);

# ifdef __cplusplus
}
# endif

#endif
