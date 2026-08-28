#include <stdint.h>
#include <string.h>

#include "auto_fallback.h"
#include "joint_planner.h"
#include "test_helpers.h"

#define GIB	((uint64_t)1024 * 1024 * 1024)
#define MIB	((uint64_t)1024 * 1024)

/* ------------------------------------------------------------------ */
/* Test doubles                                                        */
/* ------------------------------------------------------------------ */

typedef struct s_script_entry
{
	int		should_succeed;
	int		failure_class;
	int		cleanup_complete;
}	script_entry_t;

/* Scripted apply_fn: entries[] indexed by CALL ORDER (not candidate
 * index) -- the Nth call to scripted_apply() consumes entries[N].
 * Records everything a test needs to assert on afterward (call order,
 * how many times it was invoked, what each call actually saw). */
typedef struct s_apply_script
{
	script_entry_t	entries[8];
	int				n_calls;
	int				call_candidate_index[8];
	int32_t			call_gpu_layers[8];
	int				call_kv_precision[8];
	int				call_kv_placement[8];
}	apply_script_t;

static int	scripted_apply(const membrane_joint_candidate_t *c,
				int candidate_index, void *apply_ctx,
				membrane_fallback_apply_result_t *out)
{
	apply_script_t	*s = (apply_script_t *)apply_ctx;
	script_entry_t	*e = &s->entries[s->n_calls];

	s->call_candidate_index[s->n_calls] = candidate_index;
	s->call_gpu_layers[s->n_calls] = c->gpu_layers;
	s->call_kv_precision[s->n_calls] = c->kv_precision;
	s->call_kv_placement[s->n_calls] = c->kv_placement;
	s->n_calls++;
	memset(out, 0, sizeof(*out));
	out->ok = e->should_succeed;
	out->failure_class = e->failure_class;
	out->cleanup_complete = e->cleanup_complete;
	snprintf(out->detail, sizeof(out->detail), "scripted");
	return (out->ok);
}

/* Scripted refresh_fn: values[] indexed by call order, one call per
 * trace entry (skip or attempt alike) -- Section 11: "before EACH new
 * attempt". */
typedef struct s_refresh_script
{
	uint64_t	values[8];
	int			n_calls;
}	refresh_script_t;

static int	scripted_refresh(void *refresh_ctx, uint64_t *out_free_bytes)
{
	refresh_script_t	*s = (refresh_script_t *)refresh_ctx;

	*out_free_bytes = s->values[s->n_calls];
	s->n_calls++;
	return (1);
}

static void	make_candidate(membrane_joint_candidate_t *c, int32_t gpu_layers,
				int precision, int placement, uint64_t weight_bytes,
				uint64_t kv_bytes, int eligible)
{
	memset(c, 0, sizeof(*c));
	c->gpu_layers = gpu_layers;
	c->kv_precision = precision;
	c->kv_placement = placement;
	c->estimated_weight_gpu_bytes = weight_bytes;
	c->estimated_kv_gpu_bytes = kv_bytes;
	c->compatible = 1;
	c->fits_gpu = eligible;
	c->fits_host = 1;
	c->eligible = eligible;
	snprintf(c->reason_code, sizeof(c->reason_code), "%s",
		eligible ? "CANDIDATE_SELECTED" : "GPU_MEMORY_INSUFFICIENT");
}

/* ------------------------------------------------------------------ */
/* 1. First candidate succeeds -> one attempt, fallback.attempted false */
/* ------------------------------------------------------------------ */

static void	test_first_succeeds_one_attempt(void)
{
	membrane_joint_candidate_t	cands[2];
	apply_script_t				s;
	membrane_fallback_trace_t	trace;

	make_candidate(&cands[0], 10, MEMBRANE_JOINT_KV_Q8,
		MEMBRANE_JOINT_PLACEMENT_DEFAULT, 100 * MIB, 20 * MIB, 1);
	make_candidate(&cands[1], 0, MEMBRANE_JOINT_KV_Q8,
		MEMBRANE_JOINT_PLACEMENT_DEFAULT, 0, 0, 1);
	memset(&s, 0, sizeof(s));
	s.entries[0].should_succeed = 1;
	s.entries[0].cleanup_complete = 1;
	TEST_ASSERT(membrane_fallback_run(cands, 2, 0, 4 * GIB, 512 * MIB,
			scripted_apply, &s, NULL, NULL, &trace) == 1, "primary succeeds");
	TEST_ASSERT(trace.attempt_count == 1, "exactly one attempt");
	TEST_ASSERT(trace.attempted == 0,
		"attempted is false when only the primary candidate ran");
	TEST_ASSERT(strcmp(trace.reason_code,
			MEMBRANE_FALLBACK_REASON_PRIMARY_SUCCEEDED) == 0,
		"primary-succeeded reason code");
	TEST_ASSERT(strcmp(trace.final_status, "success") == 0,
		"final_status is success");
	TEST_ASSERT(trace.final_candidate_index == 0, "candidate 0 won");
	TEST_ASSERT(s.n_calls == 1, "apply_fn called exactly once");
}

/* ------------------------------------------------------------------ */
/* 2. First retryable-fails, second succeeds -> two attempts, fallback  */
/* ------------------------------------------------------------------ */

static void	test_first_retryable_fails_second_succeeds(void)
{
	membrane_joint_candidate_t	cands[2];
	apply_script_t				s;
	membrane_fallback_trace_t	trace;

	make_candidate(&cands[0], 10, MEMBRANE_JOINT_KV_Q8,
		MEMBRANE_JOINT_PLACEMENT_DEFAULT, 100 * MIB, 20 * MIB, 1);
	make_candidate(&cands[1], 5, MEMBRANE_JOINT_KV_Q8,
		MEMBRANE_JOINT_PLACEMENT_DEFAULT, 50 * MIB, 20 * MIB, 1);
	memset(&s, 0, sizeof(s));
	s.entries[0].should_succeed = 0;
	s.entries[0].failure_class = MEMBRANE_APPLY_CONTEXT_CREATE_FAILED;
	s.entries[0].cleanup_complete = 1;
	s.entries[1].should_succeed = 1;
	s.entries[1].cleanup_complete = 1;
	TEST_ASSERT(membrane_fallback_run(cands, 2, 0, 4 * GIB, 512 * MIB,
			scripted_apply, &s, NULL, NULL, &trace) == 1,
		"fallback recovers on the second candidate");
	TEST_ASSERT(trace.attempt_count == 2, "two attempts");
	TEST_ASSERT(trace.attempted == 1, "attempted is true");
	TEST_ASSERT(strcmp(trace.reason_code,
			MEMBRANE_FALLBACK_REASON_FALLBACK_SUCCEEDED) == 0,
		"fallback-succeeded reason code");
	TEST_ASSERT(trace.final_candidate_index == 1, "candidate 1 won");
	TEST_ASSERT(trace.entries[0].cleanup_complete == 1,
		"failed attempt's own cleanup completed before the next attempt");
	TEST_ASSERT(trace.entries[0].failure_class
			== MEMBRANE_APPLY_CONTEXT_CREATE_FAILED,
		"first attempt's failure class recorded");
	TEST_ASSERT(s.n_calls == 2, "apply_fn called exactly twice");
}

/* ------------------------------------------------------------------ */
/* 3. First non-retryable-fails -> no second attempt                   */
/* ------------------------------------------------------------------ */

static void	test_first_non_retryable_fails_stops(void)
{
	membrane_joint_candidate_t	cands[2];
	apply_script_t				s;
	membrane_fallback_trace_t	trace;

	make_candidate(&cands[0], 10, MEMBRANE_JOINT_KV_Q8,
		MEMBRANE_JOINT_PLACEMENT_DEFAULT, 100 * MIB, 20 * MIB, 1);
	make_candidate(&cands[1], 5, MEMBRANE_JOINT_KV_Q8,
		MEMBRANE_JOINT_PLACEMENT_DEFAULT, 50 * MIB, 20 * MIB, 1);
	memset(&s, 0, sizeof(s));
	s.entries[0].should_succeed = 0;
	s.entries[0].failure_class = MEMBRANE_APPLY_COMPAT_REJECTED;
	s.entries[0].cleanup_complete = 1;
	TEST_ASSERT(membrane_fallback_run(cands, 2, 0, 4 * GIB, 512 * MIB,
			scripted_apply, &s, NULL, NULL, &trace) == 0,
		"non-retryable failure never recovers");
	TEST_ASSERT(trace.attempt_count == 1,
		"only the primary candidate was attempted");
	TEST_ASSERT(strcmp(trace.reason_code,
			MEMBRANE_FALLBACK_REASON_NO_RETRYABLE_FAILURE) == 0,
		"no-retryable-failure reason code");
	TEST_ASSERT(s.n_calls == 1, "apply_fn never called a second time");
}

/* ------------------------------------------------------------------ */
/* 4. All candidates retryable-fail -> FALLBACK_EXHAUSTED               */
/* ------------------------------------------------------------------ */

static void	test_all_retryable_fail_exhausted(void)
{
	membrane_joint_candidate_t	cands[3];
	apply_script_t				s;
	membrane_fallback_trace_t	trace;
	int							i;

	make_candidate(&cands[0], 10, MEMBRANE_JOINT_KV_Q8,
		MEMBRANE_JOINT_PLACEMENT_DEFAULT, 100 * MIB, 20 * MIB, 1);
	make_candidate(&cands[1], 5, MEMBRANE_JOINT_KV_Q8,
		MEMBRANE_JOINT_PLACEMENT_DEFAULT, 50 * MIB, 20 * MIB, 1);
	make_candidate(&cands[2], 0, MEMBRANE_JOINT_KV_Q8,
		MEMBRANE_JOINT_PLACEMENT_DEFAULT, 0, 0, 1);
	memset(&s, 0, sizeof(s));
	i = 0;
	while (i < 3)
	{
		s.entries[i].should_succeed = 0;
		s.entries[i].failure_class = MEMBRANE_APPLY_GPU_OOM_CONFIRMED;
		s.entries[i].cleanup_complete = 1;
		i++;
	}
	TEST_ASSERT(membrane_fallback_run(cands, 3, 0, 4 * GIB, 512 * MIB,
			scripted_apply, &s, NULL, NULL, &trace) == 0,
		"every candidate failing means overall failure");
	TEST_ASSERT(trace.attempt_count == 3, "all three candidates attempted");
	TEST_ASSERT(strcmp(trace.reason_code,
			MEMBRANE_FALLBACK_REASON_EXHAUSTED) == 0, "exhausted reason code");
	TEST_ASSERT(strcmp(trace.final_status, "exhausted") == 0,
		"final_status is exhausted");
	TEST_ASSERT(trace.final_candidate_index == -1, "no winner");
}

/* ------------------------------------------------------------------ */
/* 5 / 22. Candidate no longer fits after a memory refresh -> skipped, */
/* next candidate evaluated, no random re-ranking.                     */
/* ------------------------------------------------------------------ */

static void	test_memory_shrink_skips_then_next_applies(void)
{
	membrane_joint_candidate_t	cands[2];
	apply_script_t				s;
	refresh_script_t			r;
	membrane_fallback_trace_t	trace;

	/* Candidate 0 needs 3.5 GiB of GPU budget; candidate 1 (CPU-only
	 * weights) needs none. Planner originally saw plenty of free VRAM;
	 * by apply time it has shrunk below what candidate 0 needs. */
	make_candidate(&cands[0], 10, MEMBRANE_JOINT_KV_Q8,
		MEMBRANE_JOINT_PLACEMENT_DEFAULT, (uint64_t)3 * GIB,
		(uint64_t)512 * MIB, 1);
	make_candidate(&cands[1], 0, MEMBRANE_JOINT_KV_Q8,
		MEMBRANE_JOINT_PLACEMENT_DEFAULT, 0, 0, 1);
	memset(&r, 0, sizeof(r));
	r.values[0] = 1 * GIB;	/* far below reserve + 3.5 GiB need */
	r.values[1] = 1 * GIB;	/* irrelevant: candidate 1 is CPU-only */
	memset(&s, 0, sizeof(s));
	s.entries[0].should_succeed = 1;
	s.entries[0].cleanup_complete = 1;
	TEST_ASSERT(membrane_fallback_run(cands, 2, 0, 8 * GIB, 512 * MIB,
			scripted_apply, &s, scripted_refresh, &r, &trace) == 1,
		"the CPU-only fallback still succeeds");
	TEST_ASSERT(trace.n_entries == 2, "both candidates appear in the trace");
	TEST_ASSERT(trace.entries[0].candidate_index == 0,
		"candidate 0 evaluated first (original rank order preserved)");
	TEST_ASSERT(trace.entries[0].fit_after_refresh == 0,
		"candidate 0 no longer fits after the refresh");
	TEST_ASSERT(trace.entries[0].apply_started == 0,
		"a skipped candidate is never actually applied");
	TEST_ASSERT(trace.entries[1].candidate_index == 1,
		"candidate 1 evaluated next -- no random re-ranking");
	TEST_ASSERT(trace.entries[1].apply_started == 1,
		"candidate 1 was actually attempted");
	TEST_ASSERT(trace.attempt_count == 1,
		"the skip does not itself count as an attempt");
	TEST_ASSERT(s.n_calls == 1, "apply_fn called only for candidate 1");
}

/* ------------------------------------------------------------------ */
/* 6. Same candidate never attempted twice                             */
/* ------------------------------------------------------------------ */

static void	test_same_candidate_never_attempted_twice(void)
{
	membrane_joint_candidate_t	cands[3];
	apply_script_t				s;
	membrane_fallback_trace_t	trace;
	int							i;
	int							j;

	make_candidate(&cands[0], 10, MEMBRANE_JOINT_KV_Q8,
		MEMBRANE_JOINT_PLACEMENT_DEFAULT, 100 * MIB, 20 * MIB, 1);
	make_candidate(&cands[1], 5, MEMBRANE_JOINT_KV_Q5,
		MEMBRANE_JOINT_PLACEMENT_DEFAULT, 50 * MIB, 14 * MIB, 1);
	make_candidate(&cands[2], 0, MEMBRANE_JOINT_KV_Q8,
		MEMBRANE_JOINT_PLACEMENT_DEFAULT, 0, 0, 1);
	memset(&s, 0, sizeof(s));
	i = 0;
	while (i < 3)
	{
		s.entries[i].should_succeed = 0;
		s.entries[i].failure_class = MEMBRANE_APPLY_CONTEXT_CREATE_FAILED;
		s.entries[i].cleanup_complete = 1;
		i++;
	}
	/* selected_index = 1 (the middle candidate) -- order must be
	 * [1, 0, 2] (selected first, then ascending), never repeating. */
	membrane_fallback_run(cands, 3, 1, 4 * GIB, 512 * MIB, scripted_apply,
		&s, NULL, NULL, &trace);
	TEST_ASSERT(trace.attempt_count == 3, "all three distinct candidates tried");
	TEST_ASSERT(s.call_candidate_index[0] == 1, "selected_index attempted first");
	TEST_ASSERT(s.call_candidate_index[1] == 0, "then ascending index 0");
	TEST_ASSERT(s.call_candidate_index[2] == 2, "then ascending index 2");
	i = 0;
	while (i < 3)
	{
		j = i + 1;
		while (j < 3)
		{
			TEST_ASSERT(s.call_candidate_index[i] != s.call_candidate_index[j],
				"no candidate index is ever repeated");
			j++;
		}
		i++;
	}
}

/* ------------------------------------------------------------------ */
/* 7. Hard max-attempt limit respected                                 */
/* ------------------------------------------------------------------ */

static void	test_hard_max_attempt_limit(void)
{
	membrane_joint_candidate_t	cands[MEMBRANE_JOINT_MAX_CANDIDATES];
	apply_script_t				s;
	membrane_fallback_trace_t	trace;
	int							i;

	i = 0;
	while (i < MEMBRANE_JOINT_MAX_CANDIDATES)
	{
		make_candidate(&cands[i], (int32_t)(10 - i), MEMBRANE_JOINT_KV_Q8,
			MEMBRANE_JOINT_PLACEMENT_DEFAULT, 10 * MIB, 5 * MIB, 1);
		i++;
	}
	memset(&s, 0, sizeof(s));
	i = 0;
	while (i < MEMBRANE_JOINT_MAX_CANDIDATES)
	{
		s.entries[i].should_succeed = 0;
		s.entries[i].failure_class = MEMBRANE_APPLY_CONTEXT_CREATE_FAILED;
		s.entries[i].cleanup_complete = 1;
		i++;
	}
	TEST_ASSERT(membrane_fallback_run(cands, MEMBRANE_JOINT_MAX_CANDIDATES, 0,
			4 * GIB, 512 * MIB, scripted_apply, &s, NULL, NULL, &trace) == 0,
		"still exhausted at the cap");
	TEST_ASSERT(trace.attempt_count == MEMBRANE_MAX_AUTO_ATTEMPTS,
		"never exceeds MEMBRANE_MAX_AUTO_ATTEMPTS real apply attempts");
	TEST_ASSERT(s.n_calls == MEMBRANE_MAX_AUTO_ATTEMPTS,
		"apply_fn itself was never called beyond the bound");
}

/* ------------------------------------------------------------------ */
/* 8. Deterministic order preserved across repeated calls              */
/* ------------------------------------------------------------------ */

static void	test_deterministic_repeated_calls(void)
{
	membrane_joint_candidate_t	cands[3];
	apply_script_t				s1;
	apply_script_t				s2;
	membrane_fallback_trace_t	t1;
	membrane_fallback_trace_t	t2;

	make_candidate(&cands[0], 10, MEMBRANE_JOINT_KV_Q8,
		MEMBRANE_JOINT_PLACEMENT_DEFAULT, 100 * MIB, 20 * MIB, 1);
	make_candidate(&cands[1], 5, MEMBRANE_JOINT_KV_Q8,
		MEMBRANE_JOINT_PLACEMENT_DEFAULT, 50 * MIB, 20 * MIB, 1);
	make_candidate(&cands[2], 0, MEMBRANE_JOINT_KV_Q8,
		MEMBRANE_JOINT_PLACEMENT_DEFAULT, 0, 0, 1);
	memset(&s1, 0, sizeof(s1));
	memset(&s2, 0, sizeof(s2));
	s1.entries[0].should_succeed = 0;
	s1.entries[0].failure_class = MEMBRANE_APPLY_CONTEXT_CREATE_FAILED;
	s1.entries[0].cleanup_complete = 1;
	s1.entries[1].should_succeed = 1;
	s1.entries[1].cleanup_complete = 1;
	s2 = s1;
	membrane_fallback_run(cands, 3, 0, 4 * GIB, 512 * MIB, scripted_apply,
		&s1, NULL, NULL, &t1);
	membrane_fallback_run(cands, 3, 0, 4 * GIB, 512 * MIB, scripted_apply,
		&s2, NULL, NULL, &t2);
	TEST_ASSERT(t1.attempt_count == t2.attempt_count,
		"identical attempt_count across repeated calls");
	TEST_ASSERT(t1.final_candidate_index == t2.final_candidate_index,
		"identical final candidate across repeated calls");
	TEST_ASSERT(strcmp(t1.final_status, t2.final_status) == 0,
		"identical final_status across repeated calls");
	TEST_ASSERT(s1.call_candidate_index[0] == s2.call_candidate_index[0]
			&& s1.call_candidate_index[1] == s2.call_candidate_index[1],
		"identical call order across repeated calls");
}

/* ------------------------------------------------------------------ */
/* 9/10. Cleanup: complete cleanup allows the next attempt; incomplete  */
/* cleanup is a hard STOP, never an unsafe retry.                      */
/* ------------------------------------------------------------------ */

static void	test_cleanup_failure_stops_immediately(void)
{
	membrane_joint_candidate_t	cands[2];
	apply_script_t				s;
	membrane_fallback_trace_t	trace;

	make_candidate(&cands[0], 10, MEMBRANE_JOINT_KV_Q8,
		MEMBRANE_JOINT_PLACEMENT_DEFAULT, 100 * MIB, 20 * MIB, 1);
	make_candidate(&cands[1], 5, MEMBRANE_JOINT_KV_Q8,
		MEMBRANE_JOINT_PLACEMENT_DEFAULT, 50 * MIB, 20 * MIB, 1);
	memset(&s, 0, sizeof(s));
	s.entries[0].should_succeed = 0;
	s.entries[0].failure_class = MEMBRANE_APPLY_CONTEXT_CREATE_FAILED;
	s.entries[0].cleanup_complete = 0;	/* the blocker */
	TEST_ASSERT(membrane_fallback_run(cands, 2, 0, 4 * GIB, 512 * MIB,
			scripted_apply, &s, NULL, NULL, &trace) == 0,
		"a cleanup failure is never treated as success");
	TEST_ASSERT(strcmp(trace.final_status, "cleanup_blocked") == 0,
		"final_status reports the cleanup blocker distinctly");
	TEST_ASSERT(strcmp(trace.reason_code,
			MEMBRANE_FALLBACK_REASON_CLEANUP_FAILED) == 0,
		"cleanup-failed reason code");
	TEST_ASSERT(s.n_calls == 1,
		"the second (otherwise retryable) candidate is never attempted "
		"unsafely after a cleanup failure");
}

/* ------------------------------------------------------------------ */
/* 16. No viable fallback candidate -> clear exhausted/no-alternative  */
/* ------------------------------------------------------------------ */

static void	test_no_alternative_candidate_exhausted(void)
{
	membrane_joint_candidate_t	cands[1];
	apply_script_t				s;
	membrane_fallback_trace_t	trace;

	make_candidate(&cands[0], 10, MEMBRANE_JOINT_KV_Q8,
		MEMBRANE_JOINT_PLACEMENT_DEFAULT, 100 * MIB, 20 * MIB, 1);
	memset(&s, 0, sizeof(s));
	s.entries[0].should_succeed = 0;
	s.entries[0].failure_class = MEMBRANE_APPLY_CONTEXT_CREATE_FAILED;
	s.entries[0].cleanup_complete = 1;
	TEST_ASSERT(membrane_fallback_run(cands, 1, 0, 4 * GIB, 512 * MIB,
			scripted_apply, &s, NULL, NULL, &trace) == 0,
		"a single legal candidate that fails has nothing to fall back to");
	TEST_ASSERT(trace.attempt_count == 1, "only the one candidate existed");
	/* The failure itself is a retryable class -- there is simply no
	 * OTHER legal candidate left to retry it with, so this is EXHAUSTED
	 * (every available candidate was used), distinct from
	 * NO_RETRYABLE_FAILURE (which reports a failure class that would
	 * never have been retried even if alternatives existed -- see
	 * test_first_non_retryable_fails_stops above). */
	TEST_ASSERT(strcmp(trace.reason_code,
			MEMBRANE_FALLBACK_REASON_EXHAUSTED) == 0,
		"a lone candidate's retryable failure is still a clear, honest "
		"exhausted result -- never fabricated as a false success");
	TEST_ASSERT(strcmp(trace.final_status, "exhausted") == 0,
		"final_status is exhausted");
}

/* ------------------------------------------------------------------ */
/* Ineligible candidates in the array are never attempted              */
/* ------------------------------------------------------------------ */

static void	test_ineligible_candidates_never_attempted(void)
{
	membrane_joint_candidate_t	cands[3];
	apply_script_t				s;
	membrane_fallback_trace_t	trace;

	make_candidate(&cands[0], 10, MEMBRANE_JOINT_KV_Q8,
		MEMBRANE_JOINT_PLACEMENT_DEFAULT, 100 * MIB, 20 * MIB, 1);
	make_candidate(&cands[1], 5, MEMBRANE_JOINT_KV_Q8,
		MEMBRANE_JOINT_PLACEMENT_DEFAULT, 50 * MIB, 20 * MIB, 0);
	make_candidate(&cands[2], 0, MEMBRANE_JOINT_KV_Q8,
		MEMBRANE_JOINT_PLACEMENT_DEFAULT, 0, 0, 1);
	memset(&s, 0, sizeof(s));
	s.entries[0].should_succeed = 0;
	s.entries[0].failure_class = MEMBRANE_APPLY_CONTEXT_CREATE_FAILED;
	s.entries[0].cleanup_complete = 1;
	s.entries[1].should_succeed = 1;
	s.entries[1].cleanup_complete = 1;
	membrane_fallback_run(cands, 3, 0, 4 * GIB, 512 * MIB, scripted_apply,
		&s, NULL, NULL, &trace);
	TEST_ASSERT(s.n_calls == 2, "exactly two eligible candidates attempted");
	TEST_ASSERT(s.call_candidate_index[1] == 2,
		"ineligible candidate 1 was skipped entirely -- index 2 attempted "
		"next, never index 1");
}

/* ------------------------------------------------------------------ */
/* Failure classification (Section 21)                                 */
/* ------------------------------------------------------------------ */

static void	test_failure_classification_retryable(void)
{
	TEST_ASSERT(membrane_apply_failure_is_retryable(
			MEMBRANE_APPLY_MODEL_LOAD_FAILED), "model load is retryable");
	TEST_ASSERT(membrane_apply_failure_is_retryable(
			MEMBRANE_APPLY_CONTEXT_CREATE_FAILED),
		"context create is retryable");
	TEST_ASSERT(membrane_apply_failure_is_retryable(
			MEMBRANE_APPLY_BACKEND_ALLOCATION_FAILED),
		"backend allocation is retryable");
	TEST_ASSERT(membrane_apply_failure_is_retryable(
			MEMBRANE_APPLY_GPU_OOM_CONFIRMED), "GPU OOM is retryable");
	TEST_ASSERT(membrane_apply_failure_is_retryable(
			MEMBRANE_APPLY_HOST_OOM_CONFIRMED), "host OOM is retryable");
	TEST_ASSERT(membrane_apply_failure_is_retryable(
			MEMBRANE_APPLY_DEVICE_LOST), "device lost is retryable");
}

static void	test_failure_classification_non_retryable(void)
{
	TEST_ASSERT(!membrane_apply_failure_is_retryable(
			MEMBRANE_APPLY_COMPAT_REJECTED),
		"compat rejection is never retryable -- Phase 20 already filtered "
		"this before candidate generation");
	TEST_ASSERT(!membrane_apply_failure_is_retryable(
			MEMBRANE_APPLY_UNKNOWN_FAILED),
		"an unclassifiable failure is never assumed retryable");
}

static void	test_failure_class_names_are_stable_and_safe(void)
{
	TEST_ASSERT(strcmp(membrane_apply_failure_class_name(
			MEMBRANE_APPLY_GPU_OOM_CONFIRMED), "GPU_OOM_CONFIRMED") == 0,
		"stable name for a known class");
	TEST_ASSERT(strcmp(membrane_apply_failure_class_name(9999),
			"UNKNOWN_APPLY_FAILURE") == 0,
		"an out-of-range value never crashes, falls back to a safe name");
}

/* ------------------------------------------------------------------ */
/* Section 20: explicit-constraint tests against REAL planner output   */
/* ------------------------------------------------------------------ */

static void	fill_base_request(membrane_joint_plan_request_t *req)
{
	memset(req, 0, sizeof(*req));
	req->n_layer_all = 10;
	req->bytes_per_layer = 10 * MIB;
	req->output_role_bytes = 20 * MIB;
	req->arch_name = "llama";
	req->n_embd = 512;
	req->n_head = 8;
	req->n_head_kv = 8;
	req->ctx_size = 2048;
	req->kv_bytes_native = 40 * MIB;
	req->kv_bytes_q8 = 20 * MIB;
	req->kv_bytes_q5 = 14 * MIB;
	req->precision_request = MEMBRANE_JOINT_PRECISION_REQUEST_AUTO;
	req->gpu_layers_request = MEMBRANE_JOINT_GPU_LAYERS_REQUEST_AUTO;
	req->kv_placement_mode = MEMBRANE_JOINT_PLACEMENT_DEFAULT;
}

/* 11/12. Explicit --kv native or q8: EVERY candidate the real planner
 * produced (and therefore every candidate the fallback controller could
 * ever attempt) shares the identical explicit precision -- the
 * controller has structurally no path to change it, but this proves it
 * against real planner output, not a hand-built array (Section 20). */
static void	test_explicit_precision_never_changes_across_real_candidates(void)
{
	membrane_joint_plan_request_t	req;
	membrane_joint_plan_result_t	res;
	int								i;

	fill_base_request(&req);
	req.precision_request = MEMBRANE_JOINT_KV_NATIVE;
	req.device_free_bytes = (uint64_t)(3.9 * (double)GIB);
	req.device_total_bytes = 4 * GIB;
	TEST_ASSERT(membrane_joint_plan_resolve(&req, &res) == 1,
		"real planner resolves an explicit-native request");
	i = 0;
	while (i < res.candidate_count)
	{
		TEST_ASSERT(res.candidates[i].kv_precision == MEMBRANE_JOINT_KV_NATIVE,
			"every real candidate keeps the explicit native precision");
		i++;
	}

	fill_base_request(&req);
	req.precision_request = MEMBRANE_JOINT_KV_Q8;
	req.device_free_bytes = (uint64_t)(3.9 * (double)GIB);
	req.device_total_bytes = 4 * GIB;
	TEST_ASSERT(membrane_joint_plan_resolve(&req, &res) == 1,
		"real planner resolves an explicit-q8 request");
	i = 0;
	while (i < res.candidate_count)
	{
		TEST_ASSERT(res.candidates[i].kv_precision == MEMBRANE_JOINT_KV_Q8,
			"every real candidate keeps the explicit q8 precision");
		i++;
	}
}

/* 14. Explicit --gpu-layers N: every real candidate keeps that exact N,
 * never a smaller/larger count -- fed into the fallback controller to
 * prove it never sees (and so never could pick) a different one. */
static void	test_explicit_gpu_layers_never_changes_across_real_candidates(void)
{
	membrane_joint_plan_request_t	req;
	membrane_joint_plan_result_t	res;
	apply_script_t					s;
	membrane_fallback_trace_t		trace;
	int								i;

	fill_base_request(&req);
	req.gpu_layers_request = 6;
	req.device_free_bytes = (uint64_t)(3.9 * (double)GIB);
	req.device_total_bytes = 4 * GIB;
	TEST_ASSERT(membrane_joint_plan_resolve(&req, &res) == 1,
		"real planner resolves an explicit N=6 request");
	i = 0;
	while (i < res.candidate_count)
	{
		TEST_ASSERT(res.candidates[i].gpu_layers == 6,
			"every real candidate keeps the explicit N=6 layer count");
		i++;
	}
	memset(&s, 0, sizeof(s));
	s.entries[0].should_succeed = 0;
	s.entries[0].failure_class = MEMBRANE_APPLY_CONTEXT_CREATE_FAILED;
	s.entries[0].cleanup_complete = 1;
	membrane_fallback_run(res.candidates, res.candidate_count,
		res.selected_index, req.device_total_bytes, 0, scripted_apply, &s,
		NULL, NULL, &trace);
	TEST_ASSERT(s.call_gpu_layers[0] == 6,
		"the fallback controller only ever attempted N=6, never a "
		"different layer count");
}

/* 13. Explicit --kv-placement gpu: every real candidate's resolved
 * placement outcome is either GPU or (for a genuinely CPU-only weight
 * candidate) never silently downgraded by this module -- the fallback
 * controller must never retry with a different kv_placement than what
 * Phase 20 itself already resolved for each candidate. */
static void	test_explicit_gpu_placement_never_silently_becomes_cpu(void)
{
	membrane_joint_plan_request_t	req;
	membrane_joint_plan_result_t	res;
	int								i;

	fill_base_request(&req);
	req.precision_request = MEMBRANE_JOINT_KV_Q8;
	req.gpu_layers_request = 10;
	req.kv_placement_mode = MEMBRANE_JOINT_PLACEMENT_GPU;
	req.device_free_bytes = (uint64_t)(3.9 * (double)GIB);
	req.device_total_bytes = 4 * GIB;
	TEST_ASSERT(membrane_joint_plan_resolve(&req, &res) == 1,
		"real planner resolves an explicit gpu-placement request");
	i = 0;
	while (i < res.candidate_count)
	{
		/* Every eligible candidate's placement outcome must be GPU here
		 * -- kv_residency_resolve() fails the whole candidate (never
		 * silently degrades to CPU) when GPU placement was explicitly
		 * requested and does not fit -- see kv_residency_policy.h. */
		if (res.candidates[i].eligible)
			TEST_ASSERT(res.candidates[i].kv_placement
					== MEMBRANE_JOINT_PLACEMENT_GPU,
				"an eligible candidate under explicit gpu placement is "
				"never silently CPU-resident");
		i++;
	}
}

/* 15. All-auto: the next Phase-20-ranked candidate (a genuinely
 * different, real one) is used on fallback -- not a fabricated one. */
static void	test_all_auto_uses_next_real_ranked_candidate(void)
{
	membrane_joint_plan_request_t	req;
	membrane_joint_plan_result_t	res;
	apply_script_t					s;
	membrane_fallback_trace_t		trace;

	/* Constrained budget so adaptive resolves to full-residency Q5
	 * (candidate[0]) with a real Q8 runner-up (candidate[1]) present in
	 * the same array -- see test_joint_planner.c's own
	 * test_constrained_q5_required_for_full_residency for the same
	 * derivation this reuses. */
	fill_base_request(&req);
	req.device_free_bytes = (uint64_t)(1.85 * (double)GIB);
	req.device_total_bytes = 2 * GIB;
	TEST_ASSERT(membrane_joint_plan_resolve(&req, &res) == 1,
		"real planner resolves a constrained adaptive request");
	TEST_ASSERT(res.candidate_count >= 2,
		"a real runner-up candidate exists in the array");
	memset(&s, 0, sizeof(s));
	s.entries[0].should_succeed = 0;
	s.entries[0].failure_class = MEMBRANE_APPLY_CONTEXT_CREATE_FAILED;
	s.entries[0].cleanup_complete = 1;
	s.entries[1].should_succeed = 1;
	s.entries[1].cleanup_complete = 1;
	TEST_ASSERT(membrane_fallback_run(res.candidates, res.candidate_count,
			res.selected_index, req.device_total_bytes, 0, scripted_apply,
			&s, NULL, NULL, &trace) == 1,
		"fallback recovers using the real runner-up candidate");
	TEST_ASSERT(s.call_kv_precision[1] != s.call_kv_precision[0],
		"the real runner-up differs from the primary in a genuine, "
		"Phase-20-ranked way (precision), not an invented value");
}

/* ------------------------------------------------------------------ */
/* Safety                                                              */
/* ------------------------------------------------------------------ */

static void	test_null_out_is_safe(void)
{
	membrane_joint_candidate_t	cands[1];
	apply_script_t				s;

	make_candidate(&cands[0], 10, MEMBRANE_JOINT_KV_Q8,
		MEMBRANE_JOINT_PLACEMENT_DEFAULT, 100 * MIB, 20 * MIB, 1);
	memset(&s, 0, sizeof(s));
	TEST_ASSERT(membrane_fallback_run(cands, 1, 0, 4 * GIB, 512 * MIB,
			scripted_apply, &s, NULL, NULL, NULL) == 0,
		"a NULL out pointer never crashes, just fails");
}

static void	test_invalid_selected_index_is_safe(void)
{
	membrane_joint_candidate_t	cands[1];
	apply_script_t				s;
	membrane_fallback_trace_t	trace;

	make_candidate(&cands[0], 10, MEMBRANE_JOINT_KV_Q8,
		MEMBRANE_JOINT_PLACEMENT_DEFAULT, 100 * MIB, 20 * MIB, 1);
	memset(&s, 0, sizeof(s));
	TEST_ASSERT(membrane_fallback_run(cands, 1, -1, 4 * GIB, 512 * MIB,
			scripted_apply, &s, NULL, NULL, &trace) == 0,
		"an invalid selected_index (-1, matching Phase 20's own !ok "
		"convention) never crashes, just fails closed");
	TEST_ASSERT(s.n_calls == 0, "apply_fn is never called for invalid input");
}

int	main(void)
{
	test_first_succeeds_one_attempt();
	test_first_retryable_fails_second_succeeds();
	test_first_non_retryable_fails_stops();
	test_all_retryable_fail_exhausted();
	test_memory_shrink_skips_then_next_applies();
	test_same_candidate_never_attempted_twice();
	test_hard_max_attempt_limit();
	test_deterministic_repeated_calls();
	test_cleanup_failure_stops_immediately();
	test_no_alternative_candidate_exhausted();
	test_ineligible_candidates_never_attempted();
	test_failure_classification_retryable();
	test_failure_classification_non_retryable();
	test_failure_class_names_are_stable_and_safe();
	test_explicit_precision_never_changes_across_real_candidates();
	test_explicit_gpu_layers_never_changes_across_real_candidates();
	test_explicit_gpu_placement_never_silently_becomes_cpu();
	test_all_auto_uses_next_real_ranked_candidate();
	test_null_out_is_safe();
	test_invalid_selected_index_is_safe();
	printf("test_auto_fallback: all tests passed\n");
	return (0);
}
