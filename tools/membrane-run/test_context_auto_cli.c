#include <stdint.h>
#include <string.h>

#include "context_auto_cli.h"
#include "test_helpers.h"

static void	test_minimum_basic(void)
{
	uint64_t	out = 0;

	TEST_ASSERT(membrane_ctxauto_minimum_required_context(37, 128, &out)
		== 1, "resolves");
	TEST_ASSERT(out == 37 + 128 + 8, "prompt + gen + margin, matching "
		"main.cpp's own pre-existing default-ctx formula");
}

static void	test_minimum_zero_tokens(void)
{
	uint64_t	out = 0;

	membrane_ctxauto_minimum_required_context(0, 0, &out);
	TEST_ASSERT(out == 8, "still includes the fixed margin");
}

static void	test_minimum_overflow_saturates(void)
{
	uint64_t	out = 0;

	TEST_ASSERT(membrane_ctxauto_minimum_required_context(UINT64_MAX,
		UINT64_MAX, &out) == 1, "never crashes");
	TEST_ASSERT(out == UINT64_MAX, "saturates rather than wrapping small");
}

static void	test_minimum_null_out(void)
{
	TEST_ASSERT(membrane_ctxauto_minimum_required_context(1, 1, NULL) == 0,
		"NULL out fails cleanly, never crashes");
}

static void	test_suggest_no_feasible_context_full(void)
{
	membrane_ctxauto_suggestions_t	s;

	membrane_ctxauto_suggest(MEMBRANE_CTXREC_STATUS_NO_FEASIBLE_CONTEXT,
		NULL, 128, 0, 0, &s);
	TEST_ASSERT(s.count > 0, "has suggestions");
	TEST_ASSERT(strstr(s.text[0], "gen-tokens") != NULL,
		"first suggestion mentions reducing generation tokens");
}

static void	test_suggest_never_suggests_fixed_precision(void)
{
	membrane_ctxauto_suggestions_t	s;
	size_t							i;
	int								mentions_kv_q5 = 0;

	/* --kv was already explicit -- must never suggest changing it. */
	membrane_ctxauto_suggest(MEMBRANE_CTXREC_STATUS_PLANNER_REJECTED_ALL,
		NULL, 0, /*requested_precision_explicit=*/1, 0, &s);
	for (i = 0; i < s.count; ++i)
		if (strstr(s.text[i], "--kv q5") != NULL)
			mentions_kv_q5 = 1;
	TEST_ASSERT(!mentions_kv_q5, "never suggests --kv q5 when --kv was "
		"already explicitly given");
}

static void	test_suggest_never_suggests_fixed_placement(void)
{
	membrane_ctxauto_suggestions_t	s;
	size_t							i;
	int								mentions_placement = 0;

	membrane_ctxauto_suggest(MEMBRANE_CTXREC_STATUS_PLANNER_REJECTED_ALL,
		NULL, 0, 0, /*requested_placement_explicit=*/1, &s);
	for (i = 0; i < s.count; ++i)
		if (strstr(s.text[i], "kv-placement") != NULL)
			mentions_placement = 1;
	TEST_ASSERT(!mentions_placement, "never suggests --kv-placement when "
		"it was already explicitly given");
}

static void	test_suggest_no_gen_tokens_suppresses_that_suggestion(void)
{
	membrane_ctxauto_suggestions_t	s;
	size_t							i;
	int								mentions_gen_tokens = 0;

	membrane_ctxauto_suggest(MEMBRANE_CTXREC_STATUS_NO_FEASIBLE_CONTEXT,
		NULL, /*gen_tokens=*/0, 0, 0, &s);
	for (i = 0; i < s.count; ++i)
		if (strstr(s.text[i], "gen-tokens") != NULL)
			mentions_gen_tokens = 1;
	TEST_ASSERT(!mentions_gen_tokens,
		"never suggests reducing --gen-tokens when it is already 0");
}

static void	test_suggest_minimum_exceeds_model_max(void)
{
	membrane_ctxauto_suggestions_t	s;
	size_t							i;
	int								mentions_model_choice = 0;

	membrane_ctxauto_suggest(MEMBRANE_CTXREC_STATUS_MINIMUM_EXCEEDS_MODEL_MAX,
		NULL, 128, 0, 0, &s);
	TEST_ASSERT(s.count > 0, "has suggestions");
	for (i = 0; i < s.count; ++i)
		if (strstr(s.text[i], "larger context window") != NULL)
			mentions_model_choice = 1;
	TEST_ASSERT(mentions_model_choice, "suggests a model with a larger "
		"context window among its options");
}

static void	test_suggest_host_memory_unknown_distinct(void)
{
	membrane_ctxauto_suggestions_t	s;

	/* Same top-level status as a generic rejection, but the
	 * representative_reason_code distinguishes the real cause. */
	membrane_ctxauto_suggest(MEMBRANE_CTXREC_STATUS_PLANNER_REJECTED_ALL,
		MEMBRANE_HOST_GUARD_REASON_UNKNOWN, 128, 0, 0, &s);
	TEST_ASSERT(s.count == 1, "exactly one, carefully-worded suggestion "
		"for host-memory-unknown, distinct from the generic case");
	TEST_ASSERT(strstr(s.text[0], "explicit --ctx N") != NULL,
		"suggests manual sizing as the explicit trade-off");
}

static void	test_suggest_model_max_unknown(void)
{
	membrane_ctxauto_suggestions_t	s;

	membrane_ctxauto_suggest(
		MEMBRANE_CTXREC_STATUS_MODEL_MAX_CONTEXT_UNKNOWN, NULL, 128, 0, 0,
		&s);
	TEST_ASSERT(s.count == 1, "exactly one suggestion");
	TEST_ASSERT(strstr(s.text[0], "manually") != NULL,
		"frames manual sizing as an explicit trade-off, not a plain fix");
}

static void	test_suggest_ok_status_has_no_suggestions(void)
{
	membrane_ctxauto_suggestions_t	s;

	membrane_ctxauto_suggest(MEMBRANE_CTXREC_STATUS_OK, NULL, 128, 0, 0,
		&s);
	TEST_ASSERT(s.count == 0, "a successful status has no suggestions "
		"to make");
}

static void	test_suggest_bound_never_exceeded(void)
{
	membrane_ctxauto_suggestions_t	s;

	membrane_ctxauto_suggest(MEMBRANE_CTXREC_STATUS_NO_FEASIBLE_CONTEXT,
		NULL, 128, 0, 0, &s);
	TEST_ASSERT(s.count <= MEMBRANE_CTXAUTO_MAX_SUGGESTIONS,
		"never exceeds the fixed bound");
}

static void	test_suggest_null_safety(void)
{
	membrane_ctxauto_suggest(NULL, NULL, 0, 0, 0, NULL);
	{
		membrane_ctxauto_suggestions_t	s;

		membrane_ctxauto_suggest(NULL, NULL, 0, 0, 0, &s);
		TEST_ASSERT(s.count == 0, "NULL status yields no suggestions, "
			"never crashes");
	}
}

int	main(void)
{
	test_minimum_basic();
	test_minimum_zero_tokens();
	test_minimum_overflow_saturates();
	test_minimum_null_out();
	test_suggest_no_feasible_context_full();
	test_suggest_never_suggests_fixed_precision();
	test_suggest_never_suggests_fixed_placement();
	test_suggest_no_gen_tokens_suppresses_that_suggestion();
	test_suggest_minimum_exceeds_model_max();
	test_suggest_host_memory_unknown_distinct();
	test_suggest_model_max_unknown();
	test_suggest_ok_status_has_no_suggestions();
	test_suggest_bound_never_exceeded();
	test_suggest_null_safety();
	printf("test_context_auto_cli: all tests passed\n");
	return (0);
}
