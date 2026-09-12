#include <cstdio>
#include <string>

#include "chat_finish_reason.h"
#include "test_helpers.h"

/*
 * Post-v1 product-polish: pure, deterministic unit tests for the ONE
 * shared chat-completion finish_reason decision (chat_finish_reason.h's
 * own top comment has the full contract and the real v1.0.0
 * correctness gap this closes -- finish_reason used to be inferred
 * from a token-count comparison rather than tracking the real
 * termination cause).
 */

static void	test_eog_without_stop_match_is_stop(void)
{
	TEST_ASSERT(std::string(membrane_chat_finish_reason(false, true))
			== "stop",
		"a real end-of-generation token (stopped_eog) reports \"stop\", "
		"even with no user stop sequence involved at all -- Section 6/8 "
		"of the task");
}

static void	test_user_stop_match_is_stop(void)
{
	TEST_ASSERT(std::string(membrane_chat_finish_reason(true, false))
			== "stop",
		"a matched user-supplied stop sequence reports \"stop\" -- "
		"Section 7/8 of the task, unchanged pre-existing semantics");
}

static void	test_both_is_still_stop(void)
{
	TEST_ASSERT(std::string(membrane_chat_finish_reason(true, true))
			== "stop",
		"stop-match and a genuine EOG are not mutually exclusive "
		"(a model can emit EOG whose own piece text also happens to "
		"satisfy a stop string) -- either alone is sufficient for "
		"\"stop\"");
}

static void	test_neither_is_length(void)
{
	TEST_ASSERT(std::string(membrane_chat_finish_reason(false, false))
			== "length",
		"neither a stop match nor a real EOG -- the configured token "
		"limit was reached first, so \"length\" -- Section 8: never "
		"reported merely because termination state was lost");
}

int	main(void)
{
	test_eog_without_stop_match_is_stop();
	test_user_stop_match_is_stop();
	test_both_is_still_stop();
	test_neither_is_length();
	printf("test_chat_finish_reason: all tests passed\n");
	return (0);
}
