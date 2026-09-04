#include <string>

#include "utf8_stream.h"
#include "test_helpers.h"

/*
 * Mega Phase B, PR B2: unit tests for utf8_stream.h's incomplete-UTF-8-
 * tail detector -- the exact primitive server.cpp's streaming token
 * callback uses to avoid emitting a truncated multi-byte character over
 * SSE. Pure (no httplib/llama dependency), so real multilingual content
 * is tested here directly rather than only via a real model.
 */

static void	test_empty_string_is_complete(void)
{
	TEST_ASSERT(membrane_utf8_incomplete_suffix_len("") == 0,
		"the empty string has nothing incomplete");
}

static void	test_pure_ascii_is_complete(void)
{
	TEST_ASSERT(membrane_utf8_incomplete_suffix_len("hello") == 0,
		"plain ASCII is always a complete UTF-8 sequence");
}

/* U+00E9 'e' (Latin small letter e with acute) is the 2-byte sequence
 * 0xC3 0xA9. */
static void	test_complete_2byte_sequence(void)
{
	std::string	s = "caf\xC3\xA9";	/* "café" */

	TEST_ASSERT(membrane_utf8_incomplete_suffix_len(s) == 0,
		"a complete 2-byte sequence at the end is not incomplete");
}

static void	test_incomplete_2byte_sequence(void)
{
	std::string	s = "caf\xC3";	/* the lead byte of 'e', continuation
									 * byte not arrived yet */

	TEST_ASSERT(membrane_utf8_incomplete_suffix_len(s) == 1,
		"a lone 2-byte lead byte at the end is incomplete by 1 byte");
}

/* U+4E2D CJK "middle" is the 3-byte sequence 0xE4 0xB8 0xAD. */
static void	test_complete_3byte_sequence(void)
{
	std::string	s = "\xE4\xB8\xAD";

	TEST_ASSERT(membrane_utf8_incomplete_suffix_len(s) == 0,
		"a complete 3-byte CJK sequence is not incomplete");
}

static void	test_incomplete_3byte_sequence_missing_one(void)
{
	std::string	s = "\xE4\xB8";	/* missing the final continuation byte */

	TEST_ASSERT(membrane_utf8_incomplete_suffix_len(s) == 2,
		"a 3-byte sequence missing its last continuation byte is "
		"incomplete by 2 bytes");
}

static void	test_incomplete_3byte_sequence_missing_two(void)
{
	std::string	s = "\xE4";	/* only the lead byte arrived */

	TEST_ASSERT(membrane_utf8_incomplete_suffix_len(s) == 1,
		"a bare 3-byte lead byte is incomplete by 1 byte (the lead byte "
		"itself)");
}

/* U+1F600 (grinning face emoji) is the 4-byte sequence
 * 0xF0 0x9F 0x98 0x80. */
static void	test_complete_4byte_emoji(void)
{
	std::string	s = "hi \xF0\x9F\x98\x80";

	TEST_ASSERT(membrane_utf8_incomplete_suffix_len(s) == 0,
		"a complete 4-byte emoji sequence is not incomplete");
}

static void	test_incomplete_4byte_emoji_missing_all_continuations(void)
{
	std::string	s = "hi \xF0";	/* only the lead byte */

	TEST_ASSERT(membrane_utf8_incomplete_suffix_len(s) == 1,
		"a bare 4-byte lead byte is incomplete by exactly 1 byte");
}

static void	test_incomplete_4byte_emoji_missing_last_continuation(void)
{
	std::string	s = "hi \xF0\x9F\x98";	/* missing the final byte */

	TEST_ASSERT(membrane_utf8_incomplete_suffix_len(s) == 3,
		"a 4-byte sequence missing only its last continuation byte is "
		"incomplete by 3 bytes -- the maximum possible");
}

/* Malformed input (a stray continuation byte with no lead byte at all
 * within the 3-byte lookback window) must never cause unbounded holding
 * -- capped at max_back (3). */
static void	test_malformed_input_never_unbounded(void)
{
	std::string	s = "\x80\x80\x80\x80\x80";	/* five stray continuation
												 * bytes, no lead byte
												 * anywhere */

	TEST_ASSERT(membrane_utf8_incomplete_suffix_len(s) <= 3,
		"malformed input is never held back by more than 3 bytes -- the "
		"structural maximum, never an unbounded/permanent hold");
}

/* An invalid lead byte (e.g. 0xFF, which is not a legal UTF-8 lead byte
 * under any encoding form) is treated as already complete rather than
 * held back forever. */
static void	test_invalid_lead_byte_treated_as_complete(void)
{
	std::string	s = "abc\xFF";

	TEST_ASSERT(membrane_utf8_incomplete_suffix_len(s) == 0,
		"an invalid lead byte is never held back forever");
}

int	main(void)
{
	test_empty_string_is_complete();
	test_pure_ascii_is_complete();
	test_complete_2byte_sequence();
	test_incomplete_2byte_sequence();
	test_complete_3byte_sequence();
	test_incomplete_3byte_sequence_missing_one();
	test_incomplete_3byte_sequence_missing_two();
	test_complete_4byte_emoji();
	test_incomplete_4byte_emoji_missing_all_continuations();
	test_incomplete_4byte_emoji_missing_last_continuation();
	test_malformed_input_never_unbounded();
	test_invalid_lead_byte_treated_as_complete();
	printf("test_utf8_stream: all tests passed\n");
	return (0);
}
