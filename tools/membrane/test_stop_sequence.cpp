/*
 * Unit tests for stop_sequence.h -- the OpenAI-compatible "stop" string
 * decision logic server.cpp's streaming and non-streaming paths share.
 *
 * The case that motivated pulling this out: a stop string routinely
 * spans several BPE tokens, and the streaming path can only emit
 * forward. Before membrane_stop_pending_suffix_len() existed, each
 * chunk was pushed to the client as soon as no COMPLETE match was
 * found in the accumulated text, so a stop string split across two
 * tokens ("abST" then "OPcd" for stop "STOP") had its own "ST" prefix
 * already on the wire by the time the match resolved -- the client saw
 * "abST" where the non-streaming path, truncating one fully
 * accumulated string, correctly produced "ab".
 *
 * simulate_streaming() below replays that same emit-forward-only
 * discipline over a token list and returns what a client would
 * actually receive, so these tests pin the real behavior rather than
 * just the helper's return values.
 */

#include <cstdio>
#include <string>
#include <vector>

#include "stop_sequence.h"
#include "test_helpers.h"

/* Mirrors stream_token_cb()'s stop branch: accumulate, stop at the
 * first complete match, and never emit trailing bytes that are still a
 * possible partial match. */
static std::string	simulate_streaming(const std::vector<std::string> &tokens,
						const std::vector<std::string> &stops)
{
	std::string	client;
	std::string	acc;
	size_t		sent = 0;

	for (const auto &piece : tokens)
	{
		size_t	match_pos;

		acc += piece;
		if (membrane_stop_find_match(acc, stops, &match_pos))
		{
			if (match_pos > sent)
				client += acc.substr(sent, match_pos - sent);
			return (client);
		}
		size_t	safe_end = acc.size()
			- membrane_stop_pending_suffix_len(acc, stops);

		if (safe_end > sent)
		{
			client += acc.substr(sent, safe_end - sent);
			sent = safe_end;
		}
	}
	/* End-of-generation flush: held-back bytes were not a stop after all. */
	client += acc.substr(sent);
	return (client);
}

/* The non-streaming path: accumulate everything, truncate once. */
static std::string	simulate_nonstreaming(
						const std::vector<std::string> &tokens,
						const std::vector<std::string> &stops)
{
	std::string	acc;

	for (const auto &piece : tokens)
	{
		size_t	match_pos;

		acc += piece;
		if (membrane_stop_find_match(acc, stops, &match_pos))
			return (acc.substr(0, match_pos));
	}
	return (acc);
}

static void	check_agree(const std::vector<std::string> &tokens,
				const std::vector<std::string> &stops,
				const std::string &want, const char *what)
{
	std::string	s = simulate_streaming(tokens, stops);
	std::string	n = simulate_nonstreaming(tokens, stops);

	if (s != want || n != want)
		fprintf(stderr, "  %s: streaming=\"%s\" non-streaming=\"%s\" "
			"want=\"%s\"\n", what, s.c_str(), n.c_str(), want.c_str());
	TEST_ASSERT(s == want, what);
	TEST_ASSERT(n == want, what);
	TEST_ASSERT(s == n, "streaming and non-streaming agree");
}

static void	test_stop_within_one_token(void)
{
	check_agree({"ab", "STOPcd"}, {"STOP"}, "ab",
		"stop contained in a single token is cut");
	printf("PASS test_stop_within_one_token\n");
}

/* The regression this module exists for. */
static void	test_stop_split_across_tokens(void)
{
	check_agree({"abST", "OPcd"}, {"STOP"}, "ab",
		"stop split across two tokens leaks nothing");
	check_agree({"ab", "ST", "OP", "cd"}, {"STOP"}, "ab",
		"stop split across three tokens leaks nothing");
	check_agree({"a", "b", "S", "T", "O", "P"}, {"STOP"}, "ab",
		"stop split one byte per token leaks nothing");
	/* "\n\n" is a common real stop value and is very often split. */
	check_agree({"hi\n", "\nmore"}, {"\n\n"}, "hi",
		"newline-pair stop split across tokens leaks nothing");
	printf("PASS test_stop_split_across_tokens\n");
}

/* Held-back bytes must be released when they turn out not to be a stop. */
static void	test_partial_match_released(void)
{
	check_agree({"abST", "XY"}, {"STOP"}, "abSTXY",
		"a partial match that never completes is still emitted");
	check_agree({"ab", "ST"}, {"STOP"}, "abST",
		"trailing partial match is flushed at end of generation");
	check_agree({"abc"}, {"STOP"}, "abc", "no stop present, nothing held");
	printf("PASS test_partial_match_released\n");
}

static void	test_earliest_match_wins(void)
{
	check_agree({"abXXcdYY"}, {"YY", "XX"}, "ab",
		"earliest match across several stop strings wins");
	check_agree({"ab", "XX"}, {"ZZZ", "XX"}, "ab",
		"earliest match wins when split across tokens");
	printf("PASS test_earliest_match_wins\n");
}

static void	test_empty_stop_ignored(void)
{
	size_t	pos = 12345;

	TEST_ASSERT(!membrane_stop_find_match("abc", {""}, &pos),
		"an empty stop string never matches");
	TEST_ASSERT(membrane_stop_pending_suffix_len("abc", {""}) == 0,
		"an empty stop string holds nothing back");
	check_agree({"abc"}, {""}, "abc", "empty stop leaves output untouched");
	printf("PASS test_empty_stop_ignored\n");
}

static void	test_pending_suffix_len(void)
{
	TEST_ASSERT(membrane_stop_pending_suffix_len("abST", {"STOP"}) == 2,
		"trailing \"ST\" is 2 bytes of a possible \"STOP\"");
	TEST_ASSERT(membrane_stop_pending_suffix_len("abS", {"STOP"}) == 1,
		"trailing \"S\" is 1 byte of a possible \"STOP\"");
	TEST_ASSERT(membrane_stop_pending_suffix_len("abXY", {"STOP"}) == 0,
		"no overlap holds nothing back");
	/* A COMPLETE match is find_match()'s job, not this function's. */
	TEST_ASSERT(membrane_stop_pending_suffix_len("STOP", {"STOP"}) == 0,
		"a full match is not reported as a pending prefix");
	/* Longest overlap across several stop strings wins. */
	TEST_ASSERT(membrane_stop_pending_suffix_len("xabc", {"abcd", "c!"}) == 3,
		"longest pending prefix across stop strings wins");
	/* Never more than the longest stop string minus one. */
	TEST_ASSERT(membrane_stop_pending_suffix_len("aaaa", {"aa"}) == 1,
		"pending prefix is bounded by the stop length minus one");
	/* Shorter accumulator than the stop string. */
	TEST_ASSERT(membrane_stop_pending_suffix_len("S", {"STOP"}) == 1,
		"accumulator shorter than the stop string still overlaps");
	TEST_ASSERT(membrane_stop_pending_suffix_len("", {"STOP"}) == 0,
		"empty accumulator holds nothing back");
	printf("PASS test_pending_suffix_len\n");
}

int	main(void)
{
	test_stop_within_one_token();
	test_stop_split_across_tokens();
	test_partial_match_released();
	test_earliest_match_wins();
	test_empty_stop_ignored();
	test_pending_suffix_len();
	return (0);
}
