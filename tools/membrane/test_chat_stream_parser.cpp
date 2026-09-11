#include <cstdio>
#include <string>
#include <vector>

#include "chat_stream_parser.h"
#include "test_helpers.h"

/*
 * `membrane chat`: pure, deterministic unit tests for the SSE
 * incremental parser -- no network, no server, no model. See chat_
 * stream_parser.h's own top comment for the exact wire format this
 * mirrors (server.cpp's own real `/v1/chat/completions` streaming
 * shape).
 */

static void	test_single_chunk_multiple_frames(void)
{
	membrane_chat_stream_parser_t	p;
	std::vector<std::string>		contents;
	std::vector<std::string>		finish_reasons;
	bool							done = false;

	std::string	wire =
		"data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},"
		"\"finish_reason\":null}]}\n\n"
		"data: {\"choices\":[{\"delta\":{\"content\":\", world\"},"
		"\"finish_reason\":null}]}\n\n"
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}"
		"\n\n"
		"data: [DONE]\n\n";

	membrane_chat_stream_feed(&p, wire.data(), wire.size(),
		[&](const std::string &c) { contents.push_back(c); },
		[&](const std::string &, const std::string &)
		{ TEST_ASSERT(false, "no error frame in this fixture"); },
		[&](const std::string &r) { finish_reasons.push_back(r); },
		[&]() { done = true; });

	TEST_ASSERT(contents.size() == 2, "two content deltas were dispatched");
	TEST_ASSERT(contents[0] == "Hello", "first delta content is exact");
	TEST_ASSERT(contents[1] == ", world", "second delta content is exact");
	TEST_ASSERT(finish_reasons.size() == 1 && finish_reasons[0] == "stop",
		"finish_reason \"stop\" was dispatched exactly once");
	TEST_ASSERT(done, "[DONE] was dispatched");
}

/*
 * Real TCP reads can hand the caller a chunk that ends mid-line -- split
 * the exact same wire bytes at several arbitrary, deliberately awkward
 * byte offsets (mid-JSON-key, mid-UTF-8) and confirm the assembled
 * result is identical either way (Section 18D: "multiple SSE chunks
 * concatenate correctly").
 */
static void	test_split_across_arbitrary_chunk_boundaries(void)
{
	std::string	wire =
		"data: {\"choices\":[{\"delta\":{\"content\":\"caf\\u00e9\"},"
		"\"finish_reason\":null}]}\n\n"
		"data: [DONE]\n\n";

	for (size_t split : {(size_t)1, (size_t)5, (size_t)20, (size_t)40,
			wire.size() - 1})
	{
		membrane_chat_stream_parser_t	p;
		std::string						assembled;
		bool							done = false;

		membrane_chat_stream_feed(&p, wire.data(), split, [&](
				const std::string &c) { assembled += c; }, nullptr, nullptr,
			[&]() { done = true; });
		membrane_chat_stream_feed(&p, wire.data() + split,
			wire.size() - split, [&](const std::string &c)
				{ assembled += c; }, nullptr, nullptr,
			[&]() { done = true; });

		TEST_ASSERT(assembled == "caf\xc3\xa9",
			("content survives a split at byte offset "
				+ std::to_string(split)).c_str());
		TEST_ASSERT(done, "[DONE] still recognized regardless of the "
			"split point");
	}
}

static void	test_error_frame_dispatched_and_content_never_printed(void)
{
	membrane_chat_stream_parser_t	p;
	std::string						code;
	std::string						message;
	bool							content_called = false;

	std::string	wire = "data: {\"error\":{\"code\":\"NO_FEASIBLE_CONTEXT\","
		"\"message\":\"no context/hardware plan could be resolved\"}}\n\n"
		"data: [DONE]\n\n";

	membrane_chat_stream_feed(&p, wire.data(), wire.size(),
		[&](const std::string &) { content_called = true; },
		[&](const std::string &c, const std::string &m)
		{ code = c; message = m; }, nullptr, nullptr);

	TEST_ASSERT(!content_called, "an error frame never also fires "
		"on_content");
	TEST_ASSERT(code == "NO_FEASIBLE_CONTEXT", "error code is exact");
	TEST_ASSERT(message == "no context/hardware plan could be resolved",
		"error message is exact");
}

static void	test_blank_and_malformed_lines_are_skipped_silently(void)
{
	membrane_chat_stream_parser_t	p;
	std::vector<std::string>		contents;
	bool							threw = false;

	std::string	wire =
		"\n"						/* a lone SSE frame separator */
		"event: ping\n\n"			/* a real SSE field this project's own
									 * server never emits, but a defensive
									 * parser must not choke on it */
		"data: not valid json at all\n\n"
		"data: {\"choices\":[]}\n\n"	/* well-formed JSON, empty choices */
		"data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n";

	try
	{
		membrane_chat_stream_feed(&p, wire.data(), wire.size(),
			[&](const std::string &c) { contents.push_back(c); }, nullptr,
			nullptr, nullptr);
	}
	catch (...)
	{
		threw = true;
	}
	TEST_ASSERT(!threw, "malformed/unexpected lines never throw -- Section "
		"12 of the task: never crash on a malformed response");
	TEST_ASSERT(contents.size() == 1 && contents[0] == "ok",
		"the one real, well-formed content delta still arrives despite "
		"every malformed/unexpected line around it");
}

static void	test_control_protocol_text_is_never_the_content(void)
{
	/* Section 18D's own "control protocol text not printed": a raw SSE
	 * frame ("data: ", the trailing blank line, "[DONE]" itself) must
	 * never be handed to on_content as if it were assistant text. */
	membrane_chat_stream_parser_t	p;
	std::vector<std::string>		contents;

	std::string	wire = "data: {\"choices\":[{\"delta\":{\"content\":"
		"\"data: not-a-real-frame\"}}]}\n\ndata: [DONE]\n\n";

	membrane_chat_stream_feed(&p, wire.data(), wire.size(),
		[&](const std::string &c) { contents.push_back(c); }, nullptr,
		nullptr, nullptr);

	TEST_ASSERT(contents.size() == 1
			&& contents[0] == "data: not-a-real-frame",
		"ordinary generated text that merely LOOKS like an SSE frame is "
		"still passed through byte-for-byte as real content -- only this "
		"parser's own real line-level framing is special, never the "
		"model's own generated text");
}

int	main(void)
{
	test_single_chunk_multiple_frames();
	test_split_across_arbitrary_chunk_boundaries();
	test_error_frame_dispatched_and_content_never_printed();
	test_blank_and_malformed_lines_are_skipped_silently();
	test_control_protocol_text_is_never_the_content();
	printf("test_chat_stream_parser: all tests passed\n");
	return (0);
}
