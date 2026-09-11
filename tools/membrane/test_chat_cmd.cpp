#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <functional>
#include <thread>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "chat_cmd.h"
#include "test_helpers.h"

using json = nlohmann::json;

/*
 * `membrane chat`: deterministic tests driven against a real, local
 * httplib::Server test fixture (same real-local-HTTP-round-trip
 * precedent test_server.cpp/test_use_cmd.cpp already establish) that
 * scripts exactly the responses a real MEMBRANE server would send --
 * never a real model, never a real MEMBRANE server process. Pure/no-
 * network cases (argument parsing, slash commands) need no fixture at
 * all.
 *
 * httplib::Server keeps every Post() registration for a given path in
 * an ordered list and dispatches to the FIRST match -- calling Post()
 * again per-test would never override the previous test's own handler,
 * it would just add a second, unreachable one behind it. The route is
 * therefore registered exactly ONCE, delegating to a reassignable
 * std::function each test overwrites for its own scripted response.
 */

# define TEST_PORT	18944

static httplib::Server	*g_mock;
static std::thread		*g_mock_thread;
static std::function<void(const httplib::Request &, httplib::Response &)>
	g_mock_handler;

static void	start_mock_server(void)
{
	g_mock = new httplib::Server();
	g_mock->Post("/v1/chat/completions", [](const httplib::Request &req,
			httplib::Response &res)
	{
		if (g_mock_handler)
			g_mock_handler(req, res);
	});
	g_mock_thread = new std::thread([]() { g_mock->listen("127.0.0.1",
		TEST_PORT); });

	httplib::Client	probe("127.0.0.1", TEST_PORT);
	int					attempts = 0;

	probe.set_connection_timeout(0, 50000);
	while (attempts < 100)
	{
		if (g_mock->is_running())
			return ;
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
		attempts++;
	}
	TEST_ASSERT(false, "mock server did not become ready within 2s");
}

static void	stop_mock_server(void)
{
	g_mock->stop();
	g_mock_thread->join();
	delete g_mock_thread;
	delete g_mock;
	g_mock = NULL;
	g_mock_thread = NULL;
}

/* ---- A. command parsing (Section 18A) ---- */

static void	test_parse_no_model(void)
{
	membrane_chat_opts_t	o;
	std::string				err;

	TEST_ASSERT(membrane_chat_parse_args({}, &o, &err),
		"`membrane chat` with no arguments parses cleanly");
	TEST_ASSERT(o.model.empty(), "no model was given");
	TEST_ASSERT(!o.no_stream, "streaming is the default (Section 14)");
}

static void	test_parse_with_model(void)
{
	membrane_chat_opts_t	o;
	std::string				err;

	TEST_ASSERT(membrane_chat_parse_args({"qwen2.5-1.5b-instruct"}, &o,
			&err), "`membrane chat MODEL` parses cleanly");
	TEST_ASSERT(o.model == "qwen2.5-1.5b-instruct", "the model name is "
		"captured exactly");
}

static void	test_parse_no_stream_and_overrides(void)
{
	membrane_chat_opts_t	o;
	std::string				err;

	TEST_ASSERT(membrane_chat_parse_args({"smol", "--no-stream", "--port",
			"9999", "--bind", "0.0.0.0"}, &o, &err),
		"model + --no-stream + overrides all parse together");
	TEST_ASSERT(o.model == "smol", "model still captured alongside flags");
	TEST_ASSERT(o.no_stream, "--no-stream was recognized");
	TEST_ASSERT(o.port == 9999, "--port override captured");
	TEST_ASSERT(o.bind == "0.0.0.0", "--bind override captured");
}

static void	test_parse_unknown_option_is_refused(void)
{
	membrane_chat_opts_t	o;
	std::string				err;

	TEST_ASSERT(!membrane_chat_parse_args({"--not-a-real-flag"}, &o, &err),
		"an unrecognized option is refused, never silently ignored");
	TEST_ASSERT(!err.empty(), "a real error message is produced");
}

/* ---- C. slash commands (Section 18C) ---- */

static void	test_slash_help(void)
{
	std::vector<membrane_chat_message_t>	history;
	std::string								msg;

	TEST_ASSERT(membrane_chat_handle_slash_command("/help", &history,
			"m", &msg) == MEMBRANE_CHAT_SLASH_HANDLED,
		"/help is handled");
	TEST_ASSERT(msg.find("/clear") != std::string::npos
			&& msg.find("/exit") != std::string::npos,
		"/help lists the real commands concisely");
}

static void	test_slash_clear_resets_history(void)
{
	std::vector<membrane_chat_message_t>	history
			= {{"user", "hi"}, {"assistant", "hello"}};
	std::string								msg;

	TEST_ASSERT(membrane_chat_handle_slash_command("/clear", &history, "m",
			&msg) == MEMBRANE_CHAT_SLASH_HANDLED, "/clear is handled");
	TEST_ASSERT(history.empty(), "/clear actually empties the history");
	TEST_ASSERT(!msg.empty(), "/clear confirms briefly");
}

static void	test_slash_model_shows_current_model_only(void)
{
	std::vector<membrane_chat_message_t>	history;
	std::string								msg;

	TEST_ASSERT(membrane_chat_handle_slash_command("/model", &history,
			"qwen2.5-1.5b-instruct", &msg) == MEMBRANE_CHAT_SLASH_HANDLED,
		"/model is handled");
	TEST_ASSERT(msg.find("qwen2.5-1.5b-instruct") != std::string::npos,
		"/model names the real current model");
	TEST_ASSERT(msg.find("{") == std::string::npos,
		"/model never dumps raw registry JSON (Section 7 of the task)");
}

static void	test_slash_exit_and_quit(void)
{
	std::vector<membrane_chat_message_t>	history;
	std::string								msg;

	TEST_ASSERT(membrane_chat_handle_slash_command("/exit", &history, "m",
			&msg) == MEMBRANE_CHAT_SLASH_EXIT, "/exit signals exit");
	TEST_ASSERT(membrane_chat_handle_slash_command("/quit", &history, "m",
			&msg) == MEMBRANE_CHAT_SLASH_EXIT,
		"/quit is a small, unsurprising alias for /exit");
}

static void	test_non_slash_input_is_not_a_command(void)
{
	std::vector<membrane_chat_message_t>	history;
	std::string								msg;

	TEST_ASSERT(membrane_chat_handle_slash_command("hello there", &history,
			"m", &msg) == MEMBRANE_CHAT_SLASH_NOT_A_COMMAND,
		"ordinary chat text is never mistaken for a slash command");
}

/* ---- B/D/F/G/H: real HTTP round trips against the mock server ---- */

static void	test_streaming_turn_success_and_history_ordering(void)
{
	g_mock_handler = [](const httplib::Request &req, httplib::Response &res)
	{
		json	body = json::parse(req.body);

		/* Section 18B: every prior turn's messages must actually be
		 * present in a later request. */
		TEST_ASSERT(body["messages"].is_array(), "messages is an array");
		std::string	sse =
			"data: {\"choices\":[{\"delta\":{\"content\":\"Nice to meet "
			"you, \"},\"finish_reason\":null}]}\n\n"
			"data: {\"choices\":[{\"delta\":{\"content\":\"Kadir.\"},"
			"\"finish_reason\":null}]}\n\n"
			"data: {\"choices\":[{\"delta\":{},\"finish_reason\":"
			"\"stop\"}]}\n\n"
			"data: [DONE]\n\n";
		res.set_content(sse, "text/event-stream");
	};

	std::vector<membrane_chat_message_t>	history;
	std::string								err;
	char									*buf = NULL;
	size_t									buf_size = 0;
	FILE									*capture = open_memstream(&buf,
			&buf_size);

	membrane_chat_turn_outcome_t	outcome = membrane_chat_send_turn(
			"127.0.0.1", TEST_PORT, "smol", &history, "My name is Kadir.",
			true, NULL, capture, &err);

	fclose(capture);
	TEST_ASSERT(outcome == MEMBRANE_CHAT_TURN_OK, "a scripted, valid "
		"stream reports success");
	TEST_ASSERT(std::string(buf) == "Nice to meet you, Kadir.",
		"the streamed deltas were written to `out` concatenated, in "
		"order, with no SSE framing visible");
	TEST_ASSERT(history.size() == 2, "both the user message and the "
		"completed assistant reply are now in history");
	TEST_ASSERT(history[0].role == "user"
			&& history[0].content == "My name is Kadir.",
		"the user message is first, byte-exact");
	TEST_ASSERT(history[1].role == "assistant"
			&& history[1].content == "Nice to meet you, Kadir.",
		"the assistant message is second, exactly the assembled stream");
	free(buf);

	/* A second turn: confirm the FIRST turn's own messages are actually
	 * included in the second request's own body (Section 18B: "multiple
	 * turns preserve ordering"). */
	g_mock_handler = [](const httplib::Request &req, httplib::Response &res)
	{
		json	body = json::parse(req.body);

		TEST_ASSERT(body["messages"].size() == 3, "the second request "
			"carries all three prior messages (2 from turn 1, 1 new)");
		TEST_ASSERT(body["messages"][0]["content"] == "My name is Kadir.",
			"turn 1's user message is still first, unchanged");
		TEST_ASSERT(body["messages"][1]["content"]
				== "Nice to meet you, Kadir.",
			"turn 1's assistant reply is still second, unchanged");
		TEST_ASSERT(body["messages"][2]["content"] == "What is my name?",
			"the new user message is last");
		res.set_content("data: {\"choices\":[{\"delta\":{\"content\":"
			"\"Your name is Kadir.\"},\"finish_reason\":\"stop\"}]}\n\n"
			"data: [DONE]\n\n", "text/event-stream");
	};
	outcome = membrane_chat_send_turn("127.0.0.1", TEST_PORT, "smol",
			&history, "What is my name?", true, NULL, stdout, &err);
	TEST_ASSERT(outcome == MEMBRANE_CHAT_TURN_OK, "the second turn also "
		"succeeds");
	TEST_ASSERT(history.size() == 4, "history now has all four messages");
}

static void	test_non_stream_turn_buffers_full_reply(void)
{
	g_mock_handler = [](const httplib::Request &, httplib::Response &res)
	{
		json	message;

		message["role"] = "assistant";
		message["content"] = "Hi there.";

		json	choice;

		choice["index"] = 0;
		choice["message"] = message;
		choice["finish_reason"] = "stop";

		json	body;

		body["choices"] = json::array({choice});
		res.set_content(body.dump(), "application/json");
	};

	std::vector<membrane_chat_message_t>	history;
	std::string								err;
	char									*buf = NULL;
	size_t									buf_size = 0;
	FILE									*capture = open_memstream(&buf,
			&buf_size);

	membrane_chat_turn_outcome_t	outcome = membrane_chat_send_turn(
			"127.0.0.1", TEST_PORT, "smol", &history, "hello", false, NULL,
			capture, &err);

	fclose(capture);
	TEST_ASSERT(outcome == MEMBRANE_CHAT_TURN_OK, "--no-stream still "
		"succeeds against a plain (non-SSE) JSON response");
	TEST_ASSERT(std::string(buf) == "Hi there.", "the full reply is "
		"written to `out` in one piece");
	free(buf);
}

/* ---- E/F: server unavailable / HTTP error surfaced cleanly ---- */

static void	test_connection_failure_is_human_readable(void)
{
	std::vector<membrane_chat_message_t>	history;
	std::string								err;

	/* A real closed port -- nothing is listening on TEST_PORT + 1 here. */
	membrane_chat_turn_outcome_t	outcome = membrane_chat_send_turn(
			"127.0.0.1", TEST_PORT + 1, "smol", &history, "hello", true,
			NULL, stdout, &err);

	TEST_ASSERT(outcome == MEMBRANE_CHAT_TURN_FAILED, "a real connection "
		"failure is reported as FAILED, never a crash or an uncaught "
		"exception");
	TEST_ASSERT(err.find("MEMBRANE server") != std::string::npos,
		"the error message is human-readable, not a raw exception/errno "
		"string");
	TEST_ASSERT(history.size() == 1, "the user's own message is kept "
		"(Section 13), but no assistant message was ever appended");
	TEST_ASSERT(history[0].role == "user", "the one history entry is the "
		"user's message, never a phantom assistant reply");
}

static void	test_http_error_surfaced_cleanly(void)
{
	g_mock_handler = [](const httplib::Request &, httplib::Response &res)
	{
		json	err_body;

		err_body["error"] = {{"code", "NO_FEASIBLE_CONTEXT"},
			{"message", "no context/hardware plan could be resolved for "
				"this model on this host"}};
		res.status = 500;
		res.set_content(err_body.dump(), "application/json");
	};

	std::vector<membrane_chat_message_t>	history;
	std::string								err;

	membrane_chat_turn_outcome_t	outcome = membrane_chat_send_turn(
			"127.0.0.1", TEST_PORT, "smol", &history, "hello", true, NULL,
			stdout, &err);

	TEST_ASSERT(outcome == MEMBRANE_CHAT_TURN_FAILED, "a real server "
		"error response is reported as FAILED");
	TEST_ASSERT(err.find("NO_FEASIBLE_CONTEXT") != std::string::npos,
		"the real server error code is surfaced");
	TEST_ASSERT(err.find("/clear") != std::string::npos,
		"a context-capacity error suggests /clear (Section 17)");
	TEST_ASSERT(err.find("{") == std::string::npos,
		"never a raw JSON object printed to the user");
	TEST_ASSERT(history.size() == 1 && history[0].role == "user",
		"no assistant message was appended for a failed request");
}

static void	test_mid_stream_error_frame_surfaced_cleanly(void)
{
	g_mock_handler = [](const httplib::Request &, httplib::Response &res)
	{
		res.set_content("data: {\"choices\":[{\"delta\":{\"content\":"
			"\"partial\"}}]}\n\ndata: {\"error\":{\"code\":"
			"\"GENERATION_FAILED\",\"message\":\"generation failed for "
			"this request\"}}\n\ndata: [DONE]\n\n", "text/event-stream");
	};

	std::vector<membrane_chat_message_t>	history;
	std::string								err;

	membrane_chat_turn_outcome_t	outcome = membrane_chat_send_turn(
			"127.0.0.1", TEST_PORT, "smol", &history, "hello", true, NULL,
			stdout, &err);

	TEST_ASSERT(outcome == MEMBRANE_CHAT_TURN_FAILED, "a mid-stream error "
		"frame after real partial content still reports FAILED overall");
	TEST_ASSERT(err.find("GENERATION_FAILED") != std::string::npos,
		"the real mid-stream error code is surfaced");
	TEST_ASSERT(history.size() == 1 && history[0].role == "user",
		"Section 13: a request that ultimately failed never appends an "
		"assistant message, even if real partial content had already "
		"been streamed to the terminal");
}

/* ---- G: interrupted mid-stream -> no invalid history entry ---- */

static void	test_cancelled_mid_stream_leaves_no_assistant_entry(void)
{
	g_mock_handler = [](const httplib::Request &, httplib::Response &res)
	{
		res.set_content("data: {\"choices\":[{\"delta\":{\"content\":"
			"\"partial output\"}}]}\n\ndata: [DONE]\n\n",
			"text/event-stream");
	};

	std::vector<membrane_chat_message_t>	history;
	std::string								err;
	std::atomic<bool>						cancel_flag{true};	/* already
										 * requested before the call even
										 * starts -- deterministic, no
										 * timing race */

	membrane_chat_turn_outcome_t	outcome = membrane_chat_send_turn(
			"127.0.0.1", TEST_PORT, "smol", &history, "hello", true,
			&cancel_flag, stdout, &err);

	TEST_ASSERT(outcome == MEMBRANE_CHAT_TURN_CANCELLED, "a pre-armed "
		"cancel_flag is honored -- Section 8/13 of the task");
	TEST_ASSERT(history.size() == 1 && history[0].role == "user",
		"cancellation never leaves a partial/invalid assistant entry in "
		"history, only the user's own message");
}

/* ---- H: UTF-8 round-trips through request construction ---- */

static void	test_utf8_content_round_trips(void)
{
	std::string	captured_content;

	g_mock_handler = [&](const httplib::Request &req, httplib::Response &res)
	{
		json	body = json::parse(req.body);

		captured_content = body["messages"].back()["content"]
			.get<std::string>();
		res.set_content("data: {\"choices\":[{\"delta\":{\"content\":"
			"\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
			"text/event-stream");
	};

	std::vector<membrane_chat_message_t>	history;
	std::string								err;
	std::string								utf8_message
			= "caf\xc3\xa9 \xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e \xf0\x9f"
				"\x98\x80";	/* "café 日本語 😀" */

	membrane_chat_turn_outcome_t	outcome = membrane_chat_send_turn(
			"127.0.0.1", TEST_PORT, "smol", &history, utf8_message, true,
			NULL, stdout, &err);

	TEST_ASSERT(outcome == MEMBRANE_CHAT_TURN_OK, "the request succeeds");
	TEST_ASSERT(captured_content == utf8_message, "non-ASCII UTF-8 "
		"content (accented Latin, CJK, an emoji) round-trips through "
		"request construction byte-for-byte, never truncated or "
		"mangled");
	TEST_ASSERT(history[0].content == utf8_message, "the same UTF-8 text "
		"is preserved in in-memory history too");
}

int	main(void)
{
	test_parse_no_model();
	test_parse_with_model();
	test_parse_no_stream_and_overrides();
	test_parse_unknown_option_is_refused();

	test_slash_help();
	test_slash_clear_resets_history();
	test_slash_model_shows_current_model_only();
	test_slash_exit_and_quit();
	test_non_slash_input_is_not_a_command();

	start_mock_server();
	test_streaming_turn_success_and_history_ordering();
	test_non_stream_turn_buffers_full_reply();
	test_connection_failure_is_human_readable();
	test_http_error_surfaced_cleanly();
	test_mid_stream_error_frame_surfaced_cleanly();
	test_cancelled_mid_stream_leaves_no_assistant_entry();
	test_utf8_content_round_trips();
	stop_mock_server();

	printf("test_chat_cmd: all tests passed\n");
	return (0);
}
