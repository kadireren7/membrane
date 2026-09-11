#include "chat_cmd.h"

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <unistd.h>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "chat_stream_parser.h"
#include "server_config.h"
#include "status_client.h"
#include "doctor_cmd.h"
#include "use_cmd.h"
#include "product_cli.h"

using json = nlohmann::json;

/*
 * See chat_cmd.h's own top comment for the full architectural contract.
 *
 * PR #78 (fix/service-and-fit-consistency, unmerged as of this file)
 * introduces a dedicated, shared `membrane_probe_service()`
 * (service_state.h) that several commands (doctor, `membrane use`,
 * `membrane service start`) are refactored to call instead of each
 * reading systemd/launchd/schtasks state independently. This file
 * predates that refactor (it branches from `main`, not from PR #78,
 * per this task's own instructions) and deliberately does NOT copy that
 * implementation -- print_server_not_running_guidance() below instead
 * reuses the one already-public abstraction that exists on `main`
 * today, doctor_cmd.h's own membrane_doctor_collect(), reading its
 * "service" check's real `installed`/`active_state` detail fields
 * (the exact same fields setup_cmd.cpp's own pre-existing
 * already_installed/already_active local variables already read this
 * same way). Expected integration once PR #78 merges: this function's
 * body can be simplified to a direct membrane_probe_service() call
 * instead of round-tripping through the full doctor JSON -- a small,
 * mechanical follow-up, not a redesign (see this repo's PR description
 * for the full merge-order note).
 */
static void	print_server_not_running_guidance(void)
{
	json	doctor_root;

	membrane_doctor_collect(&doctor_root);

	bool	installed = false;

	if (doctor_root.contains("checks") && doctor_root["checks"].is_array())
		for (const auto &c : doctor_root["checks"])
			if (c.value("name", std::string()) == "service")
				installed = c.contains("detail")
					&& c["detail"].value("installed", false);
	printf("MEMBRANE server is not running.\n\n");
	if (!installed)
	{
		printf("Run it in this terminal:\n  membrane serve\n\n");
		printf("Or install the background service:\n  membrane service "
			"install\n");
	}
	else
		printf("Start it with:\n  membrane service start\n");
}

bool	membrane_chat_parse_args(const std::vector<std::string> &args,
			membrane_chat_opts_t *o, std::string *err)
{
	*o = membrane_chat_opts_t();
	o->port = 0;
	for (size_t i = 0; i < args.size(); ++i)
	{
		if (args[i] == "--no-stream")
			o->no_stream = true;
		else if (args[i] == "--bind" && i + 1 < args.size())
			o->bind = args[++i];
		else if (args[i] == "--port" && i + 1 < args.size())
			o->port = atoi(args[++i].c_str());
		else if (!args[i].empty() && args[i][0] != '-' && o->model.empty())
			o->model = args[i];
		else
		{
			*err = "unknown option '" + args[i] + "'";
			return (false);
		}
	}
	return (true);
}

static std::string	format_chat_error(const std::string &code,
				const std::string &message)
{
	std::string	out = "Chat request failed";

	if (!code.empty())
		out += ": " + code;
	out += "\n";
	if (code == "CTX_TOO_SMALL_FOR_PROMPT" || code == "NO_FEASIBLE_CONTEXT")
		out += "Try `/clear` to shorten the conversation, or use a "
			"model/context plan with more capacity.\n";
	else if (code == "MODEL_NOT_FOUND")
		out += "Run `membrane model search`.\n";
	else if (code == "SERVER_BUSY")
		out += "The server is busy -- try again in a moment.\n";
	else if (!message.empty())
		out += message + "\n";
	return (out);
}

membrane_chat_turn_outcome_t	membrane_chat_send_turn(
			const std::string &bind, int port,
			const std::string &model_name,
			std::vector<membrane_chat_message_t> *history,
			const std::string &user_message, bool stream,
			const std::atomic<bool> *cancel_flag, FILE *out,
			std::string *out_error_message)
{
	/* Section 13: the ONE place this rule is enforced -- the user's own
	 * message is part of the conversation the moment it is submitted,
	 * regardless of whether the request that follows succeeds. */
	history->push_back({"user", user_message});

	json	body;
	json	messages = json::array();

	for (const auto &m : *history)
		messages.push_back({{"role", m.role}, {"content", m.content}});
	body["model"] = model_name;
	body["stream"] = stream;
	body["messages"] = messages;

	httplib::Client	cli(bind, port);

	cli.set_connection_timeout(3, 0);
	/* Generation on a CPU-only host can genuinely take a while between
	 * bytes for a long response -- generous, not indefinite (a real
	 * connection failure or a wedged server still gets reported, just
	 * not too eagerly). */
	cli.set_read_timeout(600, 0);

	int							http_status = 0;
	std::string					raw_body;	/* used only for a non-stream
											 * response or a non-200
											 * (JSON error) response body */
	membrane_chat_stream_parser_t	parser;
	std::string					assistant_text;
	bool						got_error_frame = false;
	std::string					error_code;
	std::string					error_message;
	bool						cancelled = false;

	httplib::Request	req;

	req.method = "POST";
	req.path = "/v1/chat/completions";
	req.headers = {{"Content-Type", "application/json"}};
	req.body = body.dump();
	req.response_handler = [&](const httplib::Response &res) -> bool
	{
		http_status = res.status;
		return (true);
	};
	req.content_receiver = [&](const char *data, size_t len, size_t,
			size_t) -> bool
	{
		if (cancel_flag != NULL
			&& cancel_flag->load(std::memory_order_relaxed))
		{
			cancelled = true;
			return (false);
		}
		if (!stream || http_status != 200)
		{
			raw_body.append(data, len);
			return (true);
		}
		membrane_chat_stream_feed(&parser, data, len,
			[&](const std::string &piece)
			{
				fputs(piece.c_str(), out);
				fflush(out);
				assistant_text += piece;
			},
			[&](const std::string &code, const std::string &msg)
			{
				got_error_frame = true;
				error_code = code;
				error_message = msg;
			},
			[&](const std::string &) { /* finish_reason: informational
										 * only -- the real completion
										 * signal for this function's own
										 * contract is the SSE stream
										 * ending / [DONE], not this
										 * value. */ },
			[&]() { /* [DONE]: nothing extra to do -- the content
					 * receiver naturally stops being called once the
					 * connection itself closes right after. */ });
		return (true);
	};

	httplib::Result	result = cli.send(req);

	if (!result)
	{
		if (cancelled)
			return (MEMBRANE_CHAT_TURN_CANCELLED);
		*out_error_message = "could not reach the MEMBRANE server "
			"(connection lost).";
		return (MEMBRANE_CHAT_TURN_FAILED);
	}
	if (cancelled)
		return (MEMBRANE_CHAT_TURN_CANCELLED);
	if (http_status != 200)
	{
		std::string	code;
		std::string	message;

		try
		{
			json	j = json::parse(raw_body);

			if (j.contains("error") && j["error"].is_object())
			{
				code = j["error"].value("code", std::string());
				message = j["error"].value("message", std::string());
			}
		}
		catch (const json::parse_error &)
		{
		}
		if (code.empty() && message.empty())
			message = "server returned HTTP " + std::to_string(http_status);
		*out_error_message = format_chat_error(code, message);
		return (MEMBRANE_CHAT_TURN_FAILED);
	}
	if (got_error_frame)
	{
		*out_error_message = format_chat_error(error_code, error_message);
		return (MEMBRANE_CHAT_TURN_FAILED);
	}
	if (!stream)
	{
		try
		{
			json	j = json::parse(raw_body);

			assistant_text = j["choices"][0]["message"]
				.value("content", std::string());
		}
		catch (const std::exception &)
		{
			*out_error_message = "the server's response could not be "
				"understood.";
			return (MEMBRANE_CHAT_TURN_FAILED);
		}
		fputs(assistant_text.c_str(), out);
		fflush(out);
	}
	history->push_back({"assistant", assistant_text});
	return (MEMBRANE_CHAT_TURN_OK);
}

membrane_chat_slash_result_t	membrane_chat_handle_slash_command(
			const std::string &input,
			std::vector<membrane_chat_message_t> *history,
			const std::string &model_name, std::string *out_message)
{
	if (input.empty() || input[0] != '/')
		return (MEMBRANE_CHAT_SLASH_NOT_A_COMMAND);
	if (input == "/exit" || input == "/quit")
	{
		*out_message = "Bye.";
		return (MEMBRANE_CHAT_SLASH_EXIT);
	}
	if (input == "/help")
	{
		*out_message =
			"Commands:\n"
			"  /help    show this help\n"
			"  /clear   clear the current conversation history\n"
			"  /model   show the current model\n"
			"  /exit    exit the chat session\n";
		return (MEMBRANE_CHAT_SLASH_HANDLED);
	}
	if (input == "/clear")
	{
		history->clear();
		*out_message = "Conversation history cleared.\n";
		return (MEMBRANE_CHAT_SLASH_HANDLED);
	}
	if (input == "/model")
	{
		*out_message = "Current model: " + model_name + "\n";
		return (MEMBRANE_CHAT_SLASH_HANDLED);
	}
	*out_message = "Unknown command '" + input + "' -- type /help for the "
		"list of commands.\n";
	return (MEMBRANE_CHAT_SLASH_HANDLED);
}

static void	print_chat_help(void)
{
	printf("membrane chat [MODEL] [options]\n\n");
	printf("Start an interactive chat session with the local MEMBRANE "
		"server.\n\n");
	printf("  MODEL                already-installed or catalog model "
		"name to select first\n");
	printf("                       (same resolution as `membrane use "
		"MODEL`); omit to use\n");
	printf("                       the currently configured default "
		"model\n\n");
	printf("Options:\n");
	printf("  --no-stream          wait for the complete reply instead "
		"of streaming it\n");
	printf("                       (default: streamed)\n");
	printf("  --bind ADDRESS       server address (default: "
		"server_config.json's own,\n");
	printf("                       normally 127.0.0.1)\n");
	printf("  --port N             server port (default: "
		"server_config.json's own,\n");
	printf("                       normally 8642)\n\n");
	printf("Notes:\n");
	printf("  - talks to your already-running local MEMBRANE server "
		"only -- no cloud\n");
	printf("    connection is ever made.\n");
	printf("  - conversation history is kept in memory for this session "
		"only; it is\n");
	printf("    never written to disk and is gone once you exit.\n");
	printf("  - type /help inside the session for the list of slash "
		"commands.\n");
}

/* Real, distinct outcomes of one blocking read of a line from stdin --
 * kept separate from Ctrl+D/EOF so the caller can give each its own,
 * correct UX (Section 8/9 of the task) without guessing from a single
 * bool. Deliberately no readline/libedit dependency (Section 10: "no
 * large new third-party dependency") -- a plain byte-at-a-time read()
 * is simple, has no line-editing of its own to get wrong, and is UTF-8-
 * transparent (it moves raw bytes, never interprets/truncates them). */
typedef enum e_chat_readline_result
{
	CHAT_READLINE_OK,
	CHAT_READLINE_EOF,
	CHAT_READLINE_INTERRUPTED,
}	chat_readline_result_t;

/* Set directly by the SIGINT handler below and polled by membrane_chat_
 * send_turn()'s own cancel_flag parameter -- ONE flag serves both the
 * idle-at-prompt case (chat_read_line()'s blocking read() is interrupted
 * with EINTR regardless of this flag's value, purely by the signal's
 * delivery -- see the sigaction() SA_RESTART note below) and the
 * streaming case (the content receiver explicitly checks it). A plain
 * std::atomic<bool>::store() is not formally guaranteed async-signal-
 * safe by the C++ standard, but is lock-free (hence a single atomic
 * store instruction) on every platform this project targets -- the
 * same pragmatic, extremely common pattern real-world signal-driven
 * cancellation flags use. */
static std::atomic<bool>	g_chat_cancel_requested{false};

static void	chat_sigint_handler(int)
{
	g_chat_cancel_requested.store(true, std::memory_order_relaxed);
}

static chat_readline_result_t	chat_read_line(std::string *out)
{
	out->clear();
	while (true)
	{
		char	c;
		ssize_t	n = read(STDIN_FILENO, &c, 1);

		if (n < 0)
		{
			if (errno == EINTR)
				return (CHAT_READLINE_INTERRUPTED);
			return (CHAT_READLINE_EOF);
		}
		if (n == 0)
			return (out->empty() ? CHAT_READLINE_EOF : CHAT_READLINE_OK);
		if (c == '\n')
			return (CHAT_READLINE_OK);
		out->push_back(c);
	}
}

static std::string	trim(const std::string &s)
{
	size_t	start = s.find_first_not_of(" \t\r\n");

	if (start == std::string::npos)
		return ("");
	size_t	end = s.find_last_not_of(" \t\r\n");

	return (s.substr(start, end - start + 1));
}

int	membrane_chat_cmd_dispatch(const std::vector<std::string> &args,
			bool want_json)
{
	(void)want_json;	/* an interactive REPL has no meaningful JSON
						 * mode -- unrecognized like any other option
						 * below if a caller passes --json anyway */
	for (const auto &a : args)
		if (a == "--help" || a == "-h")
			return (print_chat_help(), MEMBRANE_EXIT_SUCCESS);

	membrane_chat_opts_t	o;
	std::string				parse_err;

	if (!membrane_chat_parse_args(args, &o, &parse_err))
	{
		fprintf(stderr, "membrane chat: %s\n", parse_err.c_str());
		return (MEMBRANE_EXIT_CLI_ERROR);
	}
	/* Section 2: reuse `membrane use MODEL` verbatim for resolution --
	 * install-with-consent if not yet installed, live-switch if the
	 * server is already running, every existing error path (NOT_FOUND,
	 * NONINTERACTIVE_CONSENT_REQUIRED, ...) reported exactly as `membrane
	 * use` itself already reports it. Never a second model-lifecycle
	 * implementation. */
	if (!o.model.empty())
	{
		int	rc = membrane_use_cmd_dispatch({o.model}, false);

		if (rc != MEMBRANE_EXIT_SUCCESS)
			return (rc);
	}

	std::string						config_path
			= membrane_server_config_resolve_path();
	membrane_server_config_t		cfg = membrane_server_config_defaults();
	membrane_server_config_error_t	cfg_err;

	if (!config_path.empty())
		membrane_server_config_load(config_path, &cfg, &cfg_err);
	if (!o.bind.empty())
		cfg.listen_address = o.bind;
	if (o.port != 0)
		cfg.port = o.port;
	if (cfg.default_model.empty())
	{
		printf("No model is selected.\n\n");
		printf("Discover available models:\n  membrane model search\n\n");
		printf("Select one:\n  membrane use MODEL\n");
		return (MEMBRANE_EXIT_MODEL_ERROR);
	}

	json	status_json;

	if (!membrane_fetch_server_status(cfg.listen_address, cfg.port,
			&status_json))
	{
		print_server_not_running_guidance();
		return (MEMBRANE_EXIT_RUNTIME_ERROR);
	}

	std::string	model_name = cfg.default_model;

	printf("MEMBRANE Chat -- %s\n", model_name.c_str());
	printf("Endpoint: http://%s:%d\n", cfg.listen_address.c_str(),
		cfg.port);
	printf("Type /help for commands.\n\n");

	std::vector<membrane_chat_message_t>	history;
	struct sigaction						sa;
	struct sigaction						old_sa;

	memset(&sa, 0, sizeof(sa));
	sa.sa_handler = chat_sigint_handler;
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = 0;	/* deliberately NOT SA_RESTART -- a blocking
						 * stdin read() or the streaming HTTP read must
						 * actually be interrupted (EINTR) rather than
						 * silently resumed, so Ctrl+C is visible where
						 * it happens (Section 8 of the task) */
	sigaction(SIGINT, &sa, &old_sa);

	int	exit_code = MEMBRANE_EXIT_SUCCESS;

	while (true)
	{
		printf("You: ");
		fflush(stdout);

		std::string				line;
		chat_readline_result_t	rr = chat_read_line(&line);

		if (rr == CHAT_READLINE_INTERRUPTED)
		{
			g_chat_cancel_requested.store(false, std::memory_order_relaxed);
			printf("\n(Type /exit or press Ctrl+D to quit.)\n");
			continue ;
		}
		if (rr == CHAT_READLINE_EOF)
		{
			printf("\nBye.\n");
			break ;
		}
		std::string	trimmed = trim(line);

		if (trimmed.empty())
			continue ;

		std::string						slash_message;
		membrane_chat_slash_result_t	slash_result
				= membrane_chat_handle_slash_command(trimmed, &history,
					model_name, &slash_message);

		if (slash_result == MEMBRANE_CHAT_SLASH_EXIT)
		{
			printf("%s\n", slash_message.c_str());
			break ;
		}
		if (slash_result == MEMBRANE_CHAT_SLASH_HANDLED)
		{
			printf("%s\n", slash_message.c_str());
			continue ;
		}

		printf("Assistant: ");
		fflush(stdout);
		g_chat_cancel_requested.store(false, std::memory_order_relaxed);

		std::string						error_message;
		membrane_chat_turn_outcome_t	outcome = membrane_chat_send_turn(
				cfg.listen_address, cfg.port, model_name, &history, trimmed,
				!o.no_stream, &g_chat_cancel_requested, stdout,
				&error_message);

		if (outcome == MEMBRANE_CHAT_TURN_OK)
			printf("\n\n");
		else if (outcome == MEMBRANE_CHAT_TURN_CANCELLED)
		{
			g_chat_cancel_requested.store(false, std::memory_order_relaxed);
			printf("\n[cancelled]\n\n");
		}
		else
			printf("\n%s\n", error_message.c_str());
	}
	sigaction(SIGINT, &old_sa, NULL);
	return (exit_code);
}
