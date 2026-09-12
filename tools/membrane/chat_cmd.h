#ifndef MEMBRANE_CHAT_CMD_H
# define MEMBRANE_CHAT_CMD_H

# include <atomic>
# include <cstdio>
# include <string>
# include <vector>

/*
 * Post-v1 product-polish, prompt 3: `membrane chat [MODEL]` -- a plain-
 * terminal, human-facing REPL on top of the EXISTING OpenAI-compatible
 * server (docs/server.md). This file introduces no second inference
 * path, no second model-lifecycle policy, and no second HTTP/JSON
 * library:
 *
 *   CLI chat -> POST /v1/chat/completions (stream: true) -> the existing
 *   server/runtime, exactly like any other OpenAI-compatible client.
 *
 * Model resolution reuses `membrane use MODEL` (use_cmd.h) verbatim when
 * a MODEL argument is given (Section 2 of the task: "do not duplicate
 * model lifecycle logic") -- this file only ever reads server_config.h's
 * own default_model afterward, never re-implements catalog/registry
 * resolution. Server-reachability guidance reuses service_state.h's own
 * shared membrane_probe_service() (post-v1 product-polish, PR #78) --
 * never a second service-state probe.
 */

typedef struct s_membrane_chat_message
{
	std::string	role;		/* "user" | "assistant" */
	std::string	content;
}	membrane_chat_message_t;

typedef enum e_membrane_chat_turn_outcome
{
	MEMBRANE_CHAT_TURN_OK,			/* history now has both the user
									 * message and the completed
									 * assistant reply appended */
	MEMBRANE_CHAT_TURN_CANCELLED,	/* *cancel_flag went true mid-request
									 * (Ctrl+C) -- history has the user
									 * message only, never a partial
									 * assistant reply */
	MEMBRANE_CHAT_TURN_FAILED,		/* connection failure, a real server
									 * error response, or a malformed
									 * response -- history has the user
									 * message only; *out_error_message
									 * is a short, human-readable
									 * explanation (never raw JSON/a C++
									 * exception message) */
}	membrane_chat_turn_outcome_t;

/*
 * Sends exactly one chat turn against a real, already-reachable MEMBRANE
 * server at bind:port. Appends {role:"user", content:user_message} to
 * *history immediately (before the request is even sent) and, iff the
 * request completes successfully, appends the assembled
 * {role:"assistant", ...} reply too -- this ordering is the ONE
 * authoritative place Section 13's "append user before, assistant only
 * after success" rule is enforced; callers never append to *history
 * themselves.
 *
 * stream=true incrementally writes each new assistant-content piece to
 * `out` as it arrives (fputs + fflush -- real terminal streaming, never
 * buffered until the end); stream=false buffers the one, real, complete
 * response and writes it once. Either way `out` receives ONLY the
 * assistant's real generated text -- never a raw JSON/SSE frame (Section
 * 12), matching MEMBRANE's own Prompt-2 guarantee that a chat-control
 * token is never exposed as ordinary text either.
 *
 * cancel_flag, iff non-NULL, is polled during the streaming read
 * (mirrors runtime_session.h's own cancel_flag convention exactly) --
 * NULL means this call can never be cancelled from the outside.
 *
 * Returns MEMBRANE_CHAT_TURN_FAILED with a short, human-readable
 * *out_error_message covering: connection failure ("could not reach the
 * MEMBRANE server"), a real server error response (its own code/message,
 * with a `/clear` suggestion for a context-capacity code -- Section 17),
 * and a malformed/unparseable response. Never prints or returns a raw
 * JSON object, a raw SSE frame, or a C++ exception's own .what() text.
 */
membrane_chat_turn_outcome_t	membrane_chat_send_turn(
			const std::string &bind, int port,
			const std::string &model_name,
			std::vector<membrane_chat_message_t> *history,
			const std::string &user_message, bool stream,
			const std::atomic<bool> *cancel_flag, FILE *out,
			std::string *out_error_message);

typedef enum e_membrane_chat_slash_result
{
	MEMBRANE_CHAT_SLASH_NOT_A_COMMAND,	/* `input` did not start with '/'
										 * -- treat it as an ordinary chat
										 * message instead */
	MEMBRANE_CHAT_SLASH_HANDLED,		/* a real command ran (including an
										 * unrecognized "/whatever", which
										 * prints a short, non-fatal
										 * hint) -- *out_message is always
										 * set (never empty) for the
										 * caller to print, session
										 * continues */
	MEMBRANE_CHAT_SLASH_EXIT,			/* "/exit" (or its "/quit" alias)
										 * -- caller should print
										 * *out_message ("Bye.") and exit
										 * 0 */
}	membrane_chat_slash_result_t;

/*
 * Section 7 of the task: `/help`, `/clear`, `/model`, `/exit` -- a
 * deliberately tiny, non-extensible dispatcher, never a command
 * framework. `input` is matched verbatim (already trimmed by the
 * caller) against a fixed, small set of literal strings.
 *
 * /clear mutates *history (clears it) and returns a short confirmation.
 * /model and /help never touch *history. /exit/`/quit` returns
 * MEMBRANE_CHAT_SLASH_EXIT without mutating *history (Section 6: a chat
 * session's history is in-memory only and is simply dropped with the
 * process — there is nothing to persist or clean up on exit).
 */
membrane_chat_slash_result_t	membrane_chat_handle_slash_command(
			const std::string &input,
			std::vector<membrane_chat_message_t> *history,
			const std::string &model_name, std::string *out_message);

typedef struct s_membrane_chat_opts
{
	std::string	model;		/* "" = use the configured default */
	bool		no_stream;	/* --no-stream; default false (Section 14:
							 * "Default must be streaming") */
	std::string	bind;		/* "" = use server_config.json's own
							 * listen_address (Section 15: reuse the
							 * existing config/override mechanism
							 * `membrane serve`/`membrane status` already
							 * use, never a second one) */
	int			port;		/* 0 = use server_config.json's own port */
}	membrane_chat_opts_t;

/* Returns false (leaves *o untouched beyond defaults) and sets *err on
 * any unrecognized option or more than one positional MODEL argument --
 * a clean CLI_ERROR, never a silent misparse. */
bool	membrane_chat_parse_args(const std::vector<std::string> &args,
			membrane_chat_opts_t *o, std::string *err);

int	membrane_chat_cmd_dispatch(const std::vector<std::string> &args,
			bool want_json);

#endif
