#ifndef MEMBRANE_CHAT_STREAM_PARSER_H
# define MEMBRANE_CHAT_STREAM_PARSER_H

# include <functional>
# include <string>

/*
 * `membrane chat`: a small, pure, incremental parser for the SSE wire
 * format server.cpp's own `/v1/chat/completions` (`stream: true`)
 * already emits (docs/server.md's own "Streaming" section) -- never a
 * second HTTP/JSON library, never a full SSE spec implementation (this
 * project's own server is the only producer this ever needs to
 * understand: one `data: {...}\n\n` frame per delta, a final frame
 * carrying `finish_reason`, and the literal `data: [DONE]\n\n`
 * sentinel).
 *
 * Pure (no I/O, no httplib dependency) so it is directly unit-testable
 * with synthetic byte chunks, split at arbitrary boundaries -- a real
 * TCP read can hand the caller a chunk that ends mid-line, mid-JSON-
 * value, or even mid-UTF-8-multibyte-sequence; this parser buffers
 * whatever is incomplete and only ever invokes a callback for a
 * complete, valid `data: ` line. A line that is blank (SSE's own frame
 * separator), does not start with `data: `, or fails to parse as JSON
 * (and is not the literal "[DONE]") is silently skipped -- Section 12
 * of the task: never crash on a malformed/unexpected line.
 */

typedef struct s_membrane_chat_stream_parser
{
	std::string	pending;	/* bytes received but not yet part of a
							 * complete "\n"-terminated line */
}	membrane_chat_stream_parser_t;

/*
 * Feeds `len` new bytes into `*parser`, extracting and dispatching every
 * complete line found (including any already-buffered partial line from
 * a previous call) before returning; any trailing incomplete line is
 * left in `parser->pending` for the next call.
 *
 * For each complete `data: ` line found:
 *   - payload == "[DONE]": calls on_done() (never on_content/on_error
 *     for this same line).
 *   - a JSON object containing a top-level "error" key (server.cpp's
 *     own real mid-stream-failure frame shape, {"error":{"code",
 *     "message"}}): calls on_error(code, message).
 *   - a JSON object shaped like a real chat.completion.chunk with a
 *     non-empty choices[0].delta.content: calls on_content(text) --
 *     never called with an empty string (an empty/absent delta.content,
 *     e.g. the terminal chunk carrying only finish_reason, produces no
 *     callback at all here; see on_finish_reason below for that).
 *   - the same chunk also carrying a non-null choices[0].finish_reason:
 *     calls on_finish_reason(reason) (independently of on_content --
 *     a single chunk may carry both, content then finish_reason, in
 *     that order).
 *   - anything else (blank line, non-"data: " line, unparseable JSON):
 *     skipped silently, no callback.
 * Any callback may be an empty std::function -- never invoked in that
 * case, never a null-pointer dereference.
 */
void	membrane_chat_stream_feed(membrane_chat_stream_parser_t *parser,
			const char *data, size_t len,
			const std::function<void(const std::string &)> &on_content,
			const std::function<void(const std::string &,
				const std::string &)> &on_error,
			const std::function<void(const std::string &)>
				&on_finish_reason,
			const std::function<void(void)> &on_done);

#endif
