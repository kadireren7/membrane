# `membrane chat` — interactive terminal chat

Post-v1 product-polish, prompt 3. A real user reported that the normal
way to talk to a running `membrane serve` was to hand-compose a `curl`
`/v1/chat/completions` request — correct for API/app integration, but
a poor everyday human-facing experience. `membrane chat` is a plain-
terminal REPL on top of that same, unchanged API:

```
membrane chat
```

```
MEMBRANE Chat -- smollm2-360m-instruct
Endpoint: http://127.0.0.1:8642
Type /help for commands.

You: hello
Assistant: Hello! How can I help?

You:
```

## Architecture: a thin API client, nothing more

```
CLI chat
    |
    v
MEMBRANE HTTP API (POST /v1/chat/completions)
    |
    v
existing server/runtime
```

`membrane chat` never calls llama.cpp/the runtime directly, never
duplicates model resolution or service-lifecycle logic, and introduces
no second HTTP/JSON library:

- Model resolution reuses `membrane use MODEL` verbatim
  (`docs/model-lifecycle.md`) when a `MODEL` argument is given — the
  exact same install-with-consent-if-needed/select/live-switch flow,
  never a second implementation.
- Server-reachability guidance reuses `membrane doctor`'s own real
  service check (`membrane_doctor_collect()`) — see "Server not
  running" below for the one small, disclosed exception.
- Requests go through the standard `httplib::Client` this project
  already vendors (the same library `membrane status`/`membrane
  service status` already use for their own HTTP calls) and are parsed
  with a small, pure, purpose-built SSE parser
  (`tools/membrane/chat_stream_parser.h`) — never a second JSON/HTTP
  stack.

This keeps the chat CLI a thin, replaceable presentation layer. Future
work on hardware-aware planning, runtime abstraction, external
runtimes, observability, or adaptive memory control does not need to
touch it.

## Usage

```
membrane chat [MODEL] [options]
```

- `MODEL` (optional) — an already-installed or catalog model name,
  resolved exactly like `membrane use MODEL`. Omitted: uses the
  currently configured default model.
- `--no-stream` — wait for the complete reply instead of streaming it.
  Default: streamed (Section 14 of the task: "Default must be
  streaming").
- `--bind ADDRESS` / `--port N` — override the server address/port
  this session talks to; default comes from `server.json`
  (`docs/service.md`'s own "Server config" section), the same
  override convention `membrane serve`/`membrane status` already use.

## No model selected

```
No model is selected.

Discover available models:
  membrane model search

Select one:
  membrane use MODEL
```

Printed (never an entry into a broken REPL) whenever no `MODEL`
argument was given and `server.json`'s own `default_model` is empty.

## Server not running

```
MEMBRANE server is not running.

Run it in this terminal:
  membrane serve

Or install the background service:
  membrane service install
```

or, if a background service unit already exists but is stopped:

```
MEMBRANE server is not running.

Start it with:
  membrane service start
```

This reuses `membrane doctor`'s own existing `membrane_doctor_collect()`
(the "service" check's real `installed`/`active_state` fields) —
**not** a second service-state probe. Post-v1 product-polish prompt 1
(`fix/service-and-fit-consistency`, PR #78, unmerged as of this writing)
introduces a dedicated, shared `membrane_probe_service()` several other
commands are refactored to call instead; this file predates that
refactor (it branches from `main`, per this prompt's own instructions,
not from PR #78) and deliberately does not copy that implementation.
Expected follow-up once PR #78 merges: `chat_cmd.cpp`'s own
`print_server_not_running_guidance()` can be simplified to call
`membrane_probe_service()` directly instead of round-tripping through
the full doctor JSON — a small, mechanical change, not a redesign.

## Conversation history

In-memory only, for the current session — no database, no file, no
account system. A new `membrane chat` starts fresh every time. Prior
turns are sent as `messages` on every subsequent request, so follow-ups
work naturally:

```
You: My name is Kadir.
Assistant: Nice to meet you, Kadir.

You: What is my name?
Assistant: Your name is Kadir.
```

**Append rule** (never violated, regardless of what the request does):
the user's own message is added to history the moment it is submitted;
the assistant's reply is added only once the request has *fully and
successfully* completed. A failed, errored, or Ctrl+C-cancelled request
never leaves a partial/invalid assistant entry behind — only the user's
own message stays, exactly as they typed it, ready for another attempt.

## Slash commands

| Command | Effect |
|---|---|
| `/help` | list the commands above, concisely |
| `/clear` | clear the current conversation history, stay in the session |
| `/model` | show the current model name only — never registry internals |
| `/exit` (alias `/quit`) | exit cleanly, status 0 |

Deliberately not a command framework — a fixed, small `if`/`else`
dispatch (`membrane_chat_handle_slash_command()`), matching this
project's own "do not build infrastructure the task does not need"
convention.

## Ctrl+C / Ctrl+D

- Idle at the `You:` prompt: Ctrl+C prints a short hint
  (`(Type /exit or press Ctrl+D to quit.)`) and returns to a fresh
  prompt — it never leaves the terminal in a corrupted state and never
  exits the whole session on a single stray interrupt.
- While a reply is streaming: Ctrl+C cancels the in-flight request
  (the same real cancellation mechanism `membrane-run`'s own
  `cancel_flag` convention already establishes, applied here to the
  HTTP client's streaming read) and returns to the prompt — the
  MEMBRANE server itself is never touched, and no partial reply is ever
  added to history.
- Ctrl+D (EOF) at the prompt exits cleanly: `Bye.`, status 0. No stack
  trace, no malformed terminal state either way.

## Errors

Real server error codes are surfaced directly, never a raw JSON object,
a C++ exception, or low-level HTTP/socket detail:

```
Chat request failed: CTX_TOO_SMALL_FOR_PROMPT
Try `/clear` to shorten the conversation, or use a model/context plan
with more capacity.
```

A context-capacity error (`CTX_TOO_SMALL_FOR_PROMPT`/
`NO_FEASIBLE_CONTEXT`) specifically suggests `/clear` — conversation
history is never auto-summarized or silently mutated; it is the user's
own choice whether to shorten it.

## What this is not

No GUI, no curses/TUI dashboard, no markdown rendering, no persistent
chat history/accounts/cloud sync, no tool calling/embeddings/structured
output/RAG/web search/agent framework, no prompt-preset library, no
Ollama/vLLM adapter. A minimal, clean terminal chat experience on top
of the existing API — nothing more.
