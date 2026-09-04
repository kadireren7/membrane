# MEMBRANE local server

Mega Phase A, PR A3 (server) + PR A4 (compat polish). `membrane serve`
starts a long-lived local HTTP process exposing an OpenAI-compatible
subset on top of the same reusable runtime-session core
(`tools/membrane-run/runtime_session.h`, PR A1) and model registry
(`tools/membrane/registry_core.h`, PR A2) the CLI already uses.
`membrane-run` remains the direct inference entry point, unchanged by
this phase.

**Mega Phase B, PR B1:** for normal day-to-day use, prefer
`membrane service install && membrane service start` over running
`membrane serve` by hand in a terminal — see `docs/service.md`.
`membrane serve` itself is unchanged and still exactly what a
`membrane service install`-generated systemd unit's own `ExecStart`
invokes; it remains the right tool for foreground/debug use. With no
`--port`/`--bind` flags, it now reads `listen_address`/`port`/
`default_model` from `~/.config/membrane/server.json` first (an
explicit CLI flag always overrides) — see `docs/service.md`'s "Server
config" section for the file's shape and `membrane model use NAME` /
Section 9 below for `default_model`.

**Real compatibility evidence (PR A4):** the official Python `openai`
SDK (`pip install openai`), pointed at a running `membrane serve`
instance with no code changes beyond `base_url`/`api_key`, correctly
calls `client.models.list()`, `client.chat.completions.create()` (real
generation, real usage/finish_reason parsing), and correctly raises its
own typed `NotFoundError` from this server's 404 response — see
`results/runtime-service/validation.json`.

```
membrane model add qwen /path/to/model.gguf
membrane serve
```

Then point any OpenAI-compatible client at `http://127.0.0.1:8642/v1`.

## Security scope

- Binds `127.0.0.1` only by default. Binding anything else requires an
  explicit `--allow-non-loopback` flag, and prints a clear warning when
  used.
- **No authentication.** This is only a defensible default because the
  server is loopback-only by default — anyone who can reach the bind
  address can run inference as the user who started it. There is no API
  key check; an `Authorization` header, if a client sends one, is
  ignored.
- No telemetry, no external network calls of any kind.
- Error responses never include a local filesystem path (Section 27 of
  the Mega Phase A task) — registry-internal paths are resolved
  server-side and never echoed back to an HTTP caller.

## Endpoints

- `GET /health` — `{"status":"ok","version":"0.3.0"}`.
- `GET /v1/models` — every model currently in the registry
  (`membrane model add`), OpenAI `list` shape.
- `GET /v1/status` — membrane-specific (not an OpenAI endpoint), backs
  the `membrane status` CLI command (PR A4, Section 37 of the task): a
  thin HTTP client against an already-running `serve` instance, never a
  process-management/daemon capability this project doesn't have.
  `{"running":true,"version":"...","endpoint":"http://...","loaded_model":
  "qwen"|null,"backend":"CPU"|"Vulkan","gpu_layers":N,"kv_precision":
  "native"|"q8"|"q5","context_policy":"automatic"}` (the last four
  fields are omitted/null until the first generation request has loaded
  a model).
- `POST /v1/chat/completions` — see below.

`POST /v1/completions` (the raw-prompt, non-chat endpoint) is not
implemented this phase.

## `POST /v1/chat/completions`

Minimum request:

```json
{"model": "qwen", "messages": [{"role": "user", "content": "Hello"}]}
```

Supported request fields: `model` (required, a registered name, unless
a `default_model` is configured server-side — see below —, in which
case an omitted or empty `"model"` field falls back to it),
`messages` (required, non-empty array of `{role, content}`),
`max_tokens` / `max_completion_tokens` (optional, default 512).
`temperature`/`top_p`/other sampling fields are accepted and **ignored**
— the underlying decode loop is greedy-only (argmax) today, no sampling
support exists in this project yet; the response's own `membrane.sampling`
field says so explicitly rather than silently claiming otherwise.
`stream: true` — see "Streaming" below.

Response (OpenAI shape plus one additive `membrane` object):

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "qwen",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "..."},
    "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
  "membrane": {"context": 512, "gpu_layers": 0, "kv_precision": "q8",
    "kv_placement": "default", "sampling": "greedy (...)"}
}
```

`finish_reason` is `"length"` when generation hit the requested token
limit, `"stop"` otherwise (an inferred distinction — there is no
explicit "why did generation stop" signal from the decode loop beyond
token count vs. limit).

### Chat templates

The model's own embedded chat template (`llama_chat_apply_template()`,
real llama.cpp API, GGUF metadata) is always used — never a naive manual
`"User: ... Assistant: ..."` join. A model with no usable template
returns `500 CHAT_TEMPLATE_UNAVAILABLE`; there is no fallback formatting.

### Context, GPU layers, and KV precision — fully automatic

A client never specifies `ctx`, GPU layers, KV precision, or KV
placement. The **same** context-recommendation + host-memory-guard +
joint-planner pipeline `--ctx auto` uses (PR 33–35) runs automatically:
the fully-automatic plan a bare `membrane-run --auto` would produce for
this model is what the server resolves and applies.

**Known, disclosed limitation:** GPU layers and KV precision are decided
**once**, when a model is first loaded, using whichever request happened
to trigger that load. They cannot change again without a model reload
(reload happens only on a model switch, per the policy below). A request
with a much larger prompt than the one that triggered the load is not
replanned — it reuses the fixed GPU-layer/precision configuration and
only sizes its own `llama_context` to its own prompt (Section 5 of the
Mega Phase A task: "persistent model, new context per request"). This
can mean a later, larger request fails to fit where an earlier, smaller
one succeeded. A future phase may revisit this; documented honestly
here rather than silently accepted.

## Streaming (`stream: true`) — PR B2

Real SSE streaming, not the `400 STREAMING_NOT_SUPPORTED` earlier phases
returned. Add `"stream": true` to a chat completion request:

```
curl -N http://127.0.0.1:8642/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen", "messages": [{"role": "user", "content": "Hello"}], "stream": true}'
```

Response: `Content-Type: text/event-stream`, one `data: {...}\n\n` frame
per incremental text delta (OpenAI `chat.completion.chunk` shape),
terminated by a final `data: {"choices":[{"delta":{},"finish_reason":
"stop"|"length", ...}]}` chunk and then the literal `data: [DONE]\n\n`.
`stream_options: {"include_usage": true}` (the same real OpenAI
convention non-streaming responses always include unconditionally) adds
one extra chunk — `choices: []`, a populated `usage` object — right
before `[DONE]`; omitted by default, matching real OpenAI's own
streaming behavior.

**Architecture:** `membrane_session_generate()`'s token callback is
push-based (called synchronously per-token from inside the decode
loop); cpp-httplib's chunked content provider is pull-based. Bridged by
a dedicated generation worker thread (owns the push side) plus a
**bounded** (64-entry) thread-safe queue the HTTP connection thread's
own content-provider callback pulls from — the first token can reach
the client before generation finishes; a slow client's own queue
capacity naturally backpressures the worker thread rather than letting
memory grow unboundedly.

**Cancellation:** if the client disconnects mid-stream, the HTTP
connection thread detects it (`cpp-httplib`'s own real socket-liveness
check) and sets a shared `std::atomic<bool>` cancel flag — the *only*
concept the runtime core itself understands ("the caller asked to
stop"; the decode loop, `run_generation()` in
`tools/membrane-llama-runtime/decode_loop.h`, has no HTTP/socket
awareness of any kind). Generation genuinely stops within one decode
step of the flag being set — confirmed directly: a disconnected
client's own server process showed **zero** additional CPU time
consumed in the following two seconds, versus continuing to burn CPU
generating the rest of a 200-token request nobody was reading. A
following request (streaming or not) to the same server works
correctly immediately afterward — no leaked lock, no leaked thread, no
corrupted session state.

**UTF-8 safety:** a single generated token's own bytes do not always
align to a complete UTF-8 character (common for CJK/emoji/accented
text) — an internal accumulator holds back an incomplete trailing byte
sequence (at most 3 bytes) until it completes, rather than ever
emitting a truncated character over the wire. Verified with real
multi-byte content (Japanese, café's `é`, an emoji) through a real
model: every SSE `data:` payload parsed as valid UTF-8 JSON, and the
reconstructed text matched the model's real output exactly.

**Errors after streaming has begun:** everything that can fail with a
normal JSON status-code error (parse, request shape, unknown model, no
usable chat template, model load failure) happens *before* headers
commit to `text/event-stream`. A failure from generation itself, which
can only happen after that point, becomes a terminal SSE event instead
— `data: {"error": {"code": "...", "message": "..."}}` followed by
`data: [DONE]` — never a crash, never a silently truncated stream.

**Real evidence:** `results/background-service/validation.json`'s
`server_streaming` section — real curl wire-format capture, real
disconnect-cancellation CPU-time proof, real multilingual/emoji
content, and a real OpenAI Python SDK streaming session
(`client.chat.completions.create(..., stream=True)`, real incremental
chunks, correct `finish_reason`). `test_stream_queue.cpp` (real
multi-threaded producer/consumer tests, TSan-clean) and
`test_utf8_stream.cpp` (12 pure unit tests, including the exact 4-byte
emoji edge case that caught a real off-by-one bug during development)
cover the underlying primitives in CI.

## Default model (PR B1)

`membrane model use NAME` sets a persistent `default_model` in
`~/.config/membrane/server.json` (Section 9 of the Mega Phase B task).
It never forces a model to load — the server still starts "healthy, no
model loaded" either way — it only changes what an omitted `"model"`
field in a chat request falls back to. Takes effect on the next
`membrane serve` invocation or `membrane service restart`; an
already-running process does not pick it up live.

## Model cache policy

One active model at a time (Section 23 of the task):

- Request model A → load A.
- Next request for A → reuse the already-loaded A (no reload — proven in
  testing: three sequential requests to the same model triggered exactly
  one real weight load).
- Request model B → unload A, load B.

Every request to the currently-loaded model is served on a serialized
path (one internal mutex) — no continuous batching, no concurrent
generation. Correctness first, matching Section 24 of the task.

## Errors

JSON always: `{"error": {"code": "...", "message": "..."}}`. This
table covers every failure that can still change the HTTP status code
— all of them happen before headers commit to `text/event-stream`, so
they apply identically whether or not the request set `stream: true`.
A failure from generation itself, reachable only for a streaming
request (after headers are already committed), is instead a terminal
SSE `data: {"error": {...}}` event — see "Streaming" above.

| HTTP status | code | meaning |
|---|---|---|
| 400 | `INVALID_REQUEST` | malformed JSON or missing/malformed required fields |
| 404 | `MODEL_NOT_FOUND` | the named model is not registered |
| 500 | `CHAT_TEMPLATE_UNAVAILABLE` / `CHAT_TEMPLATE_FAILED` | the model has no usable chat template, or applying it failed |
| 500 | `MODEL_LOAD_FAILED` | the model file could not be loaded |
| 500 | (a real planner reason code) | generation failed after loading (non-streaming only) |
| 503 | `NO_FEASIBLE_CONTEXT` | no context/hardware plan could be resolved (e.g. insufficient host memory) |

## Graceful shutdown

`SIGINT`/`SIGTERM` stop the listener, join it, free the loaded model
(if any) and the llama backend, then exit 0.

## Not implemented

- `POST /v1/completions` (non-chat).
- Sampling beyond greedy decoding (temperature/top_p/etc. are accepted
  and ignored, never faked).
- Multiple simultaneously-resident models.
- No explicit model-lifecycle state machine or bounded pending-request
  queue yet — the existing full-request-serialization mutex (unchanged
  since PR A3) also serializes streaming requests behind non-streaming
  ones and vice versa; targeted for PR B3.
