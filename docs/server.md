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
generation, streaming included. Correctness first, matching Section 24
of the task. A concurrent request for a DIFFERENT model simply waits on
that same mutex — a switch is never a special case, it is the same
serialization every request already goes through.

### Model-lifecycle state machine (PR B3)

An explicit state, not "loaded bool + name string" — `empty` (never
loaded, or cleanly unloaded), `loading`, `ready`, `generating`,
`unloading`, `error`. Reported by `/v1/status` as `model_state`. All
transitions happen only while the same mutex above is held, so they are
exactly as synchronized as everything else. `error` is distinct from
`empty` — see the recovery behavior below for when it is reached.

### Model-switch failure recovery (PR B3, Section 31 of the task)

A naive "unload A, then try to load B" would leave the server with NO
model at all if B's own load fails, even though A was working a moment
ago. Instead: if B fails to load, the server automatically attempts to
**reload A** before giving up. The client's own request for B is still
correctly reported as a failure (its real error code/message, never
silently swapped for a misleading success) — only the server's own
resting state improves, ending up back at `ready` on A rather than
`empty`. Only if BOTH B's load and A's own restore attempt fail does the
server end up in the explicit `error` state.

Real evidence: with model A already loaded, a real switch attempt to an
oversized model correctly failed with `503 NO_FEASIBLE_CONTEXT` (the
host genuinely could not fit it) — `/v1/status` immediately afterward
still reported A as `loaded_model`/`ready`, and a following real request
to A succeeded normally. Reproduced twice, independently, with two
different oversized models. See
`results/background-service/validation.json`.

### Memory revalidation (Section 33 of the task)

Host memory is read **fresh, every time** a model is (re)loaded or
switched — never a cached snapshot from server startup or from the
previously-loaded model. A switch attempted after available memory has
changed (another process using more RAM, or this same server having
just freed the previous model) always sees that current reality.

### Bounded request admission (Section 29 of the task)

`POST /v1/chat/completions` admits at most 8 concurrent requests (a
single atomic counter, checked before any other work, including before
the registry lookup) — past that bound, a request gets an immediate
`503 SERVER_BUSY` with a `Retry-After` header, rather than joining an
ever-growing queue of threads blocked on the generation mutex. Given
this server's own full serialization (above), 8 is generous headroom
above the "1 active generation, the rest waiting" reality — this bound
exists to fail closed under real overload, not to constrain ordinary
use.

### Model registry hot-reload (Section 32 of the task)

The registry is polled (a single `stat()`, not a re-read) on every
`GET /v1/models` and `POST /v1/chat/completions` call; a real mtime
change triggers a real reload. `membrane model add`/`remove` while the
server is already running becomes visible on the very next request —
no restart needed. Real evidence: a model added via the real CLI while
a server was already serving requests appeared in `GET /v1/models`
without any restart. A registry file that fails to parse (e.g. caught
mid-write) is silently ignored — the previous, still-good registry is
kept rather than corrupting server state over a transient race.

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
| 503 | `SERVER_BUSY` | too many chat completion requests are already in flight (PR B3, "Bounded request admission" above) — includes a `Retry-After` header |

## Client integration (PR B4)

MEMBRANE ships no client of its own — any OpenAI-compatible client
works against `http://127.0.0.1:8642/v1` with a placeholder API key
(this server has no authentication, see "Security scope" above).

**Real, tested evidence:** the official Python `openai` SDK —
`client.models.list()`, `client.chat.completions.create()` both
non-streaming (Mega Phase A, PR A4) and streaming
(`stream=True`, PR B2) — real generation, real usage/finish_reason
parsing, real incremental chunks, all via the SDK's own typed
interface, never manual JSON handling. See
`results/background-service/validation.json`/
`results/runtime-service/validation.json`.

**Configuration instructions (not independently validated this
phase):** the following are standard OpenAI-compatible integrations
that should work against this server based on its own protocol
compliance above, but were not run end-to-end this phase — real Docker
resource pressure on the development host at the time (other unrelated
services already running, tight free memory) made starting an
additional real container unsafe to attempt without risking those
other services, so this is disclosed as configuration guidance, not a
tested claim, rather than silently skipped or falsely claimed "tested".

- **Open WebUI**: `docker run` it with its own `OPENAI_API_BASE_URL`
  environment variable set to `http://127.0.0.1:8642/v1` (or
  `http://host.docker.internal:8642/v1` if Open WebUI itself runs
  inside Docker and needs to reach the host) and any placeholder value
  for `OPENAI_API_KEY`. Open WebUI should then list and chat with
  whatever model(s) `membrane model list` shows.
- **An OpenAI-compatible editor plugin** (e.g. Continue for VS
  Code/JetBrains): configure a custom OpenAI-compatible provider with
  `apiBase` (or equivalent) set to `http://127.0.0.1:8642/v1`, model
  name matching a `membrane model add` name, and any placeholder API
  key. Since these plugins are themselves typically thin wrappers
  around the same OpenAI client conventions already validated above
  (chat completions, streaming), real incompatibility would be
  surprising, but is not itself a substitute for having actually run
  one against this server.

## Graceful shutdown

`SIGINT`/`SIGTERM` stop the listener, join it, free the loaded model
(if any) and the llama backend, then exit 0.

## Not implemented

- `POST /v1/completions` (non-chat).
- Sampling beyond greedy decoding (temperature/top_p/etc. are accepted
  and ignored, never faked).
- Multiple simultaneously-resident models.
- An idle-model timeout (unload after N minutes of no requests) —
  evaluated for PR B3 and deliberately deferred: no clear evidence yet
  that it is worth the added complexity for this project's own current
  usage pattern (a single long-lived local backend, not a multi-tenant
  service under memory pressure from many idle models). Candidate for
  v0.5+.
- A queued (not yet generating) request cannot be individually
  cancelled by its own client disconnecting — cancellation (PR B2) only
  covers a request that is ALREADY streaming/generating. A request still
  waiting on the generation mutex has no separate polling point to
  detect a dead client; implementing one would need a materially more
  complex architecture for a rare case (this server has no continuous
  batching, so the wait is normally short). Evaluated for PR B3,
  deliberately deferred with this disclosure rather than silently
  assumed to work.
