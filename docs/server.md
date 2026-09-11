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

**Post-v1 product-polish, prompt 3:** for a human at a terminal (as
opposed to an app/SDK integration), `membrane chat` is the normal way to
talk to this server — an interactive REPL client of the exact same API
this document describes, never a second inference path. See
`docs/chat.md`.

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

- `GET /health` — `{"status":"ok","version":"0.8.0"}` (the real, live
  `MEMBRANE_VERSION` — this example number tracks whatever this repo's
  own `tools/membrane-run/product_cli.h` currently defines).
- `GET /v1/models` — every model currently in the registry
  (`membrane model add`), OpenAI `list` shape.
- `GET /v1/status` — membrane-specific (not an OpenAI endpoint), backs
  the `membrane status` CLI command (PR A4, Section 37 of the task): a
  thin HTTP client against an already-running `serve` instance, never a
  process-management/daemon capability this project doesn't have.
  `{"running":true,"version":"...","endpoint":"http://...",
  "resident_model_limit":2,"resident_models":[{"model":"qwen",
  "state":"ready","pinned":false,"evictable":true,"backend":"CPU",
  "gpu_layers":N,"kv_precision":"native","estimated_model_bytes":N,
  "estimated_kv_bytes":N}, ...],"default_model":"qwen"|null,
  "context_policy":"automatic"}` (Mega Phase E, PR E2: `resident_models`
  replaces the earlier, singular `loaded_model`/`backend`/`gpu_layers`/
  `kv_precision` fields now that more than one model can be resident at
  once — see "Multi-model residency" below; the array is empty, never a
  null placeholder, until at least one model has been loaded).
- `POST /v1/chat/completions` — see below.
- `GET /membrane/v1/capabilities` (PR E3) — real, live capability
  discovery: `{"streaming":true,"stop":true,"tool_calling":false,
  "embeddings":false,"backends":["CPU", ...],"resident_model_limit":2,
  "platform":"linux"|"windows"|"macos"}`. `backends` comes from a real
  device enumeration (the same one the GPU-selection pipeline itself
  uses), never a static list. See `docs/api-v1-stability.md` for the
  frozen contract this and every other endpoint here now falls under.

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
`max_tokens` / `max_completion_tokens` (optional, default 512), `stop`
(optional — see "Stop sequences" below). `stream: true` — see
"Streaming" below.

`temperature`/`top_p`/`presence_penalty`/`frequency_penalty`/`n`/`user`/
`logprobs`/`seed` are all accepted and **ignored** (harmless — none of
them change intended behavior enough to warrant an error; see
`docs/api-contract.md`'s field table) — the underlying decode loop is
greedy-only (argmax) today, no sampling support exists in this project
yet; the response's own `membrane.sampling` field says so explicitly
rather than silently claiming otherwise. `seed` specifically is a real
no-op, not a false claim: greedy decoding with no RNG anywhere is
already fully deterministic given identical inputs.

`tools`/`tool_choice` and `response_format` (anything other than the
default `"text"`) are explicitly **rejected** (`400`), never silently
ignored — see "Not implemented" below for why.

### Stop sequences (`stop`) — PR D7

A string, or an array of up to 4 strings (OpenAI's own real limit — a
longer array is `400 INVALID_REQUEST`, never silently truncated).
Implemented as a REAL early termination, not a post-hoc truncation of
output that was already fully generated: the accumulated response text
is checked against every requested stop string as each new piece of
text becomes available, and if one matches, generation is halted right
there (reusing the same real cancellation mechanism client-disconnect
handling already uses) — the matched stop string itself, and everything
after it, is never included in the response, and never sent to the
client on the streaming path either. `finish_reason` is `"stop"` in
this case. Works identically for both `stream: false` and
`stream: true`. `usage.completion_tokens` can be at most one token
higher than what the client actually received as visible text in one
real, disclosed edge case: the one token whose own decoded text
straddles the match point is still counted as generated, even though
only the portion before the match was ever sent.

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
field in a chat request falls back to. The **config file itself** is
still only read once, at `membrane serve` startup — an already-running
process never re-reads `default_model` on its own.

Mega Phase D, PR D6 adds a separate, live path on top of this: `membrane
use MODEL` sets `default_model` exactly as above, but if a server is
already running it ALSO triggers a real, immediate model switch via a
new loopback-only admin endpoint (`POST /membrane/v1/models/activate`,
see below) — no restart needed for the *active* model to change, even
though the *config file*'s own read-once behavior is unchanged. See
`docs/model-lifecycle.md` for the full default-vs-active contract.

### `POST /membrane/v1/models/activate` (PR D6, internal/admin only)

Not part of the OpenAI-compatible surface (`/v1/...`) — a MEMBRANE-
specific, loopback-only admin route `membrane use` calls internally,
undocumented for third-party clients. Body: `{"model": "NAME"}` (a
registered name). A thin wrapper around the exact same `acquire_model_
slot()` function (Mega Phase E, PR E2 — renamed from PR E1's own
`acquire_model_slot()` when it was generalized to choose among several
resident slots) `POST /v1/chat/completions` itself uses — never a second
switch implementation, so it inherits that function's own idempotence
(already-active is a no-op, reported as `{"already_active": true}`) and
failure-recovery guarantees (see below) for free. Never evicts a model
out from under an in-progress generation, and never evicts a pinned one
— see "Multi-model residency" below for the real selection/eviction
policy this endpoint now goes through.

### `POST /membrane/v1/models/pin` / `.../unpin` (PR E2, internal/admin only)

Same admin namespace as `.../activate` above. Body: `{"model": "NAME"}`.
Pinning only ever applies to a model that is **already resident right
now** — it is not a standing preference remembered for later (that is
`default_model`'s job, above); pinning a name that is not currently
resident is a real `409 MODEL_NOT_RESIDENT`, never silently queued.
Success: `{"ok":true,"model":"NAME","pinned":true|false}`. Pinning
never violates memory safety on its own — see "Multi-model residency"
below.

## Model cache policy

Multi-model residency (Mega Phase E, PR E2) — up to
`MEMBRANE_MAX_RESIDENT_MODELS` (default 2, hard-capped at 8;
deliberately not a documented `server_config.h` setting — same "keep
minimal" convention as `MEMBRANE_MAX_CONCURRENT_DECODE`) DIFFERENT
models may be resident (loaded) at the same time, each in its own
independent slot with its own `llama_context`/KV cache/lock — a real
generalization of PR E1's own single-model design, not a second one
alongside it. Selection policy (`residency_planner.h`'s own pure
`membrane_plan_residency()`, unit-tested against synthetic slot states
in `test_residency_planner.cpp`; `acquire_model_slot()` in `server.cpp`
is the only real caller):

- **Hot switch**: requesting an already-resident model never reloads
  from disk, regardless of which OTHER models are also resident.
- **Free slot**: if fewer than the limit are resident, a genuinely new
  model just fills an empty slot.
- **LRU eviction**: once every slot is occupied, the LEAST-recently-used
  resident model among the ones that are neither pinned nor actively
  generating is evicted (a real `membrane_model_close()`, draining any
  in-flight decode against it first — bounded ~5s, `503
  MODEL_SWITCH_BUSY` on timeout, exactly PR E1's own drain-wait contract,
  now scoped to one slot instead of the server's only one).
- **Never evicts a pinned or actively-generating model.** If every
  resident slot is pinned and/or generating (no free slot, nothing
  evictable), the request fails fast with `503 RESIDENCY_EXHAUSTED` —
  deliberately BEFORE any drain-wait, since waiting up to 5s for a
  pinned model to become "evictable" would be pointless (it never will).
- **No hidden background thrashing**: eviction only ever happens
  synchronously, in direct response to one real incoming request naming
  a not-yet-resident model — never on a timer, never speculatively.

Real, real-model evidence (two SmolLM2-135M-Instruct sessions registered
under two different names — see the script's own top comment for why
the SAME small file, not two different ones, was used on this project's
own memory-constrained dev host): both resident simultaneously, each
independently servable; hot-switching back to the first reported
`already_active: true`; a slow generation in-flight correctly made a
competing switch fail `RESIDENCY_EXHAUSTED` (single-slot-forced test)
while the generation itself completed normally, and the same switch
then succeeded once idle; pinning the resident model made a competing
switch fail `RESIDENCY_EXHAUSTED`, and unpinning it let that same
switch succeed. `scripts/verify-multi-model-residency.py`,
`results/multi-model-residency/validation.json`.

`GET /v1/status`'s own `resident_models` array (see above) exposes
`pinned`/`evictable`/`state` per resident slot; `resident_model_limit`
is the real, current `MEMBRANE_MAX_RESIDENT_MODELS` value.

### Concurrent decode (Mega Phase E, PR E1)

Requests against the SAME already-loaded model now decode CONCURRENTLY,
up to a bounded limit (`MEMBRANE_MAX_CONCURRENT_DECODE`, default 2 —
deliberately conservative, real-memory-motivated: every request still
opens its own independent `llama_context`/KV cache, this project's
unchanged "persistent model, new context per request" architecture, so
N concurrent decodes means N independent KV caches resident at once).
Past that bound, an already-admitted request waits its turn (no
starvation proven for a short request behind a long one — see
`docs/soak-and-concurrency-testing.md`'s own PR E1 section for the real
evidence) rather than being rejected — `request_admission_gate_t`
(below) is the only thing that ever rejects with `503`.

This is bounded CONCURRENT execution over this project's own existing
per-request-context architecture — deliberately NOT upstream llama.cpp's
own shared-context/`seq_id`-slot continuous-batching design (reviewed
and not adopted: it would require one fixed `n_ctx`/KV-dtype shared
across every concurrent request, which conflicts with this project's own
real, working per-request adaptive planner — see `decode_concurrency.h`'s
own top comment for the full reasoning). Each request operates on its
own copy of the loaded model's session state (never the shared struct
directly), so concurrent requests cannot race on each other's own
planner output (`gpu_layers`/`kv_placement`/etc, reported in each
response's own `"membrane"` block) or corrupt each other's decode.

### Model-lifecycle state machine (PR B3)

Per-slot since Mega Phase E, PR E2 (was the server's only one, PR B3
through PR E1). An explicit state, not "loaded bool + name string" —
`empty` (never
loaded, or cleanly unloaded), `loading`, `ready`, `generating`,
`unloading`, `error`. Reported per resident slot by `/v1/status`'s own
`resident_models[].state` (a slot in state `empty` — never loaded, or
evicted back to empty — is simply omitted from that array, not reported
as a null placeholder). `generating` means "at least one request is
decoding against THIS slot" (PR E1 — it could always mean "exactly one"
before real concurrent decode existed). All transitions happen only
while that slot's own mutex is held, so they are exactly as synchronized
as everything else. `error` is distinct from `empty` — see the recovery
behavior below for when it is reached.

### Model-switch failure recovery (PR B3, Section 31 of the task)

Per-slot since Mega Phase E, PR E2 (also known as eviction, when the
new model is a THIRD name and there is no free slot — see "Multi-model
residency" above). A naive "unload A, then try to load B" would leave
that slot with NO
model at all if B's own load fails, even though A was working a moment
ago. Instead: if B fails to load, the server automatically attempts to
**reload A into the same slot** before giving up. The client's own
request for B is still correctly reported as a failure (its real error
code/message, never silently swapped for a misleading success) — only
that slot's own resting state improves, ending up back at `ready` on A
rather than `empty`. Only if BOTH B's load and A's own restore attempt
fail does that slot end up in the explicit `error` state.

Real evidence: with model A already loaded, a real switch attempt to an
oversized model correctly failed with `503 NO_FEASIBLE_CONTEXT` (the
host genuinely could not fit it) — `/v1/status` immediately afterward
still reported A resident and `ready`, and a following real request to
A succeeded normally. Reproduced twice, independently, with two
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
ever-growing queue. Since PR E1, up to `MEMBRANE_MAX_CONCURRENT_DECODE`
(default 2) of those 8 admitted requests can be genuinely decoding at
once (see "Concurrent decode" above) — 8 remains generous headroom above
that real concurrent-decode bound, not a claim that all 8 ever run
simultaneously; this bound exists to fail closed under real overload,
not to constrain ordinary use.

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
| 500 | (a real planner reason code, or `GENERATION_FAILED` if none was set) | generation failed after loading (non-streaming only) |
| 503 | `NO_FEASIBLE_CONTEXT` | no context/hardware plan could be resolved (e.g. insufficient host memory) |
| 503 | `SERVER_BUSY` | too many chat completion requests are already in flight (PR B3, "Bounded request admission" above) — includes a `Retry-After` header |
| 400 | `UNSUPPORTED_TOOL_CALLING` | the request included `"tools"`/`"tool_choice"` (PR D7, Section 15: MEMBRANE does not execute tools — rejected explicitly rather than silently ignored, since silently dropping the schema would mislead a client into expecting a tool call back) |
| 400 | `UNSUPPORTED_RESPONSE_FORMAT` | `"response_format"` requested anything other than the default (`"text"`, or the field omitted) — this server has no constrained-decoding/JSON-mode path that could actually honor it (PR D7, Section 14) |
| 400 | `CTX_TOO_SMALL_FOR_PROMPT` | the prompt is far larger (raw byte length, a cheap pre-tokenization check) than the model's own real maximum context — rejected before an expensive real tokenization attempt (PR D8, Section 15: a real, disclosed PR D7 finding that an extremely oversized prompt could make the server unresponsive for minutes on a memory-constrained host, root-caused and fixed this phase) |
| 503 | `MODEL_SWITCH_BUSY` | a switch to a different model was requested, but a request against the currently-loaded model is still decoding and didn't drain within a bounded (~5s) wait (PR E1) — the current model is left fully intact and still serving; retry shortly |
| 503 | `RESIDENCY_EXHAUSTED` | multi-model residency (PR E2): every resident slot is either pinned or actively generating, so no slot is available to load a not-yet-resident model — see "Multi-model residency" above |
| 409 | `MODEL_NOT_RESIDENT` | multi-model residency (PR E2): `POST /membrane/v1/models/pin`/`.../unpin` was called for a model that is not currently resident — pin/unpin only applies to an already-loaded model |

## Client integration (PR B4)

Expanded in PR D7 (Mega Phase D) with real Node.js SDK evidence and the
`created`-field fix described below.

MEMBRANE ships no client of its own — any OpenAI-compatible client
works against `http://127.0.0.1:8642/v1` with a placeholder API key
(this server has no authentication, see "Security scope" above). See
`docs/client-compatibility.md` for the full, current, honestly-labeled
matrix (this section stays a summary, that document is authoritative).

**Real, tested evidence:** the official Python `openai` SDK (3.8.0 as
of PR D7) and Node.js `openai` npm package (7.10.0, PR D7) — real
`client.models.list()`, real `chat.completions.create()` both
non-streaming and streaming, real usage/finish_reason parsing, real
incremental chunks, real `stop`-sequence early termination, real
`tools`/`response_format` rejection, all via each SDK's own typed
interface, never manual JSON handling. PR D7 also found and fixed a
real Python-SDK compatibility bug this way: `GET /v1/models` was
missing `created` (a field `openai.types.model.Model` declares
REQUIRED) — `client.models.list()` genuinely failed before the fix. See
`results/client-compatibility/validation.json`.

**Configuration instructions (not independently validated end-to-end):**
the following are standard OpenAI-compatible integrations that should
work against this server based on its own protocol compliance above,
but were not run end-to-end as of PR D7 — this shared development host
was under real, severe memory pressure throughout PR D7 (as low as
~150-350 MiB available RAM), with two unrelated Docker containers for a
different project already running and not to be disturbed, so starting
an additional real container (Open WebUI) was judged unsafe to attempt
without risking those other services. Disclosed as configuration
guidance, not a tested claim, rather than silently skipped or falsely
claimed "tested".

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

- `POST /v1/completions` (legacy, non-chat) and `POST /v1/embeddings` —
  evaluated for PR D7 (Mega Phase D, Section 11 of the task) and
  deliberately not added: no real target client (Python/Node SDKs, Open
  WebUI, Continue) requires either for basic chat/streaming use, and
  `/v1/embeddings` specifically would require a real, separate
  hidden-state extraction path through llama.cpp this project does not
  have yet — never faked from generation logits (Section 11: "do not
  fake embeddings from generation logits").
- Tool calling (`"tools"`/`"tool_choice"`) — evaluated for PR D7
  (Section 15) and explicitly rejected (`UNSUPPORTED_TOOL_CALLING`, see
  the Errors table below), never silently dropped. MEMBRANE does not
  execute tools; accepting the schema and silently ignoring it would
  let a client believe a tool call might come back when one never will.
- `response_format` beyond the default (`"text"`, or the field omitted)
  — evaluated for PR D7 (Section 14) and explicitly rejected
  (`UNSUPPORTED_RESPONSE_FORMAT`) rather than silently ignored: this
  server has no constrained-decoding/JSON-mode grammar path that could
  actually honor `{"type": "json_object"}`, and pretending it works via
  prompt injection alone would be dishonest.
- Sampling beyond greedy decoding (temperature/top_p/etc. are accepted
  and ignored, never faked). `seed` is likewise accepted and ignored —
  not a false determinism claim: greedy decoding with no RNG anywhere
  in the decode loop is already fully deterministic given identical
  model/prompt/parameters, a stronger guarantee than seed-based
  determinism alone would provide.
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
