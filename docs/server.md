# MEMBRANE local server

Mega Phase A, PR A3 (server) + PR A4 (compat polish). `membrane serve`
starts a long-lived local HTTP process exposing an OpenAI-compatible
subset on top of the same reusable runtime-session core
(`tools/membrane-run/runtime_session.h`, PR A1) and model registry
(`tools/membrane/registry_core.h`, PR A2) the CLI already uses.
`membrane-run` remains the direct inference entry point, unchanged by
this phase.

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

Supported request fields: `model` (required, a registered name),
`messages` (required, non-empty array of `{role, content}`),
`max_tokens` / `max_completion_tokens` (optional, default 512).
`temperature`/`top_p`/other sampling fields are accepted and **ignored**
— the underlying decode loop is greedy-only (argmax) today, no sampling
support exists in this project yet; the response's own `membrane.sampling`
field says so explicitly rather than silently claiming otherwise.
`stream: true` returns `400 STREAMING_NOT_SUPPORTED` — never faked.

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

JSON always: `{"error": {"code": "...", "message": "..."}}`.

| HTTP status | code | meaning |
|---|---|---|
| 400 | `INVALID_REQUEST` | malformed JSON or missing/malformed required fields |
| 400 | `STREAMING_NOT_SUPPORTED` | `stream: true` was requested |
| 404 | `MODEL_NOT_FOUND` | the named model is not registered |
| 500 | `CHAT_TEMPLATE_UNAVAILABLE` / `CHAT_TEMPLATE_FAILED` | the model has no usable chat template, or applying it failed |
| 500 | `MODEL_LOAD_FAILED` | the model file could not be loaded |
| 500 | (a real planner reason code) | generation failed after loading |
| 503 | `NO_FEASIBLE_CONTEXT` | no context/hardware plan could be resolved (e.g. insufficient host memory) |

## Graceful shutdown

`SIGINT`/`SIGTERM` stop the listener, join it, free the loaded model
(if any) and the llama backend, then exit 0.

## Not implemented this phase

- `POST /v1/completions` (non-chat).
- `stream: true` (SSE streaming) — reevaluated at PR A4 (Section 36 of
  the task) and deliberately deferred again, not merely carried over
  unexamined: the existing generation path (`membrane_session_generate()`,
  PR A1) drives its token callback in a **push** model (the decode loop
  calls back per-token, synchronously, on the request's own worker
  thread), while cpp-httplib's chunked content provider is **pull**-based
  (httplib calls back to ask for the next chunk). Bridging the two safely
  needs a background generation thread plus a synchronized queue and a
  real client-disconnect/cancellation story — a genuine architecture
  change, not a small addition, and out of proportion for a compat-polish
  phase per Section 36's own "if not clean, defer" allowance. `stream:
  true` still returns `400 STREAMING_NOT_SUPPORTED`, never faked.
- Sampling beyond greedy decoding (temperature/top_p/etc. are accepted
  and ignored, never faked).
- Multiple simultaneously-resident models.
