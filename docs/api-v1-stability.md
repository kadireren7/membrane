# API v1 stability contract

Mega Phase E, PR E3 (Section 18 of the task). This document is the
**authoritative stability promise** for MEMBRANE's HTTP API going into
v1.0 — `docs/api-contract.md` remains the detailed field-by-field/
versioning-policy reference and `docs/server.md` the full endpoint/
error-table reference; this page states, in one place, exactly what a
client may rely on staying true across a patch/minor MEMBRANE release.

## Stable endpoint paths

These paths, methods, and their real, current behavior are frozen for
v1.0 and beyond (additive changes only — see "Additive-change policy"
below):

- `GET /v1/models` — OpenAI `list` shape.
- `POST /v1/chat/completions` — OpenAI-compatible chat completions
  (streaming and non-streaming).
- `GET /health` — liveness probe, `{"status":"ok","version":"..."}`.

MEMBRANE-specific (never claimed as part of the OpenAI-compatible
surface, always under a namespace that says so):

- `GET /v1/status` — MEMBRANE's own status surface (kept under `/v1/`
  for historical reasons predating this contract — grandfathered in,
  not a claim that it is an OpenAI endpoint; see `docs/server.md`).
- `GET /membrane/v1/capabilities` (PR E3, new) — real, live capability
  discovery (`streaming`/`stop`/`tool_calling`/`embeddings`/`backends`/
  `resident_model_limit`/`platform`).
- `POST /membrane/v1/models/activate` / `.../pin` / `.../unpin` —
  loopback-only admin routes `membrane use`/`membrane model pin`/
  `unpin` call internally. Not intended for third-party clients
  (undocumented in `docs/client-compatibility.md`'s own client matrix
  on purpose), but their request/response shape is still covered by
  this same additive-only policy once a client does depend on them.

Not implemented, and not planned to appear at these exact paths without
a deliberate decision recorded here first: `POST /v1/completions`
(legacy, non-chat), `POST /v1/embeddings`.

## Stable error shape

Every error response is, and will remain, exactly:

```json
{"error": {"code": "SCREAMING_SNAKE_CASE_STRING", "message": "human-readable text"}}
```

- `error.code` is always present, always a string, always
  `SCREAMING_SNAKE_CASE`. A client may safely switch on it.
- `error.message` is always present, always a string, intended for a
  human, not for programmatic matching (its exact wording may change
  between releases even when `error.code` does not).
- The full, current code table lives in `docs/server.md`'s "Errors"
  section. `scripts/verify-api-contract.py` makes this genuinely
  machine-checked, bidirectionally (Mega Phase E, PR E3, closing a real
  gap the PR C3-era version of this check had — see that script's own
  top comment): every literal error code `server.cpp` can actually
  emit must appear in the docs table, AND every code the docs table
  lists must actually exist in `server.cpp`. Neither direction can
  silently drift from the other again.

## Stable required fields

A client may rely on these fields always being present with the stated
type, for every request shape this API accepts today:

- Request: `POST /v1/chat/completions` requires `messages` (non-empty
  array of `{role, content}` strings) and `model` (string) UNLESS a
  server-side `default_model` is configured, in which case an omitted
  `model` falls back to it (Mega Phase B, Section 9).
- Response (non-streaming, success): `id`, `object`, `created`,
  `model`, `choices[0].message.{role,content}`,
  `choices[0].finish_reason`, `usage.{prompt_tokens,completion_tokens,
  total_tokens}`, and the additive `membrane` object (`context`,
  `gpu_layers`, `kv_precision`, `kv_placement`, `sampling`).
- `GET /v1/models` response: `object`, and `data[].{id,object,created,
  owned_by}` for every entry — `created` specifically is REQUIRED
  because the official Python `openai` SDK's own `Model` type declares
  it required (a real compatibility bug found and fixed in Mega Phase
  D, PR D7 — see `docs/client-compatibility.md`).

## Additive-change policy

Mirrors `docs/api-contract.md`'s own versioning policy (that document
remains authoritative on this point; restated here for completeness):
`/v1/` only ever grows — new optional request fields, new response
fields inside the existing `membrane` extension object, and new error
codes may be added without a version bump or client breakage. A
genuinely breaking change to an existing field's meaning, an existing
endpoint's status-code contract, or the error shape itself would
require a new prefix (e.g. `/v2/`), never a silent change to `/v1/`
behavior underneath an existing client. No such breaking change has
happened, or is planned.

## Deprecation policy

No endpoint or field has been deprecated as of v1.0. If one ever is:

1. It is documented here, in this file, with the MEMBRANE version it
   was first marked deprecated in and the earliest version it may be
   removed in (never less than one full minor version of runway).
2. It continues to function identically until actually removed — a
   deprecation notice is never itself a behavior change.
3. Removal is itself an additive-change-policy exception and follows
   the same "new prefix, never silently changed" rule above if it would
   otherwise break an existing client mid-`/v1/`.

## Patch/minor compatibility expectations

- A PATCH release (`1.0.x`) never changes any field/endpoint covered by
  this document.
- A MINOR release (`1.x.0`) may only ADD (new optional fields, new
  endpoints, new error codes) — never remove or repurpose something
  already covered by this document.
- A MAJOR release could, in principle, break this contract — none is
  planned, and doing so would require its own dedicated decision
  document, not a routine changelog entry.

## Model ID stability (Section 25 of the task)

A client never needs to learn a separate "catalog ID" vs. "registry
ID" vs. "API ID" — there is exactly ONE canonical identifier: the
registry name a model is added/installed under (`membrane model add
NAME PATH` / `membrane use NAME`). That same string is:

- the `"model"` field a chat request sends,
- the `id` field `GET /v1/models` returns for it,
- the name every CLI subcommand and admin endpoint (`activate`/`pin`/
  `unpin`) addresses it by.

No filesystem path is ever exposed over HTTP, and no second alias
exists. See `docs/client-compatibility.md`'s own "Model IDs" section
(unchanged by this phase) and `docs/model-lifecycle.md` for the full
precedence rules around `default_model`.

## Decisions this contract deliberately does NOT relax

Both re-evaluated this phase (Sections 21/22 of the task) and left
unchanged — a real use case, not just "could implement it," is the bar
for revisiting either:

- **Tool calling**: still explicitly rejected (`400
  UNSUPPORTED_TOOL_CALLING`), never silently ignored. No real model
  fixture in this project's own test/dev-model set has been driven
  end-to-end through a real tool-call-capable chat template this
  phase, and MEMBRANE still executes no tools — implementing structured
  tool-call OUTPUT parsing without also being honest about "and then
  nothing happens with it" would be a real, confusing half-feature.
  Revisit only when both a real upstream chat-template path and a real
  end-to-end test against a real tool-call-capable model exist.
- **Embeddings**: still unimplemented, `/v1/embeddings` does not exist.
  No real embedding-extraction code path exists in this project's
  runtime core today (`membrane-run`/`membrane-llama-runtime` are
  chat/generation-shaped, not pooling-shaped) — adding a fake or
  approximate embedding endpoint would produce numerically wrong
  vectors a client could not distinguish from correct ones. Revisit
  only with a real extraction path, correct model semantics, and real
  numerical validation.
- **Structured JSON response format**: still rejected (`400
  UNSUPPORTED_RESPONSE_FORMAT`) for anything but the default `"text"`.
  No constrained-decoding/grammar path exists in the decode loop — a
  prompt-injected "please output JSON" would not be real JSON mode and
  would be dishonest to advertise as one.
