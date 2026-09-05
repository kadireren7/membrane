# API contract

Mega Phase C, PR C3. This document is the top-level contract summary;
`docs/server.md` remains the detailed, authoritative reference for
every endpoint, request/response shape, and the full error table —
this page never duplicates that table, it freezes and versions it.

## Scope: a compatible subset, never "the complete OpenAI API"

`membrane serve` implements an **OpenAI-compatible subset**: enough of
the `/v1/chat/completions`, `/v1/models` surface for real, unmodified
OpenAI SDKs and OpenAI-protocol clients to work against it (see
`docs/client-compatibility.md`). It is not, and is not claimed to be, a
complete reimplementation of OpenAI's API — `docs/server.md`'s own "Not
implemented" section lists the concrete, disclosed gaps (no
`/v1/completions`, no sampling beyond greedy decoding, no multi-model
residency, no idle-model timeout). This scope statement is itself a
regression-guarded claim (`scripts/verify-api-contract.py`): no file in
this repo may claim "complete"/"full" OpenAI API support.

## Versioning policy

- The `/v1/` prefix mirrors OpenAI's own versioning convention. As long
  as MEMBRANE targets OpenAI-protocol compatibility, `/v1/` stays
  additive-only: new optional request fields, new response fields
  inside the existing `membrane` extension object, and new error codes
  may be added without a version bump. A genuinely breaking change to
  an existing field's meaning or an existing endpoint's status-code
  contract would require a new prefix (e.g. `/v2/`) rather than
  silently changing `/v1/` behavior out from under existing clients.
- MEMBRANE's own product version (`MEMBRANE_VERSION`,
  `tools/membrane-run/product_cli.h`) and the API surface version are
  deliberately independent — a MEMBRANE release can ship product-level
  changes (packaging, CLI, docs) with zero `/v1/` API changes, as this
  phase's own PRs C1/C2 did.
- No `/v1/` breaking change has happened yet; this policy exists so a
  future one is deliberate and documented, not accidental.

## MEMBRANE extensions (non-OpenAI-standard surface)

Never silently mixed into the standard OpenAI shape — always additive
or clearly separate:

- The `membrane` object inside every chat-completion response
  (`context`, `gpu_layers`, `kv_precision`, `kv_placement`, `sampling`)
  — see `docs/server.md`'s "`POST /v1/chat/completions`" section.
- `GET /v1/status` — a MEMBRANE-specific, non-OpenAI endpoint.
- `Retry-After` header on a `503 SERVER_BUSY` response specifically
  (not on every 503 — see "Frozen error contract" below).

## Frozen error contract

Every error response is `{"error": {"code": "...", "message": "..."}}`.
The full, current table lives in `docs/server.md`'s own "Errors"
section — this PR adds a real, automated completeness check
(`scripts/verify-api-contract.py`) that parses every literal error-code
emission site in `tools/membrane/server.cpp` and confirms each one
still appears in that table, and vice versa, so the two can never
silently drift apart again. "Frozen" means: an existing code's HTTP
status and meaning will not change; a genuinely new failure mode gets a
new code, never an existing code's silently repurposed meaning.

One row is intentionally open-ended and documented as such: a
generation-time failure (500) can carry a real, dynamic planner reason
code (from `tools/membrane-run/product_cli.h`'s own `MEMBRANE_REASON_*`
constants) rather than one fixed string — `GENERATION_FAILED` is only
the fallback when no more specific reason code is available. This is
disclosed in `docs/server.md`'s table (`(a real planner reason code)`)
rather than enumerated as a closed set, since the runtime's own set of
reason codes is expected to grow independently of the HTTP contract.

**Test coverage**: `INVALID_REQUEST`, `MODEL_NOT_FOUND`, and
`SERVER_BUSY` are exercised by real, ctest-registered, CI-safe
requests (`test_server.cpp`, no GGUF model needed). `CHAT_TEMPLATE_
UNAVAILABLE`/`CHAT_TEMPLATE_FAILED`/`MODEL_LOAD_FAILED`/
`NO_FEASIBLE_CONTEXT`/`GENERATION_FAILED` all require a real model to
trigger and are validated via real, manual dev-host testing (this
project's established "no real GGUF model in CI" constraint, see
`results/product-hardening/v0.4-validation.json`).

## v0.4.0 versioning decision

Mega Phase C ships as **v0.4.0**, not v1.0. Rationale: v1.0 implies an
API/product stability promise this project cannot yet honestly make —
`docs/v1.0-readiness.md` (written at the end of this mega-phase)
classifies real-world readiness explicitly rather than assuming it. A
0.x.0 minor bump accurately reflects that this phase adds real product
maturity (onboarding, packaging, hardening) without claiming the
higher bar v1.0 implies.
