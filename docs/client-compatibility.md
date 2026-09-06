# Client compatibility

Mega Phase D, PR D7. This document records which real third-party
clients have actually been driven against a real, running `membrane
serve` instance — never "should work because the API looks similar."

Evidence states used below (see `results/client-compatibility/
validation.json` for the full detail behind each row):

- **VALIDATED_REAL** — a real client library/application was driven
  against a real, running server and its real requests/responses were
  observed.
- **CONFIGURATION_ONLY** — the client's own configuration format/
  protocol expectations were checked against this server's real
  behavior, but the client itself was not actually run end-to-end
  (disclosed reason given).
- **NOT_VALIDATED** — not attempted this phase, reason given.
- **UNSUPPORTED** — a real, confirmed incompatibility.

## Matrix

| Client | Version | Model listing | Chat | Streaming | Model switch | Notes | Status |
|---|---|---|---|---|---|---|---|
| Python `openai` SDK | 3.8.0 (PR D7) | ✅ | ✅ | ✅ | ✅ (real, via a normal `model` field change) | Validated across multiple phases (non-streaming: PR A4; streaming: PR B2; this phase: real `stop`/`tools`/`response_format` checks, real model-switch proof). Found and fixed a real bug this phase: `GET /v1/models` was missing `created` (a REQUIRED field in this SDK's `Model` type) — `client.models.list()` genuinely failed pydantic validation before the fix. See `results/runtime-service/validation.json`, `results/background-service/validation.json`, `results/client-compatibility/validation.json`. | VALIDATED_REAL |
| Node.js `openai` SDK (npm) | 7.10.0 (PR D7, new) | ✅ | ✅ | ✅ | not separately re-tested (same server, same endpoint — no Node-specific switch path exists) | Real async-iterator streaming consumption (`for await`), real `APIError.status`/`.error` shape checks, real `stop`-sequence proof. `scripts/client-compat/test-openai-node.mjs`. | VALIDATED_REAL |
| Node.js built-in `fetch` (no npm install) | n/a (PR C3, extended PR D7) | ✅ | ✅ (404 path only, no real model) | — | — | `scripts/node-client-check.mjs` — deliberately lightweight (no npm dependency) for CI/a memory-constrained dev host; extended this phase with `tools`/`response_format`/`stop`-shape/Authorization-header checks, all of which happen before model resolution so no real model is needed. Wired into a real CI job this phase (`packaging-smoke`, against a real running server). See `results/product-hardening/v0.4-validation.json`. | VALIDATED_REAL |
| curl / raw HTTP | n/a | ✅ | ✅ | ✅ (SSE) | ✅ | Used throughout this project's own real evidence gathering, including this phase's own real concurrency-bug hunt below — the lowest-level real client, no SDK abstraction to hide a wire-format bug. | VALIDATED_REAL |
| Open WebUI | — | — | — | — | — | Not attempted (this phase or prior ones): this shared dev host has been under real memory pressure across multiple phases (as low as ~200-300 MiB available, 5-7 GiB of swap in active use this phase — see `results/client-compatibility/validation.json`'s own `host_memory_pressure` note), and this phase specifically had two unrelated Docker containers (`postgres`, `redis`, belonging to a different project) already running that must not be disturbed. Pulling/running an Open WebUI container on top of that would risk destabilizing the shared host for no proportionate benefit, since the underlying OpenAI-compatible surface it would exercise (model listing, non-stream chat, streaming chat) is already VALIDATED_REAL via three independent real clients above. Deferred, not blocked — see Section 46/D8 handoff. | NOT_VALIDATED |
| Continue (VS Code extension) | — | — | — | — | — | No VS Code environment is available in this CLI-only sandbox. The real config format (`config.yaml`'s `models:` list with `apiBase`/`apiKey`/`provider: openai`) was checked against this server's real, confirmed behavior (loopback base URL + tolerated dummy Bearer token + standard `/v1/chat/completions` + `/v1/models`) — Continue's own documented OpenAI-compatible provider expects exactly this shape, so it is expected to work, but this was never run end-to-end. | CONFIGURATION_ONLY |

## Base URL

```
http://127.0.0.1:8642/v1
```

Point any OpenAI-compatible client's `base_url`/`apiBase` at exactly
this — the client library itself appends `/models`, `/chat/completions`
etc. Every real client tested above used this exact shape correctly,
never accidentally doubling the version segment (a real, common
client-library footgun this project's own docs call out explicitly).

## Model IDs

The `id` field in `GET /v1/models`, and the `"model"` field a client
sends, are the SAME string: the real registry name (`membrane model
list`/`membrane use NAME`). No filesystem path is ever exposed, and no
second "catalog alias" is exposed either — a model installed via
`membrane use qwen2.5:7b` is addressed as `"model": "qwen2.5:7b"` in a
chat request with no translation step, because the registry name IS
what `membrane use` registered it under (see `docs/model-lifecycle.md`).

## What "OpenAI-compatible" means here

This server implements an **OpenAI-compatible chat API subset** — not
the full OpenAI API surface. See `docs/api-contract.md` for the exact
field-by-field table. No `/v1/completions` (legacy), no `/v1/embeddings`,
no tool calling, no `response_format` beyond the default. These are
deliberate, disclosed scope decisions (see `docs/api-contract.md`'s own
"Not implemented" section for why), not gaps discovered by accident.
