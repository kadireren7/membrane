# Client compatibility

Mega Phase C, PR C3. `membrane serve` speaks an OpenAI-compatible
subset (`docs/api-contract.md`) at `http://127.0.0.1:8642/v1` with a
placeholder API key (no authentication, see `docs/server.md`'s
"Security scope"). This table distinguishes what was actually run
against a real server from what is disclosed configuration guidance.

| Client | Status | Evidence |
|---|---|---|
| Python `openai` SDK | **Real, tested** | `client.models.list()`, `client.chat.completions.create()` (non-streaming, PR A4; streaming, PR B2) — real generation, real usage/finish_reason parsing, correct typed `NotFoundError` on 404. See `results/runtime-service/validation.json`, `results/background-service/validation.json`. |
| Node.js built-in `fetch` | **Real, tested (PR C3)** | `scripts/node-client-check.mjs` — real `GET /health`, real `GET /v1/models` (OpenAI `list` shape), real `POST /v1/chat/completions` against an unregistered model confirming the real `404 MODEL_NOT_FOUND` shape. No npm dependency (Node's own built-in `fetch`), kept intentionally lightweight for this project's own memory-constrained dev host. See `results/product-hardening/v0.4-validation.json`. |
| `curl` | **Real, tested** | Used throughout this project's own real evidence gathering (streaming wire-format capture, health checks, CI's own packaging-smoke job). Always available in this project's own dev/CI environments; a genuinely minimal container may lack it (Mega Phase C's own first-product audit, `results/product-onboarding/validation.json`) — `membrane doctor`/`membrane status` use their own built-in HTTP client specifically so MEMBRANE's own tooling never assumes `curl` is present. |
| Open WebUI | Disclosed, not independently tested | Standard `OPENAI_API_BASE_URL` configuration should work per this server's own protocol compliance above — not run end-to-end due to real Docker resource pressure on the dev host at the time (`docs/server.md`'s own disclosure, PR B4). Unchanged this phase. |
| OpenAI-compatible editor plugins (e.g. Continue) | Disclosed, not independently tested | Same reasoning as Open WebUI — thin wrappers around the same already-validated chat-completions/streaming conventions, but not itself a substitute for having actually run one. |

No new client was independently tested this phase beyond adding the
real Node.js check — Open WebUI/editor-plugin validation remains
future work, not silently upgraded to "tested" without a real run.
