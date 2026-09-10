# MEMBRANE v1.0.0

Mega Phase E. The first release this project calls **v1.0** —
not because features stopped, but because this phase specifically
hardened concurrency, multi-model lifecycle, API stability, and upgrade
safety to the bar `docs/api-v1-stability.md`/`docs/cli-stability-
contract.md` now promise. See "Known limitations" (README.md) for
every real, disclosed gap that remains — v1.0 is a stability promise
about what ships, not a claim that nothing is missing.

## What changed since v0.8.0

- **Bounded concurrent decode** (E1) — multiple chat-completion
  requests against the same resident model now genuinely decode in
  parallel (bounded, `MEMBRANE_MAX_CONCURRENT_DECODE`, default 2 —
  real-memory-motivated, not a performance target), replacing full
  request serialization. Reviewed and deliberately did NOT adopt
  upstream llama.cpp's own shared-context/`seq_id`-slot continuous-
  batching design — it would require one fixed `n_ctx`/KV-dtype shared
  across every concurrent request, conflicting with this project's own
  real per-request adaptive planner. A real hang-class bug (an
  unsuppressed llama.cpp log stream filling a bounded pipe under
  sustained load) was found and fixed.
- **Bounded multi-model residency** (E2) — up to
  `MEMBRANE_MAX_RESIDENT_MODELS` (default 2) different models can be
  resident at once, each independently servable. Deterministic LRU
  eviction; never evicts a pinned or actively-generating model;
  `membrane model pin`/`unpin`; hot-switch to an already-resident model
  never reloads from disk.
- **API v1 maturity** (E3) — `docs/api-v1-stability.md`, the frozen v1
  contract (stable paths, error shape, required fields, additive-change
  and deprecation policy). New `GET /membrane/v1/capabilities` (real,
  live: streaming/stop/tool_calling/embeddings/backends/resident_
  model_limit/platform). A real, pre-existing error-contract
  completeness gap was found and fixed: two PR E2 error codes had never
  actually been added to the frozen Errors table.
- **Runtime hardening** (E4) — bounded soak evidence for model-switch
  cycles, multi-model hot-switch cycles, and real crash/restart
  recovery (5 real `SIGKILL`+restart cycles). Server-shutdown safety
  formally verified (source + a real empirical test): an in-flight
  generation can never race the post-shutdown model-close path.
- **External validation** (E5) — real Windows/macOS host-memory reading
  added (`--ctx auto` now works on both, previously Linux-only),
  confirmed on real CI hardware. External-validation audit: 4 real,
  independent pull requests from an outside contributor found (real
  bug fixes, under separate maintainer review, not folded into this
  release).
- **v1.0 release prep** (E6, this release) — version bump, the CLI/API
  stability contracts, a real v0.8→v1.0 upgrade test, this document,
  and the v1.0.0 release itself.

## Install once, use it

```bash
sudo apt install ./membrane_1.0.0_amd64.deb
membrane use qwen2.5:7b
```

Unchanged from v0.8.0 — two commands, no new steps.

## Concurrency and multi-model residency

`docs/soak-and-concurrency-testing.md` has the full real evidence. In
short: a request to model A no longer blocks behind a request to model
B if both are resident (PR E2); several requests to the SAME resident
model now genuinely run in parallel, bounded (PR E1). Neither is
unlimited — this release is deliberately conservative (2 concurrent
decodes, 2 resident models, both env-overridable for testing only, not
documented/supported knobs) rather than performance-maximizing.

## API stability

`docs/api-v1-stability.md` is now the authoritative frozen contract.
Summary: `GET /v1/models`, `POST /v1/chat/completions` (OpenAI-
compatible subset, unchanged wire shape since v0.8.0 — re-validated
this release against the real Python/Node `openai` SDKs with zero
client-visible regression), `GET /v1/status`, `GET /membrane/v1/
capabilities` (new), `POST /membrane/v1/models/activate|pin|unpin`
(admin, loopback-only). Tool calling, embeddings, and structured JSON
response_format all remain explicitly unimplemented/rejected — each
re-evaluated this phase and deliberately not added (no real
implementation path or real end-to-end test fixture exists yet for
any of the three).

## Platform/backend support

Unchanged evidence scope from v0.8.0 (`docs/support-matrix.md`) with
one real improvement: `--ctx auto`'s real host-memory sizing now works
on Windows and macOS, not just Linux (PR E5, confirmed on real
`windows-latest`/`macos-14` CI hardware). No official Windows/macOS
package exists yet — real per-PR CI validation only.

## Runtime hardening evidence

`results/runtime-hardening-v1/validation.json`. RSS growth under a
model-switch soak plateaus (glibc allocator-retention behavior, not a
leak) rather than growing unboundedly; thread/FD counts show zero
growth across every soak; 5/5 real crash+restart cycles recovered
cleanly; server shutdown while a generation is active is proven race-
free (never races the model-close path), with one real, disclosed
asymmetry (the non-streaming path has no shutdown-triggered
cancellation, so shutdown waits for that generation's own natural end,
bounded by `max_tokens`, never indefinite).

## Upgrade from v0.8.0

No breaking change, no schema migration — `registry_core.h`/
`server_config.h`/`model_catalog.h` all remain at their existing
`schema_version`, unchanged in shape. See
`docs/upgrade-v0.8-to-v1.0.md` for the real, tested upgrade evidence.

## Security/privacy

Unchanged: loopback-only by default, no authentication of its own, no
telemetry, HTTPS-only downloads. Re-confirmed by source audit this
phase (`docs/external-validation.md`'s own "External user data
policy").

## Validation

See `results/release-v1.0.0/readiness.json` for the complete evidence
breakdown (platform/backend/model/client/package matrices, scheduler
and residency correctness, soak results, upgrade test, test/sanitizer/
CI/CodeQL status).

## Known limitations

See `README.md`'s own "Known limitations" section — kept in one place
rather than duplicated.
