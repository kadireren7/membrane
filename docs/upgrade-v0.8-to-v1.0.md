# Upgrading from v0.8.0 to v1.0.0

No breaking change, no schema migration needed. Every schema this
project persists to disk — the model registry (`models.json`), the
server config (`server.json`), and the built-in model catalog — is
still exactly the same `schema_version` v0.8.0 wrote (confirmed this
phase by source read and by a real, container-based upgrade test, not
just read: `registry_core.h`/`server_config.h`/`model_catalog.h` gained
no field across Mega Phase E). `v1.0.0` is additive only at the
product-command/API level — see `docs/release-v1.0.0.md` for the full
release notes.

## Real, tested evidence (this phase)

A real Docker container installed the actual v0.8.0 GitHub release
asset (checksum-verified), registered a real model, captured the real
`models.json`/`server.json` contents, then upgraded in place to this
release's own `membrane_1.0.0_amd64.deb` — **both files came back
byte-for-byte identical** (`diff` produced no output). The pre-existing
model was still listed (`membrane model list --json`, `status: OK`),
and every v1.0.0-only feature — `GET /membrane/v1/capabilities`,
`membrane model pin`/`unpin`, multi-model residency status in
`membrane status` — worked immediately against the pre-upgrade
registry with zero extra setup. See `results/release-v1.0.0/
readiness.json`'s own `package_matrix.real_upgrade_test` for the full
detail.

## If you installed the `.deb` package

```bash
sudo apt install ./membrane_1.0.0_amd64.deb
```

`apt` upgrades in place. Your registry, server config, any models
already installed, and your generated systemd unit are never touched
by a package upgrade (unchanged guarantee, re-verified for real this
phase — see above). If `membrane service` was already running before
the upgrade, restart it to pick up the new binary:

```bash
membrane service restart
```

A running service automatically gets the new bounded concurrent-decode
and multi-model-residency behavior on restart — no configuration
change is required to use either; both are conservative defaults
(`MEMBRANE_MAX_CONCURRENT_DECODE=2`, `MEMBRANE_MAX_RESIDENT_MODELS=2`)
that apply automatically.

## If you built from source

Rebuild and reinstall as before (`docs/install.md`) — no new required
build flag. `--ctx auto` now also works for real on Windows and macOS
(previously Linux-only) — no flag change needed to benefit from it.

## Registry / config schema

Unchanged — see above. The registry's own real file-identity check and
every schema's own fail-closed `schema_version` check are both
untouched by this phase (`docs/schema-versioning.md`).

## New commands/behavior worth trying after upgrading

```bash
membrane model pin qwen2.5:7b     # protect a resident model from eviction
membrane model unpin qwen2.5:7b
membrane status                   # now shows every resident model, not just one
curl http://127.0.0.1:8642/membrane/v1/capabilities   # real, live capability discovery
```

All are safe to run at any time. Pinning only ever applies to a
currently-resident model (a clear `409 MODEL_NOT_RESIDENT` otherwise) —
it is never a standing preference remembered for a not-yet-loaded
model.

## API clients

No breaking change to any existing field — re-validated this phase
against the real Python/Node `openai` SDKs with zero client-visible
regression. New, additive: `GET /membrane/v1/capabilities`, two new
admin routes (`.../pin`, `.../unpin`), and two new error codes
(`RESIDENCY_EXHAUSTED`, `MODEL_NOT_RESIDENT`) a client only ever sees
if it calls the new pin/unpin routes itself. `/v1/status`'s own
`loaded_model`/`backend`/`gpu_layers`/`kv_precision` singular fields
were replaced by a `resident_models` array during Mega Phase E
(PR E2) — this is a MEMBRANE-specific, non-OpenAI endpoint (`docs/
api-v1-stability.md`), not part of the frozen OpenAI-compatible
surface, so this is not treated as a breaking API change under this
project's own additive-change policy; a client that only ever used
`/v1/chat/completions`/`/v1/models` is completely unaffected. See
`docs/api-v1-stability.md` for the complete, current, frozen contract.
