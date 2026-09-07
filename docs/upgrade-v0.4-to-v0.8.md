# Upgrading from v0.4.0 to v0.8.0

No breaking change, no schema migration needed. Every schema this
project persists to disk — the model registry (`models.json`) and the
server config (`server.json`) — is still exactly `schema_version: 1`,
byte-for-byte the same shape v0.4.0 wrote (confirmed this phase by
source read: neither `registry_core.h` nor `server_config.h` gained or
removed a field across Mega Phase D). `v0.8.0` is additive only at the
product-command/API level — see `docs/release-v0.8.0.md` for the full
release notes.

## If you installed the `.deb` package

```bash
sudo apt install ./membrane_0.8.0_amd64.deb
```

`apt` upgrades in place. Your registry
(`~/.local/share/membrane/models.json`), server config
(`~/.config/membrane/server.json`), any models already installed under
`~/.local/share/membrane/models/`, and your generated systemd unit are
**never touched** by a package upgrade (same real, container-tested
guarantee `docs/upgrade-v0.3-to-v0.4.md` already established for the
v0.3→v0.4 transition — `membrane service install` never rewrites an
existing unit, and neither registry/config file is ever written except
by an explicit `membrane model add`/`install`/`use`/`service install`
command you run yourself). If `membrane service` was already running
before the upgrade, restart it to pick up the new binary:

```bash
membrane service restart
```

(Not required for the new commands themselves — `membrane use` is a
new subcommand of the `membrane` CLI, not part of the running service
process, and works immediately after the package upgrades. A **live**
model switch via `membrane use MODEL` against an already-running
service, specifically, needs no restart at all — that is the whole
point of the new admin endpoint it uses.)

## If you built from source

Rebuild and reinstall as before (`docs/install.md`, Options B/C) — no
new required build flag. `-DGGML_CUDA=ON` is a new, optional flag if
you want the (source-build-only, one-real-device-validated) CUDA
backend — see `docs/cuda-backend.md`; it changes nothing for an
existing CPU/Vulkan build.

## Registry / config schema

Unchanged — see above. The registry's own real file-identity check
(`membrane_registry_check_identity()`) and the server config's own
`schema_version` fail-closed check are both untouched by this phase;
an unrecognized future `schema_version` still fails closed with a
clear error rather than being silently reinterpreted
(`docs/schema-versioning.md`).

## New commands/behavior worth trying after upgrading

```bash
membrane use qwen2.5:7b     # select an installed model, or install it with consent
membrane doctor             # now also flags a default/active model mismatch
membrane status             # now shows default model and active model separately
```

All three are safe to run at any time. `membrane use` only ever writes
your own explicit choice (`default_model` in `server.json`) and, if a
server is already running, live-switches it — it never deletes an
installed model or touches an unrelated one. `membrane model
uninstall` now refuses to delete a model that is currently active on a
running server, and clears (rather than leaves dangling) a matching
`default_model` when you uninstall the current default.

## API clients

No breaking change to any existing field. New, additive: real
`"stop"` sequence support, and two new explicit rejections
(`UNSUPPORTED_TOOL_CALLING`, `UNSUPPORTED_RESPONSE_FORMAT`) for fields
this server never silently honored anyway — a client that was already
omitting `tools`/`response_format` sees no change at all. See
`docs/api-contract.md` for the complete, current field table.
