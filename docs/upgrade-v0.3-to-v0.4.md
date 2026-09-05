# Upgrading from v0.3.0 to v0.4.0

No breaking change. Every existing command, flag, and JSON field keeps
its exact `v0.3.0` behavior — `v0.4.0` is additive only (new commands,
new docs, packaging/supply-chain/hardening work), see
`docs/release-v0.4.0.md` for the full release notes.

## If you installed the `.deb` package

```bash
sudo apt install ./membrane_0.4.0_amd64.deb
```

`apt` upgrades in place. Confirmed real (`docs/release-supply-chain.md`):
your registry (`~/.local/share/membrane/models.json`), server config
(`~/.config/membrane/server.json`), and generated systemd unit are
**never touched** by a package upgrade — a real container test proved
these three files are byte-identical (registry/config) or functionally
unaffected (the unit, since `membrane service install` never rewrites
an existing one) before and after an upgrade. If `membrane service`
was already running before the upgrade, restart it to pick up the new
binary:

```bash
membrane service restart
```

(Not required for the new commands themselves — `membrane setup`/
`membrane doctor` are new subcommands of the `membrane` CLI, not part
of the running service process, and work immediately after the
package upgrades.)

## If you built from source

Rebuild and reinstall as before (`docs/install.md`, Options B/C) — no
new build flag, no new dependency, no CMake option changed meaning.

## Registry / config schema

Both `models.json` and `server.json` remain `schema_version: 1` — no
migration is needed or exists (see `docs/schema-versioning.md`). The
registry now additionally *rejects* an unrecognized `schema_version`
with a clear error (a real gap this release fixed, matching the
config's own pre-existing behavior) — this only changes behavior for a
hypothetical future incompatible file, never for your existing, valid
`v0.3.0`-written files.

## New commands worth trying after upgrading

```bash
membrane doctor            # unified diagnostic: hardware, registry, config, service, HTTP
membrane setup --model /path/to/model.gguf   # if you haven't already registered a model this way
```

Both are read-mostly and safe to run at any time — `membrane doctor`
never mutates anything, and `membrane setup` is idempotent (re-running
it against already-correct state reports "already registered"/
"already installed"/"already running" and changes nothing).
