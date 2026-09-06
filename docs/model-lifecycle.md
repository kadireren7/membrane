# Model lifecycle: `membrane use`

Mega Phase D, PR D6. Makes "I want to use this model" a single command:

```
membrane use qwen2.5:7b
```

If `qwen2.5:7b` is already installed, this selects it (and live-switches
a running server to it). If it is not installed but is a known catalog
model, this previews the download, asks for consent, installs it, then
selects/activates it — the same one command either way. Nobody should
need to manually run `model install` + `model add` + `model use` +
`service restart` + `status` for the normal path.

This document describes the ORCHESTRATION `membrane use` performs. It
introduces no second registry, install pipeline, or model-lifecycle
policy of its own — see each linked doc for the real, pre-existing
primitive being reused.

## Two commands, two jobs

| Command | Job |
|---|---|
| `membrane model install NAME` | Download + register only. Never selects/activates anything. |
| `membrane model use NAME` | Set `default_model` in `server.json` only (low-level). Never installs, never live-switches. |
| `membrane use NAME` | Resolve → install-with-consent-if-needed → select → live-switch-if-running → verify → report. |

Keep using `model install`/`model use` directly when you want exactly
that one narrow effect. `membrane use` is the normal, product-facing
entry point built on top of both — see `docs/model-registry.md`.

## Resolution (Section 23 of the task)

`MODEL` is resolved with a fixed, deterministic precedence — never a
fuzzy/typo guess before a real network install:

1. An exact, already-registered registry name (`membrane model list`).
2. An exact catalog id or alias (`membrane model search`) —
   `model_catalog.h`'s own `membrane_catalog_resolve()`, matching name
   OR alias in one exact, case-insensitive step.

An unresolvable name is `NOT_FOUND`, a clean CLI error — never a
partial/ambiguous action.

## Already installed

```
$ membrane use qwen2.5:7b
Selected model: qwen2.5:7b
Active model: qwen2.5:7b
Backend: CPU
Endpoint: http://127.0.0.1:8642
```

Before anything else, the registered file's own real existence is
re-checked (`stat()`, the same real check `membrane doctor`'s own
`registry` check already performs) — a registry entry whose file has
since been deleted or moved is refused with `MODEL_FILE_MISSING`, never
silently "selected" as if it still worked.

- **Service stopped**: `default_model` is still set; reported as
  `Selected model: NAME` / `Service is not running. Start it with:
  membrane service start` — a real, successful outcome (Section 13 of
  the task), never a failure just because nothing is currently running.
- **Service running, different model active**: a real, live switch is
  attempted (see below).
- **Service running, already active**: idempotent — `Already active:
  NAME`, no reload at all (checked via a real `GET /v1/status` read
  before ever calling the switch endpoint, so a repeated `membrane use`
  against an already-correct server does zero extra work).

## Not installed (Section 5-10 of the task)

```
$ membrane use qwen2.5:7b
Model: Qwen2.5-7B-Instruct (not installed)
Recommended variant: Q4_K_M
Approx download: 4.4 GiB
Estimated hardware fit: HOST_MEMORY_FIT
Backend: GPU detected (final choice made automatically at load time)
Download and install this model? [Y/n]:
```

The recommended variant is the exact same hardware-aware selection
`membrane model install` itself uses (`docs/model-variant-selection.md`,
Mega Phase D PR D2) — reused directly, never a second "does this fit"
estimate. `--quant Q4_K_M`/`--variant Q4_K_M` overrides it; an explicit
override is always honored, never silently replaced.

**Consent** (Section 6): `--yes`/`-y` skips the prompt. Interactively,
`[Y/n]` (default yes) is asked. **Non-interactively with no `--yes`,
this fails clearly** — `NONINTERACTIVE_CONSENT_REQUIRED`, never a silent
download and never a hang:

```
$ membrane use qwen2.5:7b < /dev/null
membrane use: model is not installed. Re-run with --yes to allow
download in non-interactive mode.
```

This is deliberately DIFFERENT from `membrane setup`'s own optional
prompts (which default to yes when non-interactive) — a real network
download of several gigabytes must never happen without either a human
present or an explicit `--yes`.

After consent, the download/verify/register transaction is the exact
same one `membrane model install` runs — internally re-invoked (via the
same stdout-silencing technique `membrane setup` already uses to compose
one clean summary) with the previewed variant pinned via `--quant`, so
the file actually installed is byte-identical to what was shown and
consented to. See `docs/model-catalog.md`/D1's own download pipeline —
never a second downloader.

If the install itself fails partway (checksum mismatch, disk space,
network), `membrane use` reports the failure and stops — **no model is
selected or activated**, but a valid partial/complete download that
`model install` itself already wrote to disk is never deleted just
because this outer step failed (Section 10 — the installed file, if
valid, can remain installed; re-running `membrane use`/`membrane model
install` picks it up again without re-downloading).

If no variant is estimated to fit this host at all (and none was
explicitly forced with `--quant`), nothing is downloaded — real
alternatives (every variant considered, and why) are printed instead.

## Live switch (Section 12, 14-17 of the task)

A new, loopback-only, MEMBRANE-specific admin endpoint —
`POST /membrane/v1/models/activate` (see `docs/server.md`) — lets
`membrane use` trigger an immediate model switch on an already-running
server, with **no restart**. This is a thin wrapper around
`ensure_model_loaded()`, the exact same function `POST
/v1/chat/completions` itself already calls for every request — never a
second switch/lifecycle policy. Two guarantees fall out of that reuse
for free:

- **Idempotence** (Section 17): switching to the already-active model
  is a real no-op, reported as `already_active`.
- **Failure recovery, honestly reported** (Section 16): if the new
  model fails to load, the server automatically attempts to reload
  whatever was active before. `membrane use`'s own output always
  describes the REQUESTED switch's real outcome — a recovered previous
  model is reported as `active_model` in a still-failed result, never
  silently reframed as success:

```
$ membrane use big-model-that-does-not-fit
Selected model: big-model-that-does-not-fit (default set)
Model switch failed: no context/hardware plan could be resolved for
this model on this host
The server recovered its previous model ('qwen2.5:7b') and remains
available.
Try: membrane doctor
```

This exact failure/recovery path was exercised for real (a real HTTP
round trip through the new admin endpoint, a real `ensure_model_loaded()`
attempt) — see `results/model-lifecycle-ux/validation.json`.

**Never unloads a model beneath an active generation** (Section 15):
the admin endpoint waits (bounded, ~5s, matching the admission gate's
own spirit) on the exact same request-serializing mutex every chat
request already uses, reporting `503 SERVER_BUSY` rather than either
blocking indefinitely or racing a live generation.

**Context sizing for a switch with no real prompt yet**: the endpoint
applies a short, fixed placeholder chat turn through the model's own
real chat template (the same templating `/v1/chat/completions` uses) so
`ensure_model_loaded()` has real text to size context against. Real,
disclosed design finding: `context_recommender.c`'s own algorithm always
maximizes the recommended context to fit real host memory — the
triggering prompt's own token count is only ever a floor — so this
placeholder sizes context almost identically to what a real first
message would in the common case. Not a corner being quietly cut.

## Default vs. active (Section 11 of the task)

Two distinct, honestly-tracked things:

- **Default model** — what `server.json` currently prefers. Changed by
  `membrane use`/`membrane model use`, read once at `membrane serve`/
  service startup.
- **Active model** — whatever the running server actually has loaded
  right now. Changed live by `membrane use` when a server is running;
  otherwise stays whatever it was.

`membrane use NAME` normally makes both `NAME` when the service is
running and the switch succeeds. If the service is stopped: only the
default changes (active stays "none" — there is nothing running to have
an active model at all). `membrane status`/`membrane doctor` both report
the two separately, never collapsed into one line, so a real divergence
(e.g. right after a failed switch that recovered a different model)
stays visible.

## `membrane model uninstall` and the active/default model (Section 19)

Uninstalling a model that is currently **active** on a running server is
refused outright (`MODEL_ACTIVE`) — switch away (`membrane use OTHER`)
or stop the service first. Uninstalling the current **default** is
allowed, but `default_model` is cleared automatically (never left
dangling, pointing at a deleted file) and the outcome is reported
explicitly in the command's own output.

## `membrane doctor` integration (Section 22)

A new `model_lifecycle` check flags:

- The service is running with **neither** a default nor an active
  model configured/loaded — a chat request with no explicit `"model"`
  would be refused.
- The running server's real active model differs from the configured
  default (normal right after a live non-default `membrane use` switch,
  or before a service restart picks up a new default) — informational,
  actionable, never treated as corruption.

"Selected model file missing"/"stale catalog metadata" are already real
findings from doctor's pre-existing `registry` check (covers every
registered model, default or not) — not duplicated here.

## `membrane setup` integration (Section 21)

`membrane setup --model qwen2.5:7b` (a catalog name, not only a local
file path) routes model resolution/install/selection through the exact
same logic described above — never a second, independent install path
inside `setup` itself.

## JSON output (Section 29)

`membrane use MODEL --json` prints one JSON object:
`model`, `installed`, `downloaded`, `variant` (only when just
installed), `default_model`, `service_running`, `endpoint`,
`active_model`, `backend`, `result` (`selected` / `already_active` /
`switched` / `switch_failed` / `switch_unreachable`), and on failure an
`error: {code, message}`.

## Error codes (Section 30)

Reused verbatim where this project already has an equivalent concept
(`NOT_FOUND`, `NO_FEASIBLE_VARIANT`, registry/config `IO_ERROR` codes);
new only where none fit: `DOWNLOAD_DECLINED`,
`NONINTERACTIVE_CONSENT_REQUIRED`, `DOWNLOAD_FAILED` (a coarse wrapper —
the specific underlying cause is printed to stderr by the internally
re-invoked `membrane model install`, a disclosed simplification rather
than refactoring that command's own ~300-line transaction to return a
structured error just for this one caller), `MODEL_FILE_MISSING`,
`MODEL_SWITCH_FAILED` (only when the server itself supplies no more
specific code — it usually does, e.g. `NO_FEASIBLE_CONTEXT`/
`MODEL_LOAD_FAILED`, surfaced verbatim instead), `SERVICE_UNAVAILABLE`
(the server answered an earlier `/v1/status` but became unreachable for
the activation call itself — a real, rare race, never fatal to the
"selected" outcome since `default_model` is already saved by that
point).

## Cross-platform (Sections 24-26)

No new platform-specific orchestration — reuses the same abstractions
D3/D4/D5 already built: `registry_core.h`/`server_config.h` (already
cross-platform via `fs_util.h`'s `membrane_resolve_home_dir()`),
`status_client.h`'s httplib client (already cross-platform), and the
existing service abstraction for "start the service yourself" guidance.
Windows paths with spaces are exercised by the pre-existing
`membrane model add`/`install` GGUF path handling this module calls
into unchanged; macOS has no systemd-specific assumption anywhere in
this file.

## Known limitations

- `DOWNLOAD_FAILED`/`CHECKSUM_FAILED` are folded into one coarse JSON
  code for `membrane use` specifically (see "Error codes" above) — the
  real cause is on stderr, not in the JSON `error.code`.
- The registry does not record which *quant/variant* was installed
  (only architecture/context/file identity) — `membrane status`
  therefore cannot show "installed variant" without a registry schema
  change, which this PR does not make. `variant` is only present in
  `membrane use`'s own JSON output for the run that just performed the
  install.
- The one real not-installed→install→select→live-switch end-to-end pass
  against a real catalog entry (`smollm2-135m-instruct`) was run
  manually, not in `ctest` (this project's own existing precedent —
  `test_download_manager.cpp`'s own top comment already established that
  even a real HTTPS transfer stays out of `ctest`). See
  `results/model-lifecycle-ux/validation.json` for what was verified for
  real versus in `ctest`.
