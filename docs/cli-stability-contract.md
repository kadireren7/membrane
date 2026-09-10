# CLI stability contract

Mega Phase E, PR E6 (Section 44 of the task). The stable, public CLI
surface as of v1.0.0 — the commands below, their real subcommands, and
their real observable behavior are frozen for v1.0.0 and beyond
(additive changes only, same policy as `docs/api-v1-stability.md`'s own
API contract).

## `membrane` (the control CLI)

- `membrane setup` — guided onboarding (real doctor + model selection +
  service install, `docs/onboarding.md`).
- `membrane doctor [--json]` — unified diagnostic surface (registry,
  config, service, hardware, live server status).
- `membrane status [--json]` — thin HTTP client against an already-
  running `membrane serve`/service instance; never a process-management
  capability.
- `membrane use MODEL [--yes]` — resolves/installs (consent-gated) and
  live-switches the active/default model; idempotent if already active.
- `membrane serve [--port N] [--bind ADDR] [--allow-non-loopback]` —
  foreground HTTP server (the same runtime a `membrane service`-managed
  background process also runs).
- `membrane model add|remove|list|inspect|use|search|info|install|
  uninstall|pin|unpin ...` — registry and catalog management. `pin`/
  `unpin` (Mega Phase E, PR E2) only apply to a currently-resident
  model, never a standing preference for later.
- `membrane service install|start|stop|restart|status|uninstall|
  logs` — background-service lifecycle (systemd `--user` on Linux,
  launchd on macOS, Task Scheduler on Windows — see `docs/service.md`
  for the real, disclosed per-platform semantic differences).

Every subcommand accepts `--json` for machine-readable output where
documented; human-readable text output is not itself frozen (wording
may improve), but its presence/absence and real exit code are.

## `membrane-run` (the direct, non-HTTP inference CLI)

- `--model PATH` (required for generation), `--prompt TEXT`,
  `--ctx N|auto`, `--gpu-layers N|auto`, `--kv native|q8|q5|adaptive`,
  `--kv-placement ...`, `--auto` (joint planner), `--doctor`,
  `--list-devices`, `--json`, `--version`, `--help`.
- `--ctx auto` real host-memory sizing is now supported on Linux,
  Windows, and macOS as of v1.0.0 (PR E5) — previously Linux-only.

## Stable exit codes

- `0` — success.
- Nonzero — a real, disclosed failure category (see each command's own
  `--json` error shape, `error.code`, mirroring the HTTP API's own
  frozen error-code convention where applicable).

## Additive-change policy

Identical in spirit to the API's own policy
(`docs/api-v1-stability.md`): a MEMBRANE minor release may add new
subcommands, new flags, and new `--json` output fields — never remove
or repurpose an existing subcommand/flag's meaning within the same
major version.

## Deprecation policy

No CLI command or flag has been deprecated as of v1.0.0. If one ever
is: documented here with the version it was first marked deprecated
in, continues to function identically until actually removed (never
less than one full minor version of runway), and removal itself
follows the same "additive within a major version" rule above.

## What is NOT frozen

- Human-readable (non-`--json`) text output wording.
- Internal admin-only HTTP routes (`/membrane/v1/models/activate`/
  `pin`/`unpin`) that `membrane use`/`membrane model pin`/`unpin`
  call internally — covered by `docs/api-v1-stability.md` once a
  third party actually depends on them directly, not by this document.
- Build-from-source CMake option names/flags (`docs/build-options.md`
  if one exists, or the root `CMakeLists.txt` itself) — a developer-
  facing surface, not an end-user CLI contract.
