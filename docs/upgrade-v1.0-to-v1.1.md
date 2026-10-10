# Upgrading from v1.0.0 to v1.1.0

Nothing is migrated and nothing is rewritten. Install the new package over the
old one (`sudo apt install ./membrane_1.1.0_amd64.deb`); your model registry
(`models.json`) and `server.json` are read as they are.

Checked: the v1.1.0 binary was run against a copy of a real v1.0.0 registry
and `server.json`; read-only commands (`membrane`, `models`, `plan`,
`observe`, `advise`, `apply` without `--yes`, `adaptive status`, `run
--explain`, `doctor`) left both files byte-identical, and the existing model
was listed, planned and served without any change.

## What behaves differently

| Before | Now |
|---|---|
| bare `membrane` printed the usage text and exited `2` | prints a status screen and exits `0`; `membrane --help` / `membrane help` print the usage |
| `plan` / `observe` / `advise` printed long reports | a short view by default; the old report is `--verbose`; `--json` is the same |
| `doctor` printed `[OK] name` lines | one line per check with a summary, an overall verdict and a **Next:** step; `--json` keeps the same fields (new checks `runtimes` and `adaptive` were added) |
| `membrane use typo` said not found | suggests close names; a terminal confirms, a script gets the candidate list and a failure |

Scripts should use `--json` (never colored) rather than parse human output; the
CLI contract has always said human wording is not frozen.

## What is new and optional

* `server.json` may now contain `overrides` (written by `membrane apply`) and
  `adaptive` (written by `membrane adaptive enable`). Both are optional; an
  invalid block makes the config load fail loudly rather than be half-honored.
  Delete the block (or run `membrane apply --context auto --kv auto ...`,
  `membrane adaptive disable`) to return to the old behavior.
* New loopback admin routes: `POST /membrane/v1/models/unload`,
  `GET /membrane/v1/adaptive`.
* `~/.local/state/membrane/actions.jsonl` appears only after an applied change.

## Downgrading

Install the v1.0.0 package. v1.0.0 ignores `server.json` keys it does not
know, so nothing breaks; but anything v1.0.0 itself writes back to
`server.json` (for example `membrane service install`) is written without the
`overrides` and `adaptive` blocks, which resets those to their defaults
(automatic / off).
