# CLI reference

`membrane` is the product CLI. Human output is the default; add `--json` for
machines (never colored) and `--verbose` where a command offers it for every
detail. `--no-color` (or `NO_COLOR=1`) turns color off; color and Unicode
symbols are only ever used on a terminal. `membrane --help` lists everything;
this page groups it by what you are trying to do.

Exit codes: `0` success, `2` usage error (also: a mutation requested without
consent), `3` model error, `4` runtime error, `5` unsupported KV mode.

## Start here

| Command | What it does |
|---|---|
| `membrane` | Status screen: model, runtime, GPU, memory, server, adaptive mode, and the one or two best next commands. `--json` for the same facts. |
| `membrane help`, `--help` | The full command list. |
| `membrane doctor [--verbose]` | One-pass diagnosis (installation, which binary you are running, hardware, GPU backend, models, config, service, API, runtimes, adaptive). Overall OK / WARNING / ERROR plus **Next:** step. |
| `membrane setup` | Guided first run: register a model, install and start the service, verify. |

## Models

| Command | What it does |
|---|---|
| `membrane models` | Installed models: name, id, size, quant, loaded / default. |
| `membrane models available` | The curated catalog: purpose, size, license, and how much MEMBRANE knows (`validated` / `smoke-tested` / `metadata-only`). |
| `membrane models recommend` | What suits **this** computer, from measured RAM and catalog metadata (see [model-catalog.md](model-catalog.md)). |
| `membrane use [MODEL] [--quant Q] [--yes]` | Install (with consent) and select a model. No `MODEL` on a terminal: a numbered picker (pages, `/search`, `q`). A partial or misspelled name is **suggested, never silently chosen** (confirm on a terminal; a script gets the candidate list and a failure). |
| `membrane model add\|remove\|list\|inspect\|use\|search\|info\|install\|uninstall` | The low-level registry commands (kept for scripts). |

## Understand memory (read-only)

| Command | What it does |
|---|---|
| `membrane plan MODEL [--ctx N] [--kv P] [--gpu-layers N] [--quant Q] [--verbose]` | Will it fit? On what hardware? What will MEMBRANE do, and why? `--verbose` is the full per-variant report. |
| `membrane observe [--runtime ID] [--verbose]` | Facts right now. Default is a compact status screen; `--verbose` adds every field with its provenance. |
| `membrane advise [--runtime ID] [--verbose]` | What the facts mean: status, main finding, recommended action, why. |

## Change things (explicit)

| Command | What it does |
|---|---|
| `membrane apply [flags]` | **Dry run by default.** `--yes` (or a `y` at the prompt) applies. Flags: `--context`, `--kv`, `--kv-placement`, `--gpu-layers`, `--unload`, `--activate`, `--pin`, `--unpin`, `--reload`, `--runtime`. See [active-control.md](active-control.md). |
| `membrane adaptive status\|enable\|disable` | Opt-in, bounded adaptation. Off by default. See [adaptive-runtime.md](adaptive-runtime.md). |

## Pick a runtime

| Command | What it does |
|---|---|
| `membrane run MODEL [--runtime auto\|ID] [--intent I] [--explain]` | Chooses among membrane-native, Ollama and vLLM, prints **why**, then uses it. See [orchestration.md](orchestration.md). |
| `membrane runtime list\|inspect ID\|models ID` | What each runtime is and what MEMBRANE can do with it. |

## Serve and talk

| Command | What it does |
|---|---|
| `membrane serve [--port N] [--bind A]` | The local OpenAI-compatible server (loopback by default). |
| `membrane chat [MODEL]` | Interactive chat against the local server. With no model selected on a terminal, offers a choice. |
| `membrane status` | Is a server running, and what is loaded. |
| `membrane service install\|uninstall\|start\|stop\|restart\|status\|logs` | Run the server as a background service (systemd user unit / launchd / Task Scheduler). |

## Low-level

`membrane-run` is the direct inference CLI (no server); see
`membrane-run --help` and the README's *Advanced* section.
