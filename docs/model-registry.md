# MEMBRANE model registry

Mega Phase A, PR A2. A local name → path registry so applications don't
pass a filesystem model path with every request/CLI invocation.

```
membrane model add qwen /path/to/model.gguf
membrane model list
membrane model inspect qwen
membrane model use qwen
membrane model remove qwen
```

`membrane model use NAME` (Mega Phase B, PR B1) sets a persistent
default model for `membrane serve`/`membrane service` — see
`docs/server.md`'s "Default model" section and `docs/service.md`. It
never forces a model to load; it only changes what an omitted
`"model"` field in a chat request falls back to, and it writes to the
separate server config file (`~/.config/membrane/server.json`), not
this registry.

## Location

`$XDG_DATA_HOME/membrane/models.json`, falling back to
`$HOME/.local/share/membrane/models.json` (the standard XDG Base
Directory fallback rule). `MEMBRANE_MODELS_PATH` overrides it — used by
tests/CI so a real user registry is never touched.

## What gets stored

Per entry: `name`, the canonicalized absolute `path` (symlinks resolved
at `add` time via `realpath()`), a display `basename`, `arch_name` and
`model_max_context` (read once via the same GGUF metadata scan
`--inspect-model`/`--doctor` already use — `membrane_gpu_estimate_model()`,
never a second GGUF-parsing implementation), and the file's `size`/`mtime`
at add time (a cheap identity signature — never a hash of a multi-GB
model file).

Never stored: prompt history, secrets, API keys.

## Safety

- **Duplicate name** is rejected (the registry's own primary key).
  Duplicate *path* under a different name is allowed (a harmless alias).
- **Invalid GGUF** (`add` only) is rejected via the same real metadata
  scan `--inspect-model` uses — not a second parser.
- **Atomic writes**: every save is a temp file in the same directory,
  fsync'd, then `rename()`d into place — a reader (or a crash mid-write)
  never observes a half-written registry.
- **Stale/moved/deleted files**: `list`/`inspect` re-`stat()` the real
  file every time and report `OK`/`MISSING`/`MODIFIED`/`UNREADABLE` —
  never assumed still valid from what was cached at `add` time.
  `inspect` additionally rescans real metadata when the file changed
  (`MODIFIED`), so a stale name never keeps reporting the old
  architecture/context.

## Used by

- `membrane model ...` itself (`tools/membrane/model_cmd.cpp`).
- `membrane serve` (`docs/server.md`) — every `model` field in a chat
  request is resolved through this same registry, never a raw
  filesystem path from an HTTP client. A server that has already
  started does not see a registry change until restarted (`docs/
  service.md`'s "Model registry and default-model reload" section).
