# Quickstart

From nothing to a working local model in about five minutes. Nothing here
needs `--json`, a GGUF path, or any knowledge of GPU layers or KV caches.

## 1. Install

Ubuntu / Debian / Pop!_OS: download the `.deb` from the
[Releases page](https://github.com/kadireren7/membrane/releases) and

```bash
sudo apt install ./membrane_<version>_amd64.deb
```

Anything else (macOS, Windows, other Linux): build from source --
[install.md](install.md), [support-matrix.md](support-matrix.md).

## 2. Look around

```bash
membrane
```

```
MEMBRANE

  Model     none installed
  Runtime   Native
  GPU       NVIDIA GeForce GTX 1650 (Vulkan)
  Memory    not measured (no running server)
  Server    Stopped
  Adaptive  Off

Next:
  membrane models recommend
  membrane use
```

The bare command is a status screen. It only ever suggests commands that
would work in the state you are in.

## 3. Pick a model

```bash
membrane models recommend     # what suits THIS computer (measured RAM + catalog)
membrane use                  # on a terminal: a numbered list to choose from
membrane use qwen             # a partial name works; you are asked to confirm
```

`membrane use` previews the download (size, expected fit), asks before
downloading, installs, verifies the checksum, and selects the model. A guess
(`qwen` is not an exact id) is never acted on silently: it is shown and you
confirm it. See [model-catalog.md](model-catalog.md) for what each catalog
model is for and how much MEMBRANE knows about it.

## 4. Chat

```bash
membrane serve                # or: membrane service install && membrane service start
membrane chat
```

If something is missing, the command says what to run next.

## 5. Look at what memory is doing

```bash
membrane plan MODEL           # will it fit? what will MEMBRANE do, and why
membrane observe              # facts right now (RAM, GPU, model, server)
membrane advise               # what those facts mean, and what you could do
membrane apply                # dry run: which advice could be applied
```

`plan -> observe -> advise` never change anything. `apply` is a dry run unless
you pass `--yes` ([active-control.md](active-control.md)). Adaptation is off
unless you opt in ([adaptive-runtime.md](adaptive-runtime.md)).

## 6. Use it from your own code

```bash
curl http://127.0.0.1:8642/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5-1.5b-instruct", "messages": [{"role": "user", "content": "Hello"}]}'
```

Any OpenAI-compatible client works with base URL `http://127.0.0.1:8642/v1`
and any placeholder key. The server listens on loopback only and has no
authentication of its own -- see [security-model.md](security-model.md).

## 7. When something looks wrong

```bash
membrane doctor
```

It checks the installation, hardware, models, service and API in one pass,
says OK / WARNING / ERROR, and names the next step -- including a warning if
you are running a different `membrane` than the one on your PATH.

## A note on small models

MEMBRANE manages the runtime, not the intelligence of a model. A tiny
smoke-test model (SmolLM2-135M/360M) may echo your prompt, ramble, or
struggle with languages other than English. That is the model, not a failing
runtime -- see [model-quality.md](model-quality.md).
