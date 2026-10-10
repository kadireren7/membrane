# vLLM adapter (Milestone L) -- read-only

vLLM is a **supported inspect / advise / orchestrate runtime** in MEMBRANE.
MEMBRANE does not control it, and does not claim to.

## Audited contract

Audited against `vllm-project/vllm` **v0.31.0** (published 2026-10-05),
reading the source of the OpenAI-compatible server:

| Route | Source | What MEMBRANE uses it for |
|---|---|---|
| `GET /health` | `vllm/entrypoints/serve/instrumentator/health.py` | 200 = engine alive, **503 = engine dead** -> runtime reported unavailable |
| `GET /version` | `.../instrumentator/basic.py` | `{"version": "..."}` -> runtime version, and the "is this really vLLM" check |
| `GET /v1/models` | `vllm/entrypoints/openai/models/api_router.py`, `.../serve/engine/protocol.py` (`ModelCard`) | the served model id and `max_model_len` (the **configured** maximum context, not an allocated one) |
| `GET /load` | `.../instrumentator/basic.py` | `{"server_load": N}`: requests currently using the GPU |

That is the whole allowlist (`membrane_vllm_request_allowed()`): exact `GET`
plus path. There is no `POST` at all -- no inference request, no `/sleep`,
`/wake_up`, `/reset_prefix_cache` (dev-mode only upstream), no `/metrics`
(a Prometheus text surface MEMBRANE does not parse). A trap-route test
asserts none of them is ever reached.

## Why there is no control

vLLM fixes its model, context (`--max-model-len`), KV dtype
(`--kv-cache-dtype`), memory fraction (`--gpu-memory-utilization`) and
parallelism at process start. There is no supported runtime API to change
them, so every control capability is `unsupported`, `model_switch` and
`model_load_unload` included. A "control" that could only be applied by
restarting someone else's server is not a control.

What MEMBRANE does with it:

* `membrane runtime list | inspect vllm | models vllm | model inspect vllm ID`
* a candidate in `membrane run MODEL` ([orchestration.md](orchestration.md)),
  where it competes on facts: it is *resident* (it serves what it was started
  with), it exposes a maximum context, and its limits are disclosed.
* `membrane observe|advise|plan|apply --runtime vllm`: **not implemented**
  (they say so). Memory telemetry on vLLM lives on `/metrics`, which this
  adapter deliberately does not read; adding it later does not change the
  "no control" conclusion.

## Endpoint

`MEMBRANE_VLLM_ENDPOINT` (default `http://127.0.0.1:8000`, vLLM's default
port). Same rules as Ollama's: `http://HOST[:PORT]` only -- no https, no
credentials, no path, no redirects, bounded responses, one attempt. A
non-loopback endpoint is reported as `remote_external`, never as local.
Nothing contacts vLLM unless a command asks about it.

## Capability matrix (static contract)

Supported: model enumeration, chat completions, streaming, current model.
Partial: model metadata (only `max_model_len`), active context (the
configured maximum). Unknown: cancellation. Everything else -- including
context / GPU-layer / quant / KV-precision / KV-placement / concurrency /
device control, RAM / VRAM / KV usage telemetry, load/unload/switch and
dynamic reconfiguration -- is **unsupported**.
