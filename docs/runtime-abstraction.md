# Runtime abstraction foundation (Milestone H1)

> **Milestone H2 update:** a first external-runtime adapter now exists --
> **Ollama, read-only** -- documented in [runtime-ollama.md](runtime-ollama.md).
> H2 does NOT send inference requests and does NOT change Ollama state.
> Sections below that describe `ollama` as "reserved only" describe H1;
> the H2 notes inline say what changed.
>
> **Milestone H3 update:** negotiation is now surfaced as a read-only,
> runtime-aware assessment -- `membrane plan MODEL --runtime ID` -- see
> [runtime-plan-assessment.md](runtime-plan-assessment.md). H3 does not
> apply planner decisions.
>
> **Milestone I1 update:** the native runtime can now be *observed*
> read-only -- `membrane observe` -- see [observability.md](observability.md).
> Only membrane-native is observable in I1; no Ollama observation route.

## 1. Why this exists

MEMBRANE has, until now, been a single thing: a local inference runtime
(native llama.cpp) with a planner (Planner v2, Milestone G) layered on
top of it. The strategic direction is to turn MEMBRANE into an
inference-engine-independent hardware/memory control plane:

```
    Applications / Agents / IDEs / Open WebUI
                    |
                MEMBRANE
      planner + memory intelligence + policy
                    |
          -------------------------
          |           |           |
       native       Ollama       vLLM
      llama.cpp       ...          ...
          |           |           |
             hardware / memory
```

Getting there requires a layer that can say, precisely, what any given
runtime can and cannot do -- otherwise the planner (or a future
control-loop) has no honest way to ask "can this runtime satisfy this
plan" before an external adapter exists at all. That is the entire
scope of H1: **runtime identity + capability model + negotiation + CLI
introspection.** It does not add an Ollama or vLLM adapter, does not
change execution behavior, and does not touch Planner v2's own math or
readiness classification.

## 2. Where the naming comes from

"Runtime" (native vs. an external inference engine) is a deliberately
new, third concept in this codebase, chosen specifically to avoid two
existing, unrelated uses of the word "backend":

- `include/membrane/backend.h` -- the cold-tier **KV-block storage**
  backend (file/RAM/CXL/FPGA).
- `tools/membrane-run/gpu_device.h`'s own `backend` field -- the
  **ggml compute backend** name (CPU/CUDA/Vulkan/Metal) a single
  native runtime may use internally.

Neither of those is what "which inference engine is executing this
model" means. "Runtime" is that third, orthogonal concept.

## 3. The native runtime

H1 implements exactly one real runtime: `membrane-native`. It
describes MEMBRANE's own existing, unchanged native runtime -- the
llama.cpp engine linked directly into the `membrane`/`membrane-run`
binaries (`tools/membrane-run/runtime_session.cpp`, `tools/membrane-
llama-runtime/decode_loop.cpp`). Its execution mode is
`embedded_native`: the engine runs *inside* MEMBRANE's own process,
regardless of the fact that `membrane chat` itself is a separate
loopback-HTTP client of `membrane serve` (a MEMBRANE-internal
client/server split, not evidence the engine is external -- see
`runtime_capabilities.h`'s own top comment).

Two string identifiers were reserved in H1 for a future external-runtime
adapter: `ollama` and `vllm`. **H2:** `ollama` now has a real, read-only
adapter and is listed (see [runtime-ollama.md](runtime-ollama.md)).
`vllm` is still reserved only: `membrane runtime list` does not show it,
and `membrane runtime inspect vllm` fails with a message that
distinguishes "reserved, no adapter yet" from a genuinely unknown id.

**H2 adapter boundary.** `runtime_capabilities.h/.c` stays pure (no I/O)
and still describes only the built-in native runtime.
`tools/membrane/runtime_adapter.h` adds a fixed, statically-linked
table (`runtime_registry.cpp`): `membrane-native` (backed by the pure
module above) and `ollama` (`runtime_ollama.h/.cpp`, the only file that
speaks Ollama's HTTP API). Each row is plain function pointers:
`describe` (may make one bounded probe), `list_models` and
`inspect_model` (NULL for membrane-native, whose models are MEMBRANE's
own registry). No plugin loading, no dynamic discovery.

## 4. Capability model

`membrane_runtime_capabilities_t`
(`tools/membrane-run/runtime_capabilities.h`) is one flat struct of 23
fields, each a 4-state `membrane_capability_state_t`:

- `SUPPORTED` -- real, wired-through, verified against current code.
- `PARTIAL` -- some, but not all, of what the capability implies is
  real (e.g. a static byte estimate exists, a live measurement does
  not).
- `UNSUPPORTED` -- does not exist in this codebase.
- `UNKNOWN` -- no basis to claim either way (the zero value, so an
  unpopulated capabilities struct never accidentally claims support).

`UNKNOWN` and `PARTIAL` are both treated as "cannot be relied on" by
negotiation (see below) -- neither is ever silently read as
`SUPPORTED`.

### 4.1 Native runtime capability matrix

Every value below was set by reading this exact codebase (H1's own
audit), never guessed. See `tools/membrane-run/runtime_capabilities.c`
for the field-by-field citation comments.

| Capability | State | Basis |
|---|---|---|
| Model enumeration | SUPPORTED | `registry_core.h`/`model_catalog.h`, `membrane model list/search/info`, `/v1/models` |
| Model switch | SUPPORTED | real `POST /membrane/v1/models/activate` (`server.cpp`) |
| Model load/unload | PARTIAL | load is explicit and real; there is no explicit "unload NAME" verb, only automatic non-pinned LRU eviction |
| Model metadata | SUPPORTED | `membrane model inspect/info` |
| Context control | SUPPORTED | `decode_loop.cpp`'s `cp.n_ctx`, `membrane plan --ctx` |
| GPU layer control | SUPPORTED | `runtime_session.cpp`'s `mp.n_gpu_layers`, `membrane plan --gpu-layers` |
| Quant/variant control | SUPPORTED | `variant_selector.h`, registry/catalog variant field, `membrane model install --quant` |
| KV precision control | SUPPORTED | `decode_loop.cpp`'s `cp.type_k`/`cp.type_v`, `membrane plan --kv` |
| KV placement control | SUPPORTED | `kv_residency_policy.h`'s `MEMBRANE_KV_PLACEMENT_*`, real and wired through |
| Concurrency control | PARTIAL | `decode_concurrency.h`'s gate is real, but only via the `MEMBRANE_MAX_CONCURRENT_DECODE` env var read once at `membrane serve` startup -- no live per-request/per-plan dial, and `membrane_plan_t` has no concurrency field at all |
| Device selection | SUPPORTED | real `--device` on `membrane-run` (`membrane_select_gpu_device()`) |
| Memory/headroom telemetry (planning) | SUPPORTED | `membrane_plan_t`'s own `feasibility.host_headroom_bytes`/`max_feasible_context`, `membrane plan --json` |
| Chat completions | SUPPORTED | real `POST /v1/chat/completions` |
| Streaming | SUPPORTED | real SSE (`set_chunked_content_provider`) |
| Cancellation | SUPPORTED | real end-to-end `cancel_flag`/`gen_cancel_flag` path |
| Current model (observability) | SUPPORTED | `/v1/status`'s `resident_models[].model`/`state`, live |
| Active context (observability) | PARTIAL | a completed response echoes its own `context`/`ctx_size`; no ongoing per-resident-model context size is exposed by `/v1/status` |
| RAM usage (observability) | PARTIAL | only a static planner-time byte estimate (`estimated_model_bytes`), never a live RSS read |
| VRAM usage (observability) | PARTIAL | only device-wide free/total snapshots and a static estimate, never a live per-workload VRAM read |
| KV cache usage (observability) | PARTIAL | only `estimated_kv_bytes`, a static estimate |
| Loaded model memory (observability) | PARTIAL | only `estimated_model_bytes`, a static estimate |
| Live KV migration | UNSUPPORTED | does not exist anywhere in this codebase |
| Dynamic reconfiguration | UNSUPPORTED | a session's context/GPU-layers/KV-precision are fixed at `llama_init_from_model()` construction time; there is no live reconfiguration path |

### 4.2 Capability provenance

`membrane_capability_provenance_t` records where a capability's value
came from. H1's entire native matrix is `static_contract` -- a value
established by reading the codebase once, not by a live probe of the
running binary (no `#ifdef`/build-flag check, no call into
`gpu_device.h`/`service_state.h`). `compiled_in`, `runtime_probe`,
`api_probe`, and `version_probe` exist as an agreed vocabulary for a
future adapter's own provenance, not because H1 produces them.

**H2:** the descriptor also carries `version_provenance`. The Ollama
adapter's capability matrix is `static_contract` (derived from Ollama's
documented API, not probed), while its version is `api_probe` (read
live from `GET /api/version`). membrane-native's `version_provenance`
is `unknown`, since it reports no version.

## 5. Availability vs. running state

Four distinct questions exist and must not be conflated:

1. **Does the runtime exist** (as a concept this build knows about)?
2. **Is the runtime available** (can this build, in principle, use
   it)?
3. **Is the runtime healthy** (would it actually work right now)?
4. **Is it currently running** (is a process for it up)?

H1 only answers (1) and (2). For `membrane-native`, availability is a
**static fact**: the native engine is linked directly into the
`membrane`/`membrane-run` binaries, unconditionally -- there is no
build configuration in which the binary exists but the native runtime
does not. `membrane runtime inspect membrane-native` reports
`available` even when no `membrane serve`/background service process
is running anywhere on the host (verified by `test_runtime_cmd.cpp`'s
own `test_available_without_any_service_running`).

This module never reads or mutates service/process state at all --
`membrane status` and `membrane service status` remain the separate,
unchanged commands that answer "is the background HTTP service
actually up right now." `membrane runtime inspect` never starts,
stops, or queries either. (3) and (4), and any live health-probe
daemon, are explicitly out of scope for H1 (see &sect;8).

**H2:** the descriptor gains a separate `health` field answering (3)
for external runtimes after one bounded probe: `unreachable`,
`healthy`, `incompatible`, or `unknown` (malformed answer).
membrane-native always reports `not_probed`: it is embedded, so there is
no separate process to probe. For Ollama, "adapter exists" never means
"Ollama is running": an unreachable daemon is listed as a known runtime
whose status is `unavailable`. Still no health-probe daemon, retry
loop, or monitoring.

## 6. Negotiation

`membrane_runtime_negotiate_plan()` (`runtime_capabilities.h`) answers
"can this runtime satisfy this plan" for an already-assembled Planner
v2 `membrane_plan_t`, read-only:

- Input: a runtime's `membrane_runtime_capabilities_t` and a
  `membrane_plan_t`.
- Which dimensions the plan actually relies on is derived only from
  fields `membrane_plan_t` already has: `decisions.has_decisions`
  (context/GPU-layers/KV-precision/KV-placement are populated
  together, or not at all) and `identity.variant_known` (a specific
  quant was pinned). A plan that relies on nothing (e.g. a
  catalog-only model with no installed GGUF) always negotiates
  `FULLY_SUPPORTED` -- there is nothing to fail on.
- A dimension counts as satisfied only when the matching capability is
  exactly `SUPPORTED`; `PARTIAL` and `UNKNOWN` both count as
  unsatisfied and both appear in the outcome's `unsupported[]` list.
- Output: `FULLY_SUPPORTED` / `PARTIALLY_SUPPORTED` / `UNSUPPORTED`,
  plus the specific unsupported dimension names (`CONTEXT_CONTROL`,
  `GPU_LAYERS_CONTROL`, `QUANT_VARIANT_CONTROL`,
  `KV_PRECISION_CONTROL`, `KV_PLACEMENT_CONTROL`).

Negotiation never mutates its inputs, never calls
`membrane_plan_assemble()` or any planner function, and never applies
anything -- it is purely advisory, exactly like `membrane plan` itself.
H1 does not expose it through the CLI (no `--plan MODEL` option on
`membrane runtime inspect`); it is tested directly at the library
level (`test_runtime_capabilities.c`) and left for a later milestone to
surface, per the H1 task's own "prefer keeping H1 small" guidance.

## 7. `membrane runtime list` / `membrane runtime inspect`

```
$ membrane runtime list          # H2: Ollama adapter present, daemon down
ID                 TYPE       STATUS       VERSION
membrane-native    native     available    -
ollama             external   unavailable  -

$ membrane runtime inspect membrane-native
Runtime
  ID: membrane-native
  Type: native
  Execution mode: embedded_native
  Status: available

Capabilities
  Model lifecycle:
    Enumeration: supported
    Switch: supported
    Load/unload: partial
    Metadata: supported
  Planning control:
    Context: supported
    GPU layers: supported
    Quant/variant: supported
    KV precision: supported
    KV placement: supported
    Concurrency: partial
    Device selection: supported
    Memory/headroom telemetry (planning): supported
  Inference:
    Chat completions: supported
    Streaming: supported
    Cancellation: supported
  Observability:
    Current model: supported
    Active context: partial
    Memory telemetry (RAM/VRAM/KV/model): partial
  Advanced:
    Live KV migration: unsupported
    Dynamic reconfiguration: unsupported
```

The "Memory telemetry" line is the only rollup in the human view (a
worst-of-4 summary over `ram_usage`/`vram_usage`/`kv_cache_usage`/
`loaded_model_memory`, kept so the output stays scannable); `--json`
always itemizes all four separately, never summarized.

Both commands accept `--json` and never require a model to be loaded or
a service to be running. **H2:** they now make one bounded loopback HTTP
probe of the Ollama adapter's endpoint (`GET /api/version`); for
membrane-native they still make no network access at all. H2 also adds
`membrane runtime models ID` and `membrane runtime model inspect ID
MODEL`, which work only for external runtimes (see runtime-ollama.md).
`membrane runtime inspect membrane-native` prints exactly what it did
in H1. Health, Endpoint, Version, and Reason lines appear only when
they carry information.

## 8. JSON schema

`schema_version: 1` (`MEMBRANE_RUNTIME_SCHEMA_VERSION`). `membrane
runtime list --json`:

```json
{
  "schema_version": 1,
  "membrane_version": "1.0.0",
  "runtimes": [
    {
      "id": "membrane-native",
      "display_name": "MEMBRANE native (llama.cpp)",
      "type": "native",
      "execution_mode": "embedded_native",
      "status": "available",
      "unavailable_reason": null,
      "health": "not_probed",
      "version": null,
      "version_provenance": "unknown",
      "endpoint": null,
      "capability_provenance": "static_contract",
      "capabilities": { "...": "23 capability-name -> state string fields" }
    }
  ]
}
```

`health` and `version_provenance` were added in H2. They are additive, so
`schema_version` stays 1. `unavailable_reason` is now also filled when
`status` is `unknown` (a malformed probe answer).

`membrane runtime inspect ID --json` is the same shape, with a single
`"runtime"` object in place of `"runtimes"`. Both are produced by
`membrane_runtime_list_json()`/`membrane_runtime_inspect_json()`
(`tools/membrane/runtime_cmd.h`), which tests call directly -- no
stdout capture or human-string parsing required.

## 9. Read-only guarantee

`membrane runtime list`/`inspect` touch no filesystem state at all:
no registry, no server config, no service unit, nothing. This is
stronger than `membrane plan`'s own read-only guarantee (which reads,
but never writes, the registry) -- `runtime_cmd.cpp` has no dependency
on `registry_core.h`, `server_config.h`, or `service_state.h` at all.
`test_runtime_cmd.cpp`'s `test_dispatch_creates_no_files` proves this
by pointing `MEMBRANE_MODELS_PATH`/`MEMBRANE_SERVER_CONFIG_PATH`/
`MEMBRANE_SYSTEMD_USER_DIR` at an isolated empty directory, running
every command several times, and asserting the directory is still
empty afterward.

## 10. What is NOT implemented yet

- ~~No Ollama adapter.~~ **H2:** read-only Ollama adapter (no inference,
  no control); see runtime-ollama.md.
- **No vLLM adapter.** `vllm` is a reserved string id only.
- **No LM Studio, or any other external runtime.**
- No dynamic/plugin loading of runtimes (a fixed, static table today).
- No live health-probe daemon. H2 makes only one bounded, on-demand probe
  per command.
- ~~No CLI for negotiation.~~ **H3:** `membrane plan MODEL --runtime ID`
  (read-only assessment; see runtime-plan-assessment.md).
- No wiring of negotiation results into `membrane use`/`membrane
  serve`'s actual execution path -- this stays purely advisory.
- No change to Planner v2's own math, plan representation, or
  `READY_FOR_READ_ONLY_ONLY` classification.
- No live memory/VRAM/KV-cache usage telemetry (this was already true
  before H1; the capability model now discloses it explicitly instead
  of leaving it implicit).

These are explicitly deferred to a later milestone (H2+), not silently
dropped.
