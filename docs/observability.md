# Observation layer (Milestone I1)

> **`membrane observe` only reports facts.** It makes no recommendations and
> classifies nothing as "pressure". It keeps no history. It does not start,
> stop, load, activate, download or reconfigure anything. It does not observe
> Ollama yet.

Milestone I asks one question: *what is happening on this machine right now?*
I1 adds three things:

- a common observation model: `tools/membrane-run/observation.h/.c`, pure C
  with no I/O;
- the first native provider: `tools/membrane/observe_cmd.cpp`;
- a read-only command: `membrane observe [--runtime membrane-native] [--json]`.

## 1. Purpose

MEMBRANE already reads, estimates or records memory-related numbers in many
places: host RAM, device VRAM, the registry, GGUF headers, Planner v2 and the
running server's own status. Before I1, each number had a different meaning
depending on which command printed it. Nothing marked "the OS said so" apart
from "MEMBRANE calculated this".

The observation layer collects these numbers into one **point-in-time
snapshot**. Each number carries a **provenance**, so that measured facts and
estimates can sit side by side without being confused:

```
GPU
  VRAM free:             4.15 GiB       [measured]
KV
  Estimated footprint:   170.0 MiB      [estimated]
  Measured usage:        unknown (not_instrumented)
```

## 2. Point-in-time semantics

- A snapshot is taken once per invocation. There is no daemon, no sampling
  loop, no database and no stored history.
- `timestamp.utc` / `timestamp.unix_ms` is the **wall-clock (UTC) time at the
  start of collection**. It comes from `std::chrono::system_clock` and is
  formatted without the platform `gmtime` variants.
- `timestamp.collection_duration_ms` is measured with a monotonic clock
  (`std::chrono::steady_clock`). Every probe runs inside that window, one
  after another. A snapshot is therefore not atomic across sources, but it
  spans a short, known window (roughly 150–250 ms on the development host,
  mostly GPU enumeration).
- Individual fields carry no timestamp of their own. They all belong to the
  snapshot's window.

## 3. Provenance types

| Provenance | Meaning | Examples |
|---|---|---|
| `measured` | Read from the OS, a device or the service manager during this snapshot, or derived **only** from such reads | `/proc/meminfo` MemAvailable; ggml device memory; systemctl state; "is the server reachable" |
| `runtime_reported` | Stated by the running runtime itself | the native server's `GET /v1/status`: resident models, backend, version |
| `estimated` | Calculated by MEMBRANE; never a measurement | Planner v2's planned context, KV precision and KV bytes; the server's own load-time byte estimates |
| `static_metadata` | Facts recorded about an artifact | GGUF architecture and max context; registry file size; catalog quant match; the compiled-in runtime availability |
| `configured` | The user's persisted intent in `server.json` | default model; endpoint |
| `unknown` | Not known. The value is `null` and the `source` says why | server RSS; active context; measured KV |

`configured` is the only category added beyond the five that were required.
Without it, a configured default model would have to be mislabeled as either
live state or metadata.

### Field wrapper

Every telemetry field is a small struct: `known`, `value`, `provenance` and
`source`. In JSON it looks like this:

```json
"available_bytes": {"value": 1277952000, "known": true,
                    "provenance": "measured", "source": "proc_meminfo"}
```

The setters enforce these invariants, and `test_observation.c` asserts them:

- `known == false` implies `provenance == "unknown"` and `value == null`.
  An unknown field is **never** rendered as `0`.
- `known == true` implies a real provenance. Calling a setter with
  `UNKNOWN` provenance produces an unknown field.
- A derived field (used RAM, used VRAM, headroom) is filled only when all of
  its inputs are known, consistent and share one provenance. If the inputs
  mix provenances, the derived field is refused rather than blended, so
  "measured − estimated" is never reported as measured.

## 4. Measured vs runtime-reported vs estimated

This rule matters more than any other in I1:

- `kv.estimated_bytes` is **Planner v2 arithmetic**: the joint planner's
  selected candidate at the planned context, GPU-resident plus host-resident
  KV. It is never reported as "the KV cache currently uses N bytes".
- `kv.measured_bytes` stays `unknown` (`not_instrumented`). No live
  KV-allocation instrumentation exists yet.
- `context.planned` (estimated) and `context.active` (unknown) are separate
  fields. The native server reports `context_policy: automatic`, not a live
  `n_ctx`.
- For resident models, the server's `estimated_model_bytes` and
  `estimated_kv_bytes` are **reported by the runtime** but are still
  **estimates** (`runtime_session.cpp` computes them at load time). They are
  labeled `estimated` with source `server_status:load_time_estimate`.
- A configured default model (`model.configured`, `configured`) is not
  evidence that anything is loaded. Residency comes only from the running
  server (`model.resident_count`, `runtime_reported`). If the server cannot
  be reached, residency is `unknown`, **not** zero.

## 5. Native observation fields

| JSON path | Provenance | Source |
|---|---|---|
| `runtime.availability` | static_metadata | `runtime_capabilities` (compiled-in contract) |
| `host_memory.total_bytes`, `available_bytes` | measured | `proc_meminfo` (Linux), `GlobalMemoryStatusEx` (Windows), `mach_host_statistics64` (macOS); all through the existing `membrane_read_host_meminfo()` |
| `host_memory.used_bytes` | measured | derived: total − available |
| `host_memory.swap_total_bytes`, `swap_free_bytes` | measured on Linux; unknown elsewhere (`not_probed_on_platform`) | `proc_meminfo` |
| `host_memory.process_rss_bytes` | **unknown** | `not_instrumented` |
| `gpu.device_count` | measured | `ggml_backend_dev`: the number of GPU/iGPU devices (0 is a real measured fact) |
| `gpu.backend`, `device_name`, `device_description` | measured | `ggml_backend_dev`: the **first** GPU/iGPU, the same device `membrane plan` uses |
| `gpu.vram_total_bytes`, `vram_free_bytes` | measured | `ggml_backend_dev_memory`. Unknown when no GPU exists, or when the backend reports 0/0 |
| `gpu.vram_used_bytes` | measured | derived: total − free, **device-wide** (all processes, not only MEMBRANE) |
| `model.configured` | configured | `server_config.default_model` |
| `model.configured_registered` | static_metadata | registry |
| `model.file_size_bytes` | static_metadata | registry (the `stat()` size recorded when the model was added) |
| `model.arch` | static_metadata | GGUF header |
| `model.quant` | static_metadata | exact catalog filename match (the registry does not store quant) |
| `model.resident_count` + `model.resident_models[]` | runtime_reported (the byte figures are estimated) | `GET /v1/status` |
| `context.active` | **unknown** | `not_reported_by_runtime` |
| `context.planned` | estimated | `planner_v2` |
| `context.model_max` | static_metadata | GGUF header |
| `kv.planned_precision`, `kv.planned_placement` | estimated | `planner_v2` |
| `kv.estimated_bytes` | estimated | `planner_v2` |
| `kv.measured_bytes` | **unknown** | `not_instrumented` |
| `service.manager`, `installed`, `active` | measured | `service_state` (systemctl, launchctl or schtasks) |
| `service.endpoint` | configured | `server_config` |
| `service.reachable` | measured | `http_get_v1_status` |
| `service.server_version` | runtime_reported | `server_status` |
| `headroom.ram_bytes`, `headroom.vram_bytes` | measured | the raw available/free figures: **no** planner reserve and **no** planned footprint subtracted |

The Planner v2 estimates come from `membrane_plan_resolve_installed_v2()`. It
is the exact resolution that `membrane plan NAME` performs for a registered
model, returned in-process instead of printed. The estimates describe the
**installed** variant's plan, not a sibling variant that the planner merely
evaluated. Planner v2 itself is unchanged. The only change in
`plan_cmd.cpp` separates "resolve" from "render" for the installed-model
path, and `membrane plan`'s output is byte-for-byte the same.

### JSON document (`schema_version` 1)

```
schema_version, membrane_version, mode: "observe", ok, status,
timestamp {utc, unix_ms, clock, collection_duration_ms},
runtime {id, observable, availability},
host_memory {...}, gpu {...}, model {..., resident_models: [...]},
context {...}, kv {...}, service {...}, headroom {...},
fields {known, total, unknown: [paths]},
sources {<source>: [paths]}
```

All sections are always present. One ordered field table drives the section
fields, `fields.unknown` and `status`, so the three cannot disagree.

## 6. Partial observation semantics

The overall status is deterministic:

| Status | Rule | Exit code |
|---|---|---|
| `complete` | the runtime is observable **and** every field in the table is known | 0 |
| `partial` | the runtime is observable and at least one field is unknown | 0 |
| `unavailable` | the runtime itself cannot be observed | 4 (`MEMBRANE_EXIT_RUNTIME_ERROR`) |

An unknown field never fails the command. With membrane-native in I1,
`complete` is not reachable in practice: server RSS, the active context and
measured KV are not instrumented, so they are always unknown. Reporting
`partial` here is the honest answer, not a defect.

Asking for an unknown runtime id, or for a runtime that I1 cannot observe
(`ollama`, `vllm`), is a CLI error (exit 2). No network request is made in
that case.

## 7. Read-only guarantee

`membrane observe` uses only the existing read paths:
`membrane_read_host_meminfo()`, `membrane_gpu_list_devices()`,
`membrane_probe_service()`, `membrane_server_config_load()`,
`membrane_registry_load()`, `membrane_gpu_estimate_model()` (GGUF metadata
only, no tensor load), the Planner v2 resolution, and the bounded
`GET /v1/status` that `membrane status` already sends to MEMBRANE's **own**
configured endpoint.

It never starts or stops a service, never loads, activates, pins or downloads
a model, and never writes to the registry or config. It never calls an
inference route and adds no Ollama route. `test_observe_cmd.cpp` checks this
in three ways:

- In an isolated environment, config and registry files stay byte-identical
  and no file or unit directory is created.
- Against an in-process mock native server that traps
  `/v1/chat/completions`, `/membrane/v1/models/activate`, `pin`, `unpin`,
  `/v1/models` and `/membrane/v1/capabilities`, the only request received is
  exactly one `GET /v1/status`.
- The service probe is overridden, so no real systemctl call happens.

## 8. Not implemented yet

- **Ollama observation is not implemented in I1.** No `/api/ps` call is
  made, and the H2 Ollama route allowlist is unchanged. External-runtime
  observation belongs to I2.
- **Recommendations are not implemented in I1.** There are no
  MEMORY_PRESSURE / HEADROOM_LOW / CONTEXT_NEAR_LIMIT classifications, no
  downgrade or runtime-switch suggestions, and no policy-adjusted headroom.
  These belong to I3.
- **Historical telemetry is not implemented.** Nothing is stored between
  invocations.
- Server-process RSS is not implemented. There is no RSS probe in the
  codebase, and the observing CLI's own RSS would say nothing about the
  server.
- Live KV bytes and the active context are not implemented. The native
  server does not expose them.
- Per-process GPU memory is not implemented. `vram_used_bytes` is
  device-wide.
- Only the first GPU/iGPU is described in detail. `gpu.device_count` counts
  all of them.
