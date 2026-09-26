# Observation layer (Milestone I1 + I2)

> **`membrane observe` only reports facts.** It makes no recommendations and
> classifies nothing as "pressure". It keeps no history. It does not start,
> stop, load, activate, download or reconfigure anything. As of Milestone I2
> it can also observe Ollama (`--runtime ollama`), still read-only and still
> with no recommendations -- **Ollama observation does not change Ollama's
> runtime state**, and cross-runtime comparison/recommendation is deferred to
> Milestone I3.

Milestone I asks one question: *what is happening on this machine right now?*
I1 adds three things:

- a common observation model: `tools/membrane-run/observation.h/.c`, pure C
  with no I/O;
- the first native provider: `tools/membrane/observe_cmd.cpp`;
- a read-only command: `membrane observe [--runtime membrane-native] [--json]`.

I2 adds a second provider over the SAME model -- never a second telemetry
model (section 8 below):

- `GET /api/ps` added to the Ollama adapter's allowlist
  ([runtime-ollama.md](runtime-ollama.md));
- the Ollama observation provider: `tools/membrane/observe_ollama.cpp`;
- `membrane observe --runtime ollama`;
- `tools/membrane/observe_shared.cpp`, the runtime-agnostic host/GPU
  mapping and JSON/human rendering I1's native provider and I2's Ollama
  provider both call, pulled out of `observe_cmd.cpp` so neither provider
  depends on the other.

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

### JSON document (`schema_version` 2 as of I2)

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
`resident_models[]` entries are NOT part of that field table (they are a
variable-length array); their own shape is fixed and documented in section 9.

I2 bumped `schema_version` from 1 to 2 because it grew `resident_models[]`
entries by 7 keys (`digest`, `family`, `quant`, `reported_size_bytes`,
`reported_gpu_bytes`, `reported_context`, `expires_at`) so ONE resident-model
shape can carry both the native server's facts and Ollama's -- no existing
v1 key changed meaning, and a v1 consumer reading a v2 document still finds
every key it expects.

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
inference route and (for the NATIVE provider) never calls an Ollama route at
all. `test_observe_cmd.cpp` checks this in three ways:

- In an isolated environment, config and registry files stay byte-identical
  and no file or unit directory is created.
- Against an in-process mock native server that traps
  `/v1/chat/completions`, `/membrane/v1/models/activate`, `pin`, `unpin`,
  `/v1/models` and `/membrane/v1/capabilities`, the only request received is
  exactly one `GET /v1/status`.
- The service probe is overridden, so no real systemctl call happens.

## 8. Ollama observation (Milestone I2)

`membrane observe --runtime ollama` maps `GET /api/ps` onto the SAME
snapshot model as section 5 -- no second telemetry model. The exact
`/api/ps` contract (audited against ollama/ollama v0.34.3's own
`api/types.go` `ProcessModelResponse`, `server/routes.go` `PsHandler` and
`docs/openapi.yaml`) is documented in
[runtime-ollama.md](runtime-ollama.md) section 1; this section covers only
the MAPPING onto MEMBRANE's observation snapshot.

### 8.1 What `/api/ps` returns, and what MEMBRANE calls to get it

Two network calls, in order, and never more:

1. `GET /api/version` -- the SAME discovery probe `membrane runtime inspect
   ollama` already performs (H2). If this does not report the runtime
   AVAILABLE, `/api/ps` is never requested at all.
2. `GET /api/ps` -- only when (1) succeeded. Always answers `200` with
   `"models": []` when nothing is loaded (not an error -- section 8.5).

Each entry: `name`, `model`, `size`, `digest`,
`details.{family,quantization_level}`, `expires_at` (RFC 3339, verbatim),
`size_vram`, `context_length`. `context_length` here is the model's
CURRENTLY LOADED context, never its trained maximum (that maximum, when
known at all, only ever comes from `/api/tags`/`/api/show`, which this
command never calls -- see 8.6). Ollama omits `size_vram` entirely when it
is 0 rather than send an explicit `0` (upstream issue #4840); MEMBRANE
treats an absent `size_vram` key as a documented, known 0 bytes, and only a
PRESENT key of the wrong JSON type as unknown -- the two are not the same
kind of "missing".

### 8.2 Provenance mapping (Part 7 -- the rule that matters most)

Everything `/api/ps` reports is `runtime_reported`, **never** `measured` and
**never** `estimated`:

| Field | Provenance | Why |
|---|---|---|
| `model.resident_models[].reported_size_bytes` (`size`) | `runtime_reported` | the scheduler's own bytes accounting -- not a load-time guess the way the native server's `estimated_model_bytes` is (that field stays `estimated` for native; I2 never reuses it for Ollama's number) |
| `model.resident_models[].reported_gpu_bytes` (`size_vram`) | `runtime_reported` | THIS model's own GPU allocation, per the scheduler |
| `model.resident_models[].reported_context` (`context_length`) | `runtime_reported` | THIS model's loaded context, per the scheduler |
| `model.resident_models[].{digest,family,quant,expires_at}` | `runtime_reported` | verbatim from `/api/ps` |
| `gpu.vram_total_bytes`, `vram_free_bytes` (device-wide) | `measured` | MEMBRANE's OWN device probe (section 8.4) -- independent of Ollama entirely |
| `host_memory.*` | `measured` | MEMBRANE's OWN host probe, same as the native path |
| `runtime.availability` | `measured` | I2 just probed it live (`GET /api/version`); unlike native's compiled-in `static_metadata` availability, this is a real measurement every snapshot |
| `service.server_version` | `runtime_reported` | from `/api/version` |
| `service.endpoint` | `configured` | `MEMBRANE_OLLAMA_ENDPOINT` or the documented default |
| `service.reachable` | `measured` | this snapshot's own probe |

Fields MEMBRANE's own registry/config/Planner v2 own (`model.configured*`,
`model.file_size_bytes`, `model.quant`, `model.arch`, `context.planned`,
`context.model_max`, `kv.planned_*`, `kv.estimated_bytes`,
`service.manager/installed/active`) are honestly `unknown`, source
`not_applicable_external_runtime` -- an externally-owned Ollama model has no
MEMBRANE registry entry, no GGUF file MEMBRANE reads itself, and no
Planner v2 plan (I3 owns any future cross-runtime comparison, not I2).
`kv.measured_bytes` stays `unknown` (`not_instrumented`) for the same reason
it does for native: no live KV instrumentation exists anywhere yet.

### 8.3 Multiple loaded models

Ollama may report zero, one, or several loaded models. I1's
`resident_models[]` (`MEMBRANE_OBS_MAX_RESIDENT` = 8 entries) was already
designed as an array precisely because the native server can, in principle,
report more than one resident model too -- I2 reuses it as-is (no new
container type):

| Case | `model.resident_count` | `model.resident_models[]` | `context.active` |
|---|---|---|---|
| one native active model (I1, unchanged) | as reported by `/v1/status` | native's own fields filled, Ollama's I2 fields unknown | unknown (native never reports it) |
| zero Ollama models | `0`, known, `runtime_reported` (Part 14: NOT an error) | `[]` | unknown, `no_loaded_model` |
| one Ollama model | `1` | 1 entry, Ollama's I2 fields filled, native-only fields unknown | filled from that model's `context_length`, if reported |
| several Ollama models | the TRUE total, even beyond 8 | up to 8 entries, in Ollama's own order -- never silently dropped down to just the first | unknown, `ambiguous_multiple_loaded_models` (picking one model's context as "the" active context would be a guess) |

If more than 8 models are loaded, `resident_count` still reports the real
total; only the per-model detail array is capped at 8 (a documented
limitation, section 8.7).

### 8.4 Device-wide vs. model-specific memory (Part 8)

Two different kinds of fact, never conflated:

- **Device-wide** (`gpu.vram_total_bytes`, `vram_free_bytes`,
  `vram_used_bytes`): MEMBRANE's OWN device probe
  (`membrane_gpu_list_devices()`), `measured`, identical to the native
  path's section 5 fields, and completely independent of Ollama -- it is
  the first GPU/iGPU ggml enumerates, the same device `membrane plan` uses.
- **Model-specific** (`resident_models[].reported_gpu_bytes`): Ollama's own
  `size_vram` for THAT ONE model, `runtime_reported`.

`device used VRAM == Ollama model VRAM` is never implied or computed --
they are two separate numbers with two separate provenances, printed side
by side (see the CLI example in 8.6) but never subtracted, summed or
compared against each other.

### 8.5 Zero loaded models

A healthy, reachable Ollama daemon reporting `{"models": []}` is
`runtime_observable: true`, `status` `partial` (most fields are still
`not_applicable_external_runtime`) or `complete`, and
`model.resident_count` is a real, known `0` -- never `unavailable` and
never `unknown` (Part 14).

### 8.6 Unavailable / partial / malformed `/api/ps`

- **Ollama unreachable** (`GET /api/version` fails, times out, or answers
  incompatibly): `runtime_observable: false`, `status: unavailable`,
  nothing else is probed (no host/GPU read, no `/api/ps` call), and the
  human/JSON output never contains a raw socket error string -- the same
  `UNREACHABLE`/`INCOMPATIBLE`/`UNKNOWN` health classification H2 already
  defined (runtime-ollama.md section 2).
- **`/api/ps` itself fails** (malformed JSON, non-2xx, oversized response --
  bounded the same way as H2's other calls, 16 MiB cap, 5 s read timeout, no
  retries): `runtime_observable` stays `true` (Ollama itself WAS reached),
  and only `model.resident_count`/`resident_models`/`context.active`
  degrade to `unknown` with a `ps_<error_code>` reason
  (e.g. `ps_malformed_response`). Host/GPU/service fields, already
  collected independently, are unaffected. `status` is `partial`, never
  `unavailable` -- one broken optional call does not destroy the whole
  observation.
- Minimal `/api/ps`-only observation: this command never calls
  `/api/tags` or `/api/show` to backfill metadata `/api/ps` omits (trained
  max context, MEMBRANE registry cross-reference, ...). If a future
  milestone needs that, it must be a deliberate, justified, separately
  reviewed addition -- not something I2 does implicitly.

### 8.7 `membrane observe --runtime ollama` -- CLI shape

```
$ membrane observe --runtime ollama          # Ollama unreachable
Observation
  Runtime:               ollama
  Status:                unavailable (1/35 fields known)
  Timestamp:              ...

$ MEMBRANE_OLLAMA_ENDPOINT=http://127.0.0.1:PORT membrane observe --runtime ollama
Observation
  Runtime:               ollama
  Status:                partial (12/35 fields known)
  Timestamp:              ...

Host memory
  Total:                 5.63 GiB       [measured]
  Available:             1.14 GiB       [measured]
  ...

GPU (first GPU/iGPU enumerated -- ... device-wide, independent of the observed runtime)
  GPU devices:           0              [measured]
  ...

Model
  Configured:            unknown (not_applicable_external_runtime)
  Resident models:       2              [runtime_reported]
    - qwen2.5:7b
        size 4.36 GiB, gpu 4.36 GiB, context 4096, expires 2026-09-26T20:00:00Z [runtime_reported]
    - llama3.2:3b
        size 1.88 GiB, gpu 0.0 MiB, context 2048, expires 2026-09-26T20:05:00Z [runtime_reported]

Context
  Active:                unknown (ambiguous_multiple_loaded_models)
  ...

Service
  Endpoint:              http://127.0.0.1:PORT [configured]
  Reachable:             yes            [measured]
  Server version:        0.34.3         [runtime_reported]
```

(Produced by the real binary against a loopback fixture server, not a real
Ollama -- section 10 of runtime-ollama.md and I2's own final report have the
exact commands.) `membrane observe` and `membrane observe --runtime
membrane-native` are unaffected -- the default stays native (Part 12); an
external runtime always needs an explicit `--runtime ollama`. Exit codes
are unchanged: 0 for complete/partial, `MEMBRANE_EXIT_RUNTIME_ERROR` (4) for
unavailable, `CLI_ERROR` (2) for `--runtime vllm` (still not implemented) or
an unknown id.

### 8.8 Read-only guarantee (Ollama path)

`membrane observe --runtime ollama` never loads/unloads a model, never
changes `keep_alive`/`num_ctx`/`num_gpu`, never sends an inference request,
never pulls/deletes/creates/copies a model, never starts/stops Ollama, and
never writes to MEMBRANE's own registry, config or service state.
`test_observe_ollama.cpp` proves this the same way H2's own tests do: an
in-process mock traps `/api/generate`, `/api/chat`, `/api/embed(dings)`,
`/api/pull`, `/api/push`, `/api/create`, `/api/copy`, `/api/delete`, every
`/v1/*` route, and even `/api/tags`/`/api/show` (Part 5: minimal `/api/ps`-
only observation), and asserts the command only ever reaches exactly
`GET /api/version` and `GET /api/ps`. A separate isolated-environment test
asserts no registry/config/service file is ever created.

### 8.9 Limitations

- No per-layer GPU-offload count, no KV-cache precision, and no GPU/CPU
  "processor split" field -- none of these are in `/api/ps` at all (the
  `ollama ps` CLI's PROCESSOR column is computed client-side from
  `size`/`size_vram`; MEMBRANE does not reproduce that arithmetic).
- The per-model detail array caps at 8 entries (`MEMBRANE_OBS_MAX_RESIDENT`,
  shared with native); the reported count never does.
- No comparison, recommendation, or "which runtime is better" logic of any
  kind -- deferred to I3 in full (section 9).
- No historical telemetry, same as I1.

## 9. Not implemented yet

- **Recommendations are not implemented.** There are no
  MEMORY_PRESSURE / HEADROOM_LOW / CONTEXT_NEAR_LIMIT classifications, no
  downgrade or runtime-switch suggestions, and no policy-adjusted headroom.
  These belong to I3.
- **Historical telemetry is not implemented.** Nothing is stored between
  invocations.
- Server-process RSS is not implemented. There is no RSS probe in the
  codebase, and the observing CLI's own RSS would say nothing about the
  server.
- Live KV bytes are not implemented for any runtime. The native server's
  active context is not implemented either (it does not expose one); Ollama's
  active context IS implemented (I2, section 8), but only when exactly one
  model is loaded (section 8.3).
- Per-process GPU memory is not implemented. `vram_used_bytes` is
  device-wide.
- Only the first GPU/iGPU is described in detail. `gpu.device_count` counts
  all of them.
