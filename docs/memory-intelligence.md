# Memory intelligence (Milestone I3)

> **MEMBRANE's recommendations are advisory. No configuration is
> automatically changed.** `membrane advise` never starts/stops a service,
> loads/unloads/switches a model, changes context/GPU-layers/KV-precision/
> KV-placement, or performs inference. It reads an observation snapshot
> (I1/I2, [observability.md](observability.md)) and a runtime's own
> capability matrix (H1/H2, [runtime-abstraction.md](runtime-abstraction.md))
> and produces findings and recommendations -- text, not action.

This completes Milestone I (`observe` -> facts, `advise` -> interpretation).
Milestone J, not started, is where MEMBRANE may one day apply anything.

## 1. What memory intelligence means

I1/I2 answer *"what is happening on this machine right now?"* with plain
facts, each carrying its own provenance (measured / runtime-reported /
estimated / static-metadata / unknown). I3 asks the next question:

> Given what MEMBRANE can observe right now, what does this mean, and
> what should the user consider doing?

The answer is always two lists -- **findings** (what is true, and how
sure MEMBRANE is) and **recommendations** (what could be done about it,
and whether the observed runtime can actually do it through MEMBRANE) --
never a single "health score", never an action taken on the user's
behalf.

## 2. Observation vs. advice: two commands, on purpose

| Command            | Answers                          | Recommendations? |
|---------------------|-----------------------------------|-------------------|
| `membrane observe`  | What is true right now?          | No                |
| `membrane advise`   | What does that mean, and what could be done? | Yes (advisory only) |

`membrane observe` stays exactly as I1/I2 left it -- unchanged by this
milestone. `membrane advise` is additive: it calls the *same* collection
functions `membrane observe` calls for the same `--runtime` (never a new
probe, never a new Ollama route beyond I2's existing `GET /api/version` +
`GET /api/ps` allowlist), then hands the resulting snapshot, plus that
runtime's own capability matrix, to a pure assembler.

```
observation snapshot (I1/I2)  +  runtime capabilities (H1/H2)
        |
        v
membrane_memory_intelligence_assemble()   (tools/membrane-run/memory_intelligence.h/.c)
        |
        v
findings[] + recommendations[] + reasons[]
```

`membrane_memory_intelligence_assemble()` is pure: no I/O, no model/device
access, deterministic given the same two inputs. It is not a second
planner (it never computes a context/GPU-layer/KV figure -- those numbers
already reached the snapshot from Planner v2, tagged `estimated`) and not
a capability negotiator (H1's `membrane_runtime_negotiate_plan()`/H3's
`membrane_runtime_recommend_plan()` already answer "can this runtime
satisfy THIS plan"; I3 answers the narrower "does this runtime expose
*any* control surface for the dimension a finding is about", by reading
the matching capability field directly).

## 3. The finding model

Each finding (`membrane_intel_finding_t`) carries:

- `code` -- a stable, documented reason code (section 4)
- `severity` -- `info` / `notice` / `warning` / `critical`
- `summary` -- one plain sentence
- `evidence[]` -- the concrete numbers behind it (e.g. "Available: 420
  MiB / 5.6 GiB (7.5%)")
- `provenance` -- reused verbatim from `observation.h`'s own vocabulary
  (`measured` / `runtime_reported` / `estimated` / `static_metadata` /
  `unknown`) -- never a second provenance model
- `dimension` -- `host_memory` / `vram` / `context` / `kv` /
  `model_residency` / `runtime_capability` / `telemetry`

Severity is conservative on purpose: `critical` is reserved for measured
host/device pressure at the thresholds in section 6; an estimate-only
finding never exceeds `notice` and never outranks a measured `critical`
finding in the overall status (section 9).

## 4. Finding and recommendation codes

Implemented, each backed by a specific snapshot/capability field (see
`memory_intelligence.h`'s own comment for exactly which):

| Code | Meaning | Provenance |
|---|---|---|
| `HOST_MEMORY_HEADROOM_LOW` / `_CRITICAL` | measured RAM headroom crossed a threshold | measured |
| `VRAM_HEADROOM_LOW` / `_CRITICAL` | measured device VRAM headroom crossed a threshold | measured |
| `CONTEXT_NEAR_MODEL_MAXIMUM` / `_EXCEEDS_MODEL_MAXIMUM` | planned context vs. the model's own trained maximum | estimated (planned side) |
| `KV_FOOTPRINT_ESTIMATE_HIGH` | Planner v2's estimated KV bytes is high relative to current measured RAM headroom | estimated |
| `MULTIPLE_MODELS_RESIDENT` | more than one model is currently resident (info by default) | runtime-reported |
| `KV_PRECISION_CONTROL_UNAVAILABLE` / `KV_PLACEMENT_CONTROL_UNAVAILABLE` | a KV plan exists, but this runtime does not expose that control through MEMBRANE | static (capability contract) |
| `GPU_LAYER_CONTROL_PARTIAL` | a GPU is present and this runtime's GPU-layer control is only partial | static (capability contract) |
| `CONTEXT_CONTROL_AVAILABLE` | this runtime exposes explicit context control (a positive disclosure, native today) | static (capability contract) |
| `OBSERVATION_PARTIAL` | some observation fields were unknown | measured (a count, not a guess) |
| `TELEMETRY_INCOMPLETE` | the runtime could not be observed at all, or no memory-relevant field was known | unknown |

Two of the task's originally-listed *potential* codes were deliberately
**not** implemented, because the snapshot has no distinct evidence for
them without fabricating one:

- `MODEL_MEMORY_HIGH_RELATIVE_TO_AVAILABLE_RAM` -- `model_file_size_bytes`
  is a static fact about the file on disk, not about what is actually
  loaded; conflating the two would misreport a merely-registered model as
  memory pressure.
- `ESTIMATED_CONTEXT_MEMORY_NEAR_LIMIT` -- the snapshot has no distinct
  "estimated total footprint at the requested context" figure separate
  from `kv_estimated_bytes`; `KV_FOOTPRINT_ESTIMATE_HIGH` already covers
  the case this would cover, under its real name.

Recommendation codes: `REDUCE_CONTEXT`, `USE_LOWER_MEMORY_VARIANT`,
`REDUCE_GPU_LAYERS`, `USE_LOWER_KV_PRECISION`, `CHANGE_KV_PLACEMENT`,
`UNLOAD_UNUSED_MODEL`, `SWITCH_TO_RUNTIME_WITH_DEEPER_CONTROL`,
`NO_ACTION_NEEDED`. None is a blanket "switch to native" -- see section 7.

## 5. Provenance-aware evidence

A finding's `provenance` field is never guessed -- it is copied from the
snapshot field(s) it was derived from. This means:

```
Measured:
    Available: 420 MiB / 5.6 GiB (7.5%)      [measured]
    -> HOST_MEMORY_HEADROOM_LOW, severity warning

Runtime-reported:
    2 models resident                         [runtime_reported]
    -> MULTIPLE_MODELS_RESIDENT, severity info

Estimated:
    Estimated KV: 640 MiB vs. headroom: 900 MiB   [estimated]
    -> KV_FOOTPRINT_ESTIMATE_HIGH, severity notice (never critical)
```

MEMBRANE never states "KV currently uses 640 MiB" -- only "KV is
*estimated* at 640 MiB". An estimate-only finding can never push the
overall status above what measured evidence alone would justify (an
all-`info` result plus one `notice` estimate never outranks a real
measured `critical`, section 9).

## 6. Threshold policy (heuristic, not a guarantee)

Explicit, deterministic, documented constants in `memory_intelligence.c`
-- never scattered magic numbers. They are **warning tripwires**, not a
claim of safety: crossing one does not mean a failure is imminent, and
staying under one does not mean anything is "safe", "guaranteed" or
"optimal" (those words do not appear anywhere in this module's own
output). They are not tuned to any one host.

| Check | Critical | Warning |
|---|---|---|
| RAM headroom ratio (`headroom / total`) | `< 5%` | `< 15%` |
| RAM headroom absolute | `< 256 MiB` | `< 768 MiB` |
| VRAM headroom ratio | `< 5%` | `< 15%` |
| Context vs. model maximum | planned `>` max | planned `/` max `>= 90%` |
| KV estimate vs. RAM headroom | -- | KV estimate `>` 50% of headroom |

RAM uses both a ratio and an absolute floor (whichever is more severe
wins) so tiny-memory systems are handled sensibly: 15% of 2 GiB is only
300 MiB, and 5% of 512 GiB is 25.6 GiB. VRAM uses ratio only. Every
boundary is strict `<` (a value exactly *at* a threshold does not cross
it) except the context-maximum ratio, which is inclusive `>=` -- both are
tested exactly at, one unit below, and one unit above in
`test_memory_intelligence.c`.

## 7. Capability-aware recommendations

Every recommendation names the exact runtime capability (H1's
`membrane_capability_state_t`) it was checked against, mapped to one of
four applicabilities:

| Capability state | Applicability |
|---|---|
| `SUPPORTED` | `controllable` |
| `PARTIAL` | `partially_controllable` |
| `UNSUPPORTED` | `unavailable_on_runtime` |
| `UNKNOWN` (or no capability matrix at all) | `advisory_only` (fail-closed default -- never upgraded to `controllable`) |

This is the mechanism behind the task's central "capability-aware
recommendation filter": a recommendation is never suppressed just because
a runtime cannot apply it -- it still surfaces, honestly labeled:

```
Lower KV precision could reduce KV memory.
    Runtime support: unavailable_on_runtime
    Reason: VRAM_HEADROOM_LOW
```

`SWITCH_TO_RUNTIME_WITH_DEEPER_CONTROL` only fires for a non-native
runtime, only under real (RAM/VRAM) pressure, and only when BOTH KV
precision and KV placement control are `UNSUPPORTED` -- framed around
capability availability, never superiority:

> This runtime does not expose KV precision or KV placement control
> through MEMBRANE; MEMBRANE-native exposes both.

Never "native is better".

## 8. Native runtime behavior

For `membrane-native`, `membrane advise` combines measured host RAM/VRAM,
Planner v2's own estimated context/KV figures (already in the snapshot,
tagged `estimated`), resident-model facts from `GET /v1/status`, and
membrane-native's own capability matrix (every planning/control dimension
`SUPPORTED` today). Expect deeper findings on average: native's own
`CONTEXT_CONTROL_AVAILABLE` disclosure, and no KV-precision/KV-placement
capability-limitation findings, since native supports both.

## 9. Ollama runtime behavior

For `ollama`, `membrane advise` combines the same measured host/device
facts (MEMBRANE's own probes, never Ollama's) with `GET /api/ps`'s
runtime-reported resident-model facts (I2) and Ollama's real, static
capability matrix (`kv_precision_control`/`kv_placement_control`/
`memory_headroom_telemetry` all `UNSUPPORTED`, `gpu_layer_control`/
`quant_variant_control` `PARTIAL`). Typical findings: a capability
disclosure for KV precision/placement, `GPU_LAYER_CONTROL_PARTIAL`, and
-- if more than one model is loaded -- `MULTIPLE_MODELS_RESIDENT`, which
only becomes an `UNLOAD_UNUSED_MODEL` *recommendation* when real memory
pressure is also present (never a blanket "multiple models is bad").
MEMBRANE never fabricates exact KV bytes, per-layer memory, planner
feasibility or a safe context for Ollama -- none of that is in `/api/ps`.

## 10. Overall status

`ok` / `notice` / `warning` / `critical` / `insufficient_data`, derived
only from `findings[]` (never hand-set): the highest severity present,
with one exception -- if the runtime could not be observed at all, or no
memory-relevant field in the snapshot was known, the result is
`insufficient_data` (a single `TELEMETRY_INCOMPLETE` finding, zero
recommendations) rather than a fabricated `ok`.

## 11. Read-only guarantee

- `membrane_memory_intelligence_assemble()` performs no I/O of any kind.
- `membrane advise`'s CLI layer performs **zero new probes**: it calls
  the exact same `membrane_observe_collect_inputs()`/
  `membrane_observe_ollama_collect_inputs()` I1/I2 already ship, so the
  Ollama route allowlist is unchanged (`GET /api/version`, `GET
  /api/tags`, `POST /api/show`, `GET /api/ps` -- never a mutating/
  inference route).
- Every finding/recommendation struct carries `mutates_state` (hardcoded
  `0`, never computed) -- serialized in `--json` output too.
- `test_advise_cmd.cpp` proves the CLI end to end against an in-process
  mock Ollama server that traps every mutating route, asserting the
  request log contains only `GET /api/version`/`GET /api/ps`.

## 12. `membrane advise` CLI

```
membrane advise                       # membrane-native, human output
membrane advise --json                # machine-readable
membrane advise --runtime ollama      # Ollama, human output
membrane advise --runtime ollama --json
```

JSON schema (stable keys): `schema_version`, `timestamp`, `runtime`,
`model` (`{known, id}`), `status`, `observation_summary`, `findings[]`
(`code`, `severity`, `summary`, `provenance`, `dimension`, `evidence[]`),
`recommendations[]` (`code`, `action`, `reason_codes[]`,
`affected_runtime`, `runtime_capability`, `applicability`,
`mutates_state`), `reasons[]`, `mutates_state`.

## 13. Limitations

- No historical telemetry -- every call is a fresh, independent
  assessment (same as I1/I2).
- No per-process RAM/VRAM, no live KV bytes, no live active context for
  membrane-native -- because I1's own snapshot does not have them either
  (docs/observability.md, section 9).
- `REQUEST_EXCEEDS_PLANNED_FEASIBILITY` (a Planner v2 "requested context
  exceeds feasible" comparison) is not implemented: the observation
  snapshot carries the planner's *planned* context, not a separate
  *requested* one, so there is nothing distinct to compare without
  inventing a planner output that was not actually produced.
- Deferred to Milestone J in full: automatic context/GPU-layer/KV
  reduction, automatic model unload/switch, restart, runtime migration,
  inference routing -- anything that would apply a recommendation rather
  than state it.
