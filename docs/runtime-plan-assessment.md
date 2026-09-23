# Runtime-aware plan assessment (Milestone H3)

> **H3 does not apply planner decisions.** It sends no inference request
> and changes no runtime state. It does not touch the Ollama parameters,
> Modelfiles, tags, the model registry, or the server config. Every
> output is a recommendation or a classification. None of it is an action.

H3 completes Milestone H. It connects Planner v2 (Milestone G), the runtime
capability model (H1, [runtime-abstraction.md](runtime-abstraction.md)) and
the read-only Ollama adapter (H2, [runtime-ollama.md](runtime-ollama.md)).
Together they answer one question:

> Given this model and this runtime, what can MEMBRANE recommend, what can
> the runtime actually control, and what remains unsupported?

## 1. Architecture boundary

| Layer | Owns | H3 change |
|---|---|---|
| Planner v2 (`membrane_plan.h`, `plan_v2_resolver.h`) | variant, context, GPU layers, KV precision and KV placement decisions, plus memory math | **none**: consumed only by reading `membrane_plan_t` |
| Runtime capabilities (`runtime_capabilities.h`) | what each runtime can control or observe | none |
| External model metadata (`runtime_adapter.h`, `runtime_ollama.*`) | an Ollama model's reported architecture, parameter count, maximum context and quant | none: the adapter's existing `describe` + `inspect_model` are reused |
| **Assessment** (`runtime_plan_assessment.h/.c`, new) | classifying each plan or request dimension against a runtime | new, pure C, no I/O |
| CLI (`plan_cmd.cpp` + `runtime_plan_cmd.*`) | `membrane plan MODEL --runtime ID` | new flag; the default output is unchanged |

### Audit findings

**Runtime-neutral planner fields:**

- the model's display name and parameter-count label
- the architecture name
- `decisions.context` (a token count, which every runtime understands)
- the feasibility verdict

**Native / ggml-flavored fields:**

- `kv_precision`: native / q8 / q5, which map to llama.cpp `type_k` / `type_v`
- `kv_placement`: MEMBRANE's own residency policy
- `gpu_layers`: llama.cpp `n_gpu_layers`
- the variant: a GGUF quant string

**What Ollama's metadata is enough for:** identity, architecture,
parameter size and count, the model's trained maximum context, and the
quant that its tag fixes.

**What it is not enough for (why exact planning is impossible):**
Planner v2's memory math needs `bytes_per_layer`, the output-role bytes,
the KV formula inputs, and the *actual* KV type and slot count. The Ollama
API does not give these:

- `/api/tags` `size` is a total blob size, not a per-layer figure.
- The KV cache type is server-start environment only
  (`OLLAMA_KV_CACHE_TYPE`), and nothing observable reports it.
- `OLLAMA_NUM_PARALLEL` multiplies KV memory, and is also unobservable.
- GPU offload is decided by Ollama's own memory fitting.

Even with exact hparams, an exact Ollama memory plan would still be
fabricated.

**Device selection and concurrency:** Planner v2 makes no decision for
either. `membrane_plan_t` records the device only as a hardware snapshot,
and it has no concurrency field. Both are therefore reported as
**informational only** and are never "required".

**G2 sibling scaling:** this is **not** applied to Ollama tags. There is
no catalog anchor and no GGUF path, so byte scaling would be invented.

## 2. Planning levels

| Level | Meaning |
|---|---|
| `planner_exact` | A Planner v2 plan with context/GPU/KV decisions from **real** installed-GGUF hparams, for a variant that is not an estimate. |
| `planner_estimate` | A Planner v2 plan that is a disclosed estimate: sibling-scaled hparams, or a catalog-size fit with no context/GPU/KV decision. |
| `capability_only` | **No Planner v2 plan** for this runtime model. Only runtime capabilities, the runtime's own model metadata, and explicit flags are assessed. This is a valid result, **not an error**. It is every Ollama assessment in H3. |
| `unavailable` | Nothing can be assessed: the runtime is not available, or no model is known. |

The level names map onto the H3 task's suggested terms. `planner_exact`
corresponds to EXACT_NATIVE. `planner_estimate` covers native estimates
only: H3 never builds a metadata estimate for Ollama, and says so rather
than calling it METADATA_ESTIMATE.

## 3. Assessment model

`membrane_runtime_recommend_plan()` fills `membrane_runtime_plan_assessment_t`
with:

- the runtime id and runtime model id
- the capability provenance
- the planning level and overall actionability
- required and controllable counts
- seven dimension results, in a fixed order
- structured reasons

Each dimension result records:

- `planner_dimension`: whether Planner v2 can decide it (true for the first five)
- `required`: whether this plan or request relies on it
- `value`, `value_source` (`planner_v2`, `explicit_request`, `runtime_metadata` or `none`) and `plan_source`
- `capability`: the runtime's control capability state
- `applicability`, derived **only** from the capability matrix:

| Control capability | Applicability |
|---|---|
| supported | `controllable` |
| partial | `partially_controllable` |
| unsupported, and the matching telemetry is supported/partial | `observable_only` |
| unsupported otherwise | `unsupported` |
| unknown | `unknown` |

Matching telemetry exists only for **quant** (`model_metadata`) and
**context** (`active_context`). VRAM usage is not a layer count, and
KV-cache usage is not a precision, so no other dimension can be
`observable_only`.

- `reason`: `<H1 negotiation dimension>_<STATE>`, for example
  `CONTEXT_CONTROL_SUPPORTED`, `KV_PRECISION_CONTROL_UNSUPPORTED` or
  `GPU_LAYERS_CONTROL_PARTIAL`.

"Controllable" means the runtime **exposes** a control surface. It never
means MEMBRANE used it.

## 4. Deterministic rules

**Required dimensions** follow the same rule as H1's
`membrane_runtime_negotiate_plan()`, and the tests assert the two agree:

- With a plan: `decisions.has_decisions` requires context, GPU layers,
  KV precision and KV placement. `identity.variant_known` requires the
  quant/variant.
- Without a plan: each explicit `--ctx N`, `--gpu-layers all|N`,
  `--kv native|q8|q5` or `--quant Q` requires its dimension. `auto` and
  `adaptive` are not requests. A quant *reported by the runtime model* is
  shown, but never required: it is a fact, not a recommendation.

**Overall actionability** (no scoring):

| Condition | Result |
|---|---|
| planning level `unavailable` | `unsupported` |
| no required dimension | `advisory_only` |
| every required dimension `controllable` | `fully_actionable` |
| at least one, but not every, required dimension `controllable` | `partially_actionable` |
| no required dimension `controllable` | `advisory_only` |

`partially_controllable`, `observable_only`, `unsupported` and `unknown`
never count as controllable. This is H1's "only SUPPORTED satisfies" rule.

## 5. Native behavior

`membrane plan MODEL --runtime membrane-native` runs the **unchanged**
Planner v2 pipeline and classifies its selected plan. Planner v2's own
`--json` object is embedded verbatim as `planner_plan`.

With every Planner v2 decision present, the result is **fully_actionable
(5 of 5)**: native controls all five dimensions. Native concurrency stays
`partially_controllable` (an env var read at `membrane serve` startup),
but Planner v2 never decides it, so it is informational.

The level depends on the selected variant:

- **Installed model whose own variant is selected:** `planner_exact`.
- **A higher-quality catalog sibling is selected** (G2 policy): the plan is
  honestly `planner_estimate`. The sibling is scaled from the installed
  file, so the estimate has not been measured.
- **Catalog-only model with no install:** only the variant is required.

## 6. Ollama behavior

`membrane plan MODEL --runtime ollama` behaves as follows:

- **MODEL is the Ollama model name**, used verbatim. There is **no
  registry or catalog lookup**, even if a native model has the same name.
  The identity namespace is `runtime_inventory`.
- **Runtime I/O** is exactly H2's read path: `GET /api/version` and
  `POST /api/show`. **No new route** is used; not even `/api/ps`. Cloud
  model references are still refused before any request.
- **Planning level** is always `capability_only`. No Planner v2 plan is
  built, and no context, GPU-layer, KV or memory value is computed.
- **Without flags**, the result is `advisory_only`: nothing is planned or
  requested. The output still reports:
  - which dimensions Ollama can control
  - the model's reported architecture, parameter size, maximum context and quant
  - that the quant is fixed by the tag
- **With flags**, the requested values are classified, for example
  `--ctx 8192 --kv q8 --gpu-layers 20` gives `partially_actionable`
  (1 of 3 controllable):

  | Setting | Classification |
  |---|---|
  | context | controllable |
  | GPU layers | partially controllable |
  | KV precision | unsupported |

  A requested context above the model's reported maximum raises
  `CONTEXT_EXCEEDS_MODEL_MAXIMUM`. A `--quant` different from the tag's
  raises `REQUESTED_QUANT_DIFFERS_FROM_RUNTIME_MODEL`: a different quant
  is a different tag, which MEMBRANE never pulls.
- **Unreachable daemon:** the assessment is still printed (`unavailable`,
  `unsupported`, with reason `RUNTIME_UNAVAILABLE`) and the exit code is
  4. A missing model gives exit 3 with `MODEL_NOT_FOUND`.

The library can also assess a *native* Planner v2 plan against the Ollama
matrix: context is controllable, quant and GPU layers are partial, and KV
precision and placement are unsupported, so the result is
**partially_actionable**. The tests cover this. It shows what a native plan
would and would not carry over, without claiming either runtime is better.

## 7. CLI

```
$ membrane plan qwen2.5:7b --runtime ollama --ctx 8192 --kv q8
Runtime
  ollama (external, local_external, available, version 0.34.3)

Model
  qwen2.5:7b  (ollama runtime model -- not a MEMBRANE registry model)
  Architecture: qwen2
  Parameter size: 7.6B
  Quantization: Q4_K_M
  Context length (model maximum): 32768

Planning level
  capability_only

Requested settings
  Context: 8192 (requested)
  KV precision: q8 (requested)

Runtime applicability
  Partially actionable (1 of 2 required settings controllable)

Runtime can control
  Context

Runtime can partially control
  Quant variant  (not in this plan)
  GPU layers  (not in this plan)

Runtime cannot control
  KV precision
  KV placement  (not in this plan)
  Concurrency  (not in this plan)

Unknown
  Device selection  (not in this plan)

Why
  - [NO_PLANNER_PLAN] no Planner v2 plan exists for this runtime model; ...
  - [EXACT_MEMORY_PLAN_UNAVAILABLE] no exact context, GPU-layer or memory figure is computed: ...
  - [QUANT_FIXED_BY_RUNTIME_MODEL] quantization Q4_K_M is fixed by the runtime model itself ...

Note
  Recommendations only. No settings were changed.
```

This output came from the real binary against a loopback fixture server,
not a real Ollama. `membrane plan MODEL` without `--runtime` prints exactly
what it did before H3. `--runtime vllm` is a CLI error: vLLM is still
reserved only.

## 8. JSON schema

`schema_version: 1` (`MEMBRANE_RUNTIME_ASSESSMENT_SCHEMA_VERSION`):

```json
{
  "schema_version": 1,
  "membrane_version": "1.0.0",
  "mode": "runtime_plan_assessment",
  "ok": true,
  "mutates_state": false,
  "runtime": {"id", "type", "execution_mode", "status", "health", "version",
              "endpoint", "unavailable_reason", "capability_provenance"},
  "model": {"runtime_id", "runtime_model_id", "identity_namespace",
            "architecture", "parameter_size", "quantization",
            "max_context_length", "...": "only what is known; null otherwise"},
  "planning_level": "capability_only",
  "planner_plan": null,
  "assessment": {"actionability": "partially_actionable",
                 "required_dimensions": 2, "controllable_dimensions": 1},
  "dimensions": [
    {"name": "context", "planner_dimension": true, "required": true,
     "value": "8192", "value_source": "explicit_request", "plan_source": null,
     "capability": "supported", "applicability": "controllable",
     "reason": "CONTEXT_CONTROL_SUPPORTED"}
  ],
  "reasons": [{"code": "NO_PLANNER_PLAN", "detail": "..."}]
}
```

Details:

- `dimensions` always has 7 entries, in this order: quant_variant,
  context, gpu_layers, kv_precision, kv_placement, device_selection,
  concurrency.
- `identity_namespace` is `membrane_registry`, `membrane_catalog` or
  `runtime_inventory`.
- `planner_plan` is the existing `membrane plan --json` object on the
  native path, and `null` otherwise.
- Output is deterministic: the Ollama document is byte-identical across
  runs. The native document differs only in the live `/proc/meminfo`
  figures that Planner v2 itself reads on each call.

## 9. Read-only guarantee

- **The assessment library** does no I/O. It never modifies the plan or
  descriptor it is given; the tests check this byte for byte.
- **The native path** reads the registry and catalog through Planner v2's
  existing read-only calls. The tests assert the registry file is
  byte-identical afterwards.
- **The external path** reads no registry or config file.
- **The Ollama allowlist is unchanged:** H3 adds no HTTP route. The tests
  use a mock server that logs every request and traps `/api/chat`,
  `/api/generate`, `/api/pull`, `/api/create`, `/api/delete`, `/api/copy`,
  `/api/ps` and the `/v1` routes. They assert that only `GET /api/version`
  and `POST /api/show` were seen.
- **Planner v2 is unchanged:** `plan-v2-variant-joint-v1`, the objective
  modes and scoring are untouched, and the G3 readiness classification
  stays **`READY_FOR_READ_ONLY_ONLY`**. Nothing is wired into
  `membrane use` or `membrane serve`.

## 10. Limitations and what remains after Milestone H

**Limitations:**

- **No exact Ollama memory planning:** see §1 for the missing inputs.
- **Capability matrices are static contracts:** the Ollama matrix is for
  v0.34.3, and neither matrix is probed per daemon.
- **No identity mapping:** there is no Ollama ↔ native model comparison
  CLI (the H3 task's optional §15 was not built). The library can assess a
  native plan against the Ollama matrix, but nothing links the two
  inventories.
- **Explicit requests are classified, not validated:** on the Ollama path,
  a `--ctx`, `--kv`, `--gpu-layers` or `--quant` value is checked only
  against the capability matrix and the model's reported facts, not
  against memory.

**Deferred beyond Milestone H** (not started):

- **Applying** anything:
  - per-request `num_ctx` / `num_gpu`
  - `keep_alive`-based load/unload
  - wiring assessments into `membrane use` or `membrane serve`
- Reading `/api/ps` for live loaded-model telemetry.
- Reading an Ollama model's local GGUF blob for exact hparams. `/api/show`'s
  Modelfile names the blob path, but that is outside the documented API,
  and the KV type and slot count would still be unknown.
- A vLLM adapter.
- Any per-daemon capability probing, and any runtime-comparison or
  performance claims.
