# Planner v2 foundation (Milestone G1)

Status: architecture + product-facing introspection only. This is the
foundation for a future joint hardware-aware optimizer (Milestone G2+),
not the optimizer itself.

## Why this exists

MEMBRANE's long-term product thesis is that it is a hardware-aware
memory and inference planning layer, not "another local model runner."
The user should not need to manually reason about quantization, GPU
layer count, context size, KV-cache precision, KV-cache placement, host
memory headroom, or VRAM pressure — MEMBRANE should reason about those
jointly and be able to explain its own decisions after the fact.

Before Milestone G, that reasoning existed, but it was scattered:
several independently-evolved pure modules, each already good at its
own job, but with no single place a user (or a future joint optimizer)
could ask "what is the whole plan, and why."

## What already existed (reused, not rewritten)

Two binaries, two planning surfaces:

- **`membrane-run`** (the llama-linked inference executable) owns the
  real joint-planning math, as small, pure (llama-free), individually
  testable C modules under `tools/membrane-run/`:
  - `gpu_policy.c` — resolves `--gpu-layers {all|auto|N}` against a
    memory budget.
  - `adaptive_kv_policy.c` — the Q8-vs-Q5 KV precision decision.
  - `kv_residency_policy.c` — per-layer GPU/CPU KV placement.
  - `compat_check.c` — architecture/KV compatibility gating.
  - `joint_planner.c` — evaluates GPU-layers × KV-precision ×
    KV-placement together for one fixed context size (`joint-auto-v1`
    policy), calling the four modules above.
  - `host_memory_guard.c` — validates host-resident weight/KV bytes
    against real `/proc/meminfo` availability.
  - `context_recommender.c` — the outer loop: generates a bounded,
    geometric candidate context ladder and calls `joint_planner.c` +
    `host_memory_guard.c` once per candidate, picking the largest
    feasible one. This module was already almost exactly what a common
    plan needed, scoped to context.
  - `gpu_device.cpp` — the one place that touches real
    `ggml_backend_dev_*()`/GGUF metadata: device enumeration and a
    cheap, metadata-only model estimate (no full model load).
- **`membrane`** (the product control CLI: `model`, `use`, `serve`,
  `doctor`, ...) links most of the pure modules above directly and adds
  `variant_selector.cpp` — catalog-size-based quant selection for a
  model that is not installed yet.

## What Milestone G unifies

1. **One common plan representation** — `membrane_plan_t`
   (`tools/membrane-run/membrane_plan.h`). A pure ADAPTER, never a
   second planner: every number in it is copied or trivially mapped
   from an existing, unchanged `membrane_ctxrec_result_t`/
   `membrane_ctxrec_request_t` (itself unchanged), plus caller-supplied
   identity/variant facts those modules have no way to compute
   themselves (model name, catalog metadata, best-effort variant
   identification). See `membrane_plan_assemble()`'s own doc comment.

2. **A first-class read-only CLI command**, `membrane plan MODEL` —
   `tools/membrane/plan_cmd.cpp`. Orchestrates:
   `registry_core.h`/`model_catalog.h` (identity) →
   `variant_selector.h` (catalog-only quant estimate) →
   `gpu_device.h`/`runtime_session.h` (one real hardware snapshot) →
   `context_recommender.h` (the actual decision) → `membrane_plan_
   assemble()` (the common representation) → human or `--json` output.

3. **A small, structured provenance model** —
   `membrane_plan_source_t` (`explicit_user`, `catalog_metadata`,
   `model_metadata`, `hardware_auto`, `planner_decision`,
   `fallback_default`, `unknown`), attached to every controlled
   workload/decision field. Not a generic key/value provenance
   framework — a small, explicit enum, matching Part 3's own
   "do not overengineer" instruction.

4. **A small, structured reason-code vocabulary** for the *top-level*
   "why" (`CONTEXT_CAPPED_BY_HOST_MEMORY`, `GPU_LAYERS_CAPPED_BY_VRAM`,
   `KV_Q8_SELECTED_FOR_HEADROOM`, `USER_OVERRIDE_PRESERVED`,
   `MODEL_VARIANT_ESTIMATE_ONLY`, `HOST_MEMORY_LIMIT`, `VRAM_LIMIT`,
   `USER_FORCED_VARIANT`, `NO_FEASIBLE_PLAN`, ...) — deliberately
   narrower than, and mapped *from*, the many detailed reason codes
   `gpu_policy.h`/`kv_residency_policy.h`/`joint_planner.h`/
   `host_memory_guard.h`/`context_recommender.h` already emit. Those
   detailed codes remain available verbatim in `membrane_plan_t`'s own
   echoed `ctxrec_result` for a consumer that wants them — this module
   never replaces that established taxonomy, only adds a smaller,
   product-facing summary layer on top.

## Snapshot strategy

Every `membrane plan` invocation takes **one** real hardware/host-memory
snapshot (`membrane_read_host_meminfo()` + `membrane_gpu_list_devices()`,
called exactly once) and threads it through `membrane_ctxrec_request_t`
into every candidate the context recommender evaluates. `membrane_plan_
assemble()` never re-reads hardware itself — it only reshapes the
snapshot it is handed. This closes the "multiple independent memory
snapshots that can silently disagree within one invocation" problem the
post-v1 milestone surfaced (see `docs/planner-accuracy.md`).

An **explicit** `--ctx N` is handled by building a single-candidate
`membrane_ctxrec_request_t` (exactly one candidate: `N`) and running it
through the *same* `membrane_ctxrec_resolve()` pipeline an auto
recommendation uses — so an explicit context gets the exact same
GPU/host-memory feasibility rigor an auto one does, never a second,
looser check. If that one candidate is infeasible, `membrane plan`
reports it as infeasible with the real limiting reason — it never
silently substitutes a different context (see Part 6/10 of the G1 task).

`membrane plan` is read-only by construction: it calls only
`membrane_registry_load()`/`membrane_catalog_load()` (never `_save()`/
`_add()`), never `membrane_model_cmd_dispatch()`/`membrane_use_cmd_
dispatch()` (no install/download/activate), and never touches
`server_config.h`/`service_state.h` (no service start, no config
mutation). See `tools/membrane/test_plan_cmd.cpp`'s read-only-guarantee
tests, which assert the registry file on disk is byte-for-byte
unchanged (or, for a fresh environment, still absent) after several
`membrane plan` calls.

## `membrane plan` vs. `membrane use` vs. `membrane serve`

| Command | Mutates anything? | Purpose |
| --- | --- | --- |
| `membrane plan MODEL` | Never | Read-only planning/introspection: show the plan MEMBRANE's current planner would use, and why. |
| `membrane use MODEL` | Yes (installs/registers/activates) | Install/select/activate lifecycle. |
| `membrane serve` | Starts a server process | Execute inference. |

## Architecture, one level up

```
existing algorithms (gpu_policy, adaptive_kv_policy, kv_residency_policy,
compat_check, joint_planner, host_memory_guard, context_recommender,
variant_selector, gpu_device)
        |
        v
one shared hardware/model-metadata snapshot
        |
        v
common plan representation (membrane_plan_t / membrane_plan_assemble())
        |
        v
CLI (`membrane plan` — human text / `--json`) / future adapters
```

The representation is deliberately backend-neutral in *shape*
(identity / workload / hardware / decisions / feasibility / reasons),
which is what would let a future adapter (e.g. for a different runtime)
plug into the same common plan object — but no such adapter exists yet,
and none is implied by this document. See "Explicitly out of scope"
below.

## JSON shape

`membrane plan MODEL --json` prints one object:

```json
{
  "schema_version": 1,
  "membrane_version": "1.0.0",
  "mode": "plan",
  "ok": true,
  "identity": { "model_name": "...", "installed": true, "model_path": "...",
    "display_name": "...", "parameter_count": "...", "arch_known": true,
    "arch_name": "llama", "variant_known": false, "variant": "",
    "variant_source": "unknown", "variant_estimate_only": false },
  "workload": { "requested_context": 0, "context_source": "hardware_auto",
    "precision_request": "adaptive", "precision_source": "hardware_auto",
    "gpu_layers_request": "auto", "gpu_layers_source": "hardware_auto",
    "kv_placement_request": "default", "kv_placement_source": "fallback_default" },
  "hardware": { "host_total_bytes": 0, "host_available_bytes": 0,
    "host_available_known": true, "host_reserve_bytes": 0,
    "device_known": false, "backend": "", "device_name": "",
    "device_total_bytes": 0, "device_free_bytes": 0 },
  "decisions": { "context": 4096, "context_source": "planner_decision",
    "gpu_layers": 0, "gpu_layers_source": "planner_decision",
    "kv_precision": "native", "kv_precision_source": "planner_decision",
    "kv_placement": "default", "kv_placement_source": "planner_decision" },
  "feasibility": { "feasible": true, "limiting_resource": "",
    "max_feasible_context_known": true, "max_feasible_context": 4096,
    "host_required_bytes": 0, "host_headroom_known": true,
    "host_headroom_bytes": 0 },
  "reasons": [ { "code": "CPU_ONLY_PLAN", "detail": "..." } ],
  "explanation": "..."
}
```

`decisions` is `null` when no context/GPU/KV decision could be made at
all (a catalog-only model with no installed GGUF to plan against, or a
genuinely infeasible plan — check `feasibility.feasible` and `reasons`).
`ok` reflects whether the `membrane plan` command itself succeeded at
producing an analysis; `feasibility.feasible` is the plan's own outcome.
An infeasible plan is still `"ok": true` — the command did its job by
determining, and disclosing, that infeasibility. This is version 1 of
the schema; only additive, backward-compatible changes are expected
before any breaking revision bumps `schema_version`.

## Real smoke evidence

`results/planner-v2-foundation/validation.json` records real
`membrane plan` runs (human and `--json`) against a locally installed
tiny fixture model, plus the read-only-guarantee check (registry file
hash unchanged across several `membrane plan` invocations).

## Remaining architectural limitations (disclosed, not silent)

- The registry (`registry_core.h`) does not record which catalog quant
  an installed model's file is — `membrane plan` only reports a variant
  for an installed model when its registered basename exactly matches a
  known catalog variant's filename; otherwise the variant is reported
  as unknown rather than guessed.
- A catalog-only (not installed) model gets a variant *estimate* (from
  catalog `size_bytes`, the same proxy `variant_selector.h` already
  used pre-Milestone-G) but no context/GPU/KV plan at all — full
  planning genuinely requires the model's real GGUF metadata.
- `membrane plan` does not currently accept `--device` or
  `--kv-placement` overrides (only `--ctx`/`--kv`/`--gpu-layers`/
  `--quant`, matching this document's own illustrative examples) —
  deferred, not because it is hard, but to keep G1's flag surface
  exactly as small as what was asked for.
- The host-memory guard's own disclosed scope limitation (host-resident
  *weight and KV bytes only*, not full process/backend baseline
  overhead) is inherited unchanged — see `host_memory_guard.h`'s own
  top comment and `docs/host-memory-guard.md`.

## Explicitly out of scope for G1 (deferred to later milestones)

Ollama/vLLM adapters, external runtime discovery, active runtime
monitoring, dynamic memory/KV migration, CXL, FPGA, remote nodes, fleet
management, a cloud control plane, benchmark comparison, a new
inference backend, pricing/business features, or an "AI optimizer."
None of this is implemented, started, or implied by the representation
above.
