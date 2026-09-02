# Host memory safety guard -- Phase 34

Phase 33 built the pure/testable core of safe context recommendation
but disclosed a real gap: nothing in MEMBRANE, anywhere, had ever
checked a plan's host-resident memory against real host RAM. Phase 34
closes that gap **for the context recommendation core only** -- no CLI
surface exists yet (`--ctx auto` still does not exist), and the real,
shipped `--auto`/`--ctx N`/`--plan-only`/`--inspect-model` CLI paths in
`main.cpp` are completely unchanged.

## Problem

`context_recommender.c` (Phase 33) evaluated every candidate context
size purely against the existing joint planner's GPU-memory check.
Nothing checked whether the plan's host-resident weight or KV bytes
would actually fit in real host RAM -- `joint_planner.h`'s own
candidate field comment already said "fits_host... always 1 today (no
host-RAM capacity guard exists in the product)", and `main.cpp`'s only
host-memory signal (`MEMBRANE_WARNING_HOST_MEMORY_PRESSURE`) is
explicitly "observability only... never influenced any pass/fail
decision." A recommendation could report `OK` for a plan that would
never actually fit in real host memory.

## Why Phase 33 was not yet sufficient

Phase 33's own core explicitly could not close this gap within its own
scope: adding a host-RAM check would have meant either duplicating
memory-fit logic no existing module owned (forbidden, Section 12 of
the Phase 33 task) or inventing an unevidenced safety margin
(forbidden, Section 5). Phase 33 disclosed the gap
(`host_memory_unvalidated`) rather than papering over it. Phase 34
exists specifically to gather the missing evidence and build the
missing check as its own, separate, reusable module.

## Host memory components

| Component | Status |
|---|---|
| CPU-resident model-weight bytes | **ESTIMATED** -- derived, not duplicated (see Estimator below) |
| CPU-resident KV bytes | **KNOWN** -- reused verbatim from the existing joint planner's own `estimated_host_kv_bytes` |
| Process/runtime baseline | **ESTIMATED**, folded into the reserve floor |
| Model-loader overhead | **ESTIMATED**, folded into the same reserve floor (could not be cleanly isolated from runtime baseline in real measurements) |
| Context-creation transient compute buffer | **ESTIMATED**, folded into the same reserve floor |
| Safety reserve | the explicit, evidence-derived reserve formula below |

## Estimator

`tools/membrane-run/host_memory_guard.h`/`.c` is a new, small,
llama-free, pure module (`membrane_host_memory_guard_resolve()`) --
same testable-without-a-model/GPU pattern as `gpu_policy.h`. It does
**not** duplicate weight/KV byte arithmetic:

- **KV bytes** are reused directly from `membrane_joint_plan_result_t`'s
  own `estimated_host_kv_bytes` field -- the existing, unchanged joint
  planner already computes this correctly.
- **Weight bytes** are derived in `context_recommender.c`'s
  `host_weight_bytes_for()`: the model's real total tensor-byte
  footprint (`total_weight_bytes`, additive on
  `membrane_ctxrec_request_t`, sourced from `gpu_device.h`'s own
  `total_bytes`) minus a GPU-offload credit computed by calling the
  *existing* `membrane_joint_estimate_gpu_weight_bytes()` (Phase 20's
  own function) with `output_role_bytes` deliberately forced to `0`.

That last detail is deliberate, not an oversight. Withholding the
output-role tensor's credit is:

- **exact** for a tied-embedding model (e.g. SmolLM2): llama.cpp
  creates a *separate, GPU-eligible duplicate* of the input embedding
  tensor for the output role at runtime -- a real allocation that is
  never counted in `total_weight_bytes` at all (it isn't in the GGUF
  file). Crediting it to the GPU side would under-count the real,
  always-CPU-resident original embedding tensor that remains.
- **conservative (never unsafe)** for an untied-embedding model: the
  real `output.weight` tensor genuinely does move to the GPU at
  `gpu_layers >= 1`, so withholding its credit here over-counts the
  host-resident total by `output_role_bytes` -- the safe direction for
  a safety guard.

Concretely: **even a fully GPU-resident plan (every layer offloaded)
still shows a small, nonzero host-weight estimate** under this
formula. This is not a bug -- llama.cpp's own real policy keeps the
*input* embedding tensor CPU-resident always, "regardless of
n_gpu_layers" (`gpu_device.h`'s own field comment). A real dry-run
against a fully-GPU-resident local Vulkan run confirmed this directly:
a 63.7 MiB real residual, comfortably fitting real host RAM (see
`results/host-memory-guard/validation.json`).

## Reserve policy

```
reserve_bytes = max(MEMBRANE_HOST_RESERVE_FIXED_BYTES,
                     MEMBRANE_HOST_RESERVE_PCT% of (host_weight_bytes + host_kv_bytes))
```

`MEMBRANE_HOST_RESERVE_FIXED_BYTES = 256 MiB`. Real evidence basis
(full derivation and raw samples in
`results/host-memory-guard/validation.json`): this session's own 4
fresh `membrane-run --json` samples (SmolLM2-135M, CPU-only, native/q8
KV, ctx 2048/4096) showed a combined "loader + context-creation"
residual of 22.2-26.0 MiB. **Pre-existing Phase 19 evidence, the exact
same model+config, a different session**, showed 147.7-149.6 MiB
instead -- a real ~6x discrepancy for an identical configuration,
consistent with genuine RSS-measurement noise under this 5.6 GiB dev
host's real memory/swap pressure (this phase's own session ran with
4.3-4.6 GiB of swap already in use throughout). Rather than trust only
the more favorable, more recent sample set, the fixed floor is set to
comfortably exceed the **worse** of the two real, measured cases.

`MEMBRANE_HOST_RESERVE_PCT = 10`. Kept as a minimal, explicitly
conservative placeholder for scaling to a model larger than the one
this phase measured -- it does not bind for anything actually measured
this phase (the fixed floor dominates every real sample). This is a
disclosed limitation, not a validated claim (see Residual uncertainty).

Unlike `gpu_policy.h`'s own device-total-relative percentage, this
percentage is applied to the *candidate's own* host-resident bytes, not
to total host RAM -- loader/runtime overhead is causally connected to
how much of *this model* is host-resident, not to how much total
system memory happens to be installed.

## Recommendation integration

`context_recommender.c`'s `evaluate_one()` now calls the host-memory
guard for every candidate the joint planner itself accepts. A
candidate is only counted feasible when **both** the joint planner and
the host-memory guard accept it -- Section 11 of the Phase 34 task's
own minimum requirement: `membrane_ctxrec_resolve()` can never return
status `OK` for a candidate requiring host memory unless that host
memory was actually validated. `membrane_ctxrec_result_t` now exposes
`host_memory_checked`/`host_memory_fit`/`host_required_bytes`/
`host_available_bytes`/`host_reserve_bytes` alongside the
Phase-33-era `host_memory_unvalidated` (which keeps its original
meaning -- see `context_recommender.h`'s own field comment for exactly
how the two relate). This is unit-tested directly, including a test
that first proves the *bare, unchanged* joint planner alone would have
succeeded, then proves the full recommendation still correctly fails
once host memory is unvalidatable
(`test_recommendation_cannot_succeed_on_unvalidated_host_residency`).

## Explicit-run scope

**No CLI surface change.** `main.cpp` is not touched this phase at
all. `--ctx N`, `--auto` (still requiring an explicit `--ctx`),
`--gpu-layers 0`, `--plan-only`, `--inspect-model`, and the existing
JSON schema are all byte-for-byte unchanged. The new guard exists only
inside `context_recommender.c`, which no CLI path calls yet. When
Phase 35 wires `--ctx auto` to a real CLI surface, this guard is what
that surface will lean on for its own safety checking -- explicit
runs' own backward-compatibility policy (whether an explicit `--ctx N`
run should also gain a host-memory check, and if so, warn vs. fail) is
an open product decision for that phase, not decided here.

## CPU-only adaptive decision

Phase 33 also disclosed that adaptive (bare `--auto`) precision has no
CPU-only fallback in `joint_planner.c`'s own candidate generation at
zero GPU budget. This phase investigated the root cause as its
secondary objective:

**Root cause: accidental historical coupling (not intentional policy,
not a semantic requirement).** `joint_planner.c`'s
`resolve_adaptive_precision()` always calls
`membrane_adaptive_kv_resolve()` with `is_gpu_backend` hardcoded to
`1`. `adaptive_kv_policy.c` already has a complete, working
`is_gpu_backend=0` code path (`resolve_cpu()`). More importantly:
**`main.cpp`'s own `resolve_cpu_adaptive_kv()` (line ~1066) already
calls `membrane_adaptive_kv_resolve(..., 0, &ar)` directly**, entirely
bypassing `joint_planner.c`, for the real `gpu_layers==0` short-circuit
case. The real, shipped bare `--auto` CLI **already works correctly on
a CPU-only host today**, via this separate post-load path. The "gap"
is scoped to `joint_planner.c`'s own pre-load candidate generation
specifically (which `context_recommender.c` depends on for its
pre-load recommendation), not a real product-behavior gap in the
shipped CLI.

**Decision: deferred, not implemented this phase.** The fix is
structurally clean (mirror `resolve_cpu_adaptive_kv()`'s own simple
`cand_q8`/`cand_q5` construction inside `resolve_adaptive_precision()`
when `device_total_bytes == 0`, then call
`membrane_adaptive_kv_resolve()` with `is_gpu_backend=0`) and
semantically valid. It was not implemented this phase because
`joint_planner.c` is shared with the already-shipped `--auto` CLI path
-- any change there needs full real-CLI regression validation, which
this phase's mandatory, primary focus (host-memory safety) did not
budget for. This is exactly the deferral Section 14 of the Phase 34
task authorizes when the secondary objective is not clearly safe
enough within scope.

## Safety guarantees

- Fails closed on unknown host availability -- never assumes infinite
  host RAM (`MEMBRANE_HOST_GUARD_REASON_UNKNOWN`).
- Checked, saturating 64-bit arithmetic throughout -- a pathological
  `UINT64_MAX` requirement never wraps into a value that would
  incorrectly fit a realistic available-memory figure.
- Deterministic: identical inputs always produce an identical result.
- Never bypasses the existing joint planner's own compatibility/fit
  logic -- this guard runs strictly *after* the joint planner accepts
  a candidate, never in place of it.

## Residual uncertainty

Even with this guard, a MEMBRANE recommendation remains an **estimated
safe fit, never a guaranteed absence of OOM**. Real, disclosed
residual gaps:

1. **Process/backend baseline overhead is not modeled** for a fully
   GPU-resident plan (real evidence: ~248 MiB of real host RSS even
   with zero host-resident weight/KV bytes on a real Vulkan run, Phase
   19 evidence) -- out of this guard's scope.
2. **The transient compute-buffer question remains open at the source
   level.** This phase only answered it empirically, at small scale
   (a 0.5-2.1 MiB residual, SmolLM2-135M, ctx 2048/4096) -- whether
   `ggml`'s own context-creation compute-buffer allocation can be
   predicted or bounded *before* construction, from source, was not
   investigated this phase.
3. **All real RSS evidence this phase is for one model size**
   (SmolLM2-135M) -- the reserve policy is not validated at larger
   model scale (a real, memory-constrained 5.6 GiB dev host made a
   larger CPU-only real measurement unsafe to attempt this phase).
4. **RSS-based measurement is genuinely noisy on this dev host** -- a
   real ~6x discrepancy for the identical configuration across two
   sessions (see Reserve policy above) means any single point estimate
   should be treated skeptically; this is exactly why the guard leans
   on a generous fixed floor rather than a tight prediction.
5. Allocator fragmentation, concurrent system activity, and driver-side
   allocations can all still vary independently of anything estimated
   here.

## Phase 35 CLI handoff

Not implemented this phase. Expected scope, recorded for Phase 35:

- `--ctx auto`, reusing this exact core (Phase 33's recommender +
  Phase 34's host-memory guard together).
- The explicit-run backward-compatibility policy left open above
  (whether/how `--ctx N` should surface host-memory information).
- Real, larger-model-scale host-memory evidence, if a suitable
  environment becomes available, to validate or revise the reserve
  policy's percentage term.
- If Phase 35's own scope naturally touches `joint_planner.c` for CLI
  integration reasons, revisit the CPU-only adaptive fix documented
  above -- otherwise it remains open for a dedicated follow-up.
- Human- and JSON-facing explanation output built from this core's
  fields, no new planner ranking.
