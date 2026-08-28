# Auto-plan fallback (Phase 21)

Phase 20 (`docs/joint-planner.md`) resolves `--auto`'s GPU layer count,
KV precision, and KV placement together into ONE bounded, deterministic,
ranked candidate list. Phase 21 answers a narrower question: **what
happens if the primary candidate that list selected cannot actually be
instantiated at runtime?**

## What this is

If the primary selected complete plan cannot actually be instantiated
(model load fails, context construction fails, or a real-time memory
snapshot shows the primary candidate no longer fits), MEMBRANE retries
another already-legal, already-ranked Phase-20 candidate, in a bounded,
transparent, resource-safe way:

```text
STATIC PLAN A FAILS
  -> clean up A's own resources
  -> STATIC PLAN B (the next legal ranked candidate)
  -> retry
```

## What this is NOT

- **Not live KV migration.** There is no concept of moving a live
  context's KV state from one plan to another. A retry is a full,
  clean re-instantiation of a *different, already-ranked* candidate --
  never a partial mutation of the one that failed.
- **Not promotion/demotion.** Nothing here decides "Q5 is now better
  than Q8" or "fewer GPU layers is now preferred" based on runtime
  performance. An apply failure proves only "this candidate could not
  be instantiated right now" -- never a quality or speed judgment
  about any candidate (Section 12 of the Phase 21 task; see also
  `docs/joint-planner.md`'s own no-performance-claim discipline).
- **Not a runtime scheduler.** Nothing here relocates KV layers, model
  weights, or anything else once a candidate is successfully applied
  and generation is underway. All of that remains exactly Phase 20's
  and prior phases' territory (`docs/kv-residency.md`).
- **Not a second planner.** The fallback controller (`tools/membrane-
  run/auto_fallback.h`/`.c`) never generates a candidate. It consumes
  *exactly* the ranked array `membrane_joint_plan_resolve()` (Phase 20)
  already produced for this run -- the same struct, unmodified, no
  second candidate source.

## Explicit user constraints stay hard

The fallback controller can only ever iterate candidates Phase 20
itself already generated and ranked as legal. Since Phase 20's own
candidate generation already enforces every explicit user constraint
(an explicit `--kv q8` never produces a native or q5 candidate; an
explicit `--gpu-layers N` never produces a candidate with a different
layer count; an explicit `--kv-placement gpu` never produces a
CPU-resident candidate -- see `docs/joint-planner.md`), the fallback
controller has *no path* to violate one. It does not need its own
constraint-checking logic because there is structurally nothing for it
to override.

`tools/membrane-run/test_auto_fallback.c` proves this against **real**
planner output (not just a hand-built candidate array) for explicit
precision, explicit GPU-layer count, and explicit GPU placement.

## Candidate order and retry bound

Iteration order is `[selected_index]` (the primary candidate) followed
by every other *eligible* candidate in the array's own ascending index
order -- the exact order Phase 20's own ranking policy already
produced, never re-derived or re-ranked by this module.

The hard cap is `MEMBRANE_MAX_AUTO_ATTEMPTS = 3`, derived directly from
the candidate-generation architecture: `membrane_joint_plan_resolve()`
never produces more than 3 candidates for a single request shape
(the adaptive path's winner + runner-up + same-precision 0-layer
fallback is the largest case; the explicit-precision path produces at
most 2). 3 is therefore the natural bound, not an arbitrary number.

A memory-refit **skip** (see below) does not itself consume this
bound -- only a real instantiation attempt (a call into the apply
adapter) does.

## Candidate-set completeness (Section 6 of the Phase 21 task)

Phase 20's planner does not always retain a full cross-product of
alternatives -- e.g. an adaptive request whose primary (winner)
candidate looked eligible pre-load has no CPU-only fallback candidate
in the array *unless* the winner was already placement-ineligible at
planning time. This phase deliberately does **not** extend Phase 20's
candidate generation to guarantee one always exists: Section 4's own
rule ("Phase 20 candidates are the only fallback source... do not
invent one") takes priority. When every candidate Phase 20 actually
generated is exhausted, the honest result is `AUTO_FALLBACK_EXHAUSTED`
-- never a synthesized candidate Phase 20 itself never ranked as
legal.

## Failure classification

```text
MEMBRANE_APPLY_OK
MEMBRANE_APPLY_MODEL_LOAD_FAILED
MEMBRANE_APPLY_CONTEXT_CREATE_FAILED
MEMBRANE_APPLY_BACKEND_ALLOCATION_FAILED   (reserved, see below)
MEMBRANE_APPLY_GPU_OOM_CONFIRMED           (reserved, see below)
MEMBRANE_APPLY_HOST_OOM_CONFIRMED          (reserved, see below)
MEMBRANE_APPLY_DEVICE_LOST                 (reserved, see below)
MEMBRANE_APPLY_COMPAT_REJECTED
MEMBRANE_APPLY_UNKNOWN_FAILED
```

**Honest limitation:** `llama.h` exposes no error code or reason
string from `llama_model_load_from_file()` or
`llama_init_from_model()` -- both return a bare `NULL` on any failure,
with only free-text diagnostics on llama.cpp's own stderr. This
project's real apply adapter (`tools/membrane-run/main.cpp`'s
`real_apply_fn()`) can therefore only ever honestly produce three of
the classes above:

- `MEMBRANE_APPLY_MODEL_LOAD_FAILED` -- `llama_model_load_from_file()`
  returned `NULL` for this candidate's `gpu_layers`.
- `MEMBRANE_APPLY_CONTEXT_CREATE_FAILED` -- either
  `llama_init_from_model()` returned `NULL`, or this candidate's
  KV-placement resolution (`membrane_kv_residency_resolve()`, using
  the real post-load model shape rather than Phase 20's pre-load
  estimate) rejected it.
- `MEMBRANE_APPLY_COMPAT_REJECTED` -- a post-load
  `membrane_check_kv_compat()` re-check failed. This should be
  structurally unreachable for an auto-managed candidate (Phase 20
  already ran the identical, deterministic check pre-load on the same
  architecture/hparams), so it exists only as a defensive fail-closed
  path, never as an expected outcome.
- `MEMBRANE_APPLY_UNKNOWN_FAILED` -- a mid-generation decode failure
  (`run_kv_store_pass()`'s `decode_prompt()`/`run_generation()` stage)
  after context construction already succeeded. Deliberately
  conservative and non-retryable (Section 2/21 of the Phase 21 task:
  "do not fabricate precision" -- there is no evidence this is
  memory-related rather than a structural/model-format issue).

`GPU_OOM_CONFIRMED`, `HOST_OOM_CONFIRMED`, `DEVICE_LOST`, and
`BACKEND_ALLOCATION_FAILED` are reserved taxonomy slots for a future
llama.cpp error-introspection capability this project does not have
today -- never fabricated by the real adapter, and
`scripts/verify-auto-fallback.py` fails the build if a **REAL**
evidence run ever claims one.

### Retryable vs. non-retryable

Retryable (a different already-legal candidate might avoid the same
failure): `MODEL_LOAD_FAILED`, `CONTEXT_CREATE_FAILED`,
`BACKEND_ALLOCATION_FAILED`, `GPU_OOM_CONFIRMED`,
`HOST_OOM_CONFIRMED`, `DEVICE_LOST`.

Non-retryable: `COMPAT_REJECTED` (Phase 20 already filtered this
candidate space before generation -- see Section 26's Qwen2.5
regression requirement below), `UNKNOWN_APPLY_FAILED` (conservative
default -- an unclassifiable failure is never assumed retryable).

A **cleanup-safety blocker** (`cleanup_complete == 0` in the apply
result) always overrides an otherwise-retryable failure class and
stops the controller immediately (`AUTO_CLEANUP_FAILED`) -- Section 9
of the Phase 21 task treats resource-cleanup safety as a hard blocker,
never a policy choice.

## Memory re-snapshot before each attempt

Before every attempt (including the first), the controller re-queries
the same real GPU device query Phase 20's pre-load estimate and
`gpu_memory_observed` already use
(`membrane_gpu_list_devices()`). A GPU candidate whose
`estimated_weight_gpu_bytes + estimated_kv_gpu_bytes` no longer fits
within `refreshed_free_bytes - reserve_bytes` is marked
`SKIPPED_UPDATED_MEMORY` and the controller moves to the next
candidate in the **same, original** rank order -- never a fresh
re-ranking. A CPU-only candidate (`gpu_layers <= 0`) always trivially
fits; there is nothing to refit.

This is not a hypothetical: `results/auto-fallback/validation.json`'s
`qwen2.5_1.5b_native_gpu_placement_reload_recovery` run captured this
happening for real on this project's own memory-constrained
development host.

## Model reload

Changing `gpu_layers` changes which tensors llama.cpp offloads to the
GPU, decided once, at `llama_model_load_from_file()` time -- there is
no way to change it on an already-loaded model. The apply adapter
therefore reloads the model (a clean `llama_model_free()` +
`llama_model_load_from_file()`, never an in-place layout mutation)
whenever a candidate's `gpu_layers` differs from whatever is currently
loaded. The very first attempt always reuses the model `main()`
already loaded for the originally-selected candidate -- no reload
unless a fallback to a different `gpu_layers` count actually happens.
`model_reload_required` is reported per attempt in both the trace and
the JSON output, purely as a diagnostic (never optimized for cost in
this phase -- correctness first).

## Cleanup

`run_kv_store_pass()` (`tools/membrane-llama-runtime/decode_loop.cpp`)
already frees its own `llama_context`/collector on **every** one of
its own failure paths -- this was true before Phase 21 and required no
change. A failed model load leaves the model pointer `NULL` (nothing
to free). A rejected compat/placement check happens before any context
exists. The real apply adapter therefore always reports
`cleanup_complete = 1` on every failure path it produces -- there is
no partial state Phase 21 itself ever leaves behind. `cleanup_complete
== 0` exists in the taxonomy as a structural safety net for a future
apply adapter that might not have this guarantee, not because the
current one ever triggers it.

## JSON contract (additive only, `schema_version` stays 1)

A normal run whose primary candidate succeeds:

```json
"fallback": {"attempted": false, "attempt_count": 1, "final_status": "success"}
```

A run that never engaged the fallback controller at all (`--compare-kv`
/`--gpu-bench`, a CPU-only run, or a run where no Phase-20 candidate
array exists to iterate) reports the identical trivial shape, honestly
reflecting the one real outcome that already happened -- never a
fabricated candidate iteration.

A recovered run:

```json
"fallback": {
  "attempted": true, "attempt_count": 1,
  "initial_candidate_index": 0, "final_candidate_index": 1,
  "final_status": "success", "reason_code": "AUTO_FALLBACK_SUCCEEDED",
  "attempts": [
    {"candidate_index": 0, "gpu_layers": 28, "kv_precision": 0,
     "kv_placement": 1, "refreshed": true,
     "refreshed_available_gpu_bytes": 1147011072,
     "fit_after_refresh": false, "apply_started": false},
    {"candidate_index": 1, "gpu_layers": 0, "kv_precision": 0,
     "kv_placement": 0, "refreshed": true,
     "refreshed_available_gpu_bytes": 1147011072,
     "fit_after_refresh": true, "apply_started": true,
     "apply_ok": true, "cleanup_complete": true,
     "model_reload_required": true}
  ]
}
```

(`attempted` is `true` here even though only one real *apply* happened
-- `attempted` reflects whether the outcome ever departed from the
originally-selected candidate, and a memory-refit skip is exactly as
real a fallback event as an apply failure is. `attempt_count` counts
only real apply attempts, matching the resource-safety bound in
Section 8 of the Phase 21 task.)

An exhausted run: `"final_status": "exhausted"`, `attempts` shows every
candidate that was tried or skipped, `final_candidate_index: -1`.

## Text / verbose diagnostics

A retry is never silent (Section 15 of the Phase 21 task). Without
`--verbose`, a run whose primary candidate simply succeeded prints
nothing about fallback at all. A run that skipped or retried prints a
concise transcript to stderr, e.g.:

```text
MEMBRANE: selected plan #0
MEMBRANE: plan #0 skipped: available GPU memory changed since planning
MEMBRANE: plan #1 succeeded
```

`--verbose` additionally announces a plain primary-success outcome.

## `--plan-only` semantics

`--plan-only` never applies a runtime plan, so the fallback controller
never runs for it. Its JSON always reports the distinct
`"final_status": "not_applicable"` (`"attempted": false,
"attempt_count": 0`) -- never `"success"`, which would wrongly imply a
real attempt happened.

## Scope exclusions

- **`--compare-kv`/`--gpu-bench`** keep their exact pre-Phase-21
  code path, unchanged. Neither reaches the joint planner at all
  (Phase 20's own scope exclusion, unchanged by this phase), so
  neither ever engages the fallback controller.
- Only an auto-managed run (Phase 20's joint planner actually ran and
  produced a candidate array) engages the fallback controller. Every
  other normal run (a plain CPU-only run, or a request whose GPU
  estimate was unavailable) retains its exact pre-Phase-21
  single-attempt behavior.

## Qwen2.5 regression (Section 26)

Fallback must never turn `qwen2 + --kv adaptive` into `native` just
because native could work. This is enforced structurally, not by any
Phase-21-specific check: `compat_check.c`'s architecture gate already
rejects Qwen2.5 for q8/q5 inside Phase 20's own candidate generation,
so a bare `--auto` (adaptive) request on Qwen2.5 fails with
`NO_FEASIBLE_CANDIDATE` *before* the fallback controller is ever
invoked -- see `results/auto-fallback/validation.json`'s
`qwen2.5_1.5b_bare_auto_still_fails_closed` run.

## Real smokes

`results/auto-fallback/validation.json` -- all real, live
`membrane-run` invocations, never simulated data presented as
hardware evidence:

1. SmolLM2-135M, bare `--auto`, real Vulkan GPU: primary candidate
   applies cleanly, `fallback.attempted = false`.
2. Qwen2.5-1.5B, explicit `--kv native --kv-placement gpu`, real
   Vulkan GPU: a genuine (unplanned, reproduced twice) memory-refit
   skip and model-reload recovery to the CPU-only candidate.
3. Qwen2.5-1.5B, bare `--auto`: fails closed at the planner stage,
   confirming the fallback controller never engages for an
   incompatible request.

Deterministic apply-time failure injection (a llama-free function-
pointer seam in the fallback controller's own test file, never a
production CLI flag -- Section 17 of the Phase 21 task explicitly
forbids a `--force-oom`-style knob) lives in
`tools/membrane-run/test_auto_fallback.c` and runs under ASan/UBSan as
part of the normal test suite.

## Limitations

- The real apply adapter can only classify 3 of the 8 failure
  classes in the taxonomy (see above) -- this is an honest
  reflection of `llama.h`'s current public API surface, not a design
  gap in this module.
- A memory-refit check compares against the SAME conservative
  estimate/reserve arithmetic Phase 20 already uses -- it is not a
  stronger guarantee than that arithmetic already provides (see
  `docs/joint-planner.md`'s own disclosed limitations on the reserve
  margin).
- Model-reload cost is not optimized in this phase (Section 10 of the
  Phase 21 task: correctness first).
