# Safe context recommendation -- Phase 33 core

Phase 32 chose "safe context recommendation" as MEMBRANE's v0.4 primary
feature (`docs/v0.4-roadmap.md`). This document describes Phase 33's
deliverable: the **pure, testable core** of that feature
(`tools/membrane-run/context_recommender.h`/`.c`). **No CLI surface
exists yet** -- `--ctx auto` ships in Phase 34. `--ctx N` and `--auto`
(which still requires an explicit `--ctx`) are unchanged by this phase.

## Problem

`--auto` already resolves GPU layers, KV precision, and KV placement
jointly for a *given* context size (`docs/joint-planner.md`). The
single largest remaining "you must already know a memory-relevant
number" gap is context size itself: nothing today tells a user what
context their model and hardware can actually support. Phase 32's own
audit (`docs/product-direction.md` Section 1D) named this the primary
blocker to the v0.4 product thesis.

## Definitions

Three distinct values, never conflated:

- **`MODEL_MAX_CONTEXT`** -- the model's own GGUF metadata ceiling.
  Read from `{arch}.context_length` (source-verified against the
  pinned llama.cpp: `llama-arch.cpp`'s `LLM_KV_CONTEXT_LENGTH ==
  "%s.context_length"`, the same key `llama-model.cpp`'s own
  `ml.get_key(LLM_KV_CONTEXT_LENGTH, hparams.n_ctx_train)` reads) --
  see `gpu_device.h`'s new `model_max_context`/
  `model_max_context_available` fields.
- **`HARDWARE_FIT_CONTEXT`** -- the largest of a bounded, deterministic
  set of candidate context sizes whose plan the existing joint planner
  (`membrane_joint_plan_resolve()`) actually accepted.
- **`RECOMMENDED_CONTEXT`** -- what this module presents as its
  conservative default. Phase 33's policy
  (`RECOMMENDATION_POLICY_MAX_ESTIMATED_FIT`) sets this exactly equal
  to `HARDWARE_FIT_CONTEXT` -- see "Selection policy" below for why.

## Inputs

`membrane_ctxrec_request_t` (`context_recommender.h`) takes, as plain
data, exactly what `membrane_joint_plan_request_t` already takes
(model/device facts, explicit `--kv`/`--gpu-layers`/`--kv-placement`
constraints), plus:

- `model_max_context` / `model_max_context_known`
- `minimum_required_context` (a floor below which no candidate is ever
  generated -- reserved for a future prompt-token accounting layer,
  Phase 34+; this core never tokenizes anything itself)
- `candidates[]` -- a caller-populated array of `(ctx, kv_bytes_native,
  kv_bytes_q8, kv_bytes_q5)` tuples, one per candidate context size.

This module is **llama-free** (no `ggml`/`llama` header, no file I/O,
no model load, no GPU enumeration, no `/proc` read) -- every fact is a
caller input, exactly like every other `*_policy.h`/`joint_planner.h`
module in this project. In particular, **KV bytes are caller-supplied**
because they vary per candidate context size and require
`ggml_row_size()`, which only a caller linking `ggml` can compute (the
same real formula `main.cpp`'s own `native_kv_bytes()`/`q8_kv_bytes()`/
`q5_kv_bytes()` already use). See
`tools/membrane-run/context_recommender_dryrun.cpp` for a real,
`ggml`-linked caller.

## Candidate generation

`membrane_ctxrec_generate_candidates()`: geometric doubling starting at
`max(MEMBRANE_CTXREC_MIN_CANDIDATE, minimum_required_context)`, up to
(and always including, as the final entry) `model_max_context`.

`MEMBRANE_CTXREC_MIN_CANDIDATE = 4096` is **not an invented number** --
it is llama.cpp's own upstream `common/common.h` constant:
`fit_params_min_ctx = 4096; // minimum context size to set when trying
to reduce memory use`.

Bounded at `MEMBRANE_CTXREC_MAX_CANDIDATES = 20` (checked, saturating
`uint64_t` arithmetic throughout -- doubling stops, and
`model_max_context` is appended directly, before any multiplication
could wrap). If the effective floor already exceeds `model_max_context`
(a real fixture reproduces this: `stories15M.gguf`'s own
`llama.context_length` is 128, below the 4096 floor), zero candidates
are generated and `membrane_ctxrec_resolve()` reports
`NO_FEASIBLE_CONTEXT` rather than fabricating one.

## Planner reuse

`membrane_ctxrec_resolve()` is a **bounded outer loop around the
existing, unchanged `membrane_joint_plan_resolve()`**, called once per
candidate with that candidate's own `ctx`/KV bytes forwarded verbatim.
It never re-implements GPU-layer ranking, precision ranking, placement
ranking, compatibility gating, or memory-fit arithmetic -- and it can
never select a different GPU-layer count, precision, or placement than
the joint planner itself would have chosen for that exact candidate.
**This is not a second planner.**

Every candidate is evaluated -- **no early break**. Feasibility is not
assumed monotonic in context size (precision/placement/layer selection
can each change independently as KV cost grows), so a smaller,
infeasible candidate never causes a larger, feasible one to be skipped
(`test_no_monotonicity_assumption` proves this directly).

Explicit user constraints (`--kv`, `--gpu-layers`, `--kv-placement`) are
forwarded unchanged to every candidate -- a hard constraint of the
*whole* recommendation, never re-decided per candidate size. Explicit
`--device` is already captured by whichever device's real
`free`/`total` bytes the caller resolves before calling this module,
exactly like every existing joint-planner caller.

## Selection policy

`RECOMMENDATION_POLICY_MAX_ESTIMATED_FIT`: `RECOMMENDED_CONTEXT ==
HARDWARE_FIT_CONTEXT` exactly. Two other policies were evaluated
(one candidate of headroom below the fit; an evidence-derived reserve
adjustment before evaluation) and rejected for Phase 33 specifically
because **no empirical basis exists yet** for either margin -- inventing
one would violate this project's own "no arbitrary safety margin
without evidence" convention. The result always names which policy
produced it (`recommendation_policy` field) so a future policy is never
confused with this one.

## Safety properties

- Fails closed on missing/invalid/unknown `model_max_context`, a
  minimum that exceeds it, an empty candidate set, or every candidate
  being rejected -- see the `MEMBRANE_CTXREC_STATUS_*` taxonomy in
  `context_recommender.h`.
- Never exceeds `model_max_context`.
- Never bypasses `compat_check.c`'s architecture allowlist (a
  compressed-KV request for an unsupported architecture fails closed
  for every candidate, the same as it always has).
- Never allocates memory/model/context to "test" fit; never
  benchmarks; never probes for OOM.
- Deterministic: identical inputs always produce an identical result
  (no time, randomness, or I/O anywhere in this module) --
  `test_deterministic_repeat` compares two resolutions byte-for-byte.
- Checked, saturating 64-bit arithmetic throughout; a candidate whose
  `ctx` exceeds the joint planner's own `uint32_t ctx_size` fails that
  one candidate closed (`CTX_EXCEEDS_UINT32`) rather than truncating it.

## Failure modes

| Status | Meaning |
|---|---|
| `OK` | A feasible candidate was found. |
| `MODEL_MAX_CONTEXT_UNKNOWN` | The GGUF `{arch}.context_length` key was missing/unreadable. |
| `INVALID_MODEL_MAX_CONTEXT` | The key was present but read back as 0. |
| `MINIMUM_EXCEEDS_MODEL_MAX` | `minimum_required_context > model_max_context`. |
| `NO_FEASIBLE_CONTEXT` | No candidates could even be generated (e.g. the effective floor already exceeds `model_max_context`). |
| `PLANNER_REJECTED_ALL` | Candidates were generated and evaluated, but the joint planner rejected every one. |
| `INVALID_INPUT` | Malformed request (NULL, non-positive `n_layer_all`, an oversized `candidate_count`). |

## Known limitation: why recommendation is not yet a guarantee

Three real, disclosed gaps -- none papered over with an invented
number:

1. **Transient compute-buffer allocation is not modeled.** Fit
   estimates cover model weights and the KV cache itself, not `ggml`'s
   own transient compute-buffer allocation at context-creation time
   (Phase 32's own disclosed gap). No safe, evidence-backed headroom
   formula was found this phase to close it, so `RECOMMENDED_CONTEXT`
   remains **an estimated fit, not a guaranteed absence of OOM**.
2. **No host-RAM capacity check exists anywhere in the product.**
   Audited directly this phase: `joint_planner.h`'s own candidate
   field comment already says `fits_host... always 1 today (no
   host-RAM capacity guard exists in the product)`, and `main.cpp`'s
   only host-memory signal (`HOST_MEMORY_PRESSURE`) is explicitly
   observability-only -- "never influenced any pass/fail decision."
   This module does not invent a host-RAM safety check to paper over
   that gap (doing so would duplicate memory-fit logic no existing
   module owns, and would need an unevidenced margin). Instead,
   `membrane_ctxrec_result_t::host_memory_unvalidated` is set to `1`
   whenever the selected plan leaves any weight or KV bytes CPU/host-
   resident, so a future CLI surface can never silently claim a
   host-memory guarantee that does not exist.
3. **Adaptive (bare `--auto`) precision has no CPU-only fallback at
   zero GPU budget, today.** Source-verified this phase:
   `joint_planner.c`'s `resolve_adaptive_precision()` always calls
   `membrane_adaptive_kv_resolve()` with `is_gpu_backend` hardcoded to
   `1`, and returns `GPU_MEMORY_INSUFFICIENT` (not a CPU-only plan)
   when neither Q8 nor Q5 reaches even one GPU layer --
   `resolve_single_precision()` (explicit `--kv`) does not have this
   gap, since it always builds a `gpu_layers=0` fallback candidate. A
   genuinely GPU-less host must reach this module with an *explicit*
   precision request, matching `main.cpp`'s own real `gpu_layers==0`
   short-circuit (which never invokes the joint planner with an
   implicit adaptive request either). See
   `results/context-recommendation/core-validation.json`'s
   `known_limitations` for the exact test names that pin this down.

Phase 34/35 must preserve this wording unless new evidence improves it
(Section 31 of the Phase 33 task).

## Phase 34 CLI handoff

Not implemented this phase (Section 24 of the Phase 33 task explicitly
forbids it). Expected scope, recorded for Phase 34:

- `--ctx auto` (opt-in run path) and an extended `--inspect-model` (a
  look-before-you-leap path), both built on this exact core.
- Wiring the caller side: real GGUF `model_max_context` read (already
  additive in `gpu_device.h`/`.cpp` this phase), real per-candidate
  `kv_bytes_native/q8/q5` computation (`ggml_row_size()`-based, as
  `context_recommender_dryrun.cpp` already demonstrates), and real
  device/host facts.
- Preserving every explicit user constraint (`--kv`, `--gpu-layers`,
  `--kv-placement`, `--device`) as a hard constraint, exactly as this
  core's API already requires.
- Human- and JSON-facing explanation output, built from this core's
  `explanation`/`reason_code`/`evaluated[]` fields plus Phase 34's own
  UX-layer prose -- no new planner policy.
- Existing runtime JSON schema stays unchanged until Phase 34 actually
  adds recommendation fields to it (additive-only, per this project's
  existing `schema_version: 1` convention).
