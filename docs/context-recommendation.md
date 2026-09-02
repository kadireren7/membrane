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
  Phase 35+; this core never tokenizes anything itself)
- `candidates[]` -- a caller-populated array of `(ctx, kv_bytes_native,
  kv_bytes_q8, kv_bytes_q5)` tuples, one per candidate context size.
- **Phase 34 additions** (additive, source-compatible): `total_weight_
  bytes` (the model's real total tensor-byte footprint, from
  `gpu_device.h`'s own `total_bytes`) and `host_total_bytes`/
  `host_available_bytes`/`host_available_known` (real host memory
  facts, the same single source `main.cpp`'s own `read_host_meminfo()`
  already reads) -- see `docs/host-memory-guard.md` for what these
  feed into.

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
- **Phase 34: a candidate the joint planner accepts is then checked
  against real host RAM availability** (`docs/host-memory-guard.md`)
  whenever it leaves any weight or KV bytes host-resident -- status
  can never be `OK` for such a candidate unless that host-memory
  feasibility was actually validated. This closes (for host-resident
  weight/KV bytes specifically) the gap this document's own "Known
  limitation" section below used to describe as fully open.
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
2. **No host-RAM capacity check existed anywhere in the product --
   CLOSED this phase, for host-resident weight/KV bytes specifically.**
   Phase 34 (`docs/host-memory-guard.md`) added
   `tools/membrane-run/host_memory_guard.h`/`.c`, evidence-derived from
   real `membrane-run --json` measurements, and integrated it into
   this module: status can never be `OK` for a candidate requiring
   host memory unless that memory was actually validated. A residual
   gap remains open and disclosed: process/backend baseline overhead
   for a fully GPU-resident plan (e.g. real Vulkan driver overhead,
   ~248 MiB observed with zero host-resident weight/KV bytes) is *not*
   covered -- see `docs/host-memory-guard.md`'s own "Residual
   uncertainty" section.
3. **Adaptive (bare `--auto`) precision has no CPU-only fallback at
   zero GPU budget in `joint_planner.c`'s own pre-load candidate
   generation, today.** Source-verified in Phase 33, root-caused in
   Phase 34: `joint_planner.c`'s `resolve_adaptive_precision()` always
   calls `membrane_adaptive_kv_resolve()` with `is_gpu_backend`
   hardcoded to `1`. Phase 34 confirmed this is **accidental historical
   coupling, not product policy** -- `main.cpp`'s own
   `resolve_cpu_adaptive_kv()` already calls the same underlying
   function with `is_gpu_backend=0` for the real, shipped bare `--auto`
   CLI path, so **the real CLI already works correctly on a CPU-only
   host today**; this gap is scoped to `joint_planner.c`'s pre-load
   path specifically (which this recommendation core depends on). A
   structurally clean fix was identified but deliberately deferred
   (see `docs/host-memory-guard.md`'s "CPU-only adaptive decision") --
   a genuinely GPU-less host must still reach this module with an
   *explicit* precision request for now. See
   `results/context-recommendation/core-validation.json`'s
   `known_limitations` for the exact test names that pin this down.

This wording must be preserved unless new evidence improves it
(Section 31 of the Phase 33 task, extended by Phase 34's own
disclosure obligations).

## Phase 35 CLI handoff

Not implemented in Phase 33 or Phase 34 (both phases explicitly forbid
a CLI surface this cycle). Expected scope, recorded for Phase 35:

- `--ctx auto` (opt-in run path) and an extended `--inspect-model` (a
  look-before-you-leap path), built on this exact core plus Phase 34's
  host-memory guard.
- Wiring the caller side: real GGUF `model_max_context`/
  `total_weight_bytes` reads (already additive in `gpu_device.h`/`.cpp`
  since Phase 33), real per-candidate `kv_bytes_native/q8/q5`
  computation (`ggml_row_size()`-based, as
  `context_recommender_dryrun.cpp` already demonstrates), and real
  device/host facts (including `/proc/meminfo`, reused from `main.cpp`'s
  existing `read_host_meminfo()`).
- Preserving every explicit user constraint (`--kv`, `--gpu-layers`,
  `--kv-placement`, `--device`) as a hard constraint, exactly as this
  core's API already requires.
- The explicit-run host-memory policy Phase 34 deliberately left open
  (`docs/host-memory-guard.md`'s "Explicit-run scope"): whether/how an
  explicit `--ctx N` run should also gain a host-memory check.
- Human- and JSON-facing explanation output, built from this core's
  `explanation`/`reason_code`/`evaluated[]` fields plus Phase 35's own
  UX-layer prose -- no new planner policy.
- Existing runtime JSON schema stays unchanged until Phase 35 actually
  adds recommendation fields to it (additive-only, per this project's
  existing `schema_version: 1` convention).
- Optionally revisit the deferred CPU-only adaptive fix
  (`docs/host-memory-guard.md`'s "CPU-only adaptive decision") if
  Phase 35's own scope naturally touches `joint_planner.c` for CLI
  integration reasons.
