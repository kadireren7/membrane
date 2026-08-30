# Joint auto planner

Phase 20 replaces `--auto`'s pre-Phase-20 sequential resolution (GPU layer
count and KV precision decided pre-load; KV placement decided separately,
post-load) with one bounded, deterministic evaluation across all three
together. See `tools/membrane-run/joint_planner.h`'s own top comment for the
full mechanism and citations — this page summarizes the policy and what
changed.

## What was wrong before

`kv_residency_policy.h`'s own header comment (unchanged, still true of the
module itself) says it outright:

> Section 27: precision is decided first, placement second, strictly
> sequential, no joint search.

Concretely: the pre-load GPU-layer/precision decision had no visibility into
whether the resulting plan's KV cache would actually fit wherever
`--kv-placement` said it should go. A configuration could be selected, then
independently rejected (or silently degraded) by the placement step, with no
way for the earlier decision to reconsider.

## GPU weight-byte estimate: a real, measured, source-verified bug

Phase 19 measured (`results/planner-accuracy/measurements.json`, **REAL**)
the pre-Phase-20 estimate (`selected_layers * bytes_per_layer`)
under-counting real GPU weight residency by **+21.3% / +11.6% / +12.7%**
(SmolLM2-135M / SmolLM2-360M / Qwen2.5-1.5B —
`results/joint-planner/estimate-correction.json`'s own
`pre_phase20_error_percent` field recomputes these three figures directly
from that same committed `observed_delta_bytes`, not a separate claim).
Phase 20's own investigation (source inspection, not model/context sweeps)
found the exact mechanism in `third_party/llama.cpp/src/llama-model.cpp` and
`llama-model-loader.cpp`:

- `llama_model::n_gpu_layers()`'s own comment: `n_gpu_layers` counts the
  model's `n_layer_all` `blk.N.` layers **plus one** for the "output" role.
- The output-role tensor(s) are placed on GPU **before** any `blk.N.` layer
  (`i_gpu_start = n_layer_all + 1 - n_gpu_layers`), so requesting `V` GPU
  layers actually offloads `(V-1)` `blk.N.` layers + the output role, not `V`
  `blk.N.` layers.
- The input token-embedding tensor is **always** CPU-resident regardless of
  `n_gpu_layers` (`dev_input`'s own comment: "there is very little benefit to
  offloading the input layer, so always keep it on the CPU").
- On a **tied**-embedding model (no separate `output.weight` tensor in the
  GGUF — confirmed via `strings` on SmolLM2's `.gguf` files), llama.cpp
  duplicates `token_embd.weight`'s bytes into a second, GPU-eligible
  allocation for the output role (`llama-model-loader.cpp`'s
  `TENSOR_DUPLICATED` handling, routed to the output buffer-type list by
  name).

The corrected formula — `membrane_joint_estimate_gpu_weight_bytes()` —
is `V == 0 ? 0 : (V-1)*bytes_per_layer + output_role_bytes`, where
`output_role_bytes` (new field on `gpu_device.h`'s
`membrane_gpu_model_estimate_t`, computed during the same GGUF metadata scan
that already existed) is `output.weight`'s real bytes if present, else
`token_embd.weight`'s real bytes (the tied-duplicate stand-in), plus
`output_norm.weight`'s real bytes.

Verified against the real Phase 19 measurements and a live `membrane-run
--json` run on this repo's real Vulkan device, both captured in
`results/joint-planner/estimate-correction.json` (**REAL**, `corrections`
array — every `corrected_estimate_bytes`/`phase20_error_percent` field
there is a plain arithmetic recomputation from the real
`bytes_per_layer`/`output_role_bytes`/`observed_delta_bytes` values it also
lists, re-checkable by hand): the correction brings the residual error down
to **roughly +3.0% / +1.1% / +0.2%** for the same three models. `total_bytes`
(every tensor, including the always-CPU-resident input embedding) is
deliberately **not** used directly — Phase 20's own instruction was explicit
that this would overstate GPU residency for anything less than full offload.

## Candidate generation (bounded, never a full cross product)

- **Precision**: the request's own explicit `--kv` value (a single hard
  candidate) — or, for `--kv adaptive`/bare `--auto`, exactly `{Q8, Q5}`,
  matching `adaptive_kv_policy.h`'s own already-shipped "never native"
  contract for adaptive requests (unchanged by this phase).
- **GPU layers**: the request's own explicit `--gpu-layers N|all` value (a
  single hard candidate, must fit or the whole candidate is ineligible —
  never silently shrunk) — or, for `--gpu-layers auto`, up to two candidates
  per precision: the corrected-formula max-fit layer count, and `0`
  (CPU-only weights) as an explicit fallback so a constrained-VRAM run has a
  legal candidate within the *same* precision to fall back to.
- **Placement**: never independently generated. Each `(precision,
  gpu_layers)` candidate's placement is resolved once via
  `kv_residency_policy.h`'s own `membrane_kv_residency_resolve()`, given
  `--kv-placement`'s request mode (`default`/`gpu`/`cpu`/`auto`, unchanged) —
  this is what actually closes the "no joint search" gap: a candidate that
  can't have its KV placed as requested is ineligible, *before* ranking, not
  discovered afterward.

At most 2 raw candidates per call (the resolved precision's own best
full-layer candidate, plus — only when GPU layers are themselves `auto` and
that full candidate is ineligible — a `0`-layer same-precision fallback);
`resolve_adaptive_precision()` calls `build_candidate()` up to 3 times (the
GPU-fit winner, its runner-up when the winner's own placement needs
checking, and the fallback), all well within the 8-slot bound
`MEMBRANE_JOINT_MAX_CANDIDATES` allows. Each call to
`membrane_joint_plan_resolve()` only ever ranks candidates within a single
precision family — see the ranking policy below for why a native-vs-Q8-vs-Q5
cross-product is not what actually happens today.

## Ranking policy (`joint-auto-v1`)

1. Must satisfy every explicit constraint (enforced by generation itself).
2. Must pass `compat_check.h`'s real architecture gate (native always does;
   q8/q5 require it).
3. Must conservatively fit (GPU memory **and** the placement step, both).
4. **For an explicit precision** (native, or `--kv q8`/`--kv q5` fixed): no
   precision ordering applies at all — there is only one precision candidate.
   **For `--kv adaptive`/bare `--auto`** (Q8 vs Q5 only — adaptive never
   proposes native, matching `adaptive_kv_policy.h`'s own already-shipped
   contract): the precision choice is `adaptive_kv_policy.h`'s own
   documented, more-nuanced order — full GPU residency beats precision
   (Q5-at-full-residency beats Q8-at-partial-residency), Q8 wins only when
   the two are otherwise tied. Phase 20 does **not** override or duplicate
   that order; it only adds a placement-eligibility check on top (see next
   paragraph). Neither case is ever a performance claim (Phase 12G found no
   general GPU-KV/CPU-KV throughput advantage in the tested regime; this
   policy makes no speed assumption at all).
5. Within the same precision, prefer more GPU-resident weight layers.
6. Deterministic tie-break: candidate generation order is itself
   deterministic; no further randomness/instability is ever consulted.

For the adaptive case, `adaptive_kv_policy.h`'s own GPU-fit-based winner can
still fail the SEPARATE placement step (`kv_residency_policy.h`'s own runtime
margin can reject a candidate GPU-fit alone would have accepted) while the
runner-up precision's placement succeeds — the joint planner evaluates
placement for **both** the winner and the runner-up before finalizing, and
promotes the runner-up when the winner is placement-ineligible but the
runner-up is not (`test_joint_planner.c`'s
`test_adaptive_placement_promotes_runner_up_precision`). Given the current
CLI contract (adaptive never proposes native; an explicit precision is
always a single fixed value), native never actually competes against Q8/Q5
in the same real invocation today — the ranking function itself is written
generally regardless (and
exercised that way by `test_joint_planner.c`'s synthetic scenarios), should
that contract ever change.

## Explicit constraints are hard, always

`--auto --kv q8` → precision fixed q8, layers auto, placement auto.
`--auto --kv-placement cpu` → placement fixed cpu, precision auto, layers
auto. `--auto --gpu-layers 12` → layers fixed 12, precision auto, placement
auto. None of these get silently overridden by the planner to find a
candidate — see `test_joint_planner.c`'s
`test_explicit_gpu_layers_never_changed` /
`test_explicit_cpu_placement_never_gpu` /
`test_explicit_q8_incompatible_architecture`.

## Qwen2.5 regression (Section 14) — historical, at the Phase 20 commit

**Superseded by Phase 26** (`docs/compatibility.json`'s MC-17/MC-18/MC-19
rows are now `SUPPORTED`, `docs/compatibility.md`) — Qwen2 compressed KV
is now allowlisted. This section is kept as-written because it describes
real, immutable evidence captured *at the Phase 20 commit*
(`results/joint-planner/estimate-correction.json` is never rewritten),
not because the behavior it describes is still current. At that commit:
Qwen2.5 is `qwen2`-architecture — `compat_check.c` rejected q8/q5 for it
unconditionally. Live-verified on this repo's real Vulkan device and
captured in `results/joint-planner/estimate-correction.json`'s
`qwen2_regression_smokes` (**REAL**, real commands/exit codes/`--json`
fields, not paraphrased): `--kv q8`/`--kv adaptive` both failed closed with
`NO_FEASIBLE_CANDIDATE` (adaptive
never fell back to native — matching its own contract), while `--kv native
--gpu-layers all --kv-placement cpu` at `ctx=28500` (a real constrained-VRAM
point from prior evidence) still succeeded with all 28 layers, KV correctly
placed off-GPU. Compression was never attempted merely to make a
then-compression-incompatible model fit. See `docs/compat-expansion.md`
(Phase 26) for the current, real q8/q5/adaptive Qwen2 evidence.

## Out of scope (this phase)

- `--compare-kv`/`--gpu-bench` keep the exact pre-Phase-20
  `gpu_policy_resolve()` + `adaptive_kv_resolve()` path, unchanged — those
  modes size a *second*, native comparison pass the joint planner doesn't
  model, and `product_cli.cpp` already rejects non-default
  `--kv-placement` together with them (Phase 12H scope), so they were never
  reaching the placement-aware logic regardless.
- No runtime fallback/retry — Phase 21's job. A selected plan that fails to
  *instantiate* (as opposed to being rejected up front by this planner)
  behaves exactly as it did before Phase 20.
- No live KV migration, promotion/demotion, or dynamic scheduler — every
  plan here is still a static, pre-context decision.
- The 15%/512 MiB safety reserve is unchanged — Phase 19 found it absorbed
  every measured estimator error on the one tested device, which is not a
  general guarantee; Phase 20 does not retune it from that evidence.
