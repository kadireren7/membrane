# Static KV residency (`--kv-placement`, Product Phase 12H)

## What this is

`--kv-placement` lets you choose which backend device holds the KV
cache **separately** from which device holds the model weights
(`--gpu-layers`). The whole placement decision is made once, before
context construction — there is no runtime movement, no promotion, no
demotion, no adaptive/learned policy. This is deliberately a much
simpler and safer mechanism than the Phase 12F research dynamic
scheduler, which this phase does **not** productize.

## Why it exists

Prior research (Phase 12B–12G, see `results/phase12/`) established:

- KV placement can be chosen independently of weight placement
  (`kv_dev_override`, Phase 12B).
- A deterministic dynamic scheduler works but showed no measured
  throughput advantage over static placement on the tested
  hardware/model range (Phase 12F).
- CPU-resident KV was at parity or faster than GPU-resident KV across
  every valid Phase 12G measurement (135M/360M/1.5B, contexts up to
  65536) — so the value of KV placement on this hardware range is
  **capacity**, not a performance optimization
  (`KV_PLACEMENT_CAPACITY_ONLY_ON_TESTED_REGIME`).

Phase 12H turns that capacity finding into an opt-in product feature:
move part (or all) of the KV cache to system RAM, while keeping model
weights on GPU exactly as `--gpu-layers` already placed them, to fit
larger contexts or lower VRAM budgets.

**No universal speed claim.** This phase measured a genuinely mixed,
noisy performance picture at large context on one model (see
`results/v0.3/kv-residency-productization/performance.json`) — it is
reported as measured, not marketed as a speedup or a slowdown.

## Modes

```
--kv-placement default|gpu|cpu|auto
```

- **default** (the default) — unmodified behavior. `kv_dev_override`
  is left `NULL`; KV follows whatever device llama.cpp itself would
  already use. This is the only mode with zero behavioral change from
  before this feature existed — no opt-in, nothing changes.
- **gpu** — every KV layer on the same GPU device as the offloaded
  weights, or fails closed (`PLACEMENT_BUDGET_INSUFFICIENT`) if it
  doesn't safely fit. Requires `--gpu-layers all|auto|N`.
- **cpu** — every KV layer in system RAM. Model weights are
  unaffected (still governed entirely by `--gpu-layers`). Always
  valid, including on CPU-only builds.
- **auto** — MEMBRANE's planner keeps as many KV layers GPU-resident
  as safely fit, in ascending layer-index order, and places the rest
  in system RAM. Requires `--gpu-layers all|auto|N`. Never fails for
  lack of GPU room — an all-CPU-KV plan is a valid `auto` outcome.

`auto`'s "maximize GPU residency" objective is a **conservative
compatibility** choice (closest to pre-existing behavior when the
budget allows it), not a performance claim — see
`results/v0.3/kv-residency-productization/planner_validation.json`.

## Interaction with `--kv` and `--gpu-layers`

Precision (`--kv native|q8|q5|adaptive`) and placement are strictly
independent dimensions — `--kv-placement` never changes KV precision,
`--kv` never changes KV device residency. With `--kv adaptive`,
precision is resolved to a concrete mode **first**, then placement is
resolved **second**, using that concrete mode's real per-layer KV
byte cost (see `results/v0.3/kv-residency-productization/
adaptive_composition.json`) — no joint search.

`--gpu-layers` resolution is completely unaffected by `--kv-placement`
— the existing weight-layer guard (`gpu_policy.c`) runs first and
never sees `--kv-placement`'s value; weight placement for the same
`--gpu-layers` request is byte-identical no matter which
`--kv-placement` mode is passed
(`results/v0.3/kv-residency-productization/capacity_uplift.json`'s
`weights_unchanged_proof`).

`--kv-placement` is currently rejected together with `--compare-kv`/
`--gpu-bench` (a clear CLI error, not a silent no-op) — composing with
those two-context comparison modes is out of this phase's scope.

## A known, disclosed limitation

`gpu_policy.c`'s own pre-flight check (deliberately left unmodified —
weight placement must stay independent of KV placement) always
assumes the **full** KV cache needs GPU room when deciding whether to
grant a `--gpu-layers` request at all, regardless of what
`--kv-placement` will later do. Confirmed directly: at large enough
contexts, `--kv-placement cpu` is rejected by this pre-check
identically to `default`, even though `cpu` mode needs zero GPU KV
bytes. This means `--kv-placement` does not (yet) unlock arbitrarily
larger `--gpu-layers` requests — see `capacity_uplift.json` for what
it *does* reliably help with: a real, reproduced gap between what
`gpu_policy`'s pre-load estimate predicts and what the real allocator
can do (measured on a GTX 1650: the default all-GPU-KV path fails with
a genuine Vulkan out-of-device-memory error at a context where
`--kv-placement auto`/`cpu` succeed, at identical weight placement).

## Fail-closed reasons

`PLACEMENT_GPU_FULL`, `PLACEMENT_CPU_FULL`, `PLACEMENT_AUTO_SPLIT`,
`PLACEMENT_DEFAULT_UNCHANGED`, `PLACEMENT_BUDGET_INSUFFICIENT`,
`PLACEMENT_BACKEND_UNAVAILABLE`, `PLACEMENT_INVALID_CONFIG` — stable
strings, also emitted verbatim in `--json` output's
`kv_placement.placement_reason` field. Never a silent fallback to an
unrelated mode.

## Validation scope

Real GTX 1650 (Vulkan) and a CPU-only build (no GPU backend compiled
in) — see `results/v0.3/kv-residency-productization/`. Not validated
on CUDA or any other backend. Minimum llama.cpp patch set required:
`kv-type-override` (pre-existing) + `kv-device-override` (added this
phase) — the research-only `kv-runtime-relocate` and
`kv-buffer-retirement` patches are not part of the product build.

## Example commands

```
# Explicit all-CPU KV, weights fully on GPU
membrane-run --model model.gguf --prompt "..." --ctx 32768 \
    --gpu-layers all --kv-placement cpu

# Let MEMBRANE decide the split
membrane-run --model model.gguf --prompt "..." --ctx 32768 \
    --gpu-layers all --kv-placement auto

# Composes with adaptive precision
membrane-run --model model.gguf --prompt "..." --ctx 32768 \
    --kv adaptive --gpu-layers all --kv-placement auto
```
