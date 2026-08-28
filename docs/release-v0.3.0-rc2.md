# MEMBRANE v0.3.0-rc2 — release notes

`v0.3.0-rc1` predates Phases 13–21 and does not represent current
`main` — it shipped Vulkan GPU offload alone. `v0.3.0-rc2` supersedes
it with everything built since: Q5 KV, adaptive KV precision, static
KV residency planning, a corrected GPU weight-byte estimate, a
compatibility matrix, planner-accuracy evidence, a joint
GPU-layers/precision/placement `--auto` planner, and bounded
transparent apply-time fallback. `v0.2.0` stable's CPU-only default
behavior is unchanged by any of this.

This is a **release-readiness/audit phase** (Phase 22) — no new
product feature was added to reach RC2. The only product-behavior
changes in this phase are the version-string bump and a `--help`
addition documenting Phase 21's already-shipped fallback behavior (see
"What changed in this phase" below).

## Highlights

- Joint `--auto` planner (Phase 20): GPU layer count, KV precision,
  and KV placement resolved together as one bounded, deterministic,
  ranked candidate list — not three independent, sequential decisions.
- Corrected GPU weight-byte estimate (Phase 20): fixes a measured
  +11.6–21.3% under-count from treating `n_gpu_layers` as a flat
  `layers × bytes_per_layer` product; residual error is now
  0.2–3.0% on tested models — real, committed figures in
  `results/joint-planner/estimate-correction.json`, discussed in
  `docs/joint-planner.md`.
- Bounded, transparent apply-time fallback (Phase 21): if the primary
  auto-managed plan can't actually be instantiated at runtime (model
  load fails, context construction fails, or GPU memory changed since
  planning), MEMBRANE retries another already-legal, already-ranked
  plan — bounded at 3 attempts, never silent, never inventing a
  candidate the planner didn't already rank (`docs/auto-fallback.md`).
- Static KV residency planning (`--kv-placement default|gpu|cpu|auto`)
  and adaptive KV precision (`--kv adaptive`), both explicit-flag-
  overridable and independent of each other and of GPU layer count.
- Machine-checked compatibility matrix (`docs/compatibility.md`,
  `scripts/verify-compatibility.py`) and planner-accuracy evidence
  (`docs/planner-accuracy.md`, `scripts/verify-planner-accuracy.py`).
- Reliable install/uninstall (Phase 17): CPU and Vulkan builds install
  outside the build tree with correct RPATHs, verified by the
  `packaging-smoke` CI job on every push/PR.
- Structured, additive-only JSON diagnostics throughout: plan
  resolution (`--plan-only`), the joint planner's own candidate
  summary, and the fallback trace, all under `schema_version: 1`.

## What this does NOT claim

Unchanged from RC1's own disclosure, still true for RC2:

- No CUDA product path.
- No claim about "all llama.cpp architectures" or "all Vulkan GPUs" —
  compatibility is a per-architecture matrix
  (`docs/compatibility.md`), not a blanket claim.
- No runtime live KV migration, no dynamic KV scheduler, no per-layer
  mixed precision — Phase 21's fallback is complete-plan
  re-instantiation only (`docs/auto-fallback.md`'s own "What this is
  NOT" section).
- No production-hardware guarantee, no "zero OOM" guarantee — every
  memory estimate in this product is a documented, conservative,
  point-in-time snapshot, never an absolute guarantee (see
  `docs/joint-planner.md` and `docs/auto-fallback.md`'s own disclosed
  limitations).

## Installation

```bash
git clone --recursive https://github.com/kadireren7/membrane
cd membrane

# CPU-only
cmake -S . -B build -DMEMBRANE_ENABLE_LLAMA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2 --target membrane-run
cmake --install build --prefix /your/prefix

# Vulkan (needs libvulkan-dev, glslc, vulkan-tools, spirv-headers)
cmake -S . -B build-vulkan -DMEMBRANE_ENABLE_LLAMA=ON -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j2 --target membrane-run
cmake --install build-vulkan --prefix /your/prefix
```

Full detail: `docs/install.md` (continuously re-verified by
`packaging-smoke`, not a point-in-time log).

## What changed since RC1

`v0.3.0-rc1` → current main, 13 squash-merged PRs, 113 files changed
(+17138/−848 lines, excluding `third_party/`):

| Area | Changes |
|---|---|
| Runtime | Q5 KV mode (`d101fc8`), adaptive Q8/Q5 KV policy (`d6e6189`), static KV residency planner / `--kv-placement` (`8287e8c`) |
| Planner | Automatic-planning hardening (Phase 13.1, `e018b6d`), joint GPU-layers/precision/placement planner + corrected GPU weight estimate (Phase 20, `8be484c`) |
| Fallback | Bounded transparent apply-time fallback (Phase 21, `b97edfd`) |
| Observability | Runtime planning diagnostics (Phase 13.2, `af22d23`), planner memory-accuracy measurements (Phase 19, `6b6b966`) |
| Packaging | Reliable install/uninstall workflow with pinned `membrane_core` (Phase 17, `add0302`) |
| Compatibility | Verified compatibility matrix (Phase 18, `4e2dda2`) |
| Docs | Product README rewrite (`635d191`, `3884529`), citation metadata update (`a397dcf`) |

## What changed in this phase (Phase 22)

Release-readiness audit only, per this phase's own explicit "no new
product features" constraint:

- `MEMBRANE_VERSION` bumped from `0.3.0-rc1` to `0.3.0-rc2`
  (`tools/membrane-run/product_cli.h`) — the single source for
  `--version`, the startup banner, and every JSON `membrane_version`
  field.
- `--auto`'s `--help` text now documents Phase 21's already-shipped
  fallback behavior (it previously said nothing about it — a real
  documentation gap, not new behavior).
- README's release-status line corrected (previously described
  "product hardening... not yet in a tagged release" using a Phase-13
  feature list that predated Phases 17–21 by several phases).
- This document and `results/release-v0.3.0-rc2/` (if applicable) are
  new; no other evidence file was modified.

## Compatibility

`docs/compatibility.md` / `docs/compatibility.json`, verified by
`scripts/verify-compatibility.py`:

- 26 total rows: **16 SUPPORTED**, **9 UNSUPPORTED**, **1
  NOT_YET_VALIDATED**.
- Scope unchanged from Phase 18: `LLM_ARCH_LLAMA` verified;
  compressed KV (`q8`/`q5`/`adaptive`) requires the real architecture
  gate (`compat_check.c`) to pass — Qwen2.5 remains a real,
  machine-verified UNSUPPORTED case for compressed KV, and fails
  closed rather than silently degrading to native (see
  `docs/auto-fallback.md`'s own Qwen2.5 regression check).
- No compatibility row was added, removed, or reclassified in this
  phase — RC2 does not expand compatibility for release optics.

## Planner behavior

See `docs/joint-planner.md` for the full policy. Unchanged since Phase
20: one bounded, deterministic ranked candidate list per `--auto`
request, corrected GPU weight-byte estimate
(`membrane_joint_estimate_gpu_weight_bytes()`), explicit flags always
constrain their own dimension only. `scripts/verify-planner-accuracy.py`
— **6/6** against the immutable Phase 19 evidence
(`results/planner-accuracy/measurements.json`, 8 measurements) —
unchanged, not re-measured this phase.

## Fallback behavior

See `docs/auto-fallback.md` for the full contract. Unchanged since
Phase 21: bounded at 3 attempts, memory re-snapshotted before each
attempt, model reload only when `gpu_layers` actually changes, never
silent. `scripts/verify-auto-fallback.py` — **9/9** against
`results/auto-fallback/validation.json` (3 runs: a clean primary
success, a real captured memory-refit-skip-and-recover, and a Qwen2.5
regression check) — unchanged, not re-measured this phase.

## Measured evidence / verification

Committed, `scripts/verify-release-readiness.py`-checked evidence:
`results/release-v0.3.0-rc2/readiness.json` — captured against
`membrane_commit` `9f1d3f96493883ef2cf4c69b4792b30bd7657110`
(`release/v0.3.0-rc2-prep`, [PR
#32](https://github.com/kadireren7/membrane/pull/32)). Summary (see
that file for the full detail, including the real Vulkan smoke's exact
JSON fields and the CI run IDs it cites):

- `scripts/verify-results.py`: **73/73**, chaining compatibility
  (13/13), planner-accuracy (6/6), and auto-fallback (9/9) — **91
  total checks**, all passing.
- Full `ctest --output-on-failure`: **70/70** tests passing, Debug
  build; **70/70** again under ASan/UBSan
  (`-DMEMBRANE_ENABLE_SANITIZERS=ON`), no sanitizer report.
- CPU install: fresh configure → build → install → run `--help`/
  `--version` outside the build tree → uninstall via
  `install_manifest.txt` — every installed file removed cleanly.
- Vulkan install/uninstall: same contract, real Vulkan1 device, plus a
  real `--auto` generation run (SmolLM2-135M) confirming
  `membrane_version`/`planner`/`fallback` JSON fields are all correct
  and self-consistent end to end. The install step itself reused an
  existing local Vulkan build tree (reconfigured/rebuilt
  incrementally) rather than a from-scratch clone, to avoid an
  unnecessary heavy rebuild on this memory-constrained host — the
  fresh-clone Vulkan install contract is still exercised every push/PR
  by the `packaging-smoke` CI job, which the readiness evidence's
  `ci_runs` cites.
- GitHub CI (`build-and-test` Debug/ASan, `thread-sanitizer`,
  `packaging-smoke`, `compatibility-check`, `planner-accuracy-check`,
  `auto-fallback-check`, `fallback-controller-tests`) and CodeQL: all
  SUCCESS — exact run IDs in `results/release-v0.3.0-rc2/readiness.json`'s
  `ci_runs` field.

## Known limitations

Carried forward from RC1, still accurate, plus Phase 20/21's own
disclosed limitations:

- Vulkan validated on a small number of real local hosts/devices
  (NVIDIA GTX 1650 Mobile class hardware) — not a general claim about
  other GPUs, vendors, or hosts.
- CUDA is not a MEMBRANE product path.
- AMD/Intel GPU product validation is not claimed.
- VRAM/estimate accuracy depends on model, context size, and
  hardware — not a fixed percentage, and never an absolute OOM
  guarantee (the joint planner's safety reserve and the fallback
  controller's memory re-snapshot are both conservative, documented
  policies, not guarantees — see `docs/joint-planner.md` and
  `docs/auto-fallback.md`).
- No custom GPU kernels — this product only wires up explicit,
  product-level control over the existing, unmodified `ggml`/llama.cpp
  Vulkan backend.
- The real apply adapter behind Phase 21's fallback controller can
  only honestly classify 3 of its own 8 failure-taxonomy slots today
  (`llama.h` exposes no error code from `llama_model_load_from_file()`/
  `llama_init_from_model()`) — disclosed in `docs/auto-fallback.md`,
  not a regression in this phase.
- No FPGA/CXL hardware claim in this release (that work lives in the
  companion `kadireren7/membrane-research` repository as simulation/
  synthesis-tool-proxy evidence only).

## Upgrade notes

No breaking CLI or JSON-schema change since RC1. New CLI-visible
behavior since RC1: `--kv q5`/`adaptive`, `--kv-placement`, `--auto`,
`--plan-only`, and Phase 21's fallback retry (all additive, all
off-by-default except where an existing flag like `--auto` already
opts in). `schema_version` has stayed `1` throughout — every JSON
field added since RC1 (`adaptive`, `kv_placement`, `planner`,
`fallback`, `reason_trace`, `warnings`, ...) is additive.

## Validation status

This is a release-CANDIDATE preparation document only — `v0.3.0-rc2`
is not yet a tag or a GitHub release at the time this document is
written; both are created only after this PR merges and its own
post-merge gate passes (see the PR for exact status). Pre-release, not
the stable release line either way — `v0.2.0` remains the latest
stable tag, and this document does not change that. See the
release-prep PR for exact CI run IDs and the final merge/tag/release
verification report.
