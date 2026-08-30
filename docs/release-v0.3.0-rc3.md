# MEMBRANE v0.3.0-rc3 — release notes

`v0.3.0-rc2` shipped Phases 13–21 (joint `--auto` planner, static KV
residency, adaptive KV precision, bounded apply-time fallback). Current
`main` is materially ahead of it: real multi-backend user-validation
evidence, additive performance-profiling/attribution diagnostics, and —
the one real product-behavior change — compressed KV (`q8`/`q5`/
`adaptive`) support expanded to a second model architecture (Qwen2).
`v0.2.0` stable's CPU-only default behavior is unchanged by any of
this.

This is a **release-readiness phase** (Phase 27) — no new product
feature was added *in this phase itself*. RC3 exists specifically
because Phase 26's compressed-KV expansion is a real product-behavior
change, and this project's own policy does not promote a prerelease
straight to stable when material behavior has changed since the last
prerelease.

## Highlights

- **Qwen2 compressed KV** (Phase 26, the major delta): `q8`/`q5`/
  `adaptive` KV precision, previously `llama`-only, now also supported
  for `qwen2` — validated end to end against the real, local
  `Qwen2.5-1.5B-Instruct` fixture. See "Qwen2 compressed KV" below for
  the exact, scoped claim.
- **Performance profiling and attribution** (Phases 24–25, observability
  only): additive `timings`/`planner_stages` JSON, a fallback trace on
  a fully-exhausted `--auto` error response, and a real investigation
  concluding neither model-load nor decode has a MEMBRANE-owned hot
  loop worth optimizing on this evidence (`docs/performance-
  optimization.md`) — **no speed-changing code shipped**, see
  "Performance claims" below.
- **Real-user validation framework** (Phase 23): a repeatable,
  privacy-preserving validation protocol plus real CPU + NVIDIA-Vulkan
  + AMD-Vulkan evidence — from **one** physical developer host (see
  "External validation status" below; do not read backend diversity as
  host/user diversity).
- Everything RC2 already had: the joint `--auto` planner, corrected GPU
  weight-byte estimate, bounded transparent apply-time fallback, static
  KV residency planning, the machine-checked compatibility matrix.

## What's new since RC2

Four squash-merged PRs, `v0.3.0-rc2` → current main (see the release-prep
PR for the exact `git log`/diff-stat):

| PR | Phase | Area | Product behavior change? |
|---|---|---|---|
| [#33](https://github.com/kadireren7/membrane/pull/33) | 23 | Real-user validation framework + first report | No — evidence/docs only |
| [#34](https://github.com/kadireren7/membrane/pull/34) | 24 | Performance-profiling instrumentation | Additive JSON only (`timings`) |
| [#35](https://github.com/kadireren7/membrane/pull/35) | 25 | Planner sub-stage attribution, fallback trace on exhausted error JSON | Additive JSON only (`planner_stages`, error-path `fallback`) |
| [#36](https://github.com/kadireren7/membrane/pull/36) | 26 | Qwen2 compressed-KV compatibility expansion | **Yes — real behavior change** |

Only Phase 26 materially expands product capability; Phases 23–25 add
validation evidence and observability, not new runtime behavior.

## Qwen2 compressed KV

Precise, model-scoped claim — do not generalize beyond this:

**`Qwen2.5-1.5B-Instruct`**, `q8`/`q5`/`adaptive` KV precision, on
**CPU** and **Vulkan**, with independently-resolved KV placement
(`default`/`cpu` both validated), is `SUPPORTED`
(`docs/compatibility.json` MC-17/MC-18/MC-19; real evidence:
`results/compat-expansion/validation.json`, 8 REAL rows).

Measured memory reduction (real KV byte counts, not estimated): `q8` =
**53.125%** of native, `q5` = **37.5%** of native — matching the real
`ggml` `Q8_0`/`Q5_1` block-format formulas exactly.

This is **not**:
- "All Qwen2 models supported" — only `Qwen2.5-1.5B-Instruct` was
  tested; the underlying claim is architecture-level
  (`LLM_ARCH_QWEN2`, source-proven — see `docs/compat-expansion.md`),
  but no other Qwen2-family model has been run.
- "All llama.cpp architectures supported" — the compressed-KV allowlist
  (`compat_check.c`) is exactly `{llama, qwen2}`; every other
  architecture remains `NOT_YET_VALIDATED` for native precision and
  `UNSUPPORTED` for compressed KV.
- A performance claim — this is a compatibility/correctness expansion
  only. No throughput/latency claim is made or implied.

## Observability / profiling additions

Additive JSON only, `schema_version` unchanged at `1`:

- `timings` object (`total_ms`, `planner_ms`, `model_load_ms`,
  `tokenization_ms`, `context_create_ms`, `prefill_ms`, `decode_ms`,
  `first_token_ms`, `throughput`) — Phase 24.
- `timings.planner_stages` (`device_enumeration_ms`,
  `gguf_prescan_ms`, `joint_planner_core_ms`) — Phase 25. Real finding:
  the GGUF metadata pre-scan, not device enumeration, dominates the
  GPU-requested planner window.
- A fully-exhausted `--auto` error response (`ok: false`,
  `AUTO_FALLBACK_EXHAUSTED`) now includes the same per-attempt
  `fallback` trace the success schema already had — Phase 25.

No existing field was removed or renamed. No behavior change to
generation, planning, or fallback logic — see "Performance claims"
below for why none of this should be read as a speed improvement.

## Performance claims

**None.** Phase 25 investigated the two highest-priority measured
bottlenecks (model load, decode throughput) and found both are fully
third-party (`llama.cpp`/`ggml`-backend) cost with no MEMBRANE-owned
code inside either timed boundary — a real, evidence-backed "no safe
product-local optimization" conclusion (`docs/performance-
optimization.md`), not a shortfall. **Zero performance-changing code
was implemented.** This release notes document, and every other
release-facing doc, must not say "faster," "performance optimized," or
"improved inference speed" for this cycle — the profiling/attribution
work added observability, not speed.

## Compatibility

Current `docs/compatibility.json` (verified by
`scripts/verify-compatibility.py`, checked as part of this release's
own readiness evidence): **26 total rows — 19 SUPPORTED, 6
UNSUPPORTED, 1 NOT_YET_VALIDATED.**

This is a real change from RC2's own frozen scope (16/9/1,
`scripts/verify-release-readiness.py` continues to check that
historical figure against the `v0.3.0-rc2` tag specifically, unchanged)
— Phase 26 reclassified exactly 3 rows (MC-17/MC-18/MC-19,
`UNSUPPORTED` → `SUPPORTED`) with new, real evidence. No row was
deleted, and no row was reclassified without direct evidence.

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

## Validation

Committed, `scripts/verify-release-rc3.py`-checked evidence:
`results/release-v0.3.0-rc3/readiness.json` — see that file for exact
`membrane_commit`, CI run IDs, and per-suite pass counts. Summary:

- `scripts/verify-results.py`: **73/73** core, chaining compatibility
  (13/13), planner-accuracy (6/6), auto-fallback (9/9), release
  readiness (10/10), RC2 user-validation (9/9), performance-profiling
  (9/9), performance-optimization (8/8), and compat-expansion (7/7).
- Full `ctest --output-on-failure`, lightweight build: **67/67**.
  Llama-enabled ASan-targeted tests
  (`test_compat_check`/`test_joint_planner`/`test_auto_fallback`/
  `test_product_cli`): all passing under ASan/UBSan.
- CPU + Vulkan install/uninstall re-verified outside the build tree
  (fresh `--version`/`--help`, clean `install_manifest.txt` removal).
- A small, real, non-benchmark Qwen2/llama regression matrix (CPU
  q8/q5, Vulkan q8/adaptive for Qwen2.5-1.5B; native/q8 or adaptive for
  SmolLM2-135M) — no context/generation-length stress, no deliberate
  OOM.
- GitHub CI and CodeQL: all SUCCESS — exact run IDs in
  `results/release-v0.3.0-rc3/readiness.json`.

## Known limitations

Carried forward from RC2, still accurate, plus this cycle's own:

- Vulkan validated on a small number of real local hosts/devices
  (NVIDIA GTX 1650 Mobile class, AMD Radeon RADV RENOIR integrated) —
  not a general claim about other GPUs, vendors, or hosts.
- CUDA is not a MEMBRANE product path — evaluated in Phase 26
  (`docs/compat-expansion.md`), deferred: no validation hardware
  available, not assumed to be a bad fit.
- AMD/Intel GPU product validation is real but narrow (one AMD RADV
  integrated device, one physical host) — not a blanket claim.
- Qwen2 support is scoped to `Qwen2.5-1.5B-Instruct` specifically, not
  "all Qwen2 models" (see "Qwen2 compressed KV" above).
- VRAM/estimate accuracy depends on model, context size, and
  hardware — never an absolute OOM guarantee.
- No custom GPU kernels — this product only wires up explicit,
  product-level control over the existing, unmodified `ggml`/llama.cpp
  Vulkan backend.
- No FPGA/CXL hardware claim (companion `kadireren7/membrane-research`
  repository only, simulation/synthesis-tool-proxy evidence).

## External validation status

**Still incomplete — this is the primary reason v0.2.0 remains stable
and RC3 is not RC-final.** Phase 23's real CPU + NVIDIA-Vulkan +
AMD-Vulkan evidence (`results/rc2-user-validation/summary.json`) all
comes from **one** physical developer host — three real backends, one
real machine. Independent-host validation (a genuinely different
physical environment, ideally a different distro and a different GPU
vendor/driver stack) has not yet happened. See "Phase 28 handoff" in
the release-prep PR description for the concrete next step.

## Upgrade notes

No breaking CLI or JSON-schema change since RC2. New CLI-visible
behavior: none (Qwen2 support and the new timing fields both use
existing flags/output shape). `schema_version` has stayed `1`
throughout — every JSON field added since RC2 (`timings`,
`planner_stages`, the error-path `fallback` trace) is additive.

## Validation status

This is a release-CANDIDATE preparation document only — `v0.3.0-rc3` is
not yet a tag or a GitHub release at the time this document is written;
both are created only after this PR merges and its own post-merge gate
passes (see the PR for exact status). Pre-release, not the stable
release line either way — `v0.2.0` remains the latest stable tag, and
this document does not change that.
