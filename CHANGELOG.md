# Changelog

All notable changes to MEMBRANE, grouped by research phase. Dates are
commit dates; commit hashes are short and stable. This project does not
yet follow semantic versioning strictly (pre-1.0, research prototype) --
version numbers below track phase numbers instead.

## [Unreleased, candidate v0.2.0-research] — Q4_0 exact radix-4 divider (branch `feature/q4-radix4-divider`, not merged)

- **Not part of `v0.1.0-research`; not merged into `main`.** Pull request
  [#2](https://github.com/kadireren7/membrane/pull/2) is open against
  `main` (head `feature/q4-radix4-divider`); the change is not merged.
  Listed here for visibility only, per this project's open-development
  policy (`docs/open-development-policy.md`) -- see
  `experiments/EXP-FPGA-DIV-001/promotion-plan.md` and
  `promotion-comparison.md` on branch `experiment/fp-divider-pipeline`
  (not present on this branch) for the authoritative detail.
- Origin: `experiment/fp-divider-pipeline` (EXP-FPGA-DIV-001) characterized
  the FPGA datapath's general-purpose `membrane_fp_divider` and evaluated
  four alternatives (B1-B4); B3's completion-reorder-buffer scheduling was
  rejected (real correctness, disproportionate area cost); B2's radix-2
  iterative divider was superseded by B4's radix-4 one.
- Proposed change (this branch only): `q4_scale.sv`'s two divider call
  sites move from `membrane_fp_divider` to a new
  `membrane_fp_scale_neg_pow2.sv` (exact power-of-two shortcut, constant
  divisor) and a new `membrane_fp_divider_radix4.sv` (exact, multi-cycle,
  two-quotient-bit-per-cycle iterative divider, variable divisor), plus
  the minimal Q4_0-encode issue-serialization change that divider's
  variable latency requires in `membrane_quant_stream_top.sv`. `q8_scale.sv`
  and `membrane_quant_stream_top`'s external port list are unchanged.
- Verification reproduced fresh on the clean branch (not copied from the
  experiment): 204,128/204,128 (quick) and separately the full
  2,204,128/2,204,128 B1 differential cases, 0 mismatches;
  4,456,685/4,456,685 B4 differential cases (vs. the real
  `membrane_fp_divider`), 0 mismatches; 520,000/520,000 production
  full-datapath Verilator transactions, 0 fails; yosys generic + ECP5
  synthesis at both the standalone-divider and `q4_scale`-integration
  granularity.
- SIMULATED/SYNTHESIZED only -- no real FPGA hardware, no place-and-route,
  no Fmax claim, same disclosure convention as every other FPGA-related
  entry in this changelog.

## [0.1.0-research-candidate] — Public release audit and freeze

- Full repository consistency audit (README, CHANGELOG, CITATION.cff,
  `docs/`, `paper/`, `outreach/`, `hardware/`, `scripts/`, workflows,
  release-metadata files). Found and fixed: stale "paper skeleton"/
  "citation-needed" wording in README.md and `docs/results-summary.md`
  (both predated Phase 7.2's completed literature research); a stale
  "Phase 0 through 7.1" range in README.md and `docs/architecture.md`
  (three phases behind); `docs/phase7-hardware-outreach.md`'s
  now-outdated "Paper Build workflow has not been run" claim (corrected
  via an added note, not a silent rewrite); `paper/main.md`'s stale
  "draft manuscript" status line; `CITATION.cff`'s `version: "0.7.1"`
  with a `date-released` implying a release that never happened
  (changed to `0.1.0-research-candidate` + `commit: 58ec90b`); a
  missing `.gitignore` entry for `paper/main.pdf` and LaTeX build
  byproducts.
- Added `docs/repository-boundary.md` (public/private repository
  boundary plan — no private repository exists), `docs/research-
  release-freeze.md` (freeze baseline, scope, and unfreeze conditions),
  `docs/v0.1.0-research-release-plan.md` (proposed tag plan, no tag
  created), and `docs/public-release-audit.md` (this audit's full
  report).
- `scripts/verify-outreach.py` extended with 5 new checks (12/12 →
  17/17): no stale "paper skeleton" phrasing, no stale phase range, no
  broken workflow badge, no false private-repo-exists claim, no false
  main.pdf-tracked claim. Building these caught 3 real bugs in the
  checks themselves (see `docs/public-release-audit.md`), fixed before
  being trusted.
- Repository cleanliness re-confirmed: no committed build output,
  stale checkpoint, model file, or oversized file found (unchanged
  since the Phase 7.1 cleanliness pass).
- No tag or GitHub Release was created — per this audit's explicit
  scope, tagging is a separate, later, human-approved step.

## [0.7.3.2] — GitHub Actions verification repair (`58ec90b`)

- Fixed `test_interrupted_resume`'s hardcoded `sleep 45` (tuned to one
  dev machine's speed) that produced 0 completed checkpoint records on
  GitHub's hosted CI runner — replaced with adaptive polling for a real
  completion signal (up to 240s/60s), confirmed non-sanitizer-specific
  since it also failed under a plain Debug build. `CMakeLists.txt`'s
  test timeout raised 400s → 700s accordingly.
- Fixed `paper/build.sh`'s undefined-citation check, which scanned the
  cumulative log across all `pdflatex` passes and misreported pass 1's
  expected, transient "undefined" warnings as a real failure even after
  pass 2 resolved everything — now checks only the final pass's
  isolated output (3 passes run).
- Verified via real `gh run` logs for the failing commit (not guessed),
  and via a mock `pdflatex` reproducing both the real bug and a
  genuinely broken bibliography (no local LaTeX toolchain exists to
  test against directly).
- **Result, confirmed on the real GitHub Actions runner**: `CI` (Debug,
  ASan+UBSan, TSan) and `Paper Build` both pass; `paper/main.pdf` is now
  produced as a real CI build artifact.

## [0.7.3.1] — Authorship and AI-assistance wording correction (`bb4df95`)

- Removed "sole author"/"every line of code"/"written from scratch"/
  "independent researcher" phrasing (`outreach/`, `paper/main.md`,
  `paper/main.tex`) that could be read as claiming Kadir hand-wrote the
  codebase without AI assistance. Repositioned everywhere as: Kadir
  created and leads MEMBRANE, directing architecture, experimental
  design, and technical decisions, and reviewing every commit, while AI
  coding agents assisted with implementation and documentation. Added
  `outreach/ai-assistance-disclosure.md`.
- Corrected "production quantizer" → "ggml's own reference
  implementation"; corrected CXL-value sourcing wording (industry-typical
  assumptions informed by, not literally quoted from, the CXL
  Consortium's spec); corrected `scripts/demo.sh`'s documented duration
  from a cherry-picked "~25s" to a re-measured "~25–50s" range.
- Shortened `outreach/email-templates.md`'s four templates to ~120–180
  words each; added `outreach/primary-email-template.md`.
- `scripts/verify-outreach.py` gained 3 checks (12/12 passing) with 3
  negative tests confirmed caught then reverted.

## [0.7.3] — Phase 7.3: hardware validation outreach package (`6faa0d5`)

- Added `outreach/`: technical brief, one-page summary, four email
  templates, Kadir's research profile, demo video script, research-talk
  outline, target-selection framework, empty contact tracker, and a
  6-file lab evaluation package (`outreach/lab-package/`) — unsent
  templates; no contact made with any real person or institution.
- Added `docs/phase8-hardware-validation-plan.md` (3 levels: FPGA
  sim/impl, real board, CXL platform) and `hardware/` (board-targets,
  15-step experiment protocol, JSON results schema + a clearly-labeled
  fictional example fixture, risk register).
- Added `hardware/vendor-wrapper/`: interface-only AXI4-Stream/
  AXI4-Lite/DMA skeletons wrapping the existing, cosimulation-verified
  `membrane_quant_stream_top` — no vendor IP vendored. Elaborates
  cleanly under yosys together with the full existing `rtl/` hierarchy.
- Added `outreach/hardware-claim-gates.md` (8 gates mapping which test
  must pass to which sentence becomes allowed) and
  `scripts/verify-outreach.py` (9/9 checks at the time, later extended
  to 12/12 in `bb4df95`).
- Added `.github/workflows/paper.yml` (CI path to a real paper PDF) and
  `docs/release-candidate-checklist.md`.
- No new experiments; no physical hardware used.

## [0.7.2] — Phase 7.2: academic manuscript (`e74d719`)

- Literature research: 14 independently-verified primary sources
  (arXiv/ACM, each confirmed against its own abstract page) across all
  15 requested topic areas — `paper/related-work-matrix.md`,
  `paper/references.bib`. Related Work's `[citation needed]` markers
  from Phase 7.1 fully resolved; none remain in the manuscript body.
- Completed `paper/main.md` (abstract, 5 scoped contributions, unified
  system-design narrative, RQ1–RQ7 evaluation, visible negative-results
  section, limitations, ethics/artifact disclosure, reproducibility
  appendix) and produced `paper/main.tex` (single-file compilable
  LaTeX, generic `article` class, inline bibliography).
- Added `paper/claim-audit.md` (per-claim wording/type/source/
  limitation/allowed-vs-prohibited wording for every headline number).
- Added `paper/scripts/`: dependency-free figure generator (7 SVGs from
  real CSVs — no matplotlib available in this environment) and table
  generator (6 tables), plus `paper/scripts/verify-paper.py` (11
  checks).
- Added `paper/build.sh` and `paper/submission-options.md` (honest
  per-venue readiness analysis, no acceptance claims).
- A real inconsistency was found and fixed during the claim audit: the
  FPGA pipeline-count sensitivity result is SmolLM2-135M-only by design
  and is now stated as such everywhere it's cited.

## [0.7.1] — Phase 7.1: research release packaging (`ad6cf5f`)

- Rewrote `README.md`, added `docs/architecture.md` diagrams,
  `docs/reproduction.md`, `docs/results-summary.md`,
  `docs/licensing.md`, `docs/phase7-research-release.md`.
- Added `scripts/demo.sh`, `scripts/verify-results.py`,
  `scripts/generate-benchmark-manifest.py`, `scripts/prepare-release.sh`.
- Added `benchmarks/MANIFEST.json` (SHA-256-tracked artifact manifest).
- Added `paper/` academic-paper skeleton (citation-needed, no fabricated
  Related Work).
- Added `CITATION.cff`, `CHANGELOG.md`, `SECURITY.md`, `SUPPORT.md`,
  `CODE_OF_CONDUCT.md`.
- No new experiments; all numbers presented were measured in prior
  phases and re-verified for consistency (see
  `docs/phase7-research-release.md`).

## [0.6.5] — Out-of-core unified KV simulator (`6f26b6b`)

- Added `.attntrace3` chunked/checksummed out-of-core trace format,
  streaming reader, bounded LRU chunk cache, `/proc`-based memory guard.
- Completed the remaining 124 SmolLM2-360M unified-sweep scenarios and
  recovered both models' 900-sample tail-latency artifacts, closing the
  gap left by Phase 6.4's real OOM-driven stop at 107/231.
- Added compute-normalized latency metrics
  (`model_compute_floor_ns`/`incremental_kv_p99_ns`/
  `hidden_under_compute_fraction`) and a checkpoint/CSV integrity tool
  (`membrane-kv-exact-sim-verify`).
- See `docs/phase6-out-of-core-simulator.md`,
  `docs/phase6-unified-stress.md`.

## [0.6.4] — Unified 128K × 512 concurrency stress validation (`58fb9b5`)

- Combined 128K context and 512 concurrent sequences in a single
  discrete-event scenario for the first time (previously swept on
  separate axes only, Phase 6.3).
- Completed SmolLM2-135M's full 231/231 matrix; SmolLM2-360M reached
  107/231 before a real memory ceiling on the development machine
  (resolved in Phase 6.5, above).
- See `docs/phase6-unified-stress.md`.

## [0.6.3] — Exact sparse KV retrieval at scale (`36d7207`)

- Discrete-event exact-retrieval engine with predictor + prefetch +
  compulsory-miss fetch; 985.7x–1,281.1x bytes/token reduction vs.
  full-scan-CXL at a 4K-context capacity-bound scenario.
- See `docs/phase6-exact-sparse-retrieval.md`.

## [0.6.2] — Attention-aware KV working-set engine (`fd92476`)

- Real attention-trace-calibrated working-set analysis; measured
  4.8x–7.2x working-set reduction, and a real recall failure mode for
  naive approximate eviction policies on recall-shaped prompts.
- See `docs/phase6-attention-working-set.md`.

## [0.6.1] — CXL near-memory KV appliance simulation (`76dad80`)

- Discrete-event simulator for a near-memory/CXL appliance, calibrated
  from real captured traces; micro-batching null result.
- See `docs/phase6-cxl-near-memory.md`.

## [0.5.4] — PCIe FPGA hardware-in-the-loop runtime (`6c53736`)

- Hardware-in-the-loop PCIe emulation; disclosed finding that real PCIe
  round-trip cost would make per-block FPGA offload a net loss at
  live-decode batch sizes.
- See `docs/phase5-pcie-hardware-loop.md`.

## [0.5.3] — Fully synthesizable FPGA quantization datapath (`710a3d7`)

- Bit-exact, purely-integer FP32 arithmetic primitives; full top-level
  streaming datapath elaborates under yosys 0.33; 520,000-transaction
  Verilator cosimulation vs. the real CPU quantizer, 0 mismatches.
- See `docs/phase5-synthesizable-fpga.md`.

## [0.5.2] — Streaming FPGA quantization prototype (`173be5c`)

- First SystemVerilog RTL prototype and cycle-accurate C model.
- See `docs/phase5-hardware-datapath.md`, `docs/phase5-fpga-streaming.md`.

## [0.5.1] — High-throughput KV quantization engine (`4adc8b2`)

- SIMD-backed quantization engine, portable and independent of ggml.
- See `docs/phase5-quant-engine.md`.

## [0.4.4] — Bit-exact ggml KV quantization parity (`1d23bfb`)

- `test_ggml_quant_parity`: 100,000+ random blocks plus every documented
  edge case, verified byte-for-byte against ggml's own Q8_0/Q4_0
  reference implementation.
- See `docs/phase4-ggml-quant-parity.md`.

## [0.4.3] — Runtime KV measurement variance investigation (`a0324be`)

- Root-caused a real decode-shape bug; quantified quantize-function
  differences as the dominant source of offline-vs-runtime prediction
  gap.
- See `docs/phase4-runtime-variance.md`.

## [0.4.2] — Runtime-calibrated KV policy optimizer (`4c005cb`, `a9a1bc8`)

- Real-runtime-gated optimizer with live progress reporting for
  long-running calibration sweeps.
- See `docs/phase4-runtime-calibration.md`.

## [0.4.1] — Runtime mixed-precision KV policy engine (`667dd8c`)

- First runtime (not offline-only) mixed-precision KV policy engine.
- See `docs/phase4-runtime-policy.md`.

## [0.3.x] — Cross-model, risk-aware, and composition-aware KV policy
  (`ae6d1c9`, `f9b3332`, `22a49cd`, `df34ca3`, `b337d2a`, `7ac7209`)

- Validated policy behavior across multiple models; added risk-aware and
  composition-aware precision optimization; live Q8 KV-cache quality
  validation against real llama.cpp inference.
- See `docs/phase3-*.md`.

## [0.2.x] — Real KV-cache capture and compressibility analysis
  (`ec46214`, `45bcd62`, `c6c3635`, `72779ab`)

- First real KV-tensor capture from actual llama.cpp inference.
- **Foundational negative result**: blind byte-level lossless
  compression (RLE) achieves exactly 1.000x on real KV data (every
  block falls back to RAW) -- real KV data is near-maximum-entropy at
  the byte level. This redirected the project toward quantization.
- See `docs/phase2-kv-analysis.md`, `docs/phase2-kv-byteplane.md`,
  `docs/phase2-kv-entropy.md`, `docs/phase2-kv-predictive.md`.

## [0.1.x] — Budget-aware block store with persistent backend
  (`a3b8219`, `ecae200`, `51d86a2`, `0bfa834`, `2908960`)

- LRU-based, budget-enforced block store; file-backed cold tier with
  atomic writes and promotion/eviction under a real thread-safety
  stress test.

## [0.0.1] — Initial lossless block store (`c4e5fce` through `b5a04e2`)

- CMake skeleton, RAW/RLE codec API, `membrane-bench` CLI, CI (GitHub
  Actions), 42-Norm-style formatting pass.
