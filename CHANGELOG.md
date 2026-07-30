# Changelog

All notable changes to MEMBRANE, grouped by research phase. Dates are
commit dates; commit hashes are short and stable. This project does not
yet follow semantic versioning strictly (pre-1.0, research prototype) --
version numbers below track phase numbers instead.

## [0.7.1] — Phase 7.1: research release packaging

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
