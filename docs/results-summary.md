# MEMBRANE Results Summary

A dense technical summary of what this project measured, what it did not,
and what it explicitly failed to show. Written for a reader who wants the
substance without reading all fifteen `phase*.md` documents. Every number
here is sourced; see the linked doc section for full methodology.

## 1. Problem

LLM inference servers are usually compute-rich but memory-poor: KV-cache
size grows linearly with context length and concurrent request count, and
GPU VRAM/HBM capacity — not FLOPs — caps how much of either a given piece
of hardware can serve. The research question: can a per-block *decision
engine* (precision tier + residency) do meaningfully better than static
configuration, and where does it not? This project treats "where it does
not" as an equally valid, equally reportable outcome.

## 2. Methodology

Four layers, one shared real data source:

- **Real capture.** `membrane-kv-capture` and `membrane-kv-attn-trace-
  capture` pull actual KV tensors and attention weights out of real
  llama.cpp inference on two open checkpoints, SmolLM2-135M-Instruct and
  SmolLM2-360M-Instruct (`.kvtrace`/`.attntrace` files, committed).
- **Calibrate-once, replay-many discrete-event simulation.** Every
  latency/capacity/queue number at 128K context or 512 concurrency comes
  from a discrete-event simulator (`membrane-cxl-sim`,
  `membrane-kv-exact-sim`) calibrated from a real trace, then replayed at
  scale with real shared-resource contention — not measured on physical
  hardware, and never claimed to be.
- **Bit-exact CPU quantization**, verified against ggml's own Q8_0/Q4_0
  reference implementation, not an approximation of it
  (`test_ggml_quant_parity`, 100,000+ random blocks + every edge case).
- **RTL cosimulation**, not synthesis-to-silicon: the FPGA datapath is
  verified against the same CPU reference math via Verilator
  (`tb_top_verilator.cpp`, 520,000 transactions), and separately checked
  synthesizable under yosys 0.33 — those are two different claims, both
  disclosed as such.
- **Out-of-core execution.** The unified 128K×512 sweep's largest runs
  exceed this development machine's 5.6 GiB RAM if held naively in memory;
  a chunked/streaming trace backend and a real `/proc`-based memory guard
  (Phase 6.5) made the full sweep completable without raising the RAM
  ceiling — a real engineering constraint, not a simulated one.

Labeling discipline used throughout every `phase*.md` document and carried
into this summary: **REAL** (an actual measurement), **SIMULATED** (this
project's own discrete-event engines, actually run), **ORACLE** (fed
ground truth directly — an upper bound, not a claim a real predictor could
know the future), **ASSUMED** (a cited estimate — no real CXL/FPGA
hardware exists in this project).

## 3. Key findings

- **Unified 128K-context × 512-concurrency sweep: 462/462 scenarios
  complete** (231 SmolLM2-135M + 231 SmolLM2-360M), zero rows
  extrapolated from the other model or a smaller scale
  (`benchmarks/cxl-sim/unified-sweep.csv`,
  [phase6-unified-stress.md](phase6-unified-stress.md)).
- **KV traffic reduction vs. full-scan-CXL: 187x–405x** at the
  representative 8GiB-host/2TiB-device unified-sweep point (both models;
  99.6x–215.3x vs. the precision-matched compressed-full-scan-CXL
  baseline instead), and **985.7x–1,281.1x** at a separate, smaller
  4K-context capacity-bound scenario
  ([phase6-exact-sparse-retrieval.md](phase6-exact-sparse-retrieval.md)
  §12) — these are two different sweeps and are not interchangeable.
- **Exact retrieval never approximates a compulsory fetch.** Oracle
  precision/recall = 1.000/1.000 by construction; the real predictor
  reaches 0.90–0.98 recall. The retrieval *result* is always exact — only
  the *prefetch decision* is predicted.
- **Retrieval overhead is fully hidden under the model's own compute
  floor** on a real, scenario-dependent fraction of steps
  (`hidden_under_compute_fraction`, 0.001–0.983 across 124 real
  SmolLM2-360M rows) — generous-cache scenarios hide it almost
  completely, tight-cache scenarios expose it on nearly every step.
- **FPGA datapath: 520,000-transaction Verilator cosimulation, 0
  mismatches**, against the real CPU quantizer reference
  ([phase5-synthesizable-fpga.md](phase5-synthesizable-fpga.md) §4).
- **CPU quantization: 100,000+ random blocks, 0 mismatches** vs. ggml's
  own Q8_0/Q4_0 reference, plus every disclosed edge case (NaN/Inf/
  denormal/all-zero/constant)
  ([phase4-ggml-quant-parity.md](phase4-ggml-quant-parity.md)).
- **Quant-pipeline count dominates CXL link generation as a hardware
  sensitivity**: dropping 8→1 pipelines inflates p99 by 25.7x (15.67ms →
  402.1ms); CXL-3.0-class vs. CXL-latency-low/med/high bandwidth shows no
  measurable change at this scenario point
  ([phase6-unified-stress.md](phase6-unified-stress.md) §9).

## 4. Null / negative findings

Kept deliberately visible — these are as scientifically load-bearing as
section 3.

- **Blind lossless compression fails on real KV-cache data.** Byte-level
  RLE on real F16 KV tensors from actual inference achieves exactly
  1.000x (every block falls back to RAW storage); pure RLE without a RAW
  fallback would have *expanded* the data to 0.502x. Real KV data is
  near-maximum-entropy at the byte level (7.3–7.4 bits/byte), independent
  of prompt content, block size, or K-vs-V
  ([phase2-kv-analysis.md](phase2-kv-analysis.md)). This result is what
  redirected the whole project toward quantization instead of generic
  compression.
- **PCIe round-trip FPGA offload is a net loss at live-decode batch
  sizes.** The hardware-in-the-loop emulation charges zero real transport
  latency (~210ns measured in-emulation); a real PCIe doorbell-ring/DMA/
  completion-interrupt round trip is routinely low-single-digit
  microseconds even with an optimized driver. Real per-block CPU
  quantize/dequantize cost is only 20–180ns. A real PCIe round trip of
  even 1–2 microseconds per call would make per-block or small-batch FPGA
  offload a net loss versus keeping quantization on CPU, unless batched
  far more aggressively than live autoregressive decoding naturally
  allows (a handful of KV blocks per token, per step)
  ([phase5-pcie-hardware-loop.md](phase5-pcie-hardware-loop.md) §9–10).
  This is a composed, disclosed, unverified-on-real-hardware conclusion —
  stated as such, not as a measurement.
- **Naive approximate KV eviction breaks recall-shaped tasks.** Two
  approximate eviction policies (`sliding-window+sink`, a recency-biased
  predictive policy) were tested against 5 real prompt categories
  (recall, distractor, code, natural, longcontext). Both showed genuine,
  measured recall failures — as low as 0.04 top-1 match rate on
  recall-shaped prompts — while the same policies performed acceptably on
  natural/code prompts. This is not a tuning failure; it is a real
  demonstration that approximate eviction has a task-shape-dependent
  failure mode, motivating this project's exact-retrieval design instead
  of approximate eviction ([phase6-attention-working-set.md]
  (phase6-attention-working-set.md) §7).
- **The 10ms p99 bound is not met by any real scenario, for either
  model.** Across the full 462-scenario unified sweep, 0 of 225 real
  (non-analytical) rows meet a 10ms p99 target for SmolLM2-135M, and 0 of
  225 for SmolLM2-360M. Root cause: each model's own real decode compute
  floor (15.67ms/token for 135M, ~41.0ms/token for 360M, both measured)
  already exceeds the bound before any KV-retrieval latency is added —
  this is a model-speed ceiling, not a retrieval-quality failure
  ([phase6-unified-stress.md](phase6-unified-stress.md) §10).
- **Micro-batching shows no measurable benefit at calibrated real demand
  levels.** Phase 6.1's own micro-batching study, deliberately constructed
  to be overloaded enough to need it, found no measurable throughput or
  latency change from adding micro-batching + coalescing at this
  project's actually-calibrated attention-derived demand levels
  ([phase6-cxl-near-memory.md](phase6-cxl-near-memory.md) §8,
  [phase6-exact-sparse-retrieval.md](phase6-exact-sparse-retrieval.md)
  §8). This is a genuine null result, not an unimplemented feature — the
  mechanism was built and run.

## 5. Limitations

- No real CXL, PCIe FPGA, or GPU hardware exists anywhere in this
  project. Every CXL link latency/bandwidth figure is an assumed,
  published, industry-typical range.
- The FPGA datapath is cosimulated and confirmed synthesizable under
  yosys 0.33 at the RTL level; it has never been placed, routed, or run
  on physical silicon. No Fmax/power/area numbers exist.
- The out-of-core simulator's memory-budget enforcement carries a
  measured ~10% overshoot slack factor on this specific development
  machine's allocator/kernel behavior — not verified to generalize to
  other machines or scales.
- Worker concurrency in the largest sweeps was constrained by this
  machine's 5.6 GiB RAM, not chosen to reflect a target production
  deployment's parallelism.
- Related Work in `paper/main.md` is intentionally left with
  citation-needed markers — a literature survey is explicitly out of
  scope for this phase, to avoid citing sources not actually checked.

## 6. Hardware implications

If this system were built as real hardware, the measured sensitivities
say where engineering effort should go first:

- **Quant/dequant pipeline throughput, not CXL link generation, is the
  dominant lever.** An 8x drop in pipeline count (8→1) costs 25.7x in p99
  latency; moving from an assumed CXL-latency-low profile to a
  CXL-3.0-class profile costs nothing measurable at the same scenario
  point. A real build should over-provision quant engines before
  chasing a newer CXL generation.
- **Per-block PCIe offload is the wrong granularity.** If FPGA offload is
  pursued, batching many blocks per PCIe round trip (not per-block or
  per-small-batch, as live autoregressive decoding naturally produces) is
  a precondition for it to be worth the transport cost at all — this
  project's own emulation shows the CPU-only path winning by default at
  live-decode batch sizes.
- **Exact retrieval, not approximate eviction, is the safer default** for
  any workload that might contain recall-shaped prompts — the measured
  recall failure mode (as low as 0.04 top-1) is real and
  workload-dependent, not a corner case to tune away.

## 7. Next research steps

- Real literature survey for `paper/main.md`'s Related Work (explicitly
  deferred, not attempted with fabricated citations).
- If real CXL or FPGA hardware becomes available, replace the assumed
  link latency/bandwidth figures with measured ones — every place they
  are used is named in section 5 and in each `phase6-*.md` doc's
  disclosure section, so this is a scoped, not exploratory, follow-up.
- Investigate whether a coarser, capacity-triggered eviction policy can
  recover the working-set bytes/token reduction Phase 6.2 measured
  (4.8x–7.2x fewer resident blocks) without the recall failure mode Phase
  6.2 also measured — these two results are currently in tension and
  unresolved.
- Batch-size-aware PCIe offload heuristic (queue several blocks per
  round trip) as a direct answer to the PCIe net-loss finding in section
  4, rather than abandoning FPGA offload outright.
