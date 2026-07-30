# MEMBRANE: A Mixed-Precision, Exact-Retrieval Architecture for LLM KV-Cache Memory

**Author:** Kadir Eren Altıntaş

**Status:** Draft skeleton (Phase 7.1). Sections are populated with real,
sourced findings from this repository; Related Work is intentionally left
with `[citation needed]` markers — a literature survey is explicitly
deferred to a later phase, not attempted here with fabricated or
half-remembered citations. Figures are placeholders; see
`paper/figures/README.md`.

---

## Abstract

LLM inference servers are typically compute-rich but memory-poor: KV-cache
size grows linearly with context length and concurrent request count, and
GPU memory capacity — not FLOPs — caps achievable scale. We present
MEMBRANE, a research prototype exploring whether a per-block memory
*decision engine* (precision tier + residency) can do better than static
configuration for this problem, and where it cannot. MEMBRANE combines a
mixed-precision KV-cache runtime (FP16/Q8/Q4, verified bit-exact against
ggml's reference quantizer), an exact — non-approximate — sparse KV
retrieval engine (predictor + prefetch + compulsory-miss fetch), a
discrete-event near-memory/CXL appliance simulator calibrated from real
captured attention traces, and a synthesizable FPGA quantization datapath
cosimulated against the same CPU reference math. Across a unified
128K-context × 512-concurrency stress sweep (462 real scenarios, two
open SmolLM2 checkpoints), we measure 187x–405x KV-traffic reduction
vs. a full-scan-CXL baseline, retrieval overhead fully hidden under the
model's own compute floor on 0.1%–98.3% of steps depending on cache
provisioning, and a 520,000-transaction bit-exact FPGA cosimulation. We
also report, without suppressing, five null/negative results: blind
lossless compression fails on real KV data; PCIe-round-trip FPGA offload
is a net loss at live-decode batch sizes; naive approximate eviction
breaks recall-shaped tasks; a 10ms p99 latency target is never met,
bounded by the model's own decode speed rather than retrieval quality;
and micro-batching shows no measurable benefit at calibrated real demand
levels.

## 1. Introduction

[citation needed: LLM serving memory-wall framing — vLLM/PagedAttention,
FlexGen, and similar systems papers establish the problem this work
positions itself against; not yet surveyed.]

The research question this project asks is narrower and more falsifiable
than "build a faster KV-cache": *for a given real KV access pattern
(captured from actual inference, not synthesized), does a decision engine
that tiers precision and retrieves exactly rather than approximately
measurably reduce memory traffic without degrading retrieval quality, and
at what latency cost?* Section 8 (Evaluation) and section 9 (Negative
Results) report the answer honestly in both directions.

## 2. Background

[citation needed: KV-cache quantization prior art (e.g. INT8/INT4 KV
caches in production serving systems); CXL/near-memory computing surveys;
FPGA-accelerated quantization prior art. Deferred — see the note at the
top of this document.]

MEMBRANE's own empirical starting point (Phase 2, this repository) is
that byte-level lossless compression does not work on real KV-cache
tensors: measured Shannon entropy of 7.3–7.4 bits/byte, independent of
prompt content, motivating a move to quantization instead of generic
compression (see `docs/phase2-kv-analysis.md` and section 9 below).

## 3. System design

MEMBRANE is a from-scratch C11/C++17 implementation with four layers
sharing one real data source: captured KV/attention traces from actual
llama.cpp inference on SmolLM2-135M-Instruct and SmolLM2-360M-Instruct.
See `docs/architecture.md` for the full diagram set (end-to-end system,
KV lifecycle, FPGA datapath, exact retrieval path) — figures 1–4 in this
paper are drawn from those diagrams (see `paper/figures/README.md`).

## 4. Mixed-precision runtime

FP16 (hot) → Q8 (warm) → Q4 (cold) tiering with promotion/eviction under
a fixed budget (`include/membrane/store.h`,
`include/membrane/backend.h`). Precision transitions are verified
bit-exact against ggml's own Q8_0/Q4_0 reference implementation — not an
approximation of it — across 100,000+ randomized blocks plus every
disclosed edge case (NaN/Inf/denormal/all-zero/constant), 0 mismatches
(`tests/unit/test_ggml_quant_parity.c`; see `docs/phase4-ggml-quant-parity.md`).

## 5. FPGA datapath

A fully synthesizable, purely-integer fixed-point Q8/Q4 encode/decode
pipeline (`rtl/membrane_quant_stream_top.sv` and supporting modules) —
no `real`/`shortreal`/DPI constructs anywhere in the production datapath.
The full hierarchy elaborates under yosys 0.33. Cosimulated in Verilator
against the same CPU reference math (`src/quant/quant_simd.c`) for
520,000 transactions across all four encode/decode modes plus a
mixed-mode interleave, 0 mismatches (`rtl/tb/tb_top_verilator.cpp`; see
`docs/phase5-synthesizable-fpga.md`). This has not been placed or routed
on physical silicon — see section 10 (Limitations).

## 6. Near-memory / CXL architecture

A discrete-event simulator (`tools/membrane-cxl-sim`) modeling a
near-memory/CXL appliance, calibrated once from a real captured trace and
replayed at scale with real simulated queueing/contention
(`docs/phase6-cxl-near-memory.md`). No physical CXL hardware exists in
this project; all link latency/bandwidth figures are assumed, cited,
industry-typical ranges (section 10).

## 7. Exact sparse retrieval

An exact (non-approximate) sparse KV retrieval engine
(`tools/membrane-kv-exact-sim`): a predictor decides what to prefetch,
but every genuinely-needed block is exactly fetched on a compulsory
miss — retrieval quality is never approximated away. Calibrated once per
scenario (`calibrate()`, walking a real trace) and replayed across up to
512 concurrent sequences with real shared-resource contention
(`run_concurrent()`), never re-touching the raw trace. See
`docs/phase6-exact-sparse-retrieval.md` and
`docs/phase6-unified-stress.md`.

## 8. Evaluation

- **Scale**: unified 128K-context × 512-concurrency sweep, 462/462
  scenarios complete across both models, zero extrapolated rows
  (`benchmarks/cxl-sim/unified-sweep.csv`; `docs/phase6-unified-stress.md`).
- **KV traffic reduction**: 187x–405x vs. full-scan-CXL at the
  representative 8GiB-host/2TiB-device point (99.6x–215.3x vs. a
  precision-matched compressed baseline); 985.7x–1,281.1x at a separate,
  smaller 4K-context capacity-bound scenario
  (`docs/phase6-exact-sparse-retrieval.md`).
- **Retrieval quality**: oracle precision/recall = 1.000/1.000 by
  construction; the real predictor reaches 0.90–0.98 recall, never
  approximating a compulsory fetch result.
- **Latency composition**: `hidden_under_compute_fraction` (real, 124
  SmolLM2-360M rows) ranges 0.001–0.983 — KV retrieval overhead is fully
  hidden under the model's own decode compute floor on a
  scenario-dependent fraction of steps.
- **FPGA cosimulation**: 520,000 transactions, 0 mismatches (section 5).
- **Hardware sensitivity**: quant-pipeline count dominates CXL link
  generation — 8→1 pipelines inflates p99 by 25.7x; CXL-3.0-class vs.
  CXL-latency-low/med/high shows no measurable difference at this
  scenario point (`docs/phase6-unified-stress.md` §9).

Full sourcing for every number above: `docs/results-summary.md` and
`benchmarks/MANIFEST.json`.

## 9. Negative results

Reported without suppression, per this project's disclosure discipline
(see `docs/results-summary.md` §4 for full detail):

1. Blind lossless byte-level compression fails on real KV-cache data
   (1.000x, RAW fallback; would be 0.502x without it).
2. PCIe-round-trip FPGA offload is a net loss versus CPU-only
   quantization at live-decode (small-batch) sizes — a composed,
   disclosed, not-on-real-hardware conclusion.
3. Naive approximate KV eviction breaks recall-shaped tasks (top-1 match
   rate as low as 0.04 on recall-shaped prompts).
4. The 10ms p99 latency target is met by 0 of 225 real rows for either
   model — bounded by each model's own decode compute floor, not by
   retrieval quality.
5. Micro-batching shows no measurable throughput/latency benefit at this
   project's calibrated, real attention-derived demand levels.

## 10. Limitations

- No real CXL, PCIe FPGA, or GPU hardware exists in this project — every
  CXL figure is an assumed, cited estimate.
- The FPGA datapath is cosimulated and confirmed synthesizable under
  yosys 0.33 at the RTL level only; never placed, routed, or run on
  silicon.
- Development-machine memory constraints (5.6 GiB RAM) shaped several
  design decisions (the out-of-core simulator backend, worker
  concurrency in earlier phases).
- The out-of-core memory-budget enforcement carries a measured ~10%
  overshoot slack factor specific to this development machine.

## 11. Related work

[citation needed throughout this section — deferred to a dedicated
literature-survey phase, per the note at the top of this document. This
section intentionally contains no citations rather than fabricated or
unverified ones.]

- KV-cache quantization in production serving systems — `[citation needed]`
- Sparse/approximate attention and KV eviction policies — `[citation needed]`
- CXL/near-memory computing for LLM serving — `[citation needed]`
- FPGA-accelerated quantization for ML inference — `[citation needed]`
- Discrete-event simulation methodology for systems research —
  `[citation needed]`

## 12. Conclusion

MEMBRANE demonstrates that a per-block precision + exact-retrieval
decision engine, calibrated from real inference traces, can measure real
KV-traffic reductions at meaningful scale (128K context, 512 concurrency)
without approximating retrieval quality — while also honestly
demonstrating where this approach does not help (PCIe offload economics,
p99 latency bounded by compute, micro-batching). Both halves are
necessary for an accurate picture of where per-block KV memory management
is and isn't worth building.

---

See `paper/references.bib` (currently empty — populated once the
literature survey in section 11 is actually done) and
`paper/figures/README.md` for the figure list.
