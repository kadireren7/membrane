# MEMBRANE

[![CI](https://github.com/kadireren7/membrane/actions/workflows/ci.yml/badge.svg)](https://github.com/kadireren7/membrane/actions/workflows/ci.yml)
[![Paper Build](https://github.com/kadireren7/membrane/actions/workflows/paper.yml/badge.svg)](https://github.com/kadireren7/membrane/actions/workflows/paper.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A research prototype for LLM KV-cache memory: a mixed-precision KV-cache
runtime, exact sparse retrieval, near-memory/CXL simulation, bit-exact CPU
quantization, and a synthesizable FPGA datapath — all measured against real,
committed artifacts in this repository, not aspirational claims.

> **Status: public research release, simulation-heavy.** Every number in
> this README is sourced to a specific CSV/JSONL artifact and
> documentation section (section "Key results"). The software/simulation
> work, academic manuscript, and reproducibility tooling are complete and
> verified (see [docs/research-release-freeze.md](docs/research-release-freeze.md)).
> This is not a production system, and no claim here should be read as
> "real CXL hardware acceleration" — see
> [Technical limitations](#technical-limitations) for exactly what is and
> isn't real, and what physical FPGA/CXL validation would still add.

MEMBRANE is:

- a **mixed-precision KV-cache runtime** (FP16/Q8/Q4 tiering with
  promotion/eviction policies, calibrated against real llama.cpp inference),
- an **exact sparse retrieval** engine (predictor + prefetch + compulsory
  miss fetch, with zero approximation in the retrieval result itself),
- a **near-memory/CXL appliance simulator** (discrete-event, calibrated from
  real captured attention traces, no physical CXL hardware),
- a **bit-exact CPU quantization layer** (verified byte-for-byte against
  ggml's own Q8_0/Q4_0 reference implementation), and
- a **synthesizable FPGA datapath** (fixed-point Q8/Q4 encode/decode,
  cosimulated against the same CPU reference in Verilator, not yet placed
  or routed on a physical device).

---

## Table of contents

1. [What is MEMBRANE?](#what-is-membrane)
2. [The problem it addresses](#the-problem-it-addresses)
3. [Core idea](#core-idea)
4. [Key results](#key-results)
5. [Architecture](#architecture)
6. [Quick demo](#quick-demo)
7. [Reproduction](#reproduction)
8. [Repository layout](#repository-layout)
9. [Technical limitations](#technical-limitations)
10. [Roadmap](#roadmap)
11. [Open development](#open-development)
12. [Citation](#citation)
13. [License](#license)

## What is MEMBRANE?

MEMBRANE is a research codebase exploring one question: **for long-context,
high-concurrency LLM inference, what does it take to keep the KV-cache's
memory footprint bounded without silently degrading generation quality?**

It is a from-scratch C11/C++17 implementation — not a fork or patch of an
existing inference engine — covering four largely independent layers that
share one real data source (captured attention traces from actual
llama.cpp inference on two open SmolLM2 checkpoints):

1. A quantized KV-cache runtime with a hot/warm/cold tiering policy.
2. An exact (non-approximate) sparse KV retrieval engine with predictor +
   prefetch, discrete-event simulated at up to 128K context and 512
   concurrent sequences.
3. A near-memory/CXL appliance model and a synthesizable FPGA quant/dequant
   datapath, both built to ask "if this were built in hardware, where would
   it actually bind?"
4. An out-of-core simulation backend that makes (2) and (3)'s largest
   sweeps runnable on a memory-constrained (5.6 GiB RAM) development
   machine.

## The problem it addresses

LLM inference servers are usually compute-rich but memory-poor: KV-cache
size grows linearly with context length and concurrent request count, and
GPU VRAM/HBM capacity — not FLOPs — caps how much of either a given piece
of hardware can serve. Existing engines already offer some fixed answers
(static quantized KV types, static CPU offload). MEMBRANE's research
question is whether a **decision engine** — moving, quantizing, or exactly
retrieving each block of KV memory based on real access patterns instead of
a static, user-set knob — can do meaningfully better, and where it
genuinely cannot (the negative results in section "Key results" and
[docs/results-summary.md](docs/results-summary.md) are as load-bearing as
the positive ones).

## Core idea

Not every KV block needs to be resident, and not every resident block
needs full precision. MEMBRANE separates two independent levers:

- **Precision**: FP16 (hot) → Q8 (warm) → Q4 (cold), each verified
  bit-exact against ggml's own reference quantizer, not an approximation.
- **Residency**: exact sparse retrieval — a real predictor decides what to
  prefetch, but every genuinely-needed block is exactly fetched on a
  compulsory miss, so retrieval quality is never approximated away (unlike
  naive eviction, which the results summary shows breaks recall-shaped
  tasks).

Both levers are simulated end-to-end (discrete-event, calibrated from real
traces) before anything is committed to RTL, and the RTL itself is
cosimulated against the same CPU reference math it's meant to accelerate.

## Key results

Every row below names its source artifact, the documentation section with
full methodology, and the command to reproduce it. **135M and 360M are
SmolLM2-135M-Instruct and SmolLM2-360M-Instruct respectively — the two are
never averaged or substituted for each other below.**

| Finding | Source artifact | Doc section | Reproduce |
|---|---|---|---|
| Unified 128K-context × 512-concurrency sweep: **462/462 scenarios complete** (231 SmolLM2-135M + 231 SmolLM2-360M), zero extrapolated rows | `benchmarks/cxl-sim/unified-sweep.csv` (+ `.ckpt`) | [phase6-unified-stress.md](docs/phase6-unified-stress.md) §1, §10 | `docs/reproduction.md` Level 3 |
| KV traffic vs. full-scan-CXL: **187x–405x** reduction at the representative 8GiB-host/2TiB-device point (both models, unified 128K×512 sweep; vs. the precision-matched compressed-full-scan-CXL baseline instead, the range is 99.6x–215.3x) | `benchmarks/cxl-sim/unified-sweep.csv` | [phase6-unified-stress.md](docs/phase6-unified-stress.md) §12 | Level 3 |
| KV traffic vs. full-scan-CXL: **985.7x–1,281.1x** reduction (SmolLM2-135M, 4K-context capacity-bound scenario — a different, smaller-scale sweep, not to be conflated with the row above) | `benchmarks/cxl-sim/exact-sweep.csv` | [phase6-exact-sparse-retrieval.md](docs/phase6-exact-sparse-retrieval.md) §12 | Level 2 |
| Exact retrieval preserves full KV history: oracle precision/recall = **1.000/1.000** by construction; the real predictor reaches **0.90–0.98 recall** without ever approximating a compulsory-fetch result | `benchmarks/cxl-sim/unified-sweep.csv` | [phase6-unified-stress.md](docs/phase6-unified-stress.md) §7, §12 | Level 3 |
| With sufficient cache, KV retrieval overhead is **fully hidden under the model's own compute floor** on 0.1%–98.3% of steps depending on scenario (`hidden_under_compute_fraction`, real, 124 SmolLM2-360M rows) | `benchmarks/cxl-sim/unified-sweep.csv` | [phase6-unified-stress.md](docs/phase6-unified-stress.md) §12.1 | Level 3 |
| Full-pipeline FPGA datapath cosimulation: **520,000 transactions, 0 mismatches**, Verilator vs. the real C quantizer reference | `rtl/tb/tb_top_verilator.cpp` (run log) | [phase5-synthesizable-fpga.md](docs/phase5-synthesizable-fpga.md) §4 | `scripts/demo.sh` |
| Bit-exact CPU quantization parity: **100,000+ random blocks** + every documented edge case (NaN/Inf/denormal/all-zero/constant), 0 mismatches, vs. ggml's own Q8_0/Q4_0 reference | `tests/unit/test_ggml_quant_parity.c` (run log) | [phase4-ggml-quant-parity.md](docs/phase4-ggml-quant-parity.md) | `scripts/demo.sh` |
| FPGA quant-pipeline count dominates CXL link generation: dropping 8→1 pipelines inflates p99 by **25.7x** (15.67ms → 402.1ms); CXL 3.0 vs. CXL-latency-low/med/high shows **no measurable change** | `benchmarks/cxl-sim/unified-sweep-hardware-sensitivity.csv` | [phase6-unified-stress.md](docs/phase6-unified-stress.md) §9 | Level 3 |

For the numbers this project measured and did **not** find favorable —
blind lossless compression failing on real KV data, PCIe-round-trip FPGA
offload being a net loss at live-decode batch sizes, naive eviction
breaking recall tasks, the 10ms p99 bound never being met, and
micro-batching showing no measurable benefit — see
[docs/results-summary.md](docs/results-summary.md) §4 ("Null/negative
findings"). They are not hidden.

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full diagram set
(end-to-end system, KV lifecycle, FPGA datapath, exact sparse retrieval
path). Summary:

```
LLM runtime (llama.cpp, real inference)
        │  captured KV/attention traces (real)
        ▼
Policy engine  ──promote/evict──  Hot (FP16) / Warm (Q8) / Cold (Q4)
        │                                        │  bit-exact vs. ggml
        │ exact sparse retrieval (predictor + prefetch + compulsory fetch)
        ▼
Discrete-event simulator  ──calibrated from real traces──  CXL/near-memory
        │                                                  appliance model
        ▼
Synthesizable FPGA datapath (Q8/Q4 encode/decode, Verilator-cosimulated)
```

## Quick demo

```bash
./scripts/demo.sh --quick
```

Builds the project, runs the bit-exact quantization parity test, runs the
full 520,000-transaction FPGA Verilator cosimulation, and runs a small,
committed-fixture exact-retrieval scenario — all from small, already-
committed test data. No model download, no multi-hour sweep. Finishes in
minutes and writes `demo-results.json` / `demo-results.md`. See
[scripts/demo.sh](scripts/demo.sh) header and
[docs/reproduction.md](docs/reproduction.md) for exact flags and expected
output.

## Reproduction

Three levels, from a quick unit-test check to the full multi-hour research
sweep — each with working directory, dependencies, expected time, expected
output, and success signal: see
**[docs/reproduction.md](docs/reproduction.md)**.

| Level | What it does | Expected time |
|---|---|---|
| 1 — Quick verification | Unit tests, quant parity, small simulator replay | ~2–5 min |
| 2 — Model-backed verification | Real KV/attention trace capture + quality validation on both SmolLM2 checkpoints | ~30–90 min |
| 3 — Full research reproduction | Full 128K×512 unified sweep, out-of-core backend, checkpoint/resume | multi-hour, GBs of disk |

## Repository layout

```
include/membrane/   Public C headers (block store, codecs, quant, traces)
src/                Core C11 library implementation
tools/              CLI tools (simulators, capture, quantization benches)
rtl/                Synthesizable FPGA RTL + Verilator/Icarus testbenches
tests/unit/         C/C++ unit tests (ctest-registered)
docs/               Phase-by-phase research documentation (source of truth)
benchmarks/         Committed CSV/JSONL results + MANIFEST.json
paper/              Completed, claim-audited manuscript (main.md/main.tex,
                    references.bib, figures/tables regenerated from source
                    CSVs) -- PDF is a GitHub Actions build artifact, not
                    committed to the repo; see paper/scripts/README.md
hardware/           FPGA/CXL physical-validation plan, vendor-integration
                    interface skeletons, results schema (no real hardware
                    used yet -- see docs/phase8-hardware-validation-plan.md)
outreach/           Hardware-access outreach package (unsent templates,
                    research profile, claim gates) -- see
                    outreach/membrane-technical-brief.md; nothing has
                    been sent to anyone
scripts/            demo.sh, verify-results.py, verify-outreach.py,
                    prepare-release.sh, ...
third_party/        llama.cpp (git submodule, upstream license applies)
```

## Technical limitations

- **No real CXL/PCIe/GPU hardware exists in this project.** All CXL link
  latency/bandwidth figures are assumed, published, industry-typical
  ranges, not measurements from a physical device — see
  [docs/architecture.md](docs/architecture.md) and every `phase6-*.md`
  doc's disclosure section.
  This project does **not** claim real CXL acceleration.
- **The FPGA datapath is cosimulated (Verilator), not synthesized to a
  physical device, placed, or routed.** Synthesizability was verified
  under yosys 0.33 at the RTL level; no Fmax, power, or area numbers on
  real silicon exist.
  This project does **not** claim a working FPGA product.
- **The 10ms p99 latency target is not met** by any of the 462 real
  scenarios in the unified sweep — the model's own decode compute floor
  alone exceeds it for both SmolLM2 checkpoints (see
  [docs/phase6-unified-stress.md](docs/phase6-unified-stress.md) §10).
- **Development-machine memory constraints (5.6 GiB RAM) shape several
  design decisions** (the out-of-core simulator backend, single-worker
  execution in earlier phases) — these are disclosed as real constraints
  of this environment, not claims about production hardware requirements.
- **PCIe round-trip cost for FPGA offload is not measured on real
  hardware** — the emulation charges zero transport latency, and the
  composed, disclosed conclusion is that real PCIe overhead would make
  small-batch offload a net loss (see
  [docs/phase5-pcie-hardware-loop.md](docs/phase5-pcie-hardware-loop.md)).

## Roadmap

See [docs/architecture.md](docs/architecture.md) for the full phased
history (Phase 0 through 7.3) and
[docs/research-release-freeze.md](docs/research-release-freeze.md) for
this repository's current release status. Completed since the initial
research release: the academic manuscript is finished and
claim-audited (14 independently-verified citations, no
`[citation needed]` markers remaining — see
[paper/claim-audit.md](paper/claim-audit.md)), and `paper/main.pdf` now
builds successfully as a real GitHub Actions artifact on every push
(see the Paper Build badge above and
[docs/phase7-academic-paper.md](docs/phase7-academic-paper.md)). No new
simulation/experimental work is planned in the immediate term — the
next substantive evidence this project needs is **physical validation
on real FPGA and CXL hardware**, which this repository is not yet able
to do itself. A scoped, claim-limited outreach package for exactly that
purpose exists in [outreach/](outreach/membrane-technical-brief.md) and
[docs/phase8-hardware-validation-plan.md](docs/phase8-hardware-validation-plan.md)
(unsent — see [docs/public-release-audit.md](docs/public-release-audit.md)
for this repository's current release readiness).

One exception, in progress: `experiment/fp-divider-pipeline`
(EXP-FPGA-DIV-001) characterized the FPGA datapath's general-purpose
divider and evaluated four alternatives; branch `feature/q4-radix4-divider`
now proposes bringing the strongest result (an exact radix-4 iterative Q4_0
divider, -96.2% simulated ECP5 cells vs. baseline at the `q4_scale`
integration point, 4.4M+ differential cases and 520,000/520,000
full-datapath transactions reproduced with 0 mismatches) into `main` as a
candidate `v0.2.0-research` change — see `experiments/EXP-FPGA-DIV-001/
promotion-plan.md` and `promotion-comparison.md` on branch
`experiment/fp-divider-pipeline` (not present on this branch; kept there
per this project's own audit-docs-stay-on-the-experiment-branch
convention). Still SIMULATED/SYNTHESIZED only, no real FPGA hardware.
Pull request [#2](https://github.com/kadireren7/membrane/pull/2) is open
against `main`; the change is not merged and is not part of
`v0.1.0-research`.

## Open development

MEMBRANE remains **fully open source** — there is no private companion
repository, and none is planned. Stable, citable snapshots are tagged
releases (`v0.1.0-research` today); active work happens in public
`experiment/*`, `feature/*`, `fix/*`, and `docs/*` branches of this same
repository. Branches that haven't merged to `main` may contain
incomplete, partial, or invalid results — they are research-in-progress,
not verified claims, until reviewed and merged. See
[docs/open-development-policy.md](docs/open-development-policy.md) and
[docs/repository-boundary.md](docs/repository-boundary.md) for the full
policy, and [docs/research-release-freeze.md](docs/research-release-freeze.md)
for how this coexists with `v0.1.0-research` staying immutable.

## Citation

See [CITATION.cff](CITATION.cff).

## License

Apache License 2.0 for MEMBRANE's own code — see [LICENSE](LICENSE).
The `third_party/llama.cpp` submodule, captured benchmark traces, and any
model artifacts are under their own separate terms — see
[docs/licensing.md](docs/licensing.md) for the full boundary and what
remains unverified there.

---

Contributing: [CONTRIBUTING.md](CONTRIBUTING.md). Security: [SECURITY.md](SECURITY.md).
Support: [SUPPORT.md](SUPPORT.md). Community standards: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
