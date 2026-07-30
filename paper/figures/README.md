# Paper figures

No rendered figure files exist yet. This directory tracks what each
figure in `paper/main.md` should show and where its source data/diagram
already lives in this repository, so a figure can be generated without
re-deriving anything.

| Figure | Paper section | Source | Status |
|---|---|---|---|
| Fig. 1 — End-to-end MEMBRANE system | §3 System design | `docs/architecture.md` diagram A (Mermaid) | diagram exists, not yet rendered as a standalone figure |
| Fig. 2 — KV lifecycle state diagram | §4 Mixed-precision runtime | `docs/architecture.md` diagram B (Mermaid) | diagram exists, not yet rendered as a standalone figure |
| Fig. 3 — FPGA datapath | §5 FPGA datapath | `docs/architecture.md` diagram C (Mermaid) | diagram exists, not yet rendered as a standalone figure |
| Fig. 4 — Exact sparse retrieval sequence | §7 Exact sparse retrieval | `docs/architecture.md` diagram D (Mermaid) | diagram exists, not yet rendered as a standalone figure |
| Fig. 5 — KV traffic reduction vs. full-scan-CXL (bar chart, both models, 5 comparisons) | §8 Evaluation | `benchmarks/cxl-sim/unified-sweep.csv` | data exists, chart not yet generated |
| Fig. 6 — `hidden_under_compute_fraction` distribution (histogram, 124 real rows) | §8 Evaluation | `benchmarks/cxl-sim/unified-sweep.csv` (`hidden_under_compute_fraction` column) | data exists, chart not yet generated |
| Fig. 7 — Pipeline-count vs. p99 latency sensitivity | §8 Evaluation | `benchmarks/cxl-sim/unified-sweep-hardware-sensitivity.csv` | data exists, chart not yet generated |

No figure should be generated from anything other than the committed
CSV/JSONL artifacts (or the Mermaid diagrams already in
`docs/architecture.md`) — see `benchmarks/MANIFEST.json` for the
SHA-256-tracked source of each.
