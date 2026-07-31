# EXP-FPGA-DIV-001 -- decision log

A single, running summary of this experiment's per-phase decisions and
why, so the current state is readable without opening every phase
document. Each phase's own document (`baseline.md`, `phase-b1.md`,
`phase-b2.md`, `phase-b3.md`) is the authoritative, detailed record; this
file is a pointer/summary, not a replacement. Nothing in this file
authorizes or implies a merge into `main` -- see `experiment.md`'s own
"Promotion status" section, which is the authoritative statement on that.

| Phase | Scope | Decision | Why (one line) |
|---|---|---|---|
| A | Characterize the baseline `membrane_fp_divider` and its 4 call sites | Accepted as complete | Pure characterization, no design work; real synthesis + 520,000-txn cosim confirmed the baseline unchanged and identified 4 candidate directions for Phase B |
| B1 | Replace `q4_scale`'s constant-divisor `mx/-8.0f` with an exact power-of-two shortcut | CONTINUE | Exact and clean (2.2M+ differential cases, 0 mismatches), but the real `q4_scale`-level area win was small (-2.2%) because ABC was already sharing most of the two divider instances' cost |
| B2 | Replace `q4_scale`'s remaining variable-divisor `1/d` with an exact iterative divider | CONTINUE | Exact and clean (2.45M+ differential cases, 0 mismatches; 520,000/520,000 full-datapath), large real area win (-96.9% at `q4_scale`), but full-serialization scheduling collaterally slowed 3 untouched chains ~1.9-2.6x -- a queueing cost, not a divider-speed cost |
| B3 | Decouple issuance of the 3 unaffected chains from Q4_0 encode's in-flight status, via a small bounded completion reorder buffer | **CONTINUE** | Correct and bounded at every depth tested (0 fails/drops/duplicates/deadlocks across 7.77M+ real transaction-checks), but: (1) depths 1-2 are real REGRESSIONS vs. B2, only depth>=4 helps; (2) the reorder buffer's own area cost (14,959 ECP5 cells at depth=4) exceeds the entire `q4_scale_b2` unit B2 already shrank (2,268 cells), measurably eroding B2's -96.9% area win to an ESTIMATED ~-76.8% combined |

## Current recommendation if this experiment continues

- **Selected queue depth so far: REORDER_DEPTH=4** (see `phase-b3.md`
  section 9/10) -- smallest depth that is a real improvement over B2 on
  every measured metric.
- **Open item for a hypothetical Phase B4**: redesign
  `membrane_completion_reorder` to store a tag/pointer per outstanding
  slot instead of the full 531-bit payload, reading wide data from the
  existing per-chain output registers only at drain time. This would very
  plausibly recover most of the area Phase B3's current implementation
  costs, without touching the scheduling logic that Phase B3 already
  proved correct. Not started -- explicitly out of Phase B3's own scope.
- **`q8_scale.sv`'s two divider instances remain completely untouched**
  by every phase so far (A through B3) -- the same structural timing risk
  (wide combinational divide, no real Fmax data in this environment) Phase
  A first disclosed for them still applies, unchanged, unquantified.

## Promotion status

Not proposed for any phase (A, B1, B2, or B3). See `experiment.md`'s own
"Promotion status" section for the authoritative statement -- this remains
disclosed, research-in-progress work on `experiment/fp-divider-pipeline`,
never merged into `main`, no pull request opened.
