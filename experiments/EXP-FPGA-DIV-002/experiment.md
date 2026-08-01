# Experiment record: EXP-FPGA-DIV-002

Filled from [EXPERIMENT_TEMPLATE.md](../../EXPERIMENT_TEMPLATE.md).
Branch: `experiment/q8-divider-pipeline`.

## Experiment ID

`EXP-FPGA-DIV-002`

## Hypothesis

`rtl/q8_scale.sv`'s two parallel `membrane_fp_divider` instances
(`d = amax/127.0`, `id = 127.0/amax`) are, by direct measurement now
available from EXP-FPGA-DIV-001's own promoted Q4_0 precedent
(`experiments/EXP-FPGA-DIV-001/`), this datapath's single largest
remaining disclosed FPGA-synthesis risk: `q8_scale.sv`'s own header
comment (lines 14-19) already discloses "two divider instances run in
parallel... rather than time-multiplexing a single divider," and
`membrane_fp_divider.sv`'s own header (lines 32-37) already discloses
the underlying divider is "a single, wide combinational... operator,"
unpipelined, with an unquantified real-Fmax risk. Following
EXP-FPGA-DIV-001's own Phase A methodology exactly (characterize
first, do not design an alternative yet), this falsifiable claim is
tested in a **future Phase B**, not yet tested here: *at least one of
the two `q8_scale` divider instances can likely be replaced by an
alternative construction (shared, dual-small, or constant-optimized)
using meaningfully fewer synthesized cells and/or a shorter
combinational critical path, without changing the datapath's bit-exact
behavior.*

**Phase A scope (this record): characterize the baseline and evaluate
candidate directions on paper/differentially — no alternative divider
is designed, written, or synthesized in this phase.** See `baseline.md`
section 4 for the six candidate directions analyzed (not implemented).

## Baseline tag/commit

Tag `v0.1.0-research`, commit `8298e953b792c78aa8604c7558ef701b2b862b28`
(the current stable public release). This experiment branch,
`experiment/q8-divider-pipeline`, was created from `main` at commit
`9dbbede255dccf025cc3ecad7f17cd9f52f384a8` (the current `main` HEAD at
experiment start, several commits ahead of the tag — unrelated
CodeRabbit/CodeQL/CI-infrastructure work, no RTL changes — see
`docs/repository-boundary.md`/`docs/research-release-freeze.md` for why
`main` and the release tag are allowed to differ).

## Method

Phase A is characterization plus a **differential feasibility study**
(no new RTL divider variant written or synthesized — an explicit
constraint of this phase, unlike EXP-FPGA-DIV-001 whose later Phase B1-B4
sub-phases did write and synthesize new RTL):

1. Read `rtl/q8_scale.sv`, `rtl/membrane_fp_divider.sv`,
   `rtl/membrane_quant_stream_top.sv`, `rtl/q8_maxabs_reduce.sv`, and
   the C reference (`src/quant/quant_simd.c`) to derive the two
   divider operations' exact roles, their mathematical/floating-point
   relationship, and every edge case (zero/NaN/Inf/subnormal) —
   `results/baseline-dataflow.md`.
2. Re-ran `yosys` (generic and `synth_ecp5`) on the standalone
   `membrane_fp_divider` (cross-checking against EXP-FPGA-DIV-001's own
   published numbers), **and**, going further than EXP-FPGA-DIV-001's
   own Phase A (which had to kill this run), completed real
   `synth_ecp5` runs for `q8_scale` standalone and the full
   `membrane_quant_stream_top` — see `baseline.md` section 5 and
   `results/synthesis.csv` for whether the full-top run completed or
   was bounded/killed.
3. Re-ran `hierarchy -check -top membrane_quant_stream_top` to confirm
   the whole design still elaborates cleanly (unchanged from
   EXP-FPGA-DIV-001).
4. Re-ran the existing 520,000-transaction Verilator cosimulation
   (`rtl/tb/tb_top_verilator.cpp`) against the unmodified RTL.
5. Wrote a new differential feasibility tool,
   `rtl/tb/tb_q8_scale_feasibility.cpp`, that drives the REAL Verilated
   `membrane_fp_divider`/`membrane_fp_multiplier` RTL (never a
   hand-written approximation) to measure, at scale, whether three
   candidate shortcuts (`1/d`, `1/id`, and a constant-reciprocal
   multiply) agree bit-exactly with the current production `d`/`id`
   values, across edge cases, uniform-random `amax` bit patterns, and a
   realistic Q8 runtime `amax` distribution sample.
6. Re-ran the project's existing CI-equivalent local verification
   (Debug/Release/ASan+UBSan/TSan ctest, ggml quant parity,
   `verify-results.py`, `verify-paper.py`, `verify-outreach.py`) plus a
   CodeQL/CodeRabbit config sanity check, to confirm nothing else in
   the repository regressed.

No production RTL file was modified. No new divider variant was
written or synthesized anywhere in this phase — `tb_q8_scale_feasibility.cpp`
only re-drives the existing, unmodified `membrane_fp_divider`/
`membrane_fp_multiplier` RTL with different operand sequences.

## Environment

This project's own development machine: 5.6 GiB RAM, shared with other
concurrent local sessions/processes during this run (unlike
EXP-FPGA-DIV-001's presumably more isolated run — disclosed because it
measurably affected wall-clock time and required deliberate
sequencing of memory-heavy steps, though not peak-RSS-per-job, which
matched EXP-FPGA-DIV-001's own numbers closely). `tools/.local-yosys`
(Yosys 0.33, git sha1 2584903a060) and `tools/.local-verilator`
(locally-extracted Verilator, no root install) — the same toolchain
versions EXP-FPGA-DIV-001 and `docs/phase5-synthesizable-fpga.md` used.
No place-and-route tool, no Xilinx/Altera toolchain, no physical FPGA
board — unchanged from every prior phase's disclosure.

## Model/dataset

Not applicable in the LLM-checkpoint sense — this is a pure RTL/
synthesis-tooling experiment. The Verilator full-datapath cosimulation
uses the existing deterministic, fixed-seed golden vectors generated by
`rtl/tb/gen_top_x_vectors.c` and friends (120,000 blocks per format/
direction). The new feasibility differential
(`tb_q8_scale_feasibility.cpp`) uses (a) a full `amax` exponent/mantissa
boundary sweep plus named edge cases, (b) uniform-random 31-bit
magnitude patterns (matching `amax`'s own always-non-negative
invariant), and (c) a synthetic Q8 runtime `amax` distribution sample
built the same way `tb_membrane_fp_divider_radix4.cpp`'s own "q4
runtime d-distribution sample" stage does (32 random F16 elements per
block, `amax = max(|x|)`) — not a captured real-model trace, a
structurally-representative synthetic sample, same technique this
project's own prior divider experiment already established as
sufficient for this class of question.

## Metrics

- Divider instance count/role per call site, and their exact
  floating-point relationship (reciprocal in exact math, not
  necessarily bit-exact after independent rounding).
- Per-unit latency (cycles) and initiation interval.
- Yosys generic and ECP5-mapped cell counts: standalone divider,
  `q8_scale`, and (best-effort) the full `membrane_quant_stream_top`.
- Differential feasibility results per candidate: exact-match count,
  mismatch count, mismatch categories, max ULP, first mismatch
  examples.
- Verilator cosimulation transaction count and fail count.
- Local test-suite pass/fail counts (Debug/Release/ASan+UBSan/TSan) and
  the three `verify-*.py` script pass counts, plus ggml quant parity.

All sourced in `baseline.md`, `results/baseline-dataflow.md`, and
`results/synthesis.csv`, each number labeled MEASURED, SIMULATED,
ESTIMATED, or UNAVAILABLE.

## Success criteria (Phase A)

- The two divider operations and their exact mathematical/rounding
  relationship are correctly derived and cross-checked against at
  least two independent sources (RTL source comments + the C
  reference).
- Real (not estimated) Yosys synthesis results exist for the standalone
  divider and `q8_scale`, in both generic and technology-mapped form.
- The differential feasibility tool runs to completion at the
  requested scale (2,000,000+ cases) and reports real exact-match/
  mismatch counts for every candidate, without asserting or forcing a
  particular outcome.
- The existing 520,000-transaction Verilator cosimulation still passes
  at 0 fails against the unmodified RTL.
- Existing local verification (Debug/Release/ASan+UBSan/TSan ctest, all
  three `verify-*.py` scripts, ggml quant parity) still passes
  unchanged.

## Failure criteria (Phase A)

- Any call site or edge case (zero/NaN/Inf/subnormal handling) missed
  or mischaracterized relative to the actual RTL/C-reference source.
- Yosys synthesis failing to complete or reporting elaboration errors
  for the standalone divider or `q8_scale` (the full top-level is
  explicitly best-effort, per the task's own "mümkünse" scope — a
  bounded/killed full-top run is a disclosed limitation, not a failure).
- The feasibility tool crashing, hanging, or failing to reach its
  planned case count.
- The Verilator cosimulation reporting any of the 520,000 transactions
  as a mismatch (would indicate this characterization work itself
  introduced a regression, since no production RTL was supposed to
  change).
- Any existing test or verification script regressing.
- Presenting any exact-mismatching candidate as production-ready, or
  any generic/technology-independent cell count as if it were a real
  LUT count.

## Resource budget

Expected: a few hours of wall time (larger than EXP-FPGA-DIV-001's own
"well under an hour" Phase A budget, because this phase's synthesis
scope is larger — `q8_scale` standalone plus a full-top attempt, both
new relative to DIV-001 — and the differential tool's `--full` scope is
2,000,000+ cases against real Verilated RTL, run twice per case
through the divider). Actual figures recorded per-step in
`results/baseline-synthesis.txt` and this file's own "Results" section
below.

## Checkpoints

`scripts/run-exp-q8-divider-002.sh --resume` skips any already-built
Verilator object dir or already-produced log file it finds in
`--output-dir`, matching `scripts/verify-q4-radix4-divider.sh`'s own
`--resume` convention — relevant here because the full synthesis
matrix and the full 2M+-case differential run are each independently
resumable/skippable without re-running the other.

## Results

See `baseline.md` (full characterization + feasibility analysis),
`results/baseline-dataflow.md` (dataflow derivation),
`results/baseline-synthesis.txt` (raw synthesis summary),
`results/synthesis.csv` (structured synthesis matrix), and
`results/feasibility-differential-full.txt` (raw differential output).
Headline numbers:

- `q8_scale` = 2 parallel `membrane_fp_divider` instances (`u_div_d`,
  `u_div_id`), both `DIV_DELAY=1`. Latency 1 cycle, II 1 (parallel, not
  chained). Whole datapath: 7-cycle uniform latency (`L_MAX`), II 1.
- `amax/127` and `127/amax` are exact reciprocals in real arithmetic but
  NOT guaranteed bit-exact after independent IEEE-754 rounding — now
  quantified, not just asserted (see feasibility results below).
- Yosys: standalone `membrane_fp_divider` generic 10,234 / ECP5 73,629
  cells (3rd independent reproduction, unchanged). `q8_scale` standalone
  generic 21,800 / **ECP5 123,742 cells — NEW real measurement**,
  correcting EXP-FPGA-DIV-001's own ~75-80K extrapolation upward. Full
  top-level `synth_ecp5`: attempted, killed after a 25-minute bound,
  UNAVAILABLE.
- Verilator cosim: 520,000/520,000 transactions, 0 fails (incl. 120,000
  Q8 encode + 120,000 Q8 decode focused stages).
- Differential feasibility (2,050,239 cases each): reciprocal
  reconstruction `1/d`↔`id` 74.96% exact, `1/id`↔`d` 71.83% exact (every
  ordinary mismatch exactly 1 ULP, plus a real zero-case +Inf bug if
  unguarded); constant-reciprocal `amax*(1/127)`↔`d` 95.46% exact (also
  exactly 1 ULP per mismatch, no categorical failures).
- Local verification: Debug 28/28, Release 28/28, ASan+UBSan 30/30,
  TSan 30/30, ggml quant parity PASS, `verify-results.py` 13/13,
  `verify-paper.py` 11/11, `verify-outreach.py` 17/17.
- Decision: **NEXT_DUAL_RADIX4** (see `baseline.md` section 7).

## Limitations

- No real Fmax/timing-closure number exists for any configuration (no
  P&R tool in this environment) — unchanged from every prior phase.
- The differential feasibility tool measures **bit-value agreement**
  only; it says nothing about the relative synthesized cost of
  candidates B/C (a multiplier is generally cheaper than a divider, but
  this experiment does not synthesize a full alternative `q8_scale`
  built on any candidate — that is explicitly Phase B work).
  Candidates D/E/F (shared/dual/algebraic architectures) are evaluated
  by analysis only, not by a working alternative implementation.
- This is a characterization-and-feasibility-only phase; it makes no
  claim about whether any Phase B alternative will actually reduce cell
  count or improve timing closure in production RTL — that is exactly
  what a future Phase B would need to test, same disclosed boundary
  EXP-FPGA-DIV-001's own Phase A drew before its later Phase B1-B4 work.
- No real FPGA hardware, board, or vendor toolchain (Vivado/Quartus)
  was used anywhere in this experiment.

## Decision

**NEXT_DUAL_RADIX4.** Candidate B (reciprocal reconstruction) is
empirically rejected (~25-28% bit-mismatch rate, measured). Candidate C
(constant-reciprocal multiply) is measurably better but still not exact
(~4.5% mismatch) — kept on the list for a future dedicated Phase B, not
promoted now. Candidate E (dual exact radix-4 dividers) is the
strongest candidate: EXP-FPGA-DIV-001 Phase B4 already proved
`membrane_fp_divider_radix4` bit-exact (4,456,685 cases, 0 mismatches)
and measured its standalone cost at 1,509 ECP5 cells (-97.9% vs. this
experiment's own freshly-confirmed 73,629-cell baseline), so applying
two instances carries zero new bit-exactness risk — the open question
for a future Phase B is the real two-instance-parallel area/latency
cost, not measured this phase. See `baseline.md` section 7 for the full
reasoning, including why candidate D (shared single divider) ranks
below E.

## Promotion status

`not proposed` — this remains on `experiment/q8-divider-pipeline`,
pushed to the public repository per this project's open-development
policy (`docs/open-development-policy.md`), but **not merged into
`main`**. No pull request has been opened, per this task's own explicit
scope. Per `docs/research-release-freeze.md`, nothing here is a
verified public claim of the `v0.1.0-research` release; it is
disclosed, research-in-progress work on a public branch.
