# Experiment record template

Copy this file into your `experiment/<name>` branch (e.g. as
`docs/experiments/<name>.md`) and fill it in before opening a pull
request, whether the experiment succeeds, fails, or is abandoned
partway. Failed and null-result experiments are not deleted — they are
recorded with the same rigor as successful ones (see
`docs/repository-boundary.md`, `docs/results-summary.md` §4). Label
every number REAL, SIMULATED, EXTRAPOLATED, ORACLE, or ASSUMED, matching
this project's existing disclosure convention.

## Experiment ID

`<experiment-name>` — matches the `experiment/<name>` branch name.

## Hypothesis

What you expect to be true, stated as a falsifiable claim, not a vague
direction. ("Predictor X recovers >= 0.9 recall at 128K context with
half the prefetch bandwidth of the current predictor" — not "explore
better prefetching.")

## Baseline tag/commit

The exact tag or commit SHA this experiment started from (e.g.
`v0.1.0-research` / `8298e953b792c78aa8604c7558ef701b2b862b28`), so
results are always traceable to a known starting point.

## Method

What was built or changed, and how it was measured. Link to the actual
code/tool, not just a description.

## Environment

Hardware, OS, compiler/toolchain versions, and any resource constraints
relevant to interpreting results (prior phases of this project have run
on a RAM-constrained development machine and disclosed that explicitly —
see e.g. `docs/phase6-unified-stress.md`'s completion-history section —
record the equivalent constraint for your own run, if any).

## Model/dataset

Exact model checkpoint(s) and/or captured trace(s) used. Never averaged
or substituted across model scales without saying so explicitly.

## Metrics

What was actually measured, with units and source artifact (CSV/JSONL/
log path) for each.

## Success criteria

The specific threshold that would make the hypothesis look supported —
defined *before* running, not fitted to whatever the result turned out
to be.

## Failure criteria

The specific threshold or condition that means the hypothesis did not
hold. A record with no failure criteria is not a real experiment.

## Resource budget

Expected wall time, memory, disk, and any other constraint this run is
expected to respect (and whether it did).

## Checkpoints

Any intermediate checkpoint/resume state produced, and where it lives.

## Results

What was actually measured — real numbers with source artifacts, not
narrative. Include negative/null outcomes in full; do not omit a metric
because it came out unfavorably.

## Limitations

What this experiment does *not* establish, including anything about
scale, environment, or measurement method that limits how far the
result generalizes.

## Decision

Accepted / rejected / inconclusive / abandoned, and why, in one or two
sentences.

## Promotion status

- `not proposed` — still on the experiment branch, not yet ready for
  review.
- `proposed` — a pull request into `main` is open (link it).
- `promoted` — merged into `main` (link the merge commit).
- `not promoted` — reviewed and explicitly not merged (state why: e.g.
  a null result recorded here but not folded into a headline claim).
