# Research release freeze

## Freeze baseline

**Pre-audit baseline: commit `58ec90b`** ("fix: repair GitHub Actions
verification") — the last commit before this public-release audit
began. This audit itself (the commit that adds this document,
`docs/public-release-audit.md`, `docs/repository-boundary.md`, and the
related corrections) produces a **new commit that supersedes `58ec90b`
as the actual freeze/release-candidate baseline**. That commit's exact
SHA is not known until it is made — see
`docs/v0.1.0-research-release-plan.md` for the plan that names it once
it exists, and treat any specific SHA cited here as the state audited
*going into* this freeze, not the frozen state itself.

## Public release scope

Everything currently in the public `kadireren7/membrane` repository:
the C11/C++17/SystemVerilog implementation, the discrete-event
simulators, the bit-exact quantization verification, the unified
128K×512 sweep results, the FPGA cosimulation, the academic manuscript
(`paper/`), the reproduction/verification tooling (`scripts/`,
`paper/scripts/`), the hardware-validation plan and outreach package
(`hardware/`, `outreach/`), and the full phase-by-phase documentation
(`docs/`). See `docs/repository-boundary.md` for what is explicitly
**not** in scope (unreleased/private work, which does not currently
exist anywhere).

## Verified components (as of this freeze)

- **Software/simulation**: 462/462 unified-sweep scenarios, bit-exact
  CPU/ggml quantization parity (100,000+ blocks), FPGA/CPU Verilator
  cosimulation (520,000 transactions) — all re-verifiable via
  `scripts/verify-results.py` (13/13) and `scripts/demo.sh`.
- **Academic manuscript**: claim-audited (`paper/claim-audit.md`),
  14 independently-verified citations, `paper/scripts/verify-paper.py`
  (11/11), and `paper/main.pdf` now builds successfully as a real
  GitHub Actions artifact (workflow: `Paper Build`).
- **CI**: `CI` (Debug, ASan+UBSan, TSan) and `Paper Build` both pass on
  the real GitHub Actions runner as of `58ec90b`.
- **Outreach package**: `scripts/verify-outreach.py` (12/12), claim-gate
  enforcement (`outreach/hardware-claim-gates.md`), authorship/
  AI-assistance wording corrected and consistent (`bb4df95`).

## Open technical limitations (unchanged by this freeze)

- No real FPGA board, place-and-route result, or board bring-up exists.
- No real CXL hardware or CXL platform access exists.
- No real GPU serving-stack integration exists.
- Model scale is 135M/360M — below production LLM sizes.
- Every claim gated in `outreach/hardware-claim-gates.md` past Gate 1
  remains prohibited until its required real-hardware evidence exists.

This freeze does not change any of the above — it freezes the
*software/documentation* state as reproducible and internally
consistent, not a claim that physical validation has happened.

## What changes are acceptable in the public repo after this freeze

- Fixes to reproducibility, CI, documentation accuracy, or internal
  consistency (the same category of change this freeze itself makes).
- Corrections to a claim found to be inaccurate, overstated, or
  inconsistent — always disclosed, never silently softened.
- New, real hardware-validation results that pass their corresponding
  gate in `outreach/hardware-claim-gates.md`, promoted from
  `membrane-labs` (if one comes to exist) through the controlled
  process in `docs/repository-boundary.md`.
- Genuinely new experiments, if run with the same rigor as existing
  ones (real measurement, sourced, claim-audited, negative results
  included) — this is not prohibited by the freeze, but is not expected
  in the immediate term (see `README.md`'s Roadmap).

## What is not acceptable without an explicit, disclosed decision to lift the freeze

- Adding a new headline claim not backed by a committed artifact.
- Removing or softening a negative/null finding.
- Silently changing a verified number.
- Claiming physical hardware validation before its gate is actually
  passed.

## What work goes to labs (`membrane-labs`, not yet created)

Per `docs/repository-boundary.md`: unreleased experiments, vendor-
specific FPGA/Vivado/Vitis project files, board constraint files, raw
hardware logs, experimental RTL variants, unpublished predictors/
codecs, large-model experiments, and collaboration-sensitive materials.
None of this currently exists — this is where it would go if it did.

## Conditions under which this freeze would be lifted

- **A real hardware-validation result arrives** (Level A/B/C in
  `docs/phase8-hardware-validation-plan.md`) — the freeze is lifted for
  exactly the claims that result's corresponding gate unlocks, per
  `outreach/hardware-claim-gates.md`, not as a blanket reopening.
- **A real, disclosed error is found** in a currently-frozen claim —
  the freeze does not protect an inaccurate statement; it gets
  corrected immediately, following the same disclosure discipline as
  every prior correction in this project's history
  (`docs/phase7-hardware-outreach.md`, this document's own history).
- **Kadir decides to resume active experimentation** — a deliberate,
  disclosed decision (e.g. a new `docs/phaseN-*.md` document), not an
  implicit one.
