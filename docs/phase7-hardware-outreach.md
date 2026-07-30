# Phase 7.3: Hardware Validation Outreach Package

Baseline: commit `e74d719` (Phase 7.2, academic manuscript). This phase
produced **no new experimental results and made no outbound contact
with any real person or institution.** Its job was to prepare a
scoped, claim-limited outreach and physical-validation-planning package
that could be sent to labs/companies in the future — nothing in this
phase was sent, and no relationship, response, or prior conversation is
claimed anywhere in the produced materials.

## Packages created

- **Outreach materials** (`outreach/`): a technical brief (2-4 pages),
  a one-page summary, four email templates (university professor, FPGA
  lab, CXL/memory-systems team, company research team), a personal
  research profile for Kadir, a demo video script, a research-talk
  outline, a target-selection framework (no named targets), and an
  empty contact tracker template.
- **Hardware validation planning** (`docs/phase8-hardware-validation-plan.md`,
  `hardware/`): a 3-level plan (FPGA sim/impl, real board, CXL
  platform), candidate board/toolchain research, a real risk register,
  a sequential experiment protocol, and a JSON schema for physical
  results.
- **Vendor-integration interface** (`hardware/vendor-wrapper/`):
  AXI4-Stream adapter, AXI4-Lite control-register adapter, a
  platform-independent DMA command/completion interface, and a
  top-level wrapper — all interface-only, no vendor IP.
- **Claim governance** (`outreach/hardware-claim-gates.md`): 8 explicit
  gates mapping "which test must pass" to "which sentence becomes
  allowed," enforced automatically by `scripts/verify-outreach.py`.
- **Release/CI infrastructure**: a GitHub Actions paper-build workflow
  (`.github/workflows/paper.yml`) and a release-candidate checklist
  (`docs/release-candidate-checklist.md`, checklist only — no tag or
  release was created).

## Evidence used (nothing new; all from Phase 0-7.2)

Every technical claim in the outreach materials traces to an artifact
that already existed before this phase: `benchmarks/MANIFEST.json`'s
33 tracked artifacts, `paper/claim-audit.md`'s 11 audited claims, and
`docs/phase5-synthesizable-fpga.md`'s real (yosys/ECP5) synthesis cell
counts — reused directly in `hardware/board-targets.md` rather than
re-derived. The one genuinely new verification step this phase
performed: **elaborating the new `hardware/vendor-wrapper/*.sv` files
together with the full existing, cosimulation-verified `rtl/` hierarchy
under yosys 0.33** (`hierarchy -top membrane_core_wrapper`) — this
succeeded with 0 errors and 0 warnings attributable to the new files (all
46 unique warnings in the combined run are pre-existing, from already-
verified `rtl/` modules). This confirms the wrapper's port/parameter
wiring is structurally consistent with the verified core — it does
**not** confirm synthesizability on a real vendor toolchain, timing
closure, or anything about real silicon.

## Which hardware claims remain prohibited

Per `outreach/hardware-claim-gates.md`, as of this phase:

- **"Hardware bit-exact"** — prohibited; only "simulation bit-exact" is
  earned (Gate 0/1 passed, Gate 4 not).
- **"FPGA-deployed" / "runs on an FPGA board" / "board-verified"** —
  prohibited; no board exists (Gate 3 not passed).
- **Any specific achieved clock frequency** — prohibited; only the
  100/200/300 MHz *assumption* from `docs/phase5-synthesizable-fpga.md`
  may be cited, always labeled as an assumption (Gate 2 not passed).
- **Any throughput/latency/power number presented as a hardware
  measurement** — prohibited entirely; every such number in this
  project is a simulation or cosimulation output (Gate 5/6 not passed).
- **"Real CXL acceleration" / "CXL-accelerated"** — prohibited; only
  "CXL architecture simulation" is earned (Gate 7 not passed).

`scripts/verify-outreach.py` enforces the last two automatically (a
"gated claim" scan across `outreach/` and `hardware/`, 9/9 checks
passing as of this phase, including three negative tests: a fake
`REAL_HARDWARE`-labeled result, an injected prohibited claim, and a
removed required lab-package file were each deliberately introduced and
confirmed to be caught, then reverted).

## Resources needed from a lab

Summarized in `outreach/lab-package/required-hardware.md`; in order of
value: (1) FPGA board + Vivado/Quartus toolchain access (even
time-boxed/remote) for a real place-and-route result; (2) if that
succeeds, board bring-up + real DMA/throughput/parity measurement; (3)
a real CXL Type-3 device or emulation platform access, the single
hardest resource to obtain per `hardware/risk-register.md`.

## Physical-validation success criteria

Defined per level in `docs/phase8-hardware-validation-plan.md`:
- **Level A**: real timing closure at some frequency + bit-exact
  post-route simulation.
- **Level B**: 0 unexplained mismatches on a 100,000+-block real-
  hardware parity test, plus a real (not estimated) throughput/latency/
  power measurement, whichever direction it points.
- **Level C**: a real device successfully serving the exact-retrieval
  workload end-to-end, plus a written comparison against this project's
  simulated CXL assumptions — valuable even if the comparison shows the
  simulation was wrong.

## LaTeX/PDF build status

**Unchanged from Phase 7.2: no LaTeX toolchain exists in this
development environment** (`pdflatex`/`xelatex`/`lualatex` all absent).
`paper/main.pdf` has still never been built locally. This phase adds a
**GitHub Actions workflow** (`.github/workflows/paper.yml`) as the
disclosed, explicit path to a real PDF build: it installs TeX Live on
the CI runner, regenerates figures/tables, runs
`paper/scripts/verify-paper.py`, runs `paper/build.sh`, and uploads
`paper/main.pdf` as a build artifact. **This workflow has not been
executed** (that requires pushing to GitHub and letting Actions run,
which is outside this phase's scope) — its YAML was syntactically
validated (parses correctly as a well-formed GitHub Actions workflow:
`name`/`on`/`jobs` keys present, one job with `runs-on` and a real step
sequence) but not exercised end-to-end. No PDF is claimed to exist
anywhere in this repository.

## Verification performed

- `scripts/verify-outreach.py`: 9/9 checks (lab-package completeness,
  internal link resolution, results-schema validity, example-fixture
  schema conformance, zero `REAL_HARDWARE` records present, zero hype
  words, zero unhedged gated claims, gate-table self-consistency,
  email-template placeholder-safety).
- `paper/scripts/verify-paper.py`: 11/11 (unchanged, re-confirmed).
- `scripts/verify-results.py`: 13/13 (unchanged, re-confirmed).
- `scripts/prepare-release.sh --dry-run`: passes except the (expected,
  pre-commit) dirty-tree check.
- Existing Release test suite: 28/28 (unchanged; no C/C++ source was
  modified this phase).
- yosys elaboration of `hardware/vendor-wrapper/*.sv` + the full
  existing `rtl/` hierarchy: succeeds, 0 new errors/warnings.
- Three negative tests (fake `REAL_HARDWARE` result, prohibited
  hardware claim, missing lab-package file): all three deliberately
  introduced and confirmed caught by `scripts/verify-outreach.py`,
  then reverted.

## What's still missing (disclosed, not hidden)

- No physical FPGA board, CXL device, or vendor toolchain has been
  used — every hardware-adjacent claim in this repository remains
  simulation, cosimulation, or an assumption.
- The GitHub Actions paper-build workflow has not actually been run
  (no PDF has been produced by it yet).
- No lab or company has been contacted; the outreach materials are
  unsent templates.
- `hardware/vendor-wrapper/`'s AXI4-Lite responder is explicitly
  flagged in its own file as an unverified hand-written FSM, not a
  protocol-checker-verified implementation.
