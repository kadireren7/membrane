# Public release audit

A full-repository audit performed before considering a first tagged
release (`v0.1.0-research`, see
`docs/v0.1.0-research-release-plan.md`). Scope: `README.md`,
`CHANGELOG.md`, `CITATION.cff`, `docs/`, `paper/`, `outreach/`,
`hardware/`, `scripts/`, `benchmarks/MANIFEST.json`, GitHub Actions
workflows, `CONTRIBUTING.md`/`SECURITY.md`/`SUPPORT.md`, and license
documents. No new research result, simulator, predictor, or codec was
produced during this audit — it is a consistency and freshness pass
over what already existed.

## What was already complete and correct

- **Software/simulation**: 462/462 unified-sweep scenarios, bit-exact
  CPU/ggml quantization parity (100,000+ blocks), FPGA/CPU Verilator
  cosimulation (520,000 transactions) — unchanged, re-verified
  (`scripts/verify-results.py` 13/13).
- **Academic manuscript**: `paper/main.md`/`paper/main.tex`'s content,
  the 14-source literature research, and the claim audit
  (`paper/claim-audit.md`) were already complete and internally
  consistent (`paper/scripts/verify-paper.py` 11/11) — no change to
  their substance was needed.
- **Outreach package and claim gates**: `outreach/` and
  `outreach/hardware-claim-gates.md`'s gate table were already
  internally consistent (`scripts/verify-outreach.py` 12/12).
- **Repository cleanliness**: no committed build output, stale
  checkpoint, model file, core dump, cache directory, or oversized file
  was found — the existing `.gitignore` was already doing its job (this
  matches the same finding from the Phase 7.1 cleanliness pass, still
  holding).
- **CI**: `CI` and `Paper Build` both genuinely pass on the real GitHub
  Actions runner as of `58ec90b` (confirmed via real `gh run` logs, not
  assumed).

## What was stale or wrong

- **README.md**: the repository-layout section didn't mention
  `outreach/` or `hardware/` at all (added in Phase 7.3, never added to
  this list); `paper/` was described as "Academic paper skeleton (in
  progress, citation-needed)" — false since Phase 7.2 completed the
  literature research; the Roadmap section said "near-term work is
  citation research" (already done) and cited "Phase 0 through 7.1"
  (three phases behind).
- **`docs/architecture.md`**: said "as of Phase 7.1" and its phase-
  history list stopped at 7.1, omitting Phase 7.2 (academic manuscript)
  and Phase 7.3 (hardware outreach, wording correction, CI repair).
- **`docs/results-summary.md`**: two places still described the
  Related Work literature survey as "intentionally left with
  citation-needed markers" / "explicitly deferred" — false since Phase
  7.2 completed it with 14 verified sources.
- **`docs/phase7-hardware-outreach.md`**: stated the `Paper Build`
  workflow "has not been executed" — true when written, false after the
  CI-repair commit (`58ec90b`) actually ran and passed it.
- **`paper/main.md`**'s status line said "Phase 7.2 draft manuscript"
  without mentioning the manuscript is complete or that the PDF now
  builds on real CI.
- **`CITATION.cff`**: `version: "0.7.1"` with a `date-released` implying
  a formal release had happened — neither was true; no tag or GitHub
  Release exists yet.
- **`.gitignore`**: `paper/main.pdf` and LaTeX build byproducts
  (`.aux`/`.out`/`.fls`/etc.) were not explicitly ignored — accidental
  and undetectable-by-tooling if anyone ever committed a built PDF.
- **`CHANGELOG.md`**: had no entries for Phase 7.2 (`e74d719`), Phase
  7.3 (`6faa0d5`), the authorship correction (`bb4df95`), or the CI
  repair (`58ec90b`); the `[0.7.1]` entry itself was missing its own
  commit SHA.

## Verification tooling extended (and what building it caught)

`scripts/verify-outreach.py` gained 5 new automated checks specific to
this audit (no lingering "paper skeleton" phrasing, no stale "Phase 0
through N" phase range, workflow badges pointing at real files, no
false private-repo-exists claim, no false main.pdf-is-tracked claim) —
12/12 → 17/17. Writing these checks surfaced three real bugs in the
checks themselves before they were trusted, each found by deliberately
testing against this repository's own real, correct docs and seeing an
unexpected failure:

1. The "Phase 0 through N" regex assumed a dash-based range
   (`Phase 0–7.1`); the real wording used throughout this project is
   the spelled-out `Phase 0 through 7.1` — the original regex would
   have silently never matched anything.
2. The `membrane-labs`-hedge check first required every individual
   mention to carry its own nearby hedge (too strict — flagged
   `docs/repository-boundary.md`'s own correctly-written prose), then,
   after loosening to a whole-file check, missed an injected false
   claim entirely because an unrelated use of the word "planned"
   elsewhere in the same file satisfied the check for the wrong
   reason. Fixed by checking a real window around each occurrence,
   requiring at least one to carry a specific, narrow hedge phrase.
3. The main.pdf-tracked check used `str.find()` (first match only);
   since README.md legitimately uses the phrase "is committed"
   elsewhere ("before anything is committed to RTL"), a real injected
   violation later in the same file was silently never reached. Fixed
   with `re.finditer` to check every occurrence.

All three were caught by the negative-testing step below, fixed, and
re-confirmed — not left as latent false-positive/false-negative risk
in the checking tooling itself.

## What was fixed

- README.md: repository layout now lists `paper/`, `hardware/`, and
  `outreach/` accurately; Roadmap rewritten to state the manuscript and
  citation research are complete, the real CI-built PDF exists, and
  physical FPGA/CXL validation is the next substantive evidence needed
  — not a vague "near-term work" placeholder. Status blurb now
  describes this as a "public research release." Added a `Paper Build`
  CI badge (the `CI` and `License` badges already existed and needed no
  change).
- `docs/architecture.md`: phase-history list extended through Phase
  7.3, naming the wording-correction and CI-repair commits.
- `docs/results-summary.md`: both stale citation-needed/deferred-survey
  mentions corrected to state the survey is done (while honestly noting
  it isn't exhaustive).
- `docs/phase7-hardware-outreach.md`: left its original text as written
  (accurate for its own point in time) and appended a new, explicitly
  labeled correction section — the same disclosure pattern already
  established by that document's own prior authorship-correction
  section, not a silent rewrite.
- `paper/main.md`: status line updated to state the manuscript is
  complete and the PDF builds on real CI.
- `CITATION.cff`: `version` changed to `"0.1.0-research-candidate"`,
  `date-released` replaced with `commit: 58ec90b` (an honest anchor to
  a real commit instead of an implied release date), and `message`
  updated to point at `docs/v0.1.0-research-release-plan.md`.
- `.gitignore`: added `paper/main.pdf` and LaTeX build byproducts.
- `CHANGELOG.md`: added `[0.7.2]`, `[0.7.3]`, `[0.7.3.1]`, `[0.7.3.2]`
  entries with correct commit SHAs (`e74d719`, `6faa0d5`, `bb4df95`,
  `58ec90b` respectively); added the missing SHA to `[0.7.1]`. No
  existing entry's content was rewritten.
- New: `docs/repository-boundary.md`, `docs/research-release-freeze.md`,
  `docs/v0.1.0-research-release-plan.md`, this document.

## What was deliberately left incomplete

- **No physical FPGA/CXL hardware validation** — this audit is a
  documentation/consistency pass, not new experimental work; Level A/B/C
  in `docs/phase8-hardware-validation-plan.md` remain entirely
  unstarted.
- **No literature survey expansion** — the Phase 7.2 survey (14
  sources) stands as-is; broadening it further is listed as future work
  in `docs/results-summary.md` §7, not attempted here.
- **No `membrane-labs` repository was created** — `docs/repository-
  boundary.md` documents a plan, not a status. Nothing was moved
  anywhere.
- **No tag or GitHub Release was created** — per this task's explicit
  instruction; `docs/v0.1.0-research-release-plan.md` is a plan awaiting
  Kadir's approval.

## What claims the public repo can now carry

- The software/simulation results (462/462 scenarios, bytes/token
  reductions, compute-floor findings, negative results) — unchanged,
  re-verified.
- "A complete, claim-audited academic manuscript with 14
  independently-verified citations and no unresolved `[citation
  needed]` markers."
- "`paper/main.pdf` is built by a real GitHub Actions workflow on every
  push" — confirmed true, not aspirational.
- "This is a public research release" in the sense defined by
  `docs/research-release-freeze.md` — a frozen, internally consistent,
  reproducible software/documentation state.
- Accurate authorship framing: Kadir as creator/lead/sole owner
  directing an AI-assisted development process (per `bb4df95` and
  `outreach/ai-assistance-disclosure.md`) — consistent everywhere it
  appears, re-confirmed by this audit's own scan.

## What claims remain prohibited pending real hardware

Unchanged by this audit — see `outreach/hardware-claim-gates.md` for
the full gate table:

- "Hardware bit-exact" (only "simulation bit-exact" is earned).
- "FPGA-deployed" / "runs on an FPGA board" / any specific achieved
  clock frequency.
- Any throughput/latency/power number presented as a hardware
  measurement.
- "Real CXL acceleration" / "CXL-accelerated" (only "CXL architecture
  simulation" is earned).
- Any claim that a `membrane-labs` repository, private experiment, or
  external collaboration currently exists — none does.

## Readiness for `v0.1.0-research`

**Ready, pending Kadir's explicit approval to actually create the tag
and release** (per this task's instruction, not done here). All
verification in `docs/v0.1.0-research-release-plan.md` passes as of
this audit's own commit (see that document and the "Verification"
section of this repository's history for exact commands and results).
The one open item before tagging is procedural, not technical: someone
(Kadir) reading through this document and `docs/research-release-
freeze.md` to confirm they accurately reflect intent, per
`docs/release-candidate-checklist.md`'s final sign-off item.
