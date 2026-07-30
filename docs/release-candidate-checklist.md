# Release candidate checklist

A checklist only — **no git tag or GitHub release is created by this
document or by this phase.** Use this to decide when the project is
ready for a maintainer to cut one manually.

## Paper

- [ ] `paper/main.pdf` builds successfully via `paper/build.sh` (or the
      `Paper build` GitHub Actions workflow, `.github/workflows/paper.yml`,
      if no local LaTeX toolchain is available).
- [ ] `paper/scripts/verify-paper.py` passes all checks.
- [ ] No `[citation needed]` marker remains in the manuscript body
      (Related Work's historical-context mentions excepted, per
      `paper/scripts/verify-paper.py`'s own check).
- [ ] Bibliography cross-consistent (every `@key`/`\cite` used has a
      matching entry; every entry is cited at least once).

## Demo

- [ ] `scripts/demo.sh --quick` passes all required steps.
- [ ] `scripts/demo.sh --full` passes (if a full-suite check is desired
      before release).
- [ ] `demo-output/demo-results.json` is valid and reflects a clean run.

## Artifact hashes

- [ ] `benchmarks/MANIFEST.json` is up to date
      (`scripts/generate-benchmark-manifest.py --check` passes).
- [ ] `scripts/verify-results.py` passes all checks.

## Hardware plan

- [ ] `docs/phase8-hardware-validation-plan.md` reflects the current,
      real verification state (no level marked complete that hasn't
      actually been run on real hardware).
- [ ] `outreach/hardware-claim-gates.md`'s gate table matches reality —
      no gate marked PASSED without its required artifact actually
      existing in the repository or being independently verifiable.

## Contact package

- [ ] `outreach/` materials (technical brief, one-page summary, email
      templates, research profile, lab package) are internally
      consistent with the current claim-gate state (no outreach
      material overclaims beyond what `outreach/hardware-claim-gates.md`
      currently allows).

## Claim audit

- [ ] `paper/claim-audit.md` covers every headline number currently
      cited in `README.md` and `paper/main.md`.
- [ ] `scripts/verify-outreach.py` (prohibited-claim scan across
      `outreach/`) passes.

## Clean repository

- [ ] `git status` shows a clean tree (or only the pre-existing,
      disclosed `third_party/llama.cpp` dirty submodule state — see
      every prior phase's commit for why that's left untouched).
- [ ] No stray build output, log file, core dump, or cache directory is
      tracked by git (re-run the Phase 7.1 cleanliness check —
      `git status --porcelain --ignored=matching` — if unsure).
- [ ] `scripts/prepare-release.sh` passes (or `--dry-run` passes, at
      minimum).

## Versioning

- [ ] `CITATION.cff`'s `version` field is updated if this release
      represents a new phase milestone.
- [ ] `CHANGELOG.md` has an entry for this release candidate's changes.

## License review

- [ ] `docs/licensing.md` is current (no new third-party dependency or
      vendored code added without a corresponding licensing entry).
- [ ] `LICENSE` file unchanged/still Apache 2.0 (or, if changed, that
      change was deliberate and reviewed, not incidental).
- [ ] No vendor IP, driver, or board-support package was
      accidentally committed under `hardware/` (per
      `hardware/README.md`'s "what this directory does NOT contain").

## Final sign-off

- [ ] A human (not just automated checks) has read through
      `docs/phase7-hardware-outreach.md` (or whichever phase-summary doc
      is most recent) and confirmed it accurately reflects what's real
      vs. simulated vs. assumed vs. planned.
