# Repository development boundary

**Decision (as of `v0.1.0-research`, commit `8298e953b792c78aa8604c7558ef701b2b862b28`):
MEMBRANE will not have a private `membrane-labs` companion repository.
No private repository has been created, and none is planned.**
An earlier version of this document described a *possible* future private
repository by that name and was explicit that it did not exist yet. That plan
is now superseded — not silently deleted: Kadir decided the project stays
**fully open source**, with all new experimental work developed in public
branches of this same repository instead. See
[docs/open-development-policy.md](open-development-policy.md) for the
reasoning behind that decision. This document now describes the actual
branch and contribution structure used going forward.

This document defines the boundary between **released, verified state**
(the `main` branch and release tags) and **active, unverified research**
(everything developed in public branches) — both live in
`kadireren7/membrane`, nothing here is private.

## Branch structure

- **`main`** — only verified and reviewed changes. Every merge into `main`
  has passed CI (Debug, ASan+UBSan, thread-sanitizer, Paper Build), has
  reproduction commands that work from a clean clone, and — for anything
  touching a public claim — a claim audit and a limitations statement (see
  "Merging into `main`" below).
- **`experiment/<name>`** — research experiments: new predictors, codecs,
  simulator variants, hardware-adjacent modeling, anything whose outcome is
  not yet known. This is where MEMBRANE's actual research happens.
- **`feature/<name>`** — new functionality being built toward eventual
  `main` inclusion (tooling, CLI flags, new CI checks, new reproduction
  levels) that isn't itself a research experiment.
- **`fix/<name>`** — bug fixes, including fixes discovered while working
  an `experiment/*` or `feature/*` branch.
- **`docs/<name>`** — documentation-only changes (this change included).
- **Release tags** (`v0.1.0-research`, and future `vX.Y.Z-research` tags)
  — immutable research snapshots. See "Release tags are immutable," below.

## What experimental branches are — and are not

- **Not verified public claims.** A number, a plot, or a conclusion that
  only exists on an `experiment/*` branch is not a claim this project
  stands behind yet. Nothing on an active branch retroactively changes
  what any tagged release (starting with `v0.1.0-research`) claims.
- **Not hidden.** Experiments are developed as ordinary public branches
  and pull requests, visible in the same repository as everything else —
  there is no private staging area.
- **Failed experiments are not deleted.** A branch whose hypothesis didn't
  hold, whose results were null/negative, or that was abandoned partway is
  documented (via its `EXPERIMENT_TEMPLATE.md` record, see
  [EXPERIMENT_TEMPLATE.md](../EXPERIMENT_TEMPLATE.md)) and kept, not
  quietly removed — this matches MEMBRANE's existing practice of reporting
  negative findings with the same visibility as positive ones (see
  `docs/results-summary.md` §4, unchanged by this document).

## Merging into `main`

A pull request from any branch type into `main` must include, per
[CONTRIBUTING.md](../CONTRIBUTING.md):

1. A hypothesis or linked issue explaining why the change was made.
2. Reproduction commands that work from a clean clone.
3. Test results (`ctest` output, sanitizer runs where applicable).
4. Artifact hashes for any new/changed committed benchmark artifact.
5. A limitations section — what the change does *not* establish.
6. Any negative/null findings encountered, reported honestly.
7. Consistency with the project's AI-assistance disclosure
   (`outreach/ai-assistance-disclosure.md`) — no new text implying sole
   human authorship of AI-drafted material, and vice versa.
8. No fabricated or implied hardware claims — anything hardware-adjacent
   stays inside the gates in `outreach/hardware-claim-gates.md`.

Merges that only touch reproducibility, documentation accuracy, or CI are
held to the same bar minus what doesn't apply (e.g. a pure `docs/*` change
has no new artifact hashes to report).

## Release tags are immutable

Once a commit is tagged as a research release (`v0.1.0-research` today,
`v0.2.0-research` or later in the future), that tag and the commit it
points to never move and are never edited in place. See
[docs/research-release-freeze.md](research-release-freeze.md) for exactly
what "frozen" means and how new, verified work eventually becomes a new
release rather than a retroactive change to an old one.

## Rules

1. **Nothing is deleted or degraded to make a narrative cleaner.** A
   superseded document is replaced with something at least as accurate and
   the change is disclosed here or in `CHANGELOG.md` — reproduction and
   history must never break or disappear silently. This document's own
   supersession of the old `membrane-labs` plan follows that rule.
2. **Every experimental branch names its baseline.** Per
   `EXPERIMENT_TEMPLATE.md`'s `baseline tag/commit` field, every
   experiment states the exact tag or commit SHA it started from, so the
   relationship between released and in-progress state is always
   traceable.
3. **No file silently diverges between `main` and an active branch
   without explanation.** If a branch reworks something also present on
   `main`, the PR that eventually merges it explains what changed and why.
4. **Promotion to `main` is controlled, not automatic.** A branch that
   reaches this project's verification bar (real measurement, sourced,
   claim-audited, negative results included) is merged through ordinary
   review under Kadir's direction — the same PR requirements as any other
   change here, never a silent fast-forward.
5. **This document does not claim a private repository exists anywhere.**
   It never will under the current decision; if that decision is ever
   revisited, it will be revisited here, explicitly and disclosed, not
   implied.
