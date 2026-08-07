# Repository development boundary

**Status: Superseded by the two-repository boundary decision adopted
during the repository-focus migration.** See "The two-repository
decision" below for the current model. This document is kept in place,
not deleted or replaced, per Rule 1 and Rule 5 below — both of which
this update is a direct instance of, not an exception to.

## Historical decision (as of `v0.1.0-research`, commit `8298e953b792c78aa8604c7558ef701b2b862b28`)

**MEMBRANE will not have a private `membrane-labs` companion repository.
No private repository has been created, and none is planned.**
An earlier version of this document described a *possible* future private
repository by that name and was explicit that it did not exist yet. That plan
was superseded — not silently deleted: Kadir decided the project stays
**fully open source**, with all new experimental work developed in public
branches of this same repository instead. See
[docs/open-development-policy.md](open-development-policy.md) for the
reasoning behind that decision. The rest of this document (below "The
two-repository decision") describes the branch and contribution
structure that was used under that model, and still applies to
`kadireren7/membrane` itself.

This document originally defined the boundary between **released,
verified state** (the `main` branch and release tags) and **active,
unverified research** (everything developed in public branches) — both
living in `kadireren7/membrane`, nothing private. That boundary is now
drawn between two public repositories instead of two branch categories
within one — see below.

## The two-repository decision

**Old model**: product and research lived in one repository,
`kadireren7/membrane`, separated by branch (`main` = released/verified,
`experiment/*` = active/unverified research) rather than by repository —
the model the historical decision above put in place, explicitly
choosing it over a private companion repository.

**Why it changed**: research variants and experiment evidence expanded
substantially over the project's life — full phased experiment records,
exploratory RTL, simulators, a paper, and an outreach package, spanning
many branches. Product navigation became noisy as a result: a
contributor or reviewer looking for the maintained implementation had
to work around a large volume of research material to find it. The
maintained implementation and the research record also have genuinely
different lifecycles — a merged, reviewed change to `main` is expected
to stay correct and current; an experiment's own history is expected to
accumulate superseded phases and negative results indefinitely, and
should never be pruned to keep the product repository tidy. Reproduction
and provenance guarantees for the two also deserve independent
ownership: the maintained repository's reproduction guide should only
ever need to describe maintained flows, not caveat itself with
research-only gaps.

**New model**: `kadireren7/membrane` is the maintained implementation —
source, production RTL, tests, CI, CodeQL, CodeRabbit, and documentation
scoped to building and using the current library.
[`kadireren7/membrane-research`](https://github.com/kadireren7/membrane-research)
is the active research laboratory — every experiment (complete and
ongoing), exploratory RTL, simulator, the paper, and the outreach
package, with full SHA256-verified provenance back to this repository.

**Research is continuing.** This split is not a wind-down of research —
it is a relocation of where research happens, so it can keep expanding
without degrading this repository's own navigability. New experimental
work happens in `kadireren7/membrane-research` going forward, not in new
`experiment/*` branches here. Existing `experiment/*` branches in this
repository (`experiment/q8-divider-pipeline`,
`experiment/fp-divider-pipeline`) are preserved, not deleted — per Rule
1 below, both are also mirrored (SHA256-verified, not just referenced)
into `kadireren7/membrane-research`, so their content is available from
either repository even though this one remains their authoritative git
history.

This revisits the "public branches of this same repository" model the
historical decision above put in place — not the private-repository
question Rule 5 (below) specifically addresses, which remains
unchanged: no private repository exists, and `membrane-research` is
public. This update follows Rule 1 (nothing deleted, the change
disclosed here) and adds Rule 6 (below), the same explicit-and-disclosed
pattern Rule 5 established for its own narrower question.

## Branch structure

- **`main`** — only verified and reviewed changes. Every merge into `main`
  has passed CI (Debug, ASan+UBSan, thread-sanitizer, Paper Build), has
  reproduction commands that work from a clean clone, and — for anything
  touching a public claim — a claim audit and a limitations statement (see
  "Merging into `main`" below).
- **`experiment/<name>`** — research experiments: new predictors, codecs,
  simulator variants, hardware-adjacent modeling, anything whose outcome is
  not yet known. This is where MEMBRANE's research happened historically
  (`experiment/q8-divider-pipeline`, `experiment/fp-divider-pipeline`,
  both preserved); per Rule 6, new research branches are no longer
  created here — see "The two-repository decision" above.
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
7. Consistency with the project's AI-assistance disclosure (this
   repository's own README.md "AI-assisted development" section, and
   the full version at `kadireren7/membrane-research`'s
   `outreach/ai-assistance-disclosure.md`) — no new text implying sole
   human authorship of AI-drafted material, and vice versa.
8. No fabricated or implied hardware claims — see README.md's own
   "Limitations" section; the full outreach claim-gating discipline
   lives at `kadireren7/membrane-research`'s
   `outreach/hardware-claim-gates.md`.

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
6. **New experimental work happens in `kadireren7/membrane-research`,
   not new `experiment/*` branches here.** Existing `experiment/*`
   branches in this repository are preserved as historical git history,
   unchanged, per Rule 1 — this rule governs where *new* research
   branches are created, not a claim that old ones are removed. See
   "The two-repository decision" above.
