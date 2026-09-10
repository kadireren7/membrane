# Contributor rights governance

Why this exists, what changed, and what did not. Read
`CONTRIBUTOR_RIGHTS_AGREEMENT.md` for the actual legal terms — this page is
the design rationale and audit trail behind it, not a restatement of it.

## Revision history

- **v1.0 draft:** initial copyright-assignment-plus-fallback-license draft,
  framed as entirely optional for contributors, with plain Apache-2.0 kept
  as an always-available alternative path. Never adopted.
- **v2.0:** revises the v1.0 draft after further maintainer review before
  adoption. The most consequential change: the Agreement is no longer
  optional for external contributions — see "The mandatory flow" below.
  Also fixes a dangling cross-reference (former Section 21→22), rewrites
  the AI-disclosure clause to be operational rather than unverifiable,
  resolves the governing-law placeholder with real jurisdiction-specific
  research (Turkish moral-rights law), adds precise non-circular
  acceptance-state definitions with named edge cases, adds a
  multiple-rightsholder/co-author rule, and redesigns the enforcement
  workflow so it can actually fail a check instead of always reporting
  success. A later same-version formalities pass enumerated the specific
  economic rights assigned/licensed (FSEK Article 52), restructured moral
  rights around consent-to-exercise + non-assert rather than waiver, and
  disclosed a real open question: whether a plain GitHub acceptance
  comment satisfies Turkish law's "in writing" requirement at all.
- **v3.0 (current):** resolves that open question by **removing the
  GitHub-comment-only acceptance mechanism entirely.** A comment cannot
  reliably satisfy FSEK Article 52's written-form requirement (Turkish
  Code of Obligations Art. 14–15 ties "written form" to a handwritten or
  secure/qualified electronic signature, which an ordinary comment is
  not). Section 27 now requires an actual signature via Path A (qualified/
  secure electronic signature) or Path B (wet-ink), verified by the
  maintainer and recorded as a `VERIFIED` row in
  `docs/contributor-agreements.md` before a PR may be merged — a comment
  can still be used to start the process, never to complete it. The
  registry schema, the enforcement workflow's lookup logic, and every
  other document referencing the old comment-based mechanism were updated
  to match. Section numbers in the Agreement are unchanged between v2.0
  and v3.0; only Section 27's own mechanism and the registry schema
  changed.

## Adoption

**This document (and the Agreement it describes) is a DRAFT as of this
PR.** "Adoption" — the point at which Agreement Section 27's mandatory
flow actually starts governing merges — is a separate, later act: the
maintainer merging this PR into `main`. The effective date of adoption is
the merge commit's own date on `main`, not the date any individual commit
in this PR was authored. Until that merge happens, nothing in this PR
changes how any pull request is actually merged.

## Current legal state (audited before drafting anything)

- **License:** Apache License 2.0 (`LICENSE`, unchanged, real, standard
  text — verified by reading the file directly, not assumed).
- **No NOTICE file** exists in this repository.
- **No CLA, no DCO, no copyright-assignment agreement** existed before this
  change. `CONTRIBUTING.md` had zero mention of license, sign-off,
  contributor agreements, or copyright.
- **The current inbound-contribution model is Apache-2.0's own Section 5**
  ("Submission of Contributions"): *"Unless You explicitly state otherwise,
  any Contribution intentionally submitted for inclusion in the Work by You
  to the Licensor shall be under the terms and conditions of this License,
  without any additional terms or conditions."* In practice, this already
  gives the Project a real, broad, perpetual, worldwide, royalty-free
  license (Apache-2.0 Section 2's own copyright grant) plus a real patent
  grant with defensive termination (Section 3) for anything a contributor
  submits — **inbound = outbound**, the same mechanism many Apache-2.0
  projects rely on instead of a separate CLA.
- **What Apache-2.0-only does NOT do: it does not transfer copyright
  ownership.** A contributor who has a PR merged under plain Apache-2.0
  terms keeps their own copyright in that contribution; the Project only
  ever holds a (very broad) license to it, not title. This is the single
  most important fact this audit needs to get right — confirmed directly
  from the License text, not assumed.

## Why Apache-2.0-only is not enough for the maintainer's stated goal

The maintainer's objective (see `CONTRIBUTOR_RIGHTS_AGREEMENT.md`'s own
framing) is centralized control: the ability to relicense, dual-license,
and commercially exploit the codebase, including contributed portions,
without having to re-contact every past contributor. A pure license
(however broad) always leaves the underlying copyright with many different
people — which is legally sufficient for *distributing* the combined work
under Apache-2.0 forever, but is **not** sufficient for confidently
offering a *different, more restrictive* license (e.g. a closed-source
commercial SKU that isn't itself Apache-2.0-compatible) over contributed
code specifically, since the contributor could, in principle, also license
that same code to someone else on different terms — they never gave that
option up.

## Model comparison

| Model | Contributor keeps copyright? | Project's relicensing freedom | Overhead for contributors |
|---|---|---|---|
| A. Apache-2.0 only (current, before this change) | Yes | Broad license, but not exclusive control | None (implicit) |
| B. DCO (sign-off line) | Yes | Same as A — DCO only certifies provenance, grants no extra rights | Low (one `-s` flag) |
| C. Broad CLA (license-only) | Yes | Broader than A/B (can add explicit sublicense/relicense language) but still not full assignment | Low-medium |
| D. Copyright assignment | No — transfers to Project Owner | Maximum | Medium (real legal document, real acceptance step) |
| E. Assignment + fallback irrevocable license (chosen) | No, unless assignment is legally ineffective in the contributor's jurisdiction, in which case a fallback license (not assignment) applies for just that contributor/right | Maximum, with the same practical effect as full assignment for the contributor's usual case | Medium (same as D) |

**Chosen: E.** It gives the Project Owner what D aims for, while honestly
handling the real fact that some jurisdictions don't fully recognize
assignment of certain rights (or of future/unregistered works) — rather
than silently pretending assignment always works everywhere.

## The mandatory flow

Once adopted (see "Adoption" above), `CONTRIBUTOR_RIGHTS_AGREEMENT.md`
Section 27 makes explicit acceptance a precondition for merging any
**external** pull request — not an optional upgrade path alongside
always-available plain Apache-2.0 merging. This was a deliberate reversal
from the v1.0 draft, made because an "optional" framing let a contributor
simply decline the Agreement and still have their code merged under
Apache-2.0 terms, which defeats the entire point of maximizing the
Project Owner's relicensing control — Apache-2.0 Section 5's own inbound
licensing continues to apply to every contribution regardless (it isn't
possible to opt out of that), but it is no longer treated as *sufficient
on its own* for an external PR to be merged.

The maintainer's own commits, and any contribution the maintainer
explicitly authorizes someone to make directly to a branch in this
repository (not a fork), are not subject to this requirement — the
Agreement only ever needs to bind a rightsholder other than the Project
Owner. The narrow written-waiver exception in Section 27, item 6 exists
for genuine one-off cases (e.g. a tiny, obviously public-domain typo fix)
and is deliberately not meant to become the default path.

## Non-retroactivity and PRs #68–#71

Adoption does **not** retroactively change the terms any pre-adoption
submission was made under. Concretely, for PRs #68–#71 (all opened before
this Agreement's adoption): merely being SUBMITTED before adoption is not
itself acceptance of anything, under either the pre-adoption or
post-adoption posture — see Agreement Section 27's own closing paragraph.
If the maintainer wants to merge any of #68–#71 after this Agreement is
adopted, the author must obtain a valid Path A or Path B signature
covering the relevant PR(s), and the maintainer must record it `VERIFIED`
in `docs/contributor-agreements.md`, per Section 27, before merging — a
GitHub comment alone is never sufficient, and there is no grandfather
clause that lets an already-open PR merge under an older, weaker posture
just because it predates adoption. See the draft signing-request message
prepared (not sent) for exactly this purpose.

## What this change does NOT do

- It does **not** merge automatically, post any acceptance comment on
  anyone's behalf, or install any third-party GitHub App.
- It does **not** touch runtime code, the product version, or the v1.0.0
  tag/release.
- It does **not** add `contributor-agreement-check.yml` to branch
  protection's required-status-checks list — that remains a separate,
  later, explicit maintainer decision made in repository Settings, outside
  this PR's own file changes.

## Automation recommendation

Two real options were evaluated:

1. **A third-party CLA bot / GitHub App** (e.g. the well-known
   `cla-assistant` class of tools) — lowest maintainer effort, but
   requires installing and trusting a third-party application with
   repository access. **Not installed** by this change.
2. **A small, repo-owned GitHub Actions status check** — reads
   `docs/contributor-agreements.md` for an entry matching the PR author +
   PR number, and reports a real pass/fail result. Lowest permission (no
   write access needed, `contents: read` only), fully auditable (it's just
   a script in this repo), no third-party trust required.

**Chosen: option 2**, implemented as
`.github/workflows/contributor-agreement-check.yml`. Its design, reasoned
through explicitly rather than defaulted to:

- **Real failure, not just reporting.** The workflow `exit 1`s (with a
  `::error::` annotation) unless it finds a matching row whose
  `Verification status` is literally `VERIFIED` (or a matching waiver
  row) for an external PR, instead of unconditionally exiting 0 or
  accepting a weaker signal. It cannot itself verify a signature — that
  happens outside GitHub (Agreement Section 27) — it only checks whether
  the maintainer already recorded the result of doing so. A single
  `VERIFIED` row can cover more than one PR number (Section 27, item 7).
  This makes it something the maintainer *can* later add to branch
  protection's required-status-checks list, if and when they choose to —
  this PR does not make that change itself (see "What this change does
  NOT do").
- **`pull_request`, not `pull_request_target`.** The check needs no
  secrets and runs none of the fork's own code, so it doesn't need
  `pull_request_target`'s elevated permissions — and avoiding that trigger
  avoids its well-known risk class (a fork PR modifying workflow-triggered
  code that then runs with base-repo secrets).
- **The registry file is read from the base branch only**, via a
  dedicated `actions/checkout` step pinned to
  `github.event.pull_request.base.sha` — never from the PR's own head or
  merge commit. Without this, a malicious fork PR could add a fake
  acceptance row to its own copy of `docs/contributor-agreements.md` and
  make its own check pass; reading only the base branch's version closes
  that hole.
- **Internal (non-fork) PRs are exempt** — the Agreement only ever
  requires acceptance from a rightsholder other than the Project Owner, so
  a PR opened from a branch inside this repository itself (implicitly
  already trusted with write access) skips the check rather than
  requiring the maintainer to "accept" their own agreement with
  themselves.
- **No third-party GitHub App**, consistent with option 1 being rejected.

This workflow is **not** added to branch protection's required checks by
this PR — it exists, and can genuinely fail, but stays advisory until the
maintainer separately decides to require it.

## Contributor registry

`docs/contributor-agreements.md` — two plain Markdown tables. **Accepted**
records, per validly signed Contributor: GitHub handle, the PR(s) the
signature covers, Agreement version, Agreement commit/blob hash, signing
method (Path A/Path B), verification date (UTC), verification status
(must read `VERIFIED` for the enforcement workflow to treat it as
satisfying the requirement — a `PENDING` row is a legitimate way to track
a signature in progress without it passing the check early), and an
internal reference ID pointing at the Project Owner's own private,
secure document store. **Maintainer-recorded waivers** records the narrow
Section 27 item 6 exception: handle, PR, reason, date, and who waived it.

**The actual signed agreement (electronic-signature envelope or
wet-ink scan) is never stored in this file or anywhere in this public
repository** — see Agreement Section 27, "Signed-document storage," for
where it must be kept instead. No legal names, no signature images, no
home address, no national ID, no private email, and no e-signature
certificate material belong in this table — only the metadata above, most
of which (GitHub handle, PR numbers) is already public. If a real CLA
service is ever needed at larger scale, migrate to one rather than
extending this table indefinitely.
