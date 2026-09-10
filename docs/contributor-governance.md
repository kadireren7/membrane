# Contributor rights governance

Why this exists, what changed, and what did not. Read
`CONTRIBUTOR_RIGHTS_AGREEMENT.md` for the actual legal terms — this page is
the design rationale and audit trail behind it, not a restatement of it.

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
  most important fact this audit needs to get right, and the originating
  task explicitly warned against misstating it — confirmed directly from
  the License text, not assumed.

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

## What this change does NOT do

- It does **not** retroactively change the terms PRs #68–#71 were submitted
  under. Those remain governed by plain Apache-2.0 Section 5 unless and
  until their author separately, explicitly accepts this Agreement for
  those specific PRs (`CONTRIBUTOR_RIGHTS_AGREEMENT.md` Section 25).
- It does **not** merge automatically, post any acceptance comment on
  anyone's behalf, or install any third-party GitHub App.
- It does **not** touch runtime code, the product version, or the v1.0.0
  tag/release.

## Automation recommendation (Section 21 of the task)

Two real options were evaluated:

1. **A third-party CLA bot / GitHub App** (e.g. the well-known
   `cla-assistant` class of tools) — lowest maintainer effort, but
   requires installing and trusting a third-party application with
   repository access. **Not installed automatically** by this change, per
   explicit instruction.
2. **A small, repo-owned GitHub Actions status check** — reads
   `docs/contributor-agreements.md` (or a structured file it points to)
   for an entry matching the PR author + a real acceptance-comment link,
   and reports pass/fail as a non-blocking status check. Lowest
   permission (no write access needed, `contents: read` only), fully
   auditable (it's just a script in this repo), no third-party trust
   required.

**Recommendation: option 2**, scaffolded but **not required/blocking**
yet (`.github/workflows/contributor-agreement-check.yml`, informational
only) — the maintainer can promote it to a required check later once
comfortable with the mechanism. This is the lowest-permission trustworthy
option, consistent with the task's own instruction.

## Contributor registry

`docs/contributor-agreements.md` — a plain Markdown table: GitHub handle,
PR number, agreement version, acceptance comment URL, date. No legal
names, no signatures, no personal data beyond the public GitHub handle
already visible on the PR itself. If a real CLA service or signed legal
names are ever needed (e.g. once a company forms), migrate to a proper
CLA service rather than storing that in this repository.
