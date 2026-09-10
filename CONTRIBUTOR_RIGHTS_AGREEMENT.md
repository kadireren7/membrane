# MEMBRANE Contributor Rights Agreement

**Version:** 2.0 (supersedes the 1.0 draft; see `docs/contributor-governance.md`
for the revision history and why each change was made)
**Status:** DRAFT — pending maintainer adoption and the professional legal
review identified in Section 26. This document binds no one on its own;
see Section 27 for exactly what makes it apply to a given Contribution.

**Read this first, in plain language:** the Apache License 2.0 that governs
this repository already gives the MEMBRANE project a real, broad, perpetual,
irrevocable license to anything intentionally contributed (License Section
5) — it does **not** transfer copyright ownership, and it does not, by
itself, satisfy this Agreement's own requirements. **Once this Agreement is
adopted by the maintainer, it is the mandatory path for external code
contributions to this repository** (Section 27) — plain Apache-2.0
inbound licensing is no longer offered as an alternative for ordinary
external contributions, except through the narrow, explicit maintainer
waiver described in that section. Under this Agreement, you assign
copyright ownership of your accepted Contribution (or, wherever assignment
is not legally possible, grant the broadest license the law allows
instead), so the Project can relicense, dual-license, and commercially
exploit it without having to track you down again later. In exchange, this
Agreement is explicit that contributing never creates equity, payment,
partnership, or control rights, and that the Project cannot be forced to
accept a contribution, nor can a contributor force the Project to remove
one it already accepted. If this is not something you are comfortable
with, do not accept it — you are always free not to contribute.

---

## 1. Definitions

- **"Project"** means the MEMBRANE open-source project, its source code
  repositories (including `kadireren7/membrane` and
  `kadireren7/membrane-research`), and any successor repositories the
  Project Owner designates.
- **"Project Owner"** means Kadir Eren Altıntaş, or a successor entity
  (including a future company) to whom the Project Owner has transferred
  the rights described in this Agreement, as Section 8 makes explicitly
  possible.
- **"Contributor"** means the individual or, where Section 21/22 applies,
  the entity, submitting a Contribution under this Agreement.
- **"Contribution"** means any original work of authorship, including
  modifications or additions to existing work, that a Contributor submits
  for inclusion in, or documentation of, the Project. "Submits" includes,
  but is not limited to, a pull request, patch, issue, or any other form
  of electronic, written, or verbal communication sent to the Project or
  its maintainers, but excludes communication conspicuously marked or
  otherwise designated in writing by the Contributor as "Not a
  Contribution."
- **"Pre-Existing Material"** means material the Contributor created or
  held rights in before submitting it as part of a Contribution, and that
  is separable from the Contribution itself (see Section 19).
- **SUBMITTED.** A Contribution is **SUBMITTED** the moment it exists as an
  identifiable pull request, patch, or equivalent artifact on the
  Project's own repository (e.g., "PR #NN is open"). Being SUBMITTED, by
  itself, grants the Project Owner no rights beyond what Apache License
  2.0 Section 5 already provides, and creates no obligation on either
  side.
- **AGREEMENT ACCEPTED.** A Contribution's terms become **AGREEMENT
  ACCEPTED** only once the specific procedure in Section 27 has actually
  happened for that Contribution: a real, affirmative acceptance comment,
  naming this Agreement's version and the exact PR number, posted by a
  person or entity who is themselves a rightsholder in the Contribution
  (or properly authorized under Section 21–23), and recorded per Section
  27. This is the point at which the rights described in Sections 4–6
  become effective (see Section 28) — it does not require the PR to be
  merged, and does not, by itself, obligate the Project Owner to merge it.
- **PROJECT ACCEPTED / MERGED.** A Contribution is **PROJECT ACCEPTED**
  (used interchangeably with **MERGED** in this Agreement) when a
  maintainer actually merges it into `main` or another designated branch
  of the Project. This is a separate, later, and entirely discretionary
  act by the Project Owner (Section 24) — AGREEMENT ACCEPTED does not
  entitle a Contributor to have their Contribution merged, and MERGED
  status does not, on its own, create or expand any rights grant beyond
  what AGREEMENT ACCEPTED already established (Section 28).

These three states are independent and can occur in any order relative to
ordinary review activity (comments, CI runs, requested changes) — none of
SUBMITTED, a green CI run, or a maintainer's code-review approval is
itself AGREEMENT ACCEPTED or PROJECT ACCEPTED.

## 2. Scope of Contribution

This Agreement applies **only** to a specific Contribution once that
Contribution reaches AGREEMENT ACCEPTED status under Section 27. It does
not apply to:

- any other code, document, or material the Contributor has written or
  will write, inside or outside the Project;
- a pull request or other submission that was never AGREEMENT ACCEPTED;
- work the Contributor does for the Project under a **separate**, signed,
  written agreement that expressly says it supersedes this one (e.g. an
  employment or paid-contractor agreement).

## 3. Contributor Representations

By reaching AGREEMENT ACCEPTED status for a given Contribution, the
Contributor represents that:

- the Contributor created the Contribution, or is legally authorized to
  submit it (see Sections 21–23 for employer-owned work and multiple
  rightsholders);
- to the Contributor's actual knowledge, the Contribution does not
  knowingly infringe or misappropriate any third party's copyright,
  patent, trade secret, or other proprietary right, except for any
  third-party material disclosed under Section 20;
- if the Contributor's employer has rights to intellectual property the
  Contributor creates, the Contributor has either received permission to
  make the Contribution on the employer's behalf, or the employer has
  waived such rights, or the Contribution falls outside the scope of the
  Contributor's employment;
- the Contributor has the legal authority to enter into this Agreement.

This is a representation of the Contributor's actual, reasonable knowledge —
**not** an absolute warranty that no third party could ever assert a claim
(see Section 25).

## 4. Copyright Assignment

To the maximum extent permitted by applicable law, the Contributor hereby
**assigns to the Project Owner** all right, title, and interest, including
all copyright and related rights worldwide, in and to the Contribution.

This assignment is limited strictly to the Contribution as reaching
AGREEMENT ACCEPTED status under Section 27 — it does **not** cover:

- Pre-Existing Material (Section 19);
- any other work of the Contributor not submitted as part of this
  Contribution;
- rights that are, under the law of the relevant jurisdiction, legally
  incapable of assignment (see Section 5 and Section 7 for what happens to
  those instead).

Where assignment takes legal effect, the Project Owner becomes the
copyright owner of the Contribution and may exercise, license, sublicense,
relicense, or transfer it as described in Section 8 and Section 13, without
further consent from or payment to the Contributor, except as a separate
written agreement may otherwise provide.

## 5. Fallback License if Assignment Is Ineffective

Some jurisdictions restrict or do not recognize assignment of copyright (or
of certain categories of rights, such as future or unregistered works, or
impose formal requirements — see Section 26's own disclosure about Turkish
law's written-form and enumeration expectations for economic-rights
transfers). **Where, and only to the extent that, assignment under Section
4 is legally ineffective for a given Contribution or a given right**, the
Contributor instead grants the Project Owner, automatically and without
further action required, a:

- perpetual,
- worldwide,
- irrevocable (subject to Section 17's own limits),
- royalty-free,
- fully-paid-up,
- transferable,
- sublicensable,
- relicensable,
- non-exclusive

license to reproduce, prepare derivative works of, publicly display,
publicly perform, sublicense, relicense, and distribute the Contribution
and such derivative works, in source or object form, for any purpose,
including commercial purposes, under any license terms the Project Owner
chooses.

This fallback is intended to give the Project Owner, as nearly as legally
possible, the same practical rights an effective assignment would have
given — it is not a weaker "just in case" clause layered on top of a real
assignment; it is what governs whenever the assignment itself cannot.

## 6. Patent License

Subject to the terms of this Agreement, the Contributor grants the Project
Owner, and each recipient of software distributed by the Project, a
perpetual, worldwide, non-exclusive, royalty-free, irrevocable (except as
stated below) patent license to make, have made, use, offer to sell, sell,
import, and otherwise transfer the Contribution, alone or combined with the
Project, where such license applies only to those patent claims licensable
by the Contributor that are necessarily infringed by the Contributor's
Contribution alone, or by combination of the Contribution with the Project
to which it was submitted.

This is deliberately scoped the same way the Apache License 2.0's own
Section 3 patent grant is scoped — it does **not** require the Contributor
to assign or license any patent unrelated to the actual Contribution, and
it does not cover claims that would only be infringed by some other,
unrelated modification.

**Defensive termination:** if the Contributor (or an entity on the
Contributor's behalf) initiates patent litigation alleging that the
Project, or a Contribution incorporated within it, infringes a patent, the
patent license granted to that Contributor under this Section terminates
as of the date such litigation is filed — mirroring Apache License 2.0
Section 3's own defensive-termination mechanism.

**Patent assignment is not required** by this Agreement. Requiring patent
assignment was evaluated and rejected as excessive for an open-source
contribution model — it would require Contributors to give up rights in
inventions far broader than what they actually contributed to MEMBRANE,
for no proportionate benefit.

## 7. Moral Rights

**This section leads with the non-assert commitment, not a waiver, because
under at least one legal system realistically relevant to this Agreement
(Turkey — see Section 26), a contractual waiver of moral rights is treated
as void as a matter of public policy, regardless of what any contract
says.** A "waiver" clause presented as the primary mechanism would be
misleading about its own real effect there.

Accordingly: the Contributor agrees **not to assert** moral rights (such as
the right of attribution and the right of integrity) in the Contribution
against the Project Owner, its successors, licensees, or sublicensees, to
the maximum extent such an agreement not to assert is legally permitted in
the relevant jurisdiction.

**Separately, and only where the applicable law actually permits a
further, effective waiver of moral rights** (this varies by jurisdiction
and is not true everywhere, including not being true under Turkish law),
the Contributor additionally **waives** those moral rights, to that
jurisdiction's own maximum legal extent, as a stronger measure on top of
the non-assert commitment above.

This Agreement does **not** claim that moral rights can be eliminated
everywhere, and does not claim the non-assert commitment itself survives
every jurisdiction's own mandatory law (see Section 17's own limits) — only
that the Contributor commits not to use moral rights to block or interfere
with the uses this Agreement otherwise authorizes, to the fullest extent
each Contributor is legally able to make that commitment.

## 8. Right to Sublicense and Relicense

The Project Owner may, without further consent from or payment to the
Contributor:

- distribute the Contribution (as part of the Project or standalone) under
  the Apache License 2.0, a different open-source license, or a
  proprietary/commercial license;
- combine the Contribution with other code under different license terms;
- offer the Project, including the Contribution, under more than one
  license simultaneously (dual- or multi-licensing);
- modify, adapt, and create derivative works of the Contribution;
- **assign or transfer the rights obtained under this Agreement (whether
  held as owned copyright under Section 4, or as a license under Section
  5 or Section 6) to a successor entity**, including a future company the
  Project Owner incorporates to hold or commercialize the Project. Where
  Section 4's assignment took effect, this is simply the Project Owner
  exercising the ordinary right any copyright owner has to transfer
  owned property; where Section 5 or Section 6's license applies instead,
  this Agreement makes that license explicitly transferable so the same
  outcome is available either way. No further acceptance or notice from
  the Contributor is required for such a transfer.

## 9. Commercial and Proprietary Use

The Contribution may be included in commercial distributions, hosted
services, and proprietary products or services offered by the Project
Owner or by a party the Project Owner sublicenses to, and may be
sublicensed to the Project Owner's own customers or partners, all without
requiring further approval from, or payment to, the Contributor, except as
a separate written agreement may otherwise provide.

## 10. No Compensation or Royalties

Reaching AGREEMENT ACCEPTED or PROJECT ACCEPTED status does not entitle
the Contributor to any payment, royalty, revenue share, or other
compensation, unless a **separate, explicit, written** agreement between
the Contributor and the Project Owner says otherwise.

## 11. No Partnership

Nothing in this Agreement, and nothing about reaching AGREEMENT ACCEPTED
or PROJECT ACCEPTED status, creates a partnership between the Contributor
and the Project Owner.

## 12. No Joint Venture

Nothing in this Agreement, and nothing about reaching AGREEMENT ACCEPTED
or PROJECT ACCEPTED status, creates a joint venture between the
Contributor and the Project Owner.

## 13. No Employment

Nothing in this Agreement, and nothing about reaching AGREEMENT ACCEPTED
or PROJECT ACCEPTED status, creates an employment, agency, or contractor
relationship between the Contributor and the Project Owner, and creates no
fiduciary duty in either direction.

## 14. No Equity

Reaching AGREEMENT ACCEPTED or PROJECT ACCEPTED status does not entitle
the Contributor to equity, shares, options, or any other ownership
interest in the Project or in any entity that owns, operates, or
commercializes it.

## 15. No Revenue Share

Reaching AGREEMENT ACCEPTED or PROJECT ACCEPTED status does not entitle
the Contributor to a share of revenue, profit, or proceeds generated by
the Project or by any commercial use of it, now or in the future, unless a
separate written agreement explicitly says otherwise.

## 16. No Governance or Ownership Rights

Reaching AGREEMENT ACCEPTED or PROJECT ACCEPTED status does not grant the
Contributor:

- governance or voting rights over the Project;
- repository administration or write access (that is granted, separately
  and at the maintainer's own discretion, through normal GitHub
  permissions, and can be revoked the same way);
- veto power over any decision, release, or relicensing choice;
- any entitlement to future proceeds from the Project or from any company
  built around it.

## 17. Irrevocability of Granted Rights

Before AGREEMENT ACCEPTED status, the Project Owner may close, reject, or
ignore the pull request or submission for any reason or no reason, and
owes the Contributor no explanation.

**After** a Contribution has reached AGREEMENT ACCEPTED status, the
copyright assignment (Section 4) and/or fallback license (Section 5), and
the patent license (Section 6), remain in effect and are not revocable
merely because the Contributor later changes their mind, asks for the
Contribution to be removed, or stops supporting the Project — **subject
always to any mandatory right under applicable law that cannot be waived
or overridden by contract** (for example, moral rights under Turkish law,
Section 7, or certain consumer-protection regimes elsewhere). This
Agreement does not, and cannot, claim to override such mandatory rights;
it claims only that ordinary, voluntary "I changed my mind" withdrawal is
not, by itself, grounds for revocation.

## 18. No Right to Demand Removal After Acceptance

Consistent with Section 17, once a Contribution has reached AGREEMENT
ACCEPTED status, the Contributor does not have a right to demand that the
Project Owner remove, stop using, or stop distributing that Contribution
(if merged), or any derivative work built on it, solely because the
Contributor requests it. See Section 28 for the different, narrower case
of a Contribution that reached AGREEMENT ACCEPTED status but was never
merged at all.

## 19. Contributor Retains Rights to Pre-Existing Material

Where a Contribution incorporates Pre-Existing Material that is genuinely
separable from the new, original work submitted, this Agreement assigns
and licenses only the newly authored parts. The Contributor does not, by
reaching AGREEMENT ACCEPTED status, transfer ownership of separable
Pre-Existing Material — only a Contribution-scoped license to use that
Pre-Existing Material as incorporated (to the extent the Contributor is
able to grant one).

## 20. Third-Party and AI-Assisted Material Disclosure

This section is deliberately **operational** — it asks the Contributor to
disclose what they actually know, not to make claims they cannot verify.

The Contributor must disclose, as part of the submission (e.g. in the pull
request description):

- any **known** third-party source code, text, or other material copied or
  adapted into the Contribution, including its origin and license where
  known;
- whether AI-assisted or AI-generated output was used **materially** in
  producing the Contribution (i.e., beyond routine editor autocomplete —
  a substantial code block, algorithm, explanatory text, or similar
  drafted with material AI assistance).

The Contributor is **not** required to disclose, and is not expected to
know or investigate, what training data any AI tool they used was itself
trained on, or to make any representation about a third party's model
training practices — that information is generally not available to a
tool's own users and this Agreement does not ask for an unverifiable
claim about it. Undisclosed known third-party material (the first bullet
above) is not covered by the representations in Section 3.

## 21. Employer/Company Authorization

If the Contributor created the Contribution within the scope of their
employment, or an employer or company otherwise has rights in it, the
Contributor must either:

- obtain the employer's written permission to submit the Contribution
  under this Agreement before doing so and be able to produce evidence of
  that permission if the Project Owner reasonably asks for it, or
- have the employer itself separately authorize the Contribution as
  described in Section 22, or
- confirm, in the submission, that the Contribution falls outside the
  scope of their employment and no employer holds any right in it.

This Agreement does **not** assume that an employee personally owns work
they produce during their employment — the default assumption is the
opposite, and it is the Contributor's responsibility to confirm which
applies.

## 22. Corporate Contributions

For an individual Contributor's ordinary, incidental employer-owned
contribution, the written-authorization mechanism in Section 21 is the
expected path — it does not require a separate corporate document for
every PR.

**For a substantial contribution made on behalf of a company** (e.g. a
company directing an employee to contribute as part of their job, or a
company wishing to contribute a significant body of work), the Project
Owner may instead require a **separate Corporate Contributor Agreement**,
signed by a person with actual authority to bind that company. No such
agreement exists as of this version of this document — it is prepared
only if and when a real, substantial corporate contribution actually
appears, rather than being drafted speculatively in advance. Until then,
Section 21's authorization mechanism governs employer-owned individual
contributions.

## 23. Multiple Contributors / Co-Authors

Where a Contribution incorporates copyrightable material from more than
one individual or entity (for example: multiple commit authors on the same
pull request, paired-programming co-authorship, or several people's
separable work combined into one submission), **each** person or entity
holding rights in a separable portion of the Contribution must either:

- independently reach AGREEMENT ACCEPTED status for their own portion
  (posting their own acceptance comment, per Section 27), or
- be properly authorized to act on behalf of that other rightsholder (for
  example, under Section 21's employer-authorization mechanism, or a
  co-author's own explicit, written permission naming the accepting party
  as their agent for this specific purpose).

**One Contributor's acceptance does not, by itself, assign or license
another rightsholder's own contribution.** A pull request with commits
from more than one distinct GitHub identity is treated, for the purposes
of this Agreement, as containing more than one Contribution-portion until
each identified rightsholder has independently reached AGREEMENT ACCEPTED
status (or been properly authorized) for their own portion.

## 24. Maintainer Has No Obligation to Accept

Nothing in this Agreement obligates the Project Owner to merge any
Contribution, to review it within any particular time, or to give reasons
for rejecting it — reaching AGREEMENT ACCEPTED status is not a promise of
PROJECT ACCEPTED status.

## 25. Disclaimer / No Warranties

THE CONTRIBUTION IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, TITLE, AND NON-INFRINGEMENT. This
Agreement does not guarantee that no third party could ever bring a claim
related to the Contribution, and does not guarantee that these terms are
fully enforceable in every jurisdiction (see Section 26).

## 26. Governing Law

**This Agreement does not claim to guarantee enforceability in every
country, and the wording below is a maintainer-reviewed draft, not a
declaration that every open legal question has been resolved by counsel.**
Sections A and B below are kept deliberately separate, as requested: A is
the concrete recommended contract text; B is the reasoning and the
specific items that still need a licensed lawyer's review before this
Agreement should be treated as final.

### A. Recommended governing-law wording

> This Agreement is governed by the laws of the Republic of Türkiye,
> without regard to its conflict-of-laws principles, except that nothing
> in this Section deprives a Contributor of any protection they are
> entitled to under a mandatory provision of the law of their own country
> of residence that cannot be excluded by agreement. The courts of
> [maintainer's home city/province, Türkiye — to be filled in by the
> Project Owner] have non-exclusive jurisdiction over any dispute arising
> from this Agreement, without prejudice to the Project Owner's own right
> to seek relief in any other court of competent jurisdiction where a
> Contributor or infringing use is actually located.

### B. Reasoning and open items for counsel

- **Why the maintainer's own jurisdiction (Türkiye):** this is the
  Project Owner's own country of residence, and choosing one's own home
  jurisdiction is standard, well-established practice for individual/solo
  open-source maintainers running their own CLA — it is where the
  Project Owner can actually access courts and counsel without added
  cost, and it avoids the (worse, for a solo maintainer) alternative of
  picking an unfamiliar foreign jurisdiction purely because it is
  commercially fashionable.
- **Real, sourced finding this draft specifically accounts for:**
  research into Turkish copyright law (Law No. 5846 on Intellectual and
  Artistic Works, "FSEK") indicates that **the transfer, restriction, or
  waiver of moral rights is treated as null and void as a matter of
  Turkish public policy**, regardless of contract language — this is
  exactly why Section 7 above leads with a non-assert commitment rather
  than presenting waiver as the primary mechanism. Research also
  indicates FSEK Article 51 voids contracts purporting to transfer
  economic rights that are granted or created by **future** legislation
  (i.e., rights that do not yet exist) — a narrower point than ordinary
  assignment of currently-existing economic rights, which Turkish law
  does generally permit through licensing/transfer agreements.
- **What still genuinely needs a licensed Turkish IP/contract lawyer**,
  not general research, before this Agreement is treated as final:
  - whether FSEK imposes a **written-form / specific-enumeration**
    requirement on transfers of economic rights (a common civil-law
    pattern — requiring a transfer to name which specific economic
    rights, e.g. reproduction, adaptation, distribution, are being
    transferred, rather than a single blanket "all rights" clause being
    automatically sufficient) that Section 4's own assignment language
    should be adapted to satisfy explicitly;
  - whether software specifically receives any different treatment under
    FSEK compared to other categories of protected work;
  - the practical cross-border enforceability of a Turkish-law judgment
    against a Contributor resident in a different country, and whether an
    arbitration clause (instead of, or alongside, the courts named above)
    would serve the Project Owner better for a genuinely international
    contributor base;
  - confirming the exact city/province for the jurisdiction clause and
    reviewing the "mandatory law of the Contributor's own residence"
    carve-out for correctness under Turkish conflict-of-laws rules.
- **This draft should not be read as "final and enforceable everywhere."**
  It is a real, reasoned starting point for the Project Owner's own
  review, and for a licensed lawyer's review before the Agreement is
  adopted as binding, consistent with the other lawyer-review triggers
  already named in this document (incorporating a company, receiving
  investment, meaningful commercial licensing, or accepting a substantial
  corporate contribution — see Section 22).

## 27. Mandatory Acceptance Procedure for External Contributions

**Once the Project Owner adopts this Agreement (a decision separate from,
and later than, merging this document into the repository — see `docs/
contributor-governance.md` for how that adoption decision itself is
recorded), it becomes the required path for external code contributions.**
Concretely, from that point on:

1. **No external Contribution may be merged (reach PROJECT ACCEPTED
   status) unless it has first reached AGREEMENT ACCEPTED status** for
   every rightsholder identified in it (Section 23). "External" means
   submitted by anyone other than the Project Owner or someone the
   Project Owner has authorized to commit directly.
2. AGREEMENT ACCEPTED status requires the Contributor to post an
   affirmative, explicit comment on the exact pull request, in
   substantially this form:

   > I have read and agree to the MEMBRANE Contributor Rights Agreement
   > version `<version>` (commit `<git-hash-of-this-file>`) and confirm
   > that my contribution in PR #`<NN>` is submitted under that
   > agreement.

3. The Project Owner (or an authorized maintainer) records that
   acceptance in `docs/contributor-agreements.md` before merging.
4. **None of the following count as acceptance:** silence, merely opening
   a pull request, CI passing, a maintainer's code-review approval,
   merging without the Contributor's own affirmative comment, or
   continuing to contribute after this file was added to the repository.
5. **If the Contributor does not post the acceptance comment, the pull
   request is not merged**, regardless of its technical quality, until
   either the Contributor accepts or the maintainer records an explicit
   waiver under item 6 below.
6. **The only exception** is a documented, case-by-case decision by the
   Project Owner to waive the requirement for a specific Contribution —
   this must itself be a separate, explicit, written decision (recorded
   in `docs/contributor-agreements.md` as a "WAIVED" entry naming the PR
   and the reason), never a silent or implied exception, and never the
   Project's default posture for external contributions.

A Contribution that reached AGREEMENT ACCEPTED status **before** this
Agreement's adoption (there are none as of this version) would be governed
by whatever terms actually applied when it was accepted. A Contribution
merely SUBMITTED (e.g. an open pull request) before adoption, but never
separately AGREEMENT ACCEPTED, is governed by this Section going forward
like any other external contribution — submission before adoption is not
itself acceptance of anything, under either the old or the new posture.

## 28. Timing and Scope of the Rights Grant

The rights described in Sections 4–6 become effective **no later than**
the moment a Contribution reaches AGREEMENT ACCEPTED status for the
specific PR identified in the acceptance comment, and remain effective if
that Contribution is subsequently merged (PROJECT ACCEPTED).

To remove ambiguity about specific real situations:

- **Contributor accepts, but the PR is later rejected/closed without
  merging.** The rights grant for that specific, accepted Contribution
  remains legally valid (it was not conditioned on merging), but has no
  practical effect, since nothing was incorporated into the Project. As a
  matter of practice — not a legal requirement — the Project Owner will
  normally honor a Contributor's request to disregard/not reuse a
  rejected, unmerged Contribution, since there is nothing to un-merge.
- **Contributor accepts, then pushes further commits to the same PR
  (ordinary iteration, same author, same underlying Contribution).** The
  original acceptance continues to cover the Contribution as it evolves,
  without needing to be re-posted for every push — this is a continuation
  of the same Contribution, not a new one.
- **A different person (a co-author) adds commits to the same PR after
  the original Contributor's acceptance.** This changes the rightsholder
  scope: the new commits are a separate Contribution-portion under
  Section 23, and that person must independently reach AGREEMENT ACCEPTED
  status (or be properly authorized) for their own portion before the PR
  as a whole may be merged.
- **A maintainer edits the Contribution before merging (e.g. fixing a
  small issue found in review).** The maintainer's own edits are the
  Project Owner's own work and require no acceptance from the original
  Contributor — the original Contributor's acceptance still covers their
  own original material as accepted; it is simply combined with the
  maintainer's separately-owned edit.
- **Renewed acceptance is required only when the rightsholder scope of a
  Contribution materially changes** (a new co-author's material is added,
  or a Contribution is substantially replaced with different authorship)
  — not for every trivial push, rebase, or maintainer-requested tweak by
  the same original Contributor.

## 29. Effective Date

This Agreement, as a document, is dated by its own Git commit history. It
has no effect on any Contribution until that Contribution reaches
AGREEMENT ACCEPTED status under Section 27.
