# Contributor agreement acceptance registry

Records which Contributors have a **validly signed** copy of the
[MEMBRANE Contributor Rights Agreement](../CONTRIBUTOR_RIGHTS_AGREEMENT.md)
on file — via Path A (qualified/secure electronic signature) or Path B
(wet-ink signature), per that Agreement's own Section 27 — and which pull
requests each signature covers. **A GitHub PR comment alone is never
sufficient and must never be recorded as VERIFIED here** — see Agreement
Section 27 for why v3.0 removed the earlier, comment-only mechanism.

This file is read directly by `.github/workflows/contributor-agreement-check.yml`
(from the base branch only — see that file's own comments). Keep the table
headers, column order, and the `## Accepted` / `## Maintainer-recorded
waivers` section headings intact, since the check's lookup depends on them.

**The actual signed agreement is never stored in this file or anywhere in
this public repository.** The electronic-signature envelope (Path A) or
the scanned/physical wet-ink document (Path B) is retained privately and
securely by the Project Owner — see Agreement Section 27, "Signed-document
storage." Only the metadata below, all of it either already public
(GitHub handle, PR numbers) or non-sensitive (version, hash, method,
dates, status, an internal reference ID), is recorded here. Never record:
a signature image, a signed PDF/scan itself, a home address, a national ID
number, a private email address, e-signature certificate material, or any
other personal data beyond what's listed.

Every field is filled in for a valid entry — an incomplete row is not a
valid VERIFIED entry, and the enforcement workflow does not treat it as
one. `Covered PR(s)` may list more than one PR for a single signature
(e.g. `#68, #69, #70, #71`) when one signed agreement explicitly names all
of them (Agreement Section 27, item 7). `Verification status` must be the
literal word `VERIFIED` for the workflow to treat the row as satisfying
the requirement — any other value (e.g. `PENDING`, used while a signature
is in progress but not yet confirmed) does not pass the check. `Reference
ID` is an internal pointer (e.g. a filename or case ID in the Project
Owner's private, secure document store) — not a public URL, and not the
document itself.

## Accepted

| GitHub handle | Covered PR(s) | Agreement version | Agreement commit/blob hash | Signing method | Verification date (UTC) | Verification status | Reference ID |
|---|---|---|---|---|---|---|---|
| _(none yet)_ | | | | | | | |

## Maintainer-recorded waivers

Per Agreement Section 27, item 6: each row here must correspond to a
separate, explicit, written maintainer decision to waive the signature
requirement for one specific PR — never a routine or implied exception.
The reason column is mandatory and must not be left blank.

| GitHub handle | PR | Reason | Date | Waived by |
|---|---|---|---|---|
| _(none yet)_ | | | | |
