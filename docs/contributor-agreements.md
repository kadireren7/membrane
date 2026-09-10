# Contributor agreement acceptance registry

Records who has explicitly accepted the
[MEMBRANE Contributor Rights Agreement](../CONTRIBUTOR_RIGHTS_AGREEMENT.md)
for which pull request, and when. Only entries added through the real
acceptance procedure in that Agreement's own Section 27 belong here — never
a contributor who merely opened a PR. This file is read directly by
`.github/workflows/contributor-agreement-check.yml` (from the base branch
only — see that file's own comments) — keep both tables' column order and
the `#<number>` format in the PR column intact, since the check's lookup
depends on it.

Only public information is stored here: GitHub handle, PR number, agreement
version and commit/blob hash, a link to the contributor's own public
acceptance comment, and the acceptance timestamp. No legal names,
signatures, or other personal data. If a future need arises for signed
legal names (e.g. after incorporating a company or receiving substantial
corporate contributions), migrate to a proper CLA service instead of
extending this table.

Every field below is filled in for a valid entry — an incomplete row is not
a valid recorded acceptance. `Agreement commit/blob hash` is the exact
`CONTRIBUTOR_RIGHTS_AGREEMENT.md` commit hash the Contributor's own
acceptance comment named (see Agreement Section 27, item 2), recorded here
directly so this table is self-contained evidence and doesn't require
dereferencing the linked comment to know which version was accepted.
`Timestamp` is the acceptance comment's own GitHub-recorded creation time
(UTC), which — together with the comment URL and the platform's
authentication of the commenting account — is the practical evidentiary
basis for acceptance; see `CONTRIBUTOR_RIGHTS_AGREEMENT.md` Section 26,
Part B for the open question of whether this satisfies Turkish law's
formal "written form" requirement on its own.

## Accepted

| GitHub handle | PR | Agreement version | Agreement commit/blob hash | Acceptance comment URL | Timestamp (UTC) |
|---|---|---|---|---|---|
| _(none yet)_ | | | | | |

## Maintainer-recorded waivers

Per Agreement Section 27, item 6: each row here must correspond to a
separate, explicit, written maintainer decision to waive the acceptance
requirement for one specific PR — never a routine or implied exception. The
reason column is mandatory and must not be left blank.

| GitHub handle | PR | Reason | Date | Waived by |
|---|---|---|---|---|
| _(none yet)_ | | | | |
