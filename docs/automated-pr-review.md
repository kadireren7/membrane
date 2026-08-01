# Automated PR review and code scanning

MEMBRANE uses exactly two automated review systems, added deliberately as
a pair and nothing more — no other AI review bot, lint bot, or automated
dependency bot is configured. This document explains what each does, how
it's scoped, and how a human should treat its output.

## The two systems

### 1. CodeRabbit — advisory AI code review

Configured in [`.coderabbit.yaml`](../.coderabbit.yaml) (repo root).
Posts a PR summary, a walkthrough of the change, and line-by-line review
comments on the diff. **Advisory only**:

- No auto-approve. CodeRabbit never posts a GitHub "Approve" review.
- No auto-merge. CodeRabbit has no merge capability configured, and is
  not granted one anywhere in this repository's settings.
- No `request_changes_workflow` — findings never take the form of a
  formal blocking "Request changes" GitHub review state.
- No commit-status failures (`fail_commit_status: false`) — CodeRabbit's
  own activity can never become a required/blocking GitHub check by way
  of a failing status.
- No autofix / no direct code writes. All of CodeRabbit's own
  code-generating "finishing touches" (autofix, CI-failure autofix,
  merge-conflict autofix, docstring generation, unit-test generation) are
  explicitly disabled in the config. It comments; it does not push
  commits.
- Scoped to the current PR's diff only, with `path_filters` de-prioritizing
  generated/vendored/large-data paths (see `.coderabbit.yaml` for the
  full list: `third_party/**`, build output, model weight file
  extensions, committed benchmark CSV/JSON, the paper's built PDF and
  LaTeX byproducts, and raw experiment result artifacts) and
  `path_instructions` carrying this project's own review priorities (see
  below).
- **Installation status**: the CodeRabbit GitHub App is **not yet
  installed** on `kadireren7/membrane` as of this document being written
  (verified: no webhook, no prior CodeRabbit comment on any existing
  PR). Installing it requires Kadir to authorize it through the GitHub
  UI/marketplace — that step could not be performed as part of preparing
  this configuration. `.coderabbit.yaml` will take effect automatically
  once the app is installed; no further repository change is needed.

### 2. GitHub CodeQL — static security/quality analysis

Configured in
[`.github/workflows/codeql.yml`](../.github/workflows/codeql.yml).
Compiles this project's own C/C++ sources (`c-cpp`, same Debug
configuration as `ci.yml`'s own "Debug" job, `MEMBRANE_ENABLE_LLAMA`
left at its default OFF so `third_party/llama.cpp` is neither checked out
nor compiled here) and runs GitHub's default CodeQL query suite against
the result. Runs on every PR into `main`, every push to `main`, weekly
(Monday 05:00 UTC), and on manual dispatch.

**CodeQL is a security/static-analysis check, not a test suite, and does
not replace or overlap with this project's own `Debug`, `ASan+UBSan`, and
`ThreadSanitizer` `ctest` runs in `ci.yml`.** Those find runtime
correctness/memory/concurrency bugs by actually executing the test suite;
CodeQL finds a structurally different class of issue (pattern-based
security/quality queries over the compiled source, no test execution at
all). CodeQL's build step compiles the code and stops there — it never
runs `ctest`, never runs any `tools/membrane-*` simulator or benchmark, never
runs the 128K-context × 512-concurrency sweep, and never downloads a
model file.

GitHub's own **code-scanning "default setup"** was checked before adding
this workflow (`GET /repos/kadireren7/membrane/code-scanning/default-setup`
→ `"state":"not-configured"`) — it was not already scanning this
repository, so this advanced/custom workflow does not duplicate an
existing scan. If default setup is ever turned on for this repository in
the GitHub UI in the future, this workflow file should be removed first
(GitHub does not support running both for the same language on the same
repository).

**Permissions** (`codeql.yml`): `contents: read`, `security-events:
write`. Nothing else — no `packages`, no `actions`, no write access
beyond posting scan results.

## Bot comments do not replace human decision

Neither system can approve, merge, or push to this repository. A
CodeRabbit comment or a CodeQL alert is an input to review, not a
decision. The person merging the PR is always the one deciding whether a
finding is real and whether it's been addressed.

## Handling false positives

- **CodeRabbit**: reply directly in the review thread explaining why the
  finding doesn't apply (e.g., a flagged "exact" claim that genuinely is
  differential-tested exact, not loose wording). No special dismissal
  mechanism is required — an explicit, reasoned reply in the thread is
  the record.
- **CodeQL**: use GitHub's own alert-dismissal UI (Security → Code
  scanning alerts) with a reason (`false positive`, `won't fix`, `used in
  tests`, as appropriate) and a one-line justification. Dismissed alerts
  stay visible in the alert history — nothing is silently deleted.

## Which checks are merge blockers

| Check | Blocking? |
|---|---|
| `build-and-test (Debug)` (ci.yml) | **Required** (recommended branch protection, see below — not yet applied) |
| `build-and-test (ASan)` (ci.yml) | **Required** (recommended, not yet applied) |
| `thread-sanitizer` (ci.yml) | **Required** (recommended, not yet applied) |
| CodeQL (`codeql.yml`) | **Required**, but only after its exact check name is confirmed from a real run and added to branch protection (see below) — not yet applied |
| CodeRabbit | **Never required.** Advisory only, by design (see above). |
| Paper build (`paper.yml`) | Not applicable to most PRs (path-filtered to `paper/**`); not a proposed required check for `main` in general. |

No bot — CodeRabbit or CodeQL — has been granted auto-merge authority
anywhere in this repository.

## Branch protection: recommended, not yet applied

See this document's own "Recommended branch protection for `main`"
section for the exact rule set prepared for `main`, including the real,
verified check names for the three existing CI jobs. It has **not** been
applied via the GitHub API as part of this change — enabling required
status checks changes how every future push/PR to `main` behaves for the
whole repository, which is a decision left to Kadir to apply
consciously rather than as a side effect of adding these two review
tools. See that section for the exact settings to enable, and the one
missing manual step (installing the CodeRabbit App) that should happen
before or alongside it.

## Privacy / secrets

- CodeRabbit and CodeQL both operate only on this repository's own
  public source code and PR diffs — neither is given access to secrets,
  credentials, or any private data by this configuration.
- No new repository secret was added as part of this change.
- Keep both integrations' granted permissions at the minimum GitHub
  offers: CodeQL's workflow permissions are `contents: read` +
  `security-events: write` only (see above); when installing the
  CodeRabbit GitHub App, grant it only the repository access it actually
  needs (this repository, not an org-wide "all repositories" grant,
  unless Kadir specifically wants that for other repos too) and review
  its requested permission scope in the GitHub App install screen before
  accepting.

## Recommended branch protection for `main`

Prepared for Kadir to review and apply (or ask this session to apply,
since admin access to do so was confirmed available while preparing this
document — not exercised, per the reasoning above).

**Verified real check names** (from `GET
/repos/kadireren7/membrane/commits/{main-sha}/check-runs` against a real,
completed run):

- `build-and-test (Debug)`
- `build-and-test (ASan)`
- `thread-sanitizer`

Recommended settings:

- Require a pull request before merging.
- Required status checks: `build-and-test (Debug)`, `build-and-test
  (ASan)`, `thread-sanitizer`.
- Require branches to be up to date before merging.
- Require conversation resolution before merging.
- Block force pushes to `main`.
- Block branch deletion for `main`.

**Deliberately not required, for now** (per this change's own scope):

- CodeQL as a required check — add its exact check name (visible only
  after `codeql.yml` runs for real at least once) once confirmed; adding
  an unverified/guessed check name to branch protection can silently
  block all merges if the name is wrong.
- CodeRabbit approval — CodeRabbit is an external service with no
  formal "approve" state configured (see above); making it required
  would turn an advisory tool into a hard merge lock outside this
  project's own control.
- A minimum human-reviewer count, signed commits, auto-merge, and linear
  history — none of these were asked for, and none is added here.

## Third-party app permission hygiene

- Grant the CodeRabbit GitHub App the narrowest repository scope GitHub's
  install flow offers (this repository specifically).
- Periodically review both integrations under Settings → Integrations /
  Settings → Code security → Code scanning to confirm no permission
  scope crept beyond what's documented here.
