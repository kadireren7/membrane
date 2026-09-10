# External validation

Mega Phase E, PR E5 (Sections 34-36 of the task). This document states,
honestly, how much of MEMBRANE's real-world readiness rests on
maintainer-only testing versus genuine external validation, and how
issue-driven fixes are prioritized when they exist.

## Status labels

- **VALIDATED_MAINTAINER** — exercised for real by the project's own
  maintainer/session, on a real host or a real CI runner, but not by
  anyone outside the project.
- **VALIDATED_EXTERNAL** — exercised for real by someone who is not the
  maintainer, with real, independently-reproducible evidence (not a
  secondhand report).
- **VALIDATED_CI_REAL_HARDWARE** — a specific sub-case of maintainer
  validation worth naming separately: GitHub-hosted `windows-latest`/
  `macos-14` runners are real, physical/virtualized machines running
  the real OS (never emulation, never cross-compilation) — a passing
  real-generation step there is real hardware evidence, distinct from
  (and stronger than) a plain compile-only check, but it is still the
  project's own CI, not an external party's machine.
- **NOT_VALIDATED** — not attempted, reason disclosed.

## Platform/backend matrix

See `results/v1-external-validation/validation.json` for the full,
current matrix. Summary: Linux CPU/Vulkan are `VALIDATED_MAINTAINER`;
Linux CUDA is real-compiled every PR but never run (no discrete GPU on
any host this project has used — `COMPILE_VALIDATED_CI`, never
overclaimed further); Windows CPU and macOS Metal are each
`VALIDATED_CI_REAL_HARDWARE` (real device enumeration, real download,
real generation, real service-lifecycle, every PR touching that code
since Mega Phase D).

**No platform or backend in this matrix has ever been validated by
anyone other than this project's own maintainer/CI** — there is no
`VALIDATED_EXTERNAL` row yet for platform/backend support itself.

## GitHub issues

Zero open issues on `kadireren7/membrane` as of this phase (`gh issue
list --state open` — the repository API's own `open_issues_count`
field is not a substitute for this, since GitHub counts open pull
requests in that field too). No issue-driven fixes exist to apply for
that reason.

## Real external pull requests (found, not yet merged)

Four real, independent, well-documented pull requests exist from a
real external fork contributor (not the maintainer) as of this phase —
see `results/v1-external-validation/validation.json` for the full
list. Each includes real repro steps, real ASan/UBSan evidence, and
disclosed testing scope/limitations, matching this project's own
evidentiary standard. **These are deliberately left open and unmerged
this phase** — reviewing and (if they hold up) merging real external
contributions is real, valuable work, but it is the maintainer's own
call on timing, made explicitly for this phase: review them after the
rest of Mega Phase E, not folded into it. This is disclosed here as a
real, positive `VALIDATED_EXTERNAL`-track signal (a real outside
contributor found and fixed real bugs, independently) even though the
fixes themselves are not yet part of `main`.

## External user data policy (Section 35)

MEMBRANE collects nothing. Re-confirmed by source audit this phase, not
just repeated from an earlier phase's own claim: no code path in
`tools/membrane`/`tools/membrane-run` transmits prompts, messages,
model paths, or any other user/request data anywhere. The only
outbound network traffic anywhere in the product is the user-initiated,
user-visible model download (`download_manager.cpp`, direct to Hugging
Face) — see `docs/server.md`'s own "Security scope" section, unchanged.

If a future phase ever adds a real external-issue-intake mechanism
(crash reporting, opt-in diagnostics, etc.), the allowed field set is
fixed in advance by Section 35 of the original task and restated here
so it is never silently expanded: OS, CPU/GPU, backend, MEMBRANE
version, model ID, failure category, error code. Never prompts,
messages, private file paths, or any other personal data. No such
mechanism exists today — this is a boundary for if one is ever built,
not a description of anything currently collected.

## Issue-driven fix priorities (Section 36, for if/when real issues
exist)

Install bugs, service lifecycle, catalog/download, backend detection,
memory planning, API compatibility, model switching, Windows/macOS
path issues — in that rough order of severity, real user impact first.
No feature creep: a real external report is a bug-fix trigger, never
license to add unrelated functionality riding along with the fix.
