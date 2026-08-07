# Contributing to MEMBRANE

MEMBRANE spans a C11 core library, C++17 simulators, synthesizable
SystemVerilog RTL, and Python release tooling. See
[docs/architecture.md](docs/architecture.md) for the current system
diagram and [docs/results-summary.md](docs/results-summary.md) for what's
been measured and what hasn't, before proposing new functionality.

## Product vs. research contributions

This repository (`kadireren7/membrane`) is the **maintained
implementation** — contribute here for anything that builds, tests, or
ships as part of the current library: bug fixes, new codecs, CLI/tool
improvements, CI/build changes, documentation for what's currently
maintained.

New experiments — new predictors, RTL variants, simulator studies,
anything whose outcome isn't known yet — are contributed to
**[kadireren7/membrane-research](https://github.com/kadireren7/membrane-research)**
instead, following that repository's own `CONTRIBUTING.md`. If a
research result there reaches a point where it should become part of
the maintained product, it comes back to *this* repository as an
ordinary, reviewed PR (see `docs/repository-boundary.md` for the full
two-repository model and why it changed).

## Building

Requires CMake >= 3.16 and a C11/C++17 compiler (gcc or clang).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

For the bit-exact ggml quantization parity test, you additionally need
the `third_party/llama.cpp` submodule checked out
(`git submodule update --init --recursive`) and
`-DMEMBRANE_ENABLE_LLAMA=ON`. For RTL work, see
[docs/reproduction.md](docs/reproduction.md)'s "FPGA production-datapath
Verilator cosimulation" section. Full setup for all of the above:
[docs/reproduction.md](docs/reproduction.md).

## Building with sanitizers

All new C/C++ code is expected to pass AddressSanitizer +
UndefinedBehaviorSanitizer and ThreadSanitizer before being considered
done:

```bash
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DMEMBRANE_ENABLE_SANITIZERS=ON
cmake --build build-asan -j && ctest --test-dir build-asan --output-on-failure

cmake -S . -B build-tsan -DCMAKE_BUILD_TYPE=Debug -DMEMBRANE_ENABLE_TSAN=ON
cmake --build build-tsan -j
setarch "$(uname -m)" -R ctest --test-dir build-tsan --output-on-failure
```

(`MEMBRANE_ENABLE_ASAN` still works as a deprecated alias for
`MEMBRANE_ENABLE_SANITIZERS`. The `setarch -R` wrapper works around a
known TSan/ASLR kernel interaction — see
[docs/phase6-unified-stress.md](docs/phase6-unified-stress.md) §13.)

## Code style

- C11 for `src`/`include` (no compiler-specific extensions:
  `-std=c11`, no GNU extensions), C++17 for `tools/`.
- Build with `-Wall -Wextra -Wpedantic` and keep the build warning-free.
- Prefer small, focused translation units over large multi-purpose files.
- Every new codec/quantizer must round-trip losslessly (or be verified
  bit-exact against its reference, for lossy paths) and be exercised by
  a unit test that includes at least: empty input, single-byte input,
  random data, and corrupted/truncated data.
- New RTL must elaborate cleanly under yosys and be cosimulated against
  its C/C++ reference (see `rtl/tb/tb_top_verilator.cpp` for the
  pattern).
- Every claim in `docs/` must cite a source artifact (CSV/JSONL/log) and
  be checkable by `scripts/verify-results.py` — see that script for the
  existing check patterns before adding a new headline number anywhere.
- Commit messages follow conventional prefixes: `feat:`, `fix:`, `test:`,
  `bench:`, `docs:`, `refactor:`, `build:`, `ci:`, `chore:`, `perf:`,
  `research:`.

## Branch naming

MEMBRANE develops fully in the open — there is no private companion
repository (see [docs/repository-boundary.md](docs/repository-boundary.md)).
Name branches by what they are:

- `feature/<name>` — new functionality headed toward `main`.
- `fix/<name>` — bug fixes.
- `docs/<name>` — documentation-only changes.
- `chore/<name>` — repository maintenance (dependency bumps, CI
  changes, structural cleanup).

New research experiments are no longer started as `experiment/<name>`
branches in this repository — they go directly to
[kadireren7/membrane-research](https://github.com/kadireren7/membrane-research)
instead (see `docs/repository-boundary.md` Rule 6). Existing
`experiment/*` branches here (`experiment/q8-divider-pipeline`,
`experiment/fp-divider-pipeline`) are preserved as historical git
history, unchanged, and are mirrored (SHA256-verified) into that
repository's own `experiments/` tree.

`main` holds only verified, reviewed changes. Release tags
(`v0.1.0-research` and later) are immutable snapshots of `main` at a
point in time — see `docs/research-release-freeze.md`.

## Pull requests

Keep pull requests scoped to one logical change. Every pull request into
`main` must include:

- A hypothesis or linked issue explaining why the change was made.
- Reproduction commands that work from a clean clone.
- Test results (`ctest` output, sanitizer runs where applicable).
- Artifact hashes for any new or changed committed benchmark artifact.
- A limitations section — what the change does *not* establish.
- Any negative/null findings encountered, reported with the same
  visibility as positive ones (see `docs/results-summary.md` §4).
- Consistency with the project's AI-assistance disclosure
  (README.md's own "AI-assisted development" section, and the full
  version at `kadireren7/membrane-research`'s
  `outreach/ai-assistance-disclosure.md`) — don't introduce text that
  implies sole human authorship of AI-drafted material, or vice versa.
- No fabricated or implied hardware claims — see README.md's own
  "Limitations" section for what is and isn't real.

If you change a number in `docs/` or `README.md`, run
`scripts/verify-results.py` and include its output.

## Automated PR review

Every pull request goes through the same sequence:

1. The PR is opened.
2. Normal CI runs (`.github/workflows/ci.yml`: Debug, ASan+UBSan, TSan).
3. CodeQL runs (`.github/workflows/codeql.yml`: static security/quality
   analysis of the C/C++ sources).
4. CodeRabbit leaves review comments (PR summary, walkthrough, and
   line-by-line findings).
5. Findings are either fixed, or explicitly acknowledged and dismissed
   with a reason in the PR thread — a bot comment is not itself a
   decision.
6. The PR is merged once all required checks pass — see
   [docs/automated-pr-review.md](docs/automated-pr-review.md) for exactly
   which checks are required vs. advisory, and why CodeRabbit's own
   findings are never a merge blocker.
