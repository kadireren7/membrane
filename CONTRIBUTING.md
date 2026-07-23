# Contributing to MEMBRANE

MEMBRANE is a pre-alpha research prototype. The codebase is small and the
scope is deliberately narrow right now — see
[docs/architecture.md](docs/architecture.md) for what's in and out of scope
for the current milestone before proposing new functionality.

## Building

Requires CMake >= 3.16 and a C11 compiler (gcc or clang).

```bash
cmake -S . -B build
cmake --build build
```

## Running tests

```bash
ctest --test-dir build --output-on-failure
```

## Building with sanitizers

All new code should be verified against AddressSanitizer +
UndefinedBehaviorSanitizer before it's considered done:

```bash
cmake -S . -B build -DMEMBRANE_ENABLE_ASAN=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

## Code style

- C11, no compiler-specific extensions (`-std=c11`, no GNU extensions).
- Build with `-Wall -Wextra -Wpedantic` and keep the build warning-free.
- Prefer small, focused translation units over large multi-purpose files.
- Every new codec must round-trip losslessly and be exercised by a unit
  test that includes at least: empty input, single-byte input, random
  data, and corrupted/truncated compressed data.
- Commit messages follow conventional prefixes: `feat:`, `fix:`, `test:`,
  `bench:`, `docs:`, `refactor:`, `build:`, `ci:`, `chore:`.

## Pull requests

Keep pull requests scoped to one logical change. Include the `ctest`
output (or a summary of it) in the PR description when behavior changes.
