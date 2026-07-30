# Contributing to MEMBRANE

MEMBRANE is a research prototype spanning a C11 core library, C++17
simulators, synthesizable SystemVerilog RTL, and Python release tooling.
See [docs/architecture.md](docs/architecture.md) for the current system
diagram and [docs/results-summary.md](docs/results-summary.md) for what's
been measured and what hasn't, before proposing new functionality.

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
[docs/reproduction.md](docs/reproduction.md) Level 1.4 for the Verilator
cosimulation flow. Full setup for all of the above:
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

## Pull requests

Keep pull requests scoped to one logical change. Include the `ctest`
output (or a summary of it) in the PR description when behavior changes.
If you change a number in `docs/` or `README.md`, run
`scripts/verify-results.py` and include its output.
