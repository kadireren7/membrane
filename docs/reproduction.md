# Reproduction

This document covers the maintained verification flow only: prove the
core library, simulators, and RTL cosimulation build and behave
correctly, using small, already-committed test fixtures — no model
weights, no multi-hour sweep. Each command block states its working
directory, dependencies, expected time, expected output, and the signal
that means "this succeeded." Run these from a clean clone unless noted.

```bash
git clone --recurse-submodules <this repo>
cd membrane
```

(`--recurse-submodules` fetches `third_party/llama.cpp`'s source — a few
tens of MB of C/C++ source, not a model download.)

**Working directory:** repo root.
**Dependencies:** CMake ≥ 3.16, a C11/C++17 compiler (gcc or clang).
Optional: Verilator (for the FPGA cosimulation step — see below).

## Unit test suite

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

- **Expected time:** ~1–2 minutes on a modest machine.
- **Expected output:** `100% tests passed, 0 tests failed out of 28`.
- **Success signal:** exit code 0 from `ctest`; no `FAIL:` lines in output.

## Sanitizer builds (optional but recommended before trusting new code)

```bash
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DMEMBRANE_ENABLE_SANITIZERS=ON
cmake --build build-asan -j && ctest --test-dir build-asan --output-on-failure

cmake -S . -B build-tsan -DCMAKE_BUILD_TYPE=Debug -DMEMBRANE_ENABLE_TSAN=ON
cmake --build build-tsan -j
setarch "$(uname -m)" -R ctest --test-dir build-tsan --output-on-failure
```

- **Expected time:** ~2–3 minutes combined.
- **Expected output:** `30/30 tests passed` for both.
- **Note:** the `setarch -R` wrapper disables ASLR for the TSan run only —
  on some kernels ThreadSanitizer's shadow memory layout collides with
  ASLR and aborts with "unexpected memory mapping," which is an
  environment quirk, not a data race.

## Bit-exact CPU quantization parity (requires the llama.cpp submodule)

```bash
cmake -S . -B build-llama -DCMAKE_BUILD_TYPE=Release -DMEMBRANE_ENABLE_LLAMA=ON
cmake --build build-llama -j --target test_ggml_quant_parity
./build-llama/test_ggml_quant_parity
```

- **Expected time:** ~1–3 minutes (compiles ggml's CPU backend only —
  `LLAMA_BUILD_{TESTS,EXAMPLES,SERVER,TOOLS}` are off, so this does not
  build the full llama.cpp CLI).
- **Expected output:** `all ggml quant parity tests passed (100000+
  random blocks, all-zero, constant, extrema, NaN/Inf, denormal)`.
- **Success signal:** that exact line, exit code 0.

## FPGA production-datapath Verilator cosimulation (optional, needs Verilator)

```bash
# golden vectors (deterministic, fixed seed — generated fresh each time)
cc -O2 -I include -o /tmp/gen_top_x rtl/tb/gen_top_x_vectors.c src/codecs/f16convert.c
cc -O2 -I include -o /tmp/gen_pack rtl/tb/gen_pack_vectors.c src/codecs/f16convert.c src/quant/quant_simd.c
cc -O2 -I include -o /tmp/gen_dequant rtl/tb/gen_dequant_vectors.c src/codecs/f16convert.c src/quant/quant_simd.c
cc -O2 -I include -o /tmp/gen_q4pack rtl/tb/gen_q4pack_vectors.c src/codecs/f16convert.c src/quant/quant_simd.c
cc -O2 -I include -o /tmp/gen_q4unpack rtl/tb/gen_q4unpack_vectors.c src/codecs/f16convert.c src/quant/quant_simd.c
/tmp/gen_top_x 120000 /tmp/top_x_120k.txt
/tmp/gen_pack /tmp/top_x_120k.txt /tmp/top_q8pack_120k.txt
/tmp/gen_dequant /tmp/top_q8pack_120k.txt /tmp/top_q8dequant_120k.txt
/tmp/gen_q4pack /tmp/top_x_120k.txt /tmp/top_q4pack_120k.txt
/tmp/gen_q4unpack /tmp/top_q4pack_120k.txt /tmp/top_q4unpack_120k.txt

verilator --cc --exe --build -j 0 -Wno-fatal --Mdir /tmp/membrane-verilator-obj \
  --top-module membrane_quant_stream_top \
  rtl/membrane_fp_pkg.sv rtl/membrane_fp_adder.sv rtl/membrane_fp_divider.sv \
  rtl/membrane_fp_divider_radix4.sv rtl/membrane_fp_scale_neg_pow2.sv \
  rtl/membrane_fp_multiplier.sv rtl/stream_fifo.sv rtl/valid_delay_line.sv \
  rtl/q4_pack.sv rtl/q4_scale.sv rtl/q4_scan.sv rtl/q4_unpack.sv \
  rtl/q8_dequantize.sv rtl/q8_maxabs_reduce.sv rtl/q8_quantize_pack.sv \
  rtl/q8_scale.sv rtl/membrane_quant_stream_top.sv "$PWD/rtl/tb/tb_top_verilator.cpp" \
  -o Vtop
/tmp/membrane-verilator-obj/Vtop
```

(The testbench path is passed absolute above — Verilator's generated
Makefile runs from `--Mdir`, so a relative testbench path resolves
against the wrong working directory otherwise. `rtl/membrane_fp_divider_radix4.sv`
and `rtl/membrane_fp_scale_neg_pow2.sv` are required because `q4_scale.sv`
instantiates both, since the EXP-FPGA-DIV-001 promotion — both real,
previously-missing dependencies of this exact command, confirmed by
running it, not assumed from the file list alone.)

- **Expected time:** ~15–20 seconds total (build ~13s, vector generation
  <1s, cosimulation run ~8s).
- **Expected output:** ends with `PASS: membrane_quant_stream_top
  Verilator cosim, 520000 transactions, 0 fails, <N>s`.
- **Success signal:** that line, exit code 0. If Verilator is not
  installed, this step is optional and skippable.
- `scripts/demo.sh --quick` runs the unit tests, quant parity, and this
  cosimulation automatically (the cosimulation step is skipped with a
  clear message if Verilator is unavailable).

## Q4_0 radix-4 divider regression gate

**Working directory:** repo root.
**Dependencies:** Yosys and Verilator (same as the cosimulation step
above).

```bash
scripts/verify-q4-radix4-divider.sh --quick
```

Production verification + performance-regression gate for the promoted
Q4_0 radix-4 divider integration (`rtl/membrane_fp_divider_radix4.sv`,
`rtl/q4_scale.sv`). `--quick` is CI-sized; `--full` reproduces the
original research-scale differential scope (millions of cases) — see
the script's own header for exact counts and expected time. The
research record this promotion came from
(`EXP-FPGA-DIV-001`) is not duplicated here — see "Research
reproduction" below.

- **Expected time:** ~1 minute in `--quick` mode (measured: 66s).
- **Expected output:** ends with `=== ALL CHECKS PASSED (quick mode) ===`.
- **Success signal:** that line, exit code 0.

## Maintained release verification

**Working directory:** repo root.
**Dependencies:** none beyond what the earlier sections already need
(`--dry-run` doesn't build or test anything itself — it previews the
checklist a real run would execute).

```bash
scripts/prepare-release.sh --dry-run
```

`--dry-run` is a **preview only**: it skips the test suite (step 2/8)
and reports what a full run would do, without building or running
anything itself. It still executes steps 3-8 for real: clean-tree
check, `scripts/verify-results.py` (headline-claim vs. artifact
checks), `benchmarks/MANIFEST.json` freshness, a repository-wide
relative-markdown-link check, a large-file check, and release-metadata
presence.

- **Expected time:** a few seconds (measured: 2.6s) for `--dry-run`;
  the test suite step alone (skipped here) is the same as the "Unit
  test suite" section above, ~1-2 minutes.
- **Expected output:** `Checks: N/6 passed (--dry-run: test suite
  skipped)`, `N`=6 only on a clean working tree (uncommitted local
  changes are the one thing `--dry-run` still fails on, by design).
- Run `scripts/prepare-release.sh` without `--dry-run` to execute the
  complete checklist for real, including the test suite. See
  `docs/release-candidate-checklist.md` for the full release process
  this script is one gate of.

## Research reproduction

Advanced experiments, CXL/near-memory simulation, the multi-hour
128K-context × 512-concurrency unified sweep, exploratory FPGA RTL
(including the related but separate Q8_0 divider/scheduler
investigation), canonical research results, the project paper, and the
outreach package all have their own full reproduction guide at:

**[kadireren7/membrane-research](https://github.com/kadireren7/membrane-research)**
— see that repository's own `reproduction.md`.
