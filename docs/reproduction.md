# Reproduction

Three levels, ordered by cost. Each command block states its working
directory, dependencies, expected time, expected output, and the signal
that means "this succeeded." Run these from a clean clone unless noted.

All levels assume:

```bash
git clone --recurse-submodules <this repo>
cd membrane
```

(`--recurse-submodules` fetches `third_party/llama.cpp`'s source — a few
tens of MB of C/C++ source, not a model download.)

---

## Level 1 — Quick verification

**Goal:** prove the core library, simulators, and RTL cosimulation build
and behave correctly, using only small, already-committed test fixtures.
No model weights, no multi-hour sweep.

**Working directory:** repo root.
**Dependencies:** CMake ≥ 3.16, a C11/C++17 compiler (gcc or clang).
Optional: Verilator (for the FPGA cosimulation step — see below).

### 1.1 Unit test suite

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

- **Expected time:** ~1–2 minutes on a modest machine.
- **Expected output:** `100% tests passed, 0 tests failed out of 28`.
- **Success signal:** exit code 0 from `ctest`; no `FAIL:` lines in output.

### 1.2 Sanitizer builds (optional but recommended before trusting new code)

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
  environment quirk, not a data race (see
  [phase6-unified-stress.md](phase6-unified-stress.md) §13).

### 1.3 Bit-exact CPU quantization parity (requires the llama.cpp submodule)

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

### 1.4 FPGA full-pipeline Verilator cosimulation (optional, needs Verilator)

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
  rtl/membrane_fp_multiplier.sv rtl/stream_fifo.sv rtl/valid_delay_line.sv \
  rtl/q4_pack.sv rtl/q4_scale.sv rtl/q4_scan.sv rtl/q4_unpack.sv \
  rtl/q8_dequantize.sv rtl/q8_maxabs_reduce.sv rtl/q8_quantize_pack.sv \
  rtl/q8_scale.sv rtl/membrane_quant_stream_top.sv rtl/tb/tb_top_verilator.cpp \
  -o Vtop
/tmp/membrane-verilator-obj/Vtop
```

- **Expected time:** ~15–20 seconds total (build ~13s, vector generation
  <1s, cosimulation run ~8s).
- **Expected output:** ends with `PASS: membrane_quant_stream_top
  Verilator cosim, 520000 transactions, 0 fails, <N>s`.
- **Success signal:** that line, exit code 0. If Verilator is not
  installed, this step is optional and skippable — it is not required for
  Level 1 to otherwise pass.
- `scripts/demo.sh --quick` runs 1.1, 1.3, and 1.4 automatically (1.4
  skipped with a clear message if Verilator is unavailable).

---

## Level 2 — Model-backed verification

**Goal:** validate real KV/attention trace capture and quality-preserving
quantization against actual llama.cpp inference on both SmolLM2
checkpoints.

**Working directory:** repo root.
**Dependencies:** everything in Level 1's `-DMEMBRANE_ENABLE_LLAMA=ON`
build, plus two real GGUF checkpoints under `models/` (not committed —
`*.gguf` is gitignored, see [licensing.md](licensing.md)):

```bash
# One-time: obtain and convert the real checkpoints via llama.cpp's own
# converter (already vendored as the third_party/llama.cpp submodule).
# HuggingFaceTB/SmolLM2-135M-Instruct and HuggingFaceTB/SmolLM2-360M-Instruct
# are the real, public checkpoints used throughout this project's docs.
python3 third_party/llama.cpp/convert_hf_to_gguf.py \
  <path-to-downloaded-SmolLM2-135M-Instruct> --outfile models/SmolLM2-135M-Instruct-f16.gguf --outtype f16
python3 third_party/llama.cpp/convert_hf_to_gguf.py \
  <path-to-downloaded-SmolLM2-360M-Instruct> --outfile models/SmolLM2-360M-Instruct-f16.gguf --outtype f16
```

### 2.1 Real KV/attention trace capture

```bash
cmake --build build-llama -j --target membrane-kv-capture membrane-kv-attn-trace-capture
./build-llama/tools/membrane-kv-capture/membrane-kv-capture \
  --model models/SmolLM2-135M-Instruct-f16.gguf \
  --prompt-file benchmarks/kv/prompts/natural.txt \
  --out /tmp/repro-135m.kvdump --n-tokens 128
```

- **Expected time:** ~10–30 seconds per model on CPU (135M), a few minutes
  for 360M.
- **Expected output:** a `.kvdump`/`.kvtrace`/`.attntrace` file with real
  captured tensors; the tool prints per-layer tensor shapes as it runs.
- **Success signal:** process exits 0 and the output file is non-empty;
  compare against the committed `benchmarks/cxl-sim/traces/*.kvtrace` for
  the expected shape.

### 2.2 Quality validation (real inference, both models)

```bash
./build-llama/tools/membrane-kv-quality/membrane-kv-quality \
  --model models/SmolLM2-135M-Instruct-f16.gguf \
  --prompt-file benchmarks/kv/prompts/recall.txt \
  --n-tokens 256 --gen-tokens 64 --runs 3 \
  --out /tmp/repro-135m-quality.jsonl
```

Repeat with `models/SmolLM2-360M-Instruct-f16.gguf`.

- **Expected time:** a few minutes per model (real greedy decoding, 3
  runs for determinism checking).
- **Expected output:** a JSONL file, one record per run, with quantized
  vs. FP16 quality-comparison fields.
- **Success signal:** all 3 runs per model produce identical greedy output
  under the same seed (the tool reports this directly); compare against
  `benchmarks/results/phase3-kv-quality/quality.jsonl` for the expected
  shape and prior real results.

---

## Level 3 — Full research reproduction

**Goal:** reproduce the full 128K-context × 512-concurrency unified sweep
(462 scenarios across both models) using the out-of-core simulator
backend, including checkpoint/resume and artifact integrity verification.

**Working directory:** repo root.
**Dependencies:** Level 1's plain build (`-DMEMBRANE_ENABLE_LLAMA=OFF` is
sufficient — the unified sweep only needs the already-committed
`.attntrace` captures, not live model inference).
**Expected disk:** ~1–2 GiB transient (`.attntrace3` out-of-core synthetic
traces, regenerated deterministically, gitignored — see `.gitignore`'s
Phase 6.5 section).
**Expected RAM:** designed to run within a few hundred MiB–2 GiB budget
(`--memory-budget-mib`), verified on a 5.6 GiB machine; see
[phase6-out-of-core-simulator.md](phase6-out-of-core-simulator.md) §3.
**Expected time:** multi-hour (this sweep took multiple real sessions to
complete originally, including real OOM-driven restarts — see
[phase6-unified-stress.md](phase6-unified-stress.md)'s "Completion
history" section). Not something to run casually; checkpoint/resume makes
it safe to interrupt.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target membrane-kv-exact-sim membrane-kv-exact-sim-verify

./build/tools/membrane-kv-exact-sim/membrane-kv-exact-sim \
  --trace-135m-long benchmarks/cxl-sim/traces/SmolLM2-135M-Instruct-f16-long.attntrace \
  --trace-360m-long benchmarks/cxl-sim/traces/SmolLM2-360M-Instruct-f16-long.attntrace \
  --out benchmarks/cxl-sim/unified-sweep.csv \
  --checkpoint benchmarks/cxl-sim/unified-sweep.ckpt \
  --backend streaming --memory-budget-mib 768 --trace-cache-mib 256 \
  --workers 1
```

- **Checkpoint/resume:** interrupt with `Ctrl-C` or `kill` at any point
  and re-run the identical command — it resumes from the last checkpointed
  scenario, verified by trace/config hash staleness detection (a stale or
  mismatched checkpoint is refused and the sweep restarts fresh, not
  silently reused).
- **Artifact verification:**
  ```bash
  ./build/tools/membrane-kv-exact-sim/membrane-kv-exact-sim-verify \
    --checkpoint benchmarks/cxl-sim/unified-sweep.ckpt \
    --csv benchmarks/cxl-sim/unified-sweep.csv
  ```
  **Success signal:** `0 problem(s) found; 462 unique scenarios in
  checkpoint, 462 CSV data rows`. Any duplicate, missing, truncated, or
  stale row is reported explicitly with the offending scenario id, not
  silently repaired.
- **Full detail** on the out-of-core backend, memory-guard behavior,
  worker scaling, and every real bug found while building this path:
  [phase6-out-of-core-simulator.md](phase6-out-of-core-simulator.md).

### 3.1 Hardware-sensitivity sub-sweep (smaller, still real)

```bash
./build/tools/membrane-cxl-sim/membrane-cxl-sim \
  --trace benchmarks/cxl-sim/traces/SmolLM2-135M-Instruct-f16.kvtrace \
  --out /tmp/repro-hardware-sensitivity.csv
```

- **Expected time:** ~2–3 minutes (this specific sweep is much smaller
  than the full unified sweep; it is not out-of-core and does not need
  the streaming backend).
- **Expected output:** a CSV with the 10-point hardware-sensitivity
  matrix; compare against
  `benchmarks/cxl-sim/unified-sweep-hardware-sensitivity.csv`.
- **Success signal:** the tool's own printed "success criteria
  evaluation" section, reported as-is (met/not-met), never adjusted after
  the fact — see [phase6-cxl-near-memory.md](phase6-cxl-near-memory.md)
  §12.
