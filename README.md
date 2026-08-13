# MEMBRANE

[![CI](https://github.com/kadireren7/membrane/actions/workflows/ci.yml/badge.svg)](https://github.com/kadireren7/membrane/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kadireren7/membrane/actions/workflows/codeql.yml/badge.svg)](https://github.com/kadireren7/membrane/actions/workflows/codeql.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

MEMBRANE is an experimental KV-cache memory runtime for local LLM
inference, built on an open-source research project spanning CPU
quantization, synthesizable RTL, and near-memory/CXL simulation.

It can run llama.cpp-compatible models with a Q8_0 KV cache instead of
the native full-precision cache, reducing KV-cache memory while
preserving a usable inference path — measured, not assumed (see
"Measured Example" below). `v0.2.0`, single model/host verified so
far; see "Limitations."

## Quick Start

```bash
git clone --recursive https://github.com/kadireren7/membrane
cd membrane
cmake -S . -B build-llama -DMEMBRANE_ENABLE_LLAMA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-llama -j --target membrane-run

./build-llama/tools/membrane-run/membrane-run \
  --model model.gguf --prompt "Hello" --ctx 4096 --kv q8
```

`--kv q8` is opt-in — the default is `native` (unmodified llama.cpp
behavior), so nothing about model output changes unless you ask for it.
`membrane-run --help` documents every flag, the native/q8 trade-off,
and exit codes. See `docs/live-runtime.md` for the mechanism and
`tools/membrane-llama-runtime/` (`membrane-llama-run`) for the
diagnostic shadow/injection tooling this is built on top of.

## Measured Example

`results/v0.2/smollm2-q8-memory.json` (committed, machine-checkable via
`scripts/verify-results.py`) — SmolLM2-135M, one prompt, CPU, single
host:

| context | native KV | q8 KV | native RSS after ctx | q8 RSS after ctx | reduction |
|---|---|---|---|---|---|
| 2048 | 45.00 MiB | 23.91 MiB | 335,964 kB | 314,644 kB | 6.35% |
| 8192 | 180.00 MiB | 95.62 MiB | 474,480 kB | 388,268 kB | 18.17% |

Reduction grows with context size (full 5-point sweep in the artifact);
token identity and top1 preservation stayed perfect at these sizes on
this prompt, logit drift stayed small. **Single model, single prompt,
single CPU host — not a general claim.** Reproduce with
`scripts/benchmark-v0.2.sh MODEL.gguf`.

## Why MEMBRANE

LLM inference servers are usually compute-rich but memory-poor: KV-cache
size grows linearly with context length and concurrent request count,
and memory/bandwidth — not FLOPs — is often what runs out first.
MEMBRANE explores whether quantizing and exactly retrieving KV memory
based on real access patterns, instead of a static per-deployment knob,
helps — and where it doesn't. The core library here is a from-scratch
C11 implementation: bit-exact Q8_0/Q4_0 quantization (verified against
ggml's own reference math), a block-oriented KV store, and a
synthesizable FPGA datapath for the quantization step, with one
component already promoted to production RTL after research validation.

## What is maintained here

| Component | Status | Validation |
|---|---|---|
| Block/store API (`src/block`, `src/store`) | Maintained | `ctest` unit suite |
| RAW/RLE/byteplane/Huffman codecs (`src/codecs`) | Maintained | `ctest` unit suite |
| Q8_0/Q4_0 CPU quantization (`src/quant`) | Maintained | Bit-exact vs. ggml — `test_ggml_quant_parity` |
| Production FPGA datapath RTL (`rtl/membrane_quant_stream_top.sv`) | Maintained | Verilator cosimulation vs. C reference, 0 mismatches |
| Q4_0 radix-4 divider (promoted from research) | Maintained | 4,456,685 differential cases, 0 mismatches — [PR #2](https://github.com/kadireren7/membrane/pull/2) |
| Sanitizer + CodeQL CI | Maintained | Debug/ASan+UBSan/TSan + static analysis, every push |

Everything else this project has explored — the Q8_0 dual-divider and
scheduler research, CXL/near-memory simulation tooling's own research
usage, the academic paper, and the outreach package — is not maintained
*here*. It's active, not abandoned — see "Research record" below.

## Architecture

```text
LLM runtime (llama.cpp, real inference)
        |  captured KV/attention traces
        v
Quantization  --Q8_0/Q4_0, bit-exact vs. ggml-->  Block store
        |
        v
Synthesizable FPGA datapath (Q8/Q4 encode/decode, Verilator-cosimulated)
```

Full diagram set (KV lifecycle, FPGA datapath detail): `docs/architecture.md`.

## Repository layout

```text
include/membrane/   Public C headers (block store, codecs, quant, traces)
src/                Core C11 library implementation
tools/              CLI tools (simulators, capture, quantization benches --
                    several are wired into the maintained build/test suite
                    even where their own subject matter is research-flavored)
rtl/                Synthesizable FPGA RTL + Verilator/Icarus testbenches
tests/unit/         C/C++ unit tests (ctest-registered)
docs/               Architecture, reproduction, and process documentation
benchmarks/         Committed CSV/JSONL artifacts + MANIFEST.json, read by
                    maintained tests and scripts/verify-results.py
paper/              Manuscript source, built by .github/workflows/paper.yml
hardware/           FPGA/CXL physical-validation plan (no real hardware
                    used yet -- see docs/phase8-hardware-validation-plan.md)
scripts/            demo.sh, verify-results.py, prepare-release.sh,
                    verify-q4-radix4-divider.sh, ...
third_party/        llama.cpp (git submodule, upstream license applies)
```

`tools/`, `benchmarks/`, and `paper/` still contain research-flavored
material by subject matter — they stay in this repository because the
maintained build (`CMakeLists.txt`), maintained tests, or maintained CI
(`.github/workflows/paper.yml`) genuinely depend on them, not because
they were missed by the split above. See `docs/repository-boundary.md`
for the full account of what moved and what a dependency audit found
had to stay.

## Verified results

| Result | Value | Evidence type | Scope |
|---|---|---|---|
| ggml quantization parity | 100,000+ random blocks + all documented edge cases, 0 mismatches vs. ggml's Q8_0/Q4_0 reference | MEASURED_BY_TOOL | Maintained |
| Production RTL full-pipeline cosimulation | 520,000 transactions, 0 mismatches, Verilator vs. real C quantizer reference | MEASURED_BY_TOOL | Maintained |
| Q4_0 radix-4 divider exactness (promoted) | 4,456,685 differential cases, 0 mismatches | MEASURED_BY_TOOL | Maintained, [PR #2](https://github.com/kadireren7/membrane/pull/2) |
| Q4_0 radix-4 divider area | -96.2% ECP5 cells at the `q4_scale` integration point (74,382 → 2,836), synthesis-tool proxy | MEASURED_BY_TOOL (proxy, not physical) | Maintained, promoted |
| Sparse-retrieval traffic reduction (simulated) | 187x-405x vs. full-scan-CXL at the 8GiB-host/2TiB-device point (unified 128K-context x 512-concurrency sweep); retrieval itself is exact (non-approximate) by construction, the traffic-reduction number is simulated | SIMULATED | Maintained data (`benchmarks/`) |
| Q8_0 dual-divider area reduction — **RESEARCH ONLY, not merged** | -97.76% ECP5 cells at the `q8_scale` integration point, bit-exact (4,052,224 cases, 0 mismatches) — scheduler collateral cost unresolved | MEASURED_BY_TOOL (proxy, not physical) | Research only — [kadireren7/membrane-research](https://github.com/kadireren7/membrane-research), `EXP-FPGA-DIV-002` |

No result above implies physical FPGA hardware, physical CXL hardware,
measured Fmax, timing closure, or measured power — see "Limitations."

## Live llama.cpp runtime (Phase 5-8)

`membrane-run` (Product Phase 8, see "Quick Start" above) is the
user-facing entry point: one normal decode pass by default (no hidden
comparison work), or `--compare-kv` for the full native-vs-q8
memory/quality/performance comparison. It's built on
`tools/membrane-llama-runtime/` (`membrane-llama-run`, kept as the
diagnostic/legacy tool), which drives real llama.cpp inference with
MEMBRANE observing (`shadow-*`), authoritatively injecting
reconstructed values (`inject-*`), or — `--kv-store q8` — replacing the
KV cache's own allocated tensor type with genuinely compressed Q8_0
storage (no `third_party/llama.cpp` patch required; see
`docs/live-runtime.md`).

## GPU / Vulkan (development branch, not yet in v0.2.0 stable)

`v0.2.0` stable was CPU-first: every claim in this document up to this
section is CPU-only. The `feature/gpu-vulkan-runtime` development
branch adds explicit, opt-in GPU runtime controls to `membrane-run` on
top of the existing, unmodified `ggml`/llama.cpp Vulkan backend — no
custom GPU kernels, no `third_party/llama.cpp` changes.

**What was found (Phase 9A/9B), on one tested host** (Pop!_OS 24.04,
NVIDIA GeForce GTX 1650 Mobile, 4 GB VRAM, driver 580.159.03): with the
Vulkan backend, the Q8_0 KV cache is genuinely allocated in GPU VRAM
(not host RAM), and its VRAM footprint measured lower than native F16
KV at every tested context on SmolLM2-135M — from roughly 2% at small
contexts up to roughly 25% at `ctx=16384`. **This is one model family,
one GPU, one driver, one host — not a general GPU/VRAM claim.**
Measured reduction depends on model, context size, and hardware, and
will differ elsewhere. The GPU path also has a measured throughput
cost on this host: q8 generation speed measured roughly 7-18% lower
than native across the same SmolLM2-135M context sweep — reported as
the measured range, not a single asserted number.

CUDA is not yet a supported MEMBRANE product path — this section is
about the Vulkan backend only.

### Build

```bash
cmake -S . -B build-vulkan \
  -DMEMBRANE_ENABLE_LLAMA=ON \
  -DGGML_VULKAN=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j
```

Vulkan is an optional, non-default build path — a plain
`-DMEMBRANE_ENABLE_LLAMA=ON` build (no `-DGGML_VULKAN=ON`) remains
CPU-only, unchanged. Building the Vulkan backend needs Vulkan
development headers, the `glslc` shader compiler, and SPIR-V headers
already on the system — MEMBRANE never installs system packages
automatically. Required package names vary by distribution; for
Pop!_OS/Ubuntu (24.04, the only distribution this was verified on),
the packages actually confirmed necessary in this project are:

```bash
sudo apt install libvulkan-dev glslc vulkan-tools spirv-headers
```

Do not assume these exact package names apply to other distributions.

### CLI

GPU use is **explicit only** — `membrane-run` never uses a GPU unless
asked to, even when built with `-DGGML_VULKAN=ON`:

- `--gpu-layers 0` — CPU-only. Default, and also the explicit
  CPU-forcing form, equivalent to omitting the flag entirely.
- `--gpu-layers all` — offload every layer. Rejected (exit 5) if the
  estimated memory requirement exceeds a safe budget on the selected
  device — see "Memory-aware offload" below.
- `--gpu-layers N` — offload `N` layers, clamped to the model's real
  layer count if `N` exceeds it (the common llama.cpp `-ngl 99`-style
  idiom for "offload everything" without knowing the exact count).
  Same budget check as `all`.
- `--gpu-layers auto` (Phase 9B.1) — MEMBRANE picks a layer count from
  real device-free-memory and model-size information, leaving an
  explicit safety margin.
- `--device NAME` — select a GPU device by a case-insensitive
  substring match against its name or description (e.g. `nvidia`,
  `radeon`, `1650`). Requires `--gpu-layers all|auto|N`. Fails clearly
  if no device matches, or if more than one does.

Any nonzero `--gpu-layers` value — `all`, `auto`, or an explicit `N`
— requires an explicit `--ctx`: the memory guard can't estimate a real
KV budget before the context size is known, and auto-sizing `--ctx`
from the prompt needs a model already loaded, which is a decision GPU
resolution has to make first, before load.

Requesting GPU layers on a build with no GPU backend compiled in, or
with no GPU device found at runtime, fails clearly (exit 5) — it never
silently falls back to CPU, for `all`/`auto`/`N` alike.

**"auto" is conservative, not an OOM guarantee.** It uses real
information from public APIs — `ggml_backend_dev_memory()` for device
free/total bytes, and `gguf_init_from_file()` (no full model load) for
real per-tensor byte sizes and hparams — reserves a safety margin
(the larger of a 512 MiB fixed floor or 15% of device total memory),
subtracts an estimated KV requirement for the requested context/KV
type, and picks the largest layer count that fits the remainder. It
does **not** model `ggml`'s actual allocator overhead exactly
(alignment/scratch/fragmentation), and it has no visibility into
memory other processes might claim after it decides — a desktop
compositor, browser, or game claiming VRAM after MEMBRANE starts can
still exhaust a margin that looked safe at decision time. Treat it as
a helpful default, not a promise.

**Memory-aware offload / guard**: the same budget check applies to
`all`/`N` too, not just `auto` — an explicit request that clearly
exceeds the estimated safe budget is rejected before any inference
begins (exit 5, no partial run, no crash), not silently reduced.
Verified on the test host: `--gpu-layers all` at a context large
enough to make the estimated requirement exceed the budget is
rejected outright, while `--gpu-layers auto` at that *same* context
succeeds by choosing a genuinely smaller layer count instead — see
`results/v0.3/gpu-vulkan-validation.json`'s `auto_policy.
guard_boundary_test` for the exact numbers.

**`--gpu-bench`** runs an explicit native-vs-q8 comparison under a GPU
configuration (requires `--gpu-layers all|auto|N`, mutually exclusive
with `--compare-kv`): selected device/layers, KV bytes, throughput,
and quality (token identity, first divergence, top1 preservation,
logit rel-L2, delta NLL) side by side. Q8 measurably trades some
throughput for the memory reduction on the tested host (see below) —
this is reported as measured, not asserted as universal.

`--json`/human output report a `gpu` block (whether GPU use was
requested, whether the build supports it, the requested vs *selected*
layer count, and the backend/device membrane-run explicitly selected
via public `ggml_backend_dev_*()` enumeration — never left to
llama.cpp's own implicit upstream default) and, whenever a memory
estimate was used, a `gpu_policy` block (device free/total bytes,
safety reserve, estimated weight/KV bytes). This reflects what was
*requested and selected/estimated*, not an independently-confirmed
post-load allocation — no public API exposes that without scraping
runtime logs, so none is claimed.

Vulkan was tested on exactly one host, one GPU (NVIDIA GeForce GTX
1650 Mobile). There is no universal CUDA or AMD support claim yet —
this is a Vulkan-only, single-host validation.

## Build from source (llama-free library only)

```bash
git clone --recurse-submodules https://github.com/kadireren7/membrane
cd membrane
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

This builds the portable quantization/storage/FPGA-cosim library and
tools with no `third_party/llama.cpp` submodule and no model file
needed — see "Quick Start" above for the llama.cpp-backed
`membrane-run` build. `scripts/demo.sh --quick` additionally runs the
bit-exact quantization parity test and the full FPGA Verilator
cosimulation from small, already-committed fixtures — no model
download, finishes in minutes.

## Try the demo

`membrane-demo` is a self-contained CLI (no model download, no network)
that runs a deterministic synthetic KV-block workload through the
maintained Q4_0/Q8_0 quantization path — real per-block precision
selection, real packed-byte storage accounting against an FP32
baseline, real decode and error-bound validation:

```bash
./build/tools/membrane-demo/membrane-demo
```
```text
Precision policy (Q4 accepted if rel-L2 error <= 0.05)
  Q4 blocks               6456
  Q8 blocks               1736
Storage (baseline: FP32 bytes, same logical element count)
  Reduction               83.29%
Validation (lossy reconstruction -- lower is better, not bit-exact)
  Result                  PASS
```

Machine-readable output: `./build/tools/membrane-demo/membrane-demo --blocks 16384 --seed 42 --json`.
See `tools/membrane-demo/demo_core.h` for exactly what each field measures.

## Benchmark

`membrane-quant-policy-bench` measures storage/accuracy/runtime trade-offs
across 4 workloads (all **synthetic** — calibrated to exercise a real range of
Q4_0/Q8_0 quantization outcomes, not captured or modeled LLM traces) and 3
precision policies (`q4-only`, `q8-only`, `adaptive`), using the same
maintained quantization engine as the demo:

```bash
./build/tools/membrane-quant-policy-bench/membrane-quant-policy-bench --matrix
```

This prints one row per workload x policy combination (12 rows: 4 workloads x
`q4-only`/`q8-only`/`adaptive`), shaped like:

```text
Workload                 Policy       Storage%    MeanErr     MedianMs     Blocks/s
synthetic-<kind>         <policy>     <baseline-vs-encoded reduction, %>
                                       <block-count-weighted mean rel-L2 error>
                                                    <median of --iterations, ms>
                                                                 <blocks/s>
```

No fixed numbers are reproduced here: `Storage%`/`MeanErr` are deterministic
for a given `--blocks`/`--seed`/`--policy` (see
`tools/membrane-workload-core/test_workload_core.c` and
`tools/membrane-quant-policy-bench/test_membrane_bench.c` for that guarantee)
but will naturally shift if the generator or thresholds ever change;
`MedianMs`/`Blocks/s` are local-machine timing and vary by host.

Machine-readable: `--json` or `--csv` (add `--matrix` for all 12 cells, or
omit it and pass `--workload`/`--policy` for one). Timing is **local CPU wall
time only** (median of `--iterations`, after `--warmup` discarded passes) —
not LLM end-to-end inference performance, and no FPGA/CXL hardware is
measured or claimed anywhere here. See `--help` for the exact timed-region
definition.

### Benchmark a captured KV trace

```bash
./build/tools/membrane-quant-policy-bench/membrane-quant-policy-bench \
  --trace your_trace.memkv --policy adaptive
```

The trace is captured separately (`tools/membrane-kv-capture`, needs
`-DMEMBRANE_ENABLE_LLAMA=ON` and a `.gguf` model) and converted with
`tools/membrane-kv-trace-export`; the benchmark itself still runs fully
offline, with no llama.cpp dependency, reading only the real numerical KV
block data you supply — no prompt/token text is required or read. `--trace
file --matrix` runs exactly the 3 policies against that one trace (not the
4-workload synthetic sweep). Numbers depend entirely on the model/layer/
input the trace came from — see `docs/kv-trace-format.md` for the format
and full pipeline. As with the synthetic benchmark: no end-to-end LLM
performance claim, and no physical FPGA/CXL measurement, either mode.

### Benchmark many layers/tensors from one capture

```bash
mkdir -p traces/
./build/tools/membrane-kv-trace-export/membrane-kv-trace-export \
  --input capture.kvdump --output-dir traces/
./build/tools/membrane-quant-policy-bench/membrane-quant-policy-bench \
  --trace-dir traces/ --matrix
```

`--output-dir` must already exist (the tool never creates it) and
batch-exports every compatible F16 K/V record from one `.kvdump`
capture as `layer-NNN-k.memkv` / `layer-NNN-v.memkv` (optional
`--layer-start`/`--layer-end`/`--tensor k|v|both` filters). `--trace-dir`
then benchmarks every `.memkv` file directly inside that directory — all
3 policies per trace, plus a block-weighted (never naively averaged)
cross-trace adaptive summary: total storage reduction, pooled mean/max
reconstruction error, and neutral facts like the min/max/median adaptive
Q4 ratio across traces. `--json`/`--csv` give one row per trace x policy
(3N rows for N traces) plus the aggregate. Answers "does Q4/Q8
suitability vary across layers/tensors on one real model execution" —
never a claim about models or prompts in general, and never an
end-to-end LLM or physical FPGA/CXL performance claim.

### Live llama.cpp shadow/injection runtime

```bash
cmake -S . -B build-llama -DCMAKE_BUILD_TYPE=Release -DMEMBRANE_ENABLE_LLAMA=ON
cmake --build build-llama -j --target membrane-llama-run
./build-llama/tools/membrane-llama-runtime/membrane-llama-run \
  --model your_model.gguf --prompt-file your_prompt.txt --mode shadow-adaptive
```

`membrane-llama-run` observes (`shadow-q8`/`shadow-adaptive`) and, for
`inject-q8`/`inject-adaptive`, authoritatively writes back
reconstructed live K/V values *during* real generation (via
`ggml_backend_sched_eval_callback`/`ggml_backend_tensor_set`, no
`.kvdump`/`.memkv` round-trip, no llama.cpp source modification).
**SHADOW modes never replace native llama KV**: `baseline`,
`shadow-q8`, and `shadow-adaptive` never alter what llama itself
stores or reads. **INJECT modes replace only the tensor feeding the
KV-cache write** (scoped via `--inject-layer`/`--inject-tensor`/
`--inject-token-start`/`-end`) — native cache *allocation* is
unchanged either way, so neither mode is ever a process-memory
reduction claim. `--json`'s `token_ids`/`injection`/`divergence`/
`behavior` fields let a local run check token identity, block
coverage, and logit/NLL drift directly. See `docs/live-runtime.md` for
the full architecture, semantics, and limitations.

## Test

```bash
# Normal build + unit suite
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j && ctest --test-dir build --output-on-failure

# Sanitizers (ASan+UBSan, then TSan)
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DMEMBRANE_ENABLE_SANITIZERS=ON
cmake --build build-asan -j && ctest --test-dir build-asan --output-on-failure

cmake -S . -B build-tsan -DCMAKE_BUILD_TYPE=Debug -DMEMBRANE_ENABLE_TSAN=ON
cmake --build build-tsan -j
setarch "$(uname -m)" -R ctest --test-dir build-tsan --output-on-failure
```

Full reproduction guide, including the Q4_0 divider regression gate and
release verification: `docs/reproduction.md`.

## Project status

**Maintained** (this repository): the CPU quantization/storage
implementation, the production FPGA RTL (including the promoted Q4_0
radix-4 divider), CI (sanitizers + CodeQL), and this repository's own
release process.

**Research** (not maintained here, but active): the Q8_0 dual-divider
and scheduler investigation, CXL/near-memory simulation research, the
academic paper, and outreach material — all at
**[kadireren7/membrane-research](https://github.com/kadireren7/membrane-research)**.

## Limitations

- **No physical FPGA implementation.** Every synthesis number is a
  Yosys 0.33 generic or `synth_ecp5` synthesis-tool proxy result, not a
  measurement from real silicon.
- **No physical CXL hardware.** CXL/near-memory results are software
  simulations, calibrated from real captured traces, never a physical
  device measurement.
- **No vendor timing closure, no measured Fmax, no measured FPGA
  power** exist anywhere in this project.
- **Limited model/workload scope**: two open SmolLM2 checkpoints
  (135M/360M), not a broad model sweep.
- **Some conclusions are simulation- or synthesis-based**, explicitly
  labeled as such — see `docs/reproduction.md` and each result's own
  "Evidence type" above.
- **`membrane-run --kv q8` (Product Phase 8)**: verified on one model
  (SmolLM2-135M), one prompt, CPU-only, single host — the "Measured
  Example" numbers above are not a general RAM-reduction claim across
  models/prompts/hosts. Only `LLM_ARCH_LLAMA` models are supported;
  `membrane-run` checks compatibility before creating a q8 context and
  fails clearly (never silently falls back to native) if unsupported.
  No speedup claim — performance is reported as measured in both
  directions across local runs.

## Research record

MEMBRANE used to keep all research in this repository's own branches.
As the research record grew — full phased experiment histories,
simulators, a paper, an outreach package — that made this repository
harder to navigate for its actual purpose: building and using the
maintained implementation. Research now lives in
[kadireren7/membrane-research](https://github.com/kadireren7/membrane-research)
instead, with full SHA256-verified provenance back to this repository
and nothing deleted — including results that didn't pan out. See
`docs/repository-boundary.md` for the full decision record, including
why this reverses an earlier, deliberate decision to keep everything in
one repository.

## AI-assisted development

Kadir Eren Altıntaş leads project architecture, experiment selection,
validation criteria, release decisions, and repository direction. AI
coding agents have assisted implementation, analysis automation,
documentation, and review. Results promoted by the project are
validated through tests, CI, reproducible experiments, or explicitly
classified as estimates/simulation. Full disclosure:
[kadireren7/membrane-research](https://github.com/kadireren7/membrane-research)'s
`outreach/ai-assistance-disclosure.md`.

## Citation

See [CITATION.cff](CITATION.cff).

## License

Apache License 2.0 for MEMBRANE's own code — see [LICENSE](LICENSE).
The `third_party/llama.cpp` submodule and any model artifacts are under
their own separate terms — see [docs/licensing.md](docs/licensing.md).

---

Contributing: [CONTRIBUTING.md](CONTRIBUTING.md). Security: [SECURITY.md](SECURITY.md).
Support: [SUPPORT.md](SUPPORT.md). Community standards: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
