# MEMBRANE v0.3.0-rc1 — release notes

Adds explicit, opt-in Vulkan GPU runtime support to `membrane-run` on
top of the unmodified `ggml`/llama.cpp Vulkan backend (no custom GPU
kernels, no `third_party/llama.cpp` changes), plus a VRAM-aware
automatic offload policy and a memory safety guard on top of it.
`v0.2.0` stable's CPU-only default behavior is unchanged.

## What works

- `--gpu-layers 0|N|all|auto` — GPU layer offload. Default `0`
  (explicit CPU-only — also the CPU-forcing form; never implied by a
  GPU-capable build). `N` offloads `N` layers, clamped to the model's
  real layer count if `N` exceeds it. `all` offloads every layer.
  `auto` picks the largest layer count that fits an estimated memory
  budget (`ggml_backend_dev_memory()` for device free/total bytes,
  real per-tensor byte sizes from GGUF metadata for the per-layer
  estimate, a safety reserve of the larger of 512 MiB or 15% of device
  total, an estimated KV requirement for the requested `--ctx`/KV
  type) — conservative, not an OOM guarantee. Any nonzero request
  requires an explicit `--ctx`.
- `--device NAME` — select a GPU device by case-insensitive substring
  match against its name or description.
- Memory safety guard: an explicit request that doesn't fit the
  estimated memory budget (`all`, explicit `N`, or `auto` with nothing
  fitting) is rejected before model load — clear error, exit 5, no
  partial inference, no silent fallback to CPU, and never a silent
  *memory-budget-driven* reduction of an explicit request (distinct
  from the model-depth clamp on `N` above, which is reported in
  telemetry, not silent).
- Partial GPU offload: `auto` genuinely selects fewer than all layers
  when the full set doesn't fit, rather than only ever being
  full-offload-or-nothing (see "Verified result" below for the
  concrete boundary case).
- `--gpu-bench` — explicit native-vs-Q8 comparison under a GPU
  configuration: selected device/layers, KV bytes, throughput,
  quality where available. Own JSON schema (`mode":"gpu_bench"`), no
  fabricated VRAM figure (real process VRAM can't be measured
  portably from inside MEMBRANE without shelling out).
- CPU-only default preserved: no `--gpu-layers` flag behaves exactly
  as `v0.2.0` stable; GPU requests on a CPU-only build fail closed
  (exit 5, clear message), never silently proceed on CPU.

## Quick start

```bash
git clone --recursive https://github.com/kadireren7/membrane
cd membrane
cmake -S . -B build-vulkan -DMEMBRANE_ENABLE_LLAMA=ON -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j --target membrane-run

./build-vulkan/tools/membrane-run/membrane-run \
  --model model.gguf --prompt "Hello" --ctx 4096 --gpu-layers auto --kv q8
```

Vulkan build prerequisites (Pop!\_OS/Ubuntu; do not assume these exact
package names apply to other distributions):

```bash
sudo apt install libvulkan-dev glslc vulkan-tools spirv-headers
```

## Verified result

`results/v0.3/gpu-vulkan-validation.json` (committed;
`scripts/verify-results.py` — 37/37, including a guard-boundary check
and a README-to-artifact numeric tie-back). GTX 1650 Mobile (4 GB
VRAM) / Vulkan, real `nvidia-smi`-measured peak VRAM:

SmolLM2-135M, `--gpu-layers auto`:

| ctx | native VRAM | q8 VRAM | reduction |
|---|---|---|---|
| 2048 | 352 MiB | 331 MiB | 5.97% |
| 8192 | 482 MiB | 400 MiB | 17.01% |
| 16384 | 656 MiB | 493 MiB | 24.85% |

SmolLM2-360M, ctx=2048, `--gpu-layers auto`: 32/32 layers, 808 → 772
MiB (4.46% reduction).

Partial offload, proven near the guard boundary (not forced OOM):
at ctx=145000, `--gpu-layers all` is rejected (exit 5, before decode)
while `--gpu-layers auto` succeeds *at the same context* by selecting
17 of 30 layers — concrete evidence `auto` is not merely
full-offload-or-nothing.

The table above and the partial-offload figures are the committed,
`scripts/verify-results.py`-checked numbers from
`results/v0.3/gpu-vulkan-validation.json`. The following `--gpu-bench`
throughput characterization, by contrast, is from this release's own
local pre-release validation sprint (13 independent `--gpu-bench`
repetitions across both models on the same host) and is **not**
committed as a machine-verified artifact -- reported here as observed,
not as a checked claim: mean generation delta ≈ −9 to −10% (q8 vs
native), run-to-run range roughly −6% to −16% -- q8's decode
throughput was measurably noisier run to run than native's on this
host; quality metrics were bit-identical across all repetitions of a
given configuration.

## Known limitations

- Vulkan validated on one NVIDIA GTX 1650 Mobile host — not a general
  claim about other GPUs, vendors, or hosts.
- CUDA is not yet a MEMBRANE-supported product path.
- AMD/Intel GPU product validation is not claimed (device selection
  and matching were exercised against a local AMD iGPU as the
  *non-selected* device in this validation, not as a supported target).
- Q8 may reduce generation throughput relative to native (measured,
  not assumed — see above); it is not a universal speed win, only a
  memory win.
- VRAM savings depend on model, context size, and hardware — not a
  fixed percentage.
- `auto`'s safety margin is a conservative estimate from public
  `ggml`/GGUF APIs, not an absolute OOM guarantee: it does not model
  allocator overhead exactly and has no visibility into memory other
  processes claim after it decides.
- No custom GPU kernels — this release only wires up explicit,
  product-level control over the existing, unmodified `ggml`/llama.cpp
  Vulkan backend.
- No FPGA/CXL hardware claim in this release.
- `test_mem_guard` (`tools/membrane-kv-exact-sim/`) is a known,
  pre-existing, host-memory-pressure-dependent flake on
  memory-constrained local dev hosts — does not reproduce on CI.

## Validation status

Local dev host (this branch, this commit):

- llama-free Release: 63/63
- llama-free ASan/UBSan: 63/63
- llama-enabled CPU Release: 64/65 (`test_mem_guard` only -- known,
  pre-existing, host-memory-pressure-dependent flake, see "Known
  limitations")
- llama-enabled CPU ASan/UBSan: 64/65 (same)
- Vulkan Release smoke (native/q8/`--gpu-layers auto`/`--gpu-bench`):
  all exit 0, no fallback, correct device selection
- `scripts/verify-results.py`: 37/37

GitHub CI (`build-and-test` Debug/ASan, `thread-sanitizer`, CodeQL):
green, including a clean `test_mem_guard` pass -- this test's failures
are specific to this memory-constrained local host and do not
reproduce on CI runners.

## Model/backend scope

Vulkan GPU path verified against two local `LLM_ARCH_LLAMA` models,
SmolLM2-135M and SmolLM2-360M, on one Vulkan-capable host (NVIDIA GTX
1650 Mobile, AMD Radeon iGPU present as the non-selected device). CPU
backend scope is unchanged from `v0.2.0` stable.

This is a release-candidate tag preparation document only — no tag has
been created yet.
