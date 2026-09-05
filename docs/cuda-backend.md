# CUDA backend

Mega Phase D, PR D3. NVIDIA CUDA is now a real, tested product GPU
backend — `-DGGML_CUDA=ON` — alongside the existing Vulkan backend.
Real hardware validation, not a build-only claim: see "Real evidence"
below.

## Building with CUDA

```bash
cmake -S . -B build-cuda \
  -DMEMBRANE_ENABLE_LLAMA=ON -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j2 --target membrane-run membrane
```

Requires a real **CUDA Toolkit** (`nvcc`, `cudart`, `cublas`) — a real
NVIDIA GPU + driver alone (`nvidia-smi` working) is **not** sufficient;
the compiler toolkit is a separate install. If CMake can't find one,
configure fails clearly (`GGML_CUDA=ON but no CUDA Toolkit was found`)
before ever descending into `third_party/llama.cpp`'s own CUDA CMake —
the same fail-fast, MEMBRANE-specific-error pattern the Vulkan build
already uses (`README.md`'s Vulkan build section).

`CMAKE_CUDA_ARCHITECTURES` is auto-detected from the real, installed
GPU (`GGML_NATIVE`, CMake ≥ 3.24 + CUDA Toolkit ≥ 11.6) — confirmed
directly: a real GTX 1650 (Turing, compute capability 7.5) configured
to `75-real` with zero manual architecture flags.

### Non-standard Toolkit install location

If the Toolkit isn't at the standard `/usr/local/cuda` path (e.g. a
user-local, no-root install — this project's own real validation used
exactly this, see "Real evidence" below):

```bash
cmake -S . -B build-cuda \
  -DMEMBRANE_ENABLE_LLAMA=ON -DGGML_CUDA=ON \
  -DCUDAToolkit_ROOT=/path/to/toolkit \
  -DCMAKE_CUDA_COMPILER=/path/to/toolkit/bin/nvcc \
  -DCMAKE_BUILD_TYPE=Release
```

**A real, found-and-fixed build issue**: the final executable link can
fail with `undefined reference` to every `cudart`/`cublas` symbol, even
though `ggml-cuda.so` itself built successfully — `ld` names the real
cause directly (`libcudart.so.12, needed by ..., not found (try using
-rpath or -rpath-link)`): the CUDA runtime libraries aren't on the
linker's default search path when installed to a non-standard prefix.
Fix: set `LD_LIBRARY_PATH` to the Toolkit's `lib64` directory for the
build step:

```bash
export LD_LIBRARY_PATH=/path/to/toolkit/lib64:$LD_LIBRARY_PATH
cmake --build build-cuda -j2 --target membrane-run membrane
```

The same variable is needed at runtime too, for the same reason,
unless the Toolkit is at a standard, `ldconfig`-known location.

## What changed

**Nothing, in the runtime/planner code.** `gpu_device.h`'s own design
comment predicted this exactly ("a future CUDA/ROCm backend needs no
change here beyond a new CMake build flag") — confirmed directly:
`gpu_policy.c`, `context_recommender.c`, `compat_check.c`, and the
joint planner have zero backend-name-specific code, and this PR's own
real generation test succeeded with zero changes to any of them.

Only two small, CMake-level additions:
- `CMakeLists.txt`: a `GGML_CUDA` pre-check (mirrors the existing
  `GGML_VULKAN` one) — fails clearly, at configure time, if the
  Toolkit is missing.
- `tools/membrane-run/CMakeLists.txt`: `ggml-cuda` joins the existing,
  `TARGET`-guarded install/RPATH `foreach` loop the same way
  `ggml-vulkan` already does — a build without `GGML_CUDA=ON` is
  completely unaffected (confirmed: a full CPU/Vulkan rebuild + the
  entire local test suite still pass unchanged after this PR).

## Real evidence

Real hardware: **NVIDIA GeForce GTX 1650** (Turing, compute capability
7.5, 4096 MiB VRAM), driver 580.159.03, real CUDA 12.6.20 Toolkit
(installed to a user-local, no-root prefix via NVIDIA's own installer's
`--toolkit --toolkitpath=` mode — this dev host had no sudo password
available this session).

- `membrane-run --list-devices`: real `CUDA0 (NVIDIA GeForce GTX 1650)`
  entry with real memory figures, cross-checked directly against a real
  `nvidia-smi` invocation (exact device name/total memory match).
- `membrane-run --doctor`: `GPU backend compiled in`, `GPU device:
  CUDA0 (NVIDIA GeForce GTX 1650)` — no code change needed to report
  this; the existing, backend-agnostic diagnostic just worked.
- Real generation, `SmolLM2-135M-Instruct`, both native (F16) and
  adaptive (selected `q8`) KV precision, `--gpu-layers auto` → 30/30
  layers GPU-resident (full residency): real completion, real
  driver-reported GPU memory delta before/after, decode throughput
  167.8–175.7 tok/s. The exact same joint planner/fallback machinery
  every other backend uses ran unmodified (`planner.used: true`,
  `fallback.attempted: false` — the primary candidate succeeded
  immediately).

See `results/cuda-backend/validation.json` for full detail, and
`docs/compatibility.json`'s MC-23/MC-27/MC-28/MC-29 rows.

## Packaging

**No official CUDA-enabled `.deb` package exists yet.** Unlike Vulkan
(a single, small, always-in-standard-repos `libvulkan1` runtime
dependency `dpkg-shlibdeps` already auto-detects), a real CUDA runtime
is a much larger, versioned system dependency — hundreds of MB,
typically installed via NVIDIA's own repository or a user-local prefix
exactly like this PR's own real validation — not something
`dpkg-shlibdeps` can resolve into a simple `Depends:` line the way
`libvulkan1` already is. Building from source with `-DGGML_CUDA=ON`
(this PR's own real, tested path) is the supported route for now; a
real packaged CUDA variant is deferred, disclosed as a known
limitation rather than invented without a concrete distribution
strategy.

## Known limitations

- Tested against exactly **one** real device (GTX 1650) — not a claim
  about any other NVIDIA GPU/architecture, the same disclosed scope
  Vulkan's own validation already carries.
- `q5` compressed KV was not separately exercised on CUDA this phase
  (MC-29) — only native and adaptive-selected-`q8` were tested.
- No official `.deb` package — see "Packaging" above.
- No real GPU hardware in CI — the CI build-only leg (an
  `apt`-installed CUDA Toolkit on a GitHub-hosted runner) confirms real
  *compilation* only, never real generation; this dev host's own local
  run is the only source of real hardware evidence.
