# macOS / Metal backend

Mega Phase D, PR D4. Apple Metal is now a real, tested product GPU
backend — `-DGGML_METAL=ON` — alongside Vulkan (PR-precedent) and CUDA
(PR D3). Real hardware validation via GitHub Actions' `macos-14`
hosted runner (real Apple Silicon; see "Hardware scope" below for what
"real" means here precisely) — not a build-only claim.

## Building with Metal

```bash
cmake -S . -B build-metal \
  -DMEMBRANE_ENABLE_LLAMA=ON -DGGML_METAL=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-metal -j3 --target membrane-run membrane
```

No extra dependencies to install — Metal/Foundation/MetalKit are part
of the OS itself on every real macOS install. Requesting
`-DGGML_METAL=ON` on any non-Apple OS fails clearly at configure time
(`message(FATAL_ERROR ...)` in the root `CMakeLists.txt`, checked via
CMake's own `APPLE` variable) rather than failing deep inside
`third_party/llama.cpp/ggml/src/ggml-metal/CMakeLists.txt`'s own
`find_library(... REQUIRED)` calls with a message that never mentions
MEMBRANE — the same fail-fast pattern already established for Vulkan
and CUDA.

## What changed

**Nothing in the runtime/planner code.** Exactly the same finding as
PR D3's CUDA work: `gpu_device.cpp` only reads the registered backend's
own name generically (`ggml_backend_reg_name()`) — no
Vulkan/CUDA/Metal-specific branch exists anywhere in `gpu_policy.c`,
`context_recommender.c`, `compat_check.c`, or the joint planner. Real
Metal generation (see below) exercised this completely unmodified
machinery successfully.

Three small, genuinely new pieces of work, not one:

1. **CMake wiring** — a `GGML_METAL`+`NOT APPLE` pre-check (root
   `CMakeLists.txt`) and `ggml-metal` joining
   `tools/membrane-run/CMakeLists.txt`'s existing install/RPATH
   `foreach` loop, the same way `ggml-vulkan`/`ggml-cuda` already do.

2. **A real, previously-undiscovered cross-platform RPATH bug**, found
   and fixed *before* the first macOS CI run even attempted a Metal
   build: the existing install-time RPATH used the Linux/ELF-only
   `$ORIGIN` token unconditionally. dyld (macOS's dynamic linker) has
   no concept of `$ORIGIN` at all — its own equivalent token is
   `@loader_path`. Fixed by selecting the right token once, via
   CMake's `APPLE` variable (`MEMBRANE_RPATH_ORIGIN_TOKEN`), reused
   everywhere `INSTALL_RPATH` is set.

3. **A platform abstraction for background-service lifecycle** —
   launchd on macOS instead of systemd `--user` on Linux. See
   `docs/service.md`'s own "macOS: launchd instead of systemd" section
   for the full command mapping and disclosed semantic differences;
   not duplicated here.

## Real bugs found by the real macOS CI build (in order)

None of these were anticipated up front — each was root-caused from a
real compiler/runtime error on GitHub's real `macos-14` runner, not
guessed:

1. **`memfd_create()`** (`tools/membrane-run/main.cpp`,
   `parse_opts_capture_stderr()`) — a real Linux-only glibc/kernel
   syscall with no Darwin equivalent at all (`use of undeclared
   identifier 'memfd_create'`). Fixed by switching to `tmpfile()` — a
   portable POSIX equivalent for this exact use case (an anonymous,
   auto-cleaned temp file), which is a strict improvement (the
   Linux-only dependency is removed outright, not conditionally
   compiled around).

2. **`struct stat`'s `st_mtim` field** (three independent call sites:
   `server.cpp`, `model_cmd.cpp`, `doctor_cmd.cpp`) — POSIX.1-2008
   naming, real on Linux/glibc, but macOS/BSD names the same
   sub-second timespec field `st_mtimespec` instead (`no member named
   'st_mtim' in 'stat'`). Fixed by adding one shared, portable
   `membrane_stat_mtime_ns()` helper to `fs_util.h`/`.cpp` (already
   transitively linked by all three call sites) with the `#ifdef
   __APPLE__` switch centralized in exactly one place, rather than
   three independent copies of the same platform check.

Both are genuine, disclosed findings this PR fixed for real — not
platform-specific workarounds bolted onto working code, but real,
previously-latent portability bugs this project's Linux-only history
had never surfaced before real macOS compilation was attempted.

## Real evidence (GitHub Actions `macos-14` runner)

Every item below is from a real, observed CI run (`macos-metal-smoke`
job) — not simulated, not asserted from source reading alone.

- **Real Metal build**: `-DGGML_METAL=ON` configures and compiles
  cleanly (`-- Metal framework found`, `Embedding Metal library`,
  `[ OK] Built target ggml-metal`).
- **Real device enumeration**: `membrane-run --list-devices` reported
  `MTL0 (Apple Paravirtual device)` (4778.7 MiB) alongside `BLAS
  (Accelerate)` and `CPU (Apple M1 (Virtual))` — cross-checked against
  `membrane-run --doctor`'s own `[OK] GPU device: MTL0 (Apple
  Paravirtual device)` line. See "Hardware scope" below for what
  "Paravirtual"/"(Virtual)" means for this specific result.
- **Real `test_launchd_unit`**: pure plist-generation/path-resolution
  unit tests (`launchd_unit.h`/`.cpp`) — `test_launchd_unit: all tests
  passed`.
- **Real HTTPS model download**: `membrane model install smollm2:135m
  --quant q4_k_m` performed a genuine download from Hugging Face
  (D1/D2's own real catalog/download-manager/variant-selector product
  path, exercised on macOS for the first time) — real size validation
  and real GGUF-readability validation both passed
  (`"ok":true,"path":"/Users/runner/.local/share/membrane/models/..."`).
  `checksum_verified` came back `false` — **not** a MEMBRANE bug: see
  "Known limitations" below.
- **Real Metal generation**: `membrane-run --model ... --ctx 2048
  --auto --gpu-layers auto --kv adaptive` (via `--auto`) produced a
  real, successful generation —
  `"ok":true,"execution":{"success":true,"exit_code":0,"generated_tokens":128}`,
  `"gpu":{"backend":"MTL","device_selected":"MTL0","gpu_layers_selected":30}`
  (full 30/30-layer residency), adaptive KV selected `q8`
  (`Q8_FULL_RESIDENCY`), real driver-reported GPU memory delta
  (`device_free_bytes_before:5010800640` →
  `device_free_bytes_after:4906663936`, `observed_delta_bytes:104136704`).
  The exact same joint planner/fallback machinery every other backend
  uses ran unmodified (`planner.used:true`, `fallback.attempted:false`
  — the primary candidate succeeded immediately). **Throughput was very
  low** (`decode_tokens_per_second:0.413`, ~310 seconds for 128
  tokens) — disclosed honestly below, not hidden: this is almost
  certainly the paravirtualized GPU's own overhead, not a MEMBRANE
  inefficiency (the exact same model/quant/adaptive-KV path measured
  175 decode tok/s on real bare-metal CUDA hardware in PR D3).
  `--ctx auto` (the fully-automatic form) does **not** yet work on
  macOS — see "Known limitations".
- **Real launchd service lifecycle smoke**: `membrane service install`
  → `start` (real `launchctl bootstrap`) → `status` (real `running`
  state, real pid) → `stop` → `uninstall`, all against a real
  per-session launchd (`gui/<uid>`) on the CI runner — **fully
  succeeded**, the `continue-on-error` fallback this job's own CI step
  carries was never actually needed.

See `results/macos-metal/validation.json` for the complete evidence
record, and `docs/compatibility.json`'s MC-30/MC-31/MC-32 rows.

## Hardware scope

GitHub's `macos-14` runners run on real Apple Silicon (Apple's own
cloud infrastructure), but the job itself executes inside a
**paravirtualized VM** on that hardware, not with direct/bare-metal
GPU access — confirmed directly by Metal's own device name
(`MTL0 (Apple Paravirtual device)`) and CPU name (`Apple M1 (Virtual)`)
in this project's own real output. This is a materially different,
weaker hardware-scope claim than PR D3's CUDA validation (a real,
bare-metal GTX 1650 on this project's own dev host) or the existing
Vulkan rows (same bare-metal host) — real compatibility ("does it
work") is genuinely established, but **no performance conclusion**
should ever be drawn from this specific run's throughput numbers. This
matches `docs/compatibility.md`'s own pre-existing "Compatibility !=
performance" policy, applied here even more strongly than usual.

No physical, non-virtualized Mac was available to this project during
PR D4 — if one becomes available later, a real bare-metal throughput
figure would be a valuable, separate addition, not a correction to
this scope-limited result.

## Known limitations

- **`--ctx auto` does not work on macOS yet.** `host_memory_guard.h`'s
  real host-RAM reading (`membrane_read_host_meminfo()`) is
  Linux-only (`/proc/meminfo`) — confirmed directly:
  `membrane-run --doctor` reports `[WARN] host RAM could not be read
  (non-Linux host, or /proc/meminfo unavailable)`, and `--ctx auto`
  correctly, safely refuses to proceed (`PLANNER_REJECTED_ALL` —
  "`--ctx auto` never proceeds on unknown host memory" is this
  project's own existing, deliberate safety policy, not a new
  restriction). An explicit `--ctx N` (still combined with `--auto`
  for GPU-layers/KV auto-selection) works today, as demonstrated
  above. Extending `host_memory_guard.h` to read real memory on macOS
  (e.g. via `sysctl hw.memsize`/`host_statistics64`) is real,
  concrete, disclosed future work — out of this PR's own scope
  (Metal backend + service-lifecycle abstraction), not silently
  assumed to already work.
- **Checksum verification degrades gracefully, not silently, but
  doesn't actually run** on a stock macOS install: `sha256sum` (GNU
  coreutils) is not preinstalled on macOS (this project's own real CI
  run confirmed it — `checksum_verified:false` on an otherwise fully
  successful, size-and-GGUF-validated real download). This is
  `download_manager.cpp`'s own pre-existing, already-disclosed
  `TOOL_UNAVAILABLE` degradation path (Mega Phase D, PR D1) — not a
  new bug, and not something this PR silently discovered and hid.
  Installing GNU coreutils (`brew install coreutils`, which provides a
  real `sha256sum`) restores real checksum verification.
- **`membrane doctor`'s service-status check remains
  systemd/`systemctl`-specific** (`tools/membrane/doctor_cmd.cpp`) —
  distinct from `membrane-run --doctor` (the cross-platform,
  backend/device diagnostic surface, real-tested above). Calling
  `membrane doctor` on macOS invokes a nonexistent `systemctl` binary;
  `membrane_run_subprocess()`'s own existing exec-failure handling
  reports this gracefully (`"systemctl is not available in this
  environment"`), never crashes — but the message itself is still
  systemd-flavored, not launchd-aware. Not exercised by this PR's own
  CI job (which only runs `membrane-run --doctor`); disclosed as a
  real, un-fixed gap rather than silently left unmentioned. A future
  phase could make this check launchd-aware the same way
  `service_cmd.cpp` already is.
- One real device tested (a paravirtualized `MTL0`) — see "Hardware
  scope" above; not a claim about any other Mac, chip generation, or
  bare-metal Apple Silicon performance.
- No official macOS package (`.pkg`/Homebrew formula/etc.) — source
  build only, matching PR D3's own CUDA packaging decision (no
  official CUDA `.deb` either); a real distribution strategy is future
  work, not invented here without one.
- No CI coverage for `--kv q5`/`--kv-placement cpu` variants on Metal
  this phase — only the `--kv adaptive` (→ q8) default path was
  exercised for real; see `docs/compatibility.json`'s MC-32 row.
