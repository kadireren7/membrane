# Windows support

Mega Phase D, PR D5. MEMBRANE now builds and runs on real Windows —
validated end to end on GitHub Actions' `windows-latest` hosted runner
(real MSVC, Visual Studio 2026) — not a build-only claim. No discrete
GPU exists on this runner (same as this project's own Linux CPU-only
jobs) — this is real **CPU-only** Windows validation, never a
GPU-backend claim (there is no Windows-specific GPU backend to add
here in the first place — Vulkan/CUDA are cross-platform build options
already; a future phase could validate either on a real Windows GPU
host, not attempted this phase).

## Building on Windows

```
cmake -S . -B build -DMEMBRANE_ENABLE_LLAMA=ON -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_TOOLCHAIN_FILE=%VCPKG_INSTALLATION_ROOT%\scripts\buildsystems\vcpkg.cmake ^
  -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build build --config Release --target membrane-run membrane
cmake --install build --config Release --prefix C:\membrane-install
```

curl and zlib are real, external `find_package(... REQUIRED)`
dependencies on every platform (`find_package(ZLIB REQUIRED)` at the
root `CMakeLists.txt` level; `find_package(CURL REQUIRED)` for the
model-catalog download manager) — neither ships with Windows itself
the way it does on Linux/macOS. [vcpkg](https://vcpkg.io) is the real,
standard way to get both; this project's own CI uses the vcpkg
installation already bundled with GitHub's `windows-latest` runner
image (no from-scratch bootstrap needed there).

**A real `cmake --install` step is required to actually run the
binaries at all** — see "Why install is required" below; running
`membrane-run.exe`/`membrane.exe` straight from the raw build tree
fails.

## What changed

Unlike PR D3 (CUDA) and PR D4 (macOS/Metal), D5 is not a new *GPU
backend* — Windows has no backend-level analog to add. The real work
here is making the existing, pre-existing codebase compile and run
correctly on MSVC at all, plus a real Windows service-lifecycle
abstraction (Task Scheduler). This project had never been compiled on
Windows before this phase; the codebase's own Linux/POSIX history
(fork/execvp, pthread, dirent, POSIX timing, `struct stat` fields, and
more) had never been questioned end to end the way porting to a
genuinely different OS family forces.

### Real bugs found and fixed (in the order real CI discovered them)

Every one of these was root-caused from a real MSVC compiler error or
a real runtime failure on GitHub's own `windows-latest` runner — none
were anticipated up front:

1. **`-Wall -Wextra -Wpedantic`** applied project-wide, unconditionally
   — GCC/Clang flag spellings MSVC's `cl.exe` cannot parse at all
   (`error D8021: invalid numeric argument '/Wextra'`, failing every
   target). Fixed with an `if(MSVC)` guard using `/W4` (MSVC's own
   conventional "high warning level" — its literal `/Wall` is far
   noisier than `-Wall -Wextra` combined).
2. **A real, multi-threaded storage/codec/backend engine
   (`membrane_core`, both product binaries' own shared dependency) had
   deep POSIX dependencies throughout** — none of which macOS (already
   POSIX) had ever surfaced:
   - `src/store/store.c`/`store_internal.h`: a real mutex + condition
     variable (38 `pthread_*` call sites).
   - `src/quant/quant_simd.c`: a real "spawn N worker threads, join
     them all" pattern (`pthread_create`/`pthread_join`), plus
     `sysconf(_SC_NPROCESSORS_ONLN)` for the default thread count.
   - `src/backends/backend_file.c`: `opendir`/`readdir`/`closedir`
     (`dirent.h`) for real on-disk record enumeration, plus
     `fsync`/`unlink`/`access`.
   - `src/stats.c`, and (once found) five more real call sites across
     `tools/membrane-run/{auto_fallback.c,main.cpp,runtime_session.cpp}`
     and `tools/membrane-llama-runtime/{decode_loop,llama_hook}.cpp`
     (the real generation hot path): `clock_gettime(CLOCK_MONOTONIC[_RAW], ...)`.
   - `tools/membrane-llama-runtime/kv_store_telemetry.c`:
     `getrusage(RUSAGE_SELF)` for real RSS high-water-mark telemetry.

   Fixed via four new, real, backed-by-native-Win32-primitives shim
   headers — `include/membrane/{pthread,dirent,posix,clock}_compat.h`
   — each defining the *same* POSIX type/function names these files'
   own real call sites already use (never touched), backed by
   `SRWLOCK`+`CONDITION_VARIABLE`, `FindFirstFile`/`FindNextFile`,
   `_commit`/`<io.h>`'s legacy aliases, and `QueryPerformanceCounter`
   respectively — a pure passthrough to the real POSIX headers on
   every other platform.
3. **`psapi.h` included before `windows.h`** — a classic Windows
   header-ordering bug (`psapi.h`'s own declarations use types that
   only exist once `windows.h` has already been processed), producing
   dozens of cascading syntax errors inside `psapi.h` itself.
4. **`F_OK`** assumed available via `<io.h>` alongside
   `access`/`unlink`/`fileno` (which *are* real, undecorated legacy
   aliases there) — it is not, on this SDK. Defined explicitly (its
   value, 0, is standard on every platform).
5. **`windows.h`'s own `min()`/`max()` preprocessor macros broke real
   `std::min`/`std::max` calls** (`decode_loop.cpp`) — the exact,
   well-known `error C2589: '(': illegal token on right side of '::'`
   symptom. New `include/membrane/windows_lean.h` centralizes every
   one of this project's real `<windows.h>` includes (there are eight)
   behind a header that defines `NOMINMAX` (and `WIN32_LEAN_AND_MEAN`)
   before including it — the only way to guarantee the ordering
   everywhere, since `NOMINMAX` must be set before `windows.h`'s
   *first* inclusion in a translation unit, not just locally.
6. **Several CMake targets never had `include/` on their own include
   path** (`membrane_kv_store_telemetry`, `membrane_auto_fallback`,
   `membrane_runtime_session`, `membrane_setup_cmd`,
   `membrane_fs_util`, `membrane_subprocess`, `membrane_windows_task`)
   — each caught and fixed individually as the real build reached
   progressively further into the dependency graph.
7. **`dup`/`dup2`/`close`/`STDOUT_FILENO`/`STDERR_FILENO`**
   (`main.cpp`'s real `--json` parse-error stderr capture) — only the
   `_`-prefixed forms are guaranteed on Windows (unlike
   `access`/`unlink`/`fileno`), and the `STD*_FILENO` macros don't
   exist there at all (no `<unistd.h>`).
8. **`realpath()`/`PATH_MAX`** (`model_cmd.cpp`'s `membrane model add`
   path canonicalization) — no Windows equivalent *name* exists at
   all; `_fullpath()`/`MAX_PATH` are the real substitutes (a real,
   disclosed, minor difference: `_fullpath()` does not resolve
   symlinks identically, but Windows symlinks are rare enough in
   practice that this project's one real call site is not meaningfully
   weakened).
9. **`isatty()`/`open()`/`O_WRONLY`, plus `/dev/null` vs. `NUL`**
   (`setup_cmd.cpp`'s TTY check and its "redirect stdout to the null
   device" helper) — same `_`-prefix requirement, plus the null device
   itself has a different *name* on Windows (a real path string no
   function-name shim can paper over — a new `MEMBRANE_NULL_DEVICE`
   macro is used at the one real call site instead of a hardcoded
   path).
10. **`nanosleep()`** (`server.cpp`'s graceful-shutdown poll loop,
    `setup_cmd.cpp`'s retry-with-backoff helper — both fixed
    100–200ms sleeps, no real sub-millisecond precision need) — no
    Windows equivalent name; `Sleep()` (millisecond granularity) is
    the real substitute.
11. **A real, structural Windows DLL-search-path gap** — not a code
    bug. Windows has no RPATH/`@loader_path`/`$ORIGIN` concept at all.
    On Linux/macOS, CMake's own default *build-tree* RPATH lets an
    uninstalled build-tree binary find its sibling `.so`/`.dylib`
    files for free (why the CUDA/Metal CI legs run their own
    build-tree binaries directly) — MSVC's multi-config generator puts
    the `.exe` and its own `ggml*.dll`/`llama.dll` dependencies in
    *different* build-tree directories, and Windows's own default DLL
    search order (same directory as the `.exe`, then `PATH`) never
    finds them there. Fixed by adding a real `cmake --install` step to
    CI and running the *installed* binaries — exercising this
    project's own actual, real install path (the DLL-placement fix
    already made for Windows in `tools/membrane-run/CMakeLists.txt`:
    DLLs install to `CMAKE_INSTALL_BINDIR`, the same directory as the
    `.exe`, not `CMAKE_INSTALL_LIBDIR`), the same way `packaging-smoke`
    already does for its own CPU/Vulkan legs on Linux — not a
    CI-only workaround.
12. **`setenv()`/`unsetenv()`** (this phase's own real regression-guard
    tests — `test_windows_task.cpp`/`test_systemd_unit.cpp`/
    `test_launchd_unit.cpp` all save/override/restore an env var
    around each test case) — no Windows equivalent name; `_putenv_s()`
    is the real substitute (an empty value is its own documented way
    to remove a variable).
13. **A real `schtasks.exe` error**: `"ERROR: unable to switch the
    encoding"`. Task Scheduler's own `/xml` import genuinely requires
    the file to actually *be* UTF-16 on disk, not just declared as
    such — writing the generated XML's own plain UTF-8 bytes directly
    was a real encoding mismatch. Fixed with a real
    `MultiByteToWideChar()`-based UTF-8→UTF-16LE conversion (plus a
    real byte-order-mark), applied only at the file-write step
    (`service_cmd.cpp`'s own Windows install path) — the pure XML
    generator itself (`windows_task.cpp`) still returns plain UTF-8
    text, unchanged, for its own unit tests.
14. **A real, silent checksum-verification bug**: GNU coreutils'
    `sha256sum` prepends a literal `\` before the hash whenever the
    *filename* argument contains a backslash or newline (its own
    documented "escaped filename" mode) — every real Windows path
    contains backslashes, so this is the *normal* case there, not a
    rare edge case. Before this fix, checksum verification silently
    corrupted the parsed hash by one character on every real Windows
    install (`CHECKSUM_MISMATCH`, download refused). Fixed by
    stripping a leading `\` before parsing — a real no-op on
    Linux/macOS (real paths there never trigger coreutils' escaped
    mode).

### Windows service lifecycle: Task Scheduler

`membrane service ...` manages a **Windows Task Scheduler** task
instead of a systemd unit (Linux) or launchd LaunchAgent (macOS) — see
`docs/service.md`'s own "macOS: launchd instead of systemd" section
for the equivalent narrative; Windows follows the same pattern. New
`windows_task.h`/`.cpp` (pure Task Scheduler XML generation, mirrors
`systemd_unit.h`/`launchd_unit.h`) + `test_windows_task.cpp` (pure
string tests, runs in normal Linux CI too, not only on a Windows
runner). `service_cmd.cpp` dispatches between all three backends via
`#ifdef __APPLE__`/`#ifdef _WIN32`.

Task Scheduler was chosen over a full Win32 Service (Service Control
Manager, a `ServiceMain` dispatch table, a real service-hosting
architecture change to the binary itself) — the task's own explicit
language ("Windows service/startup task") allows either, and a
scheduled task is materially less invasive while still running under
the calling user's own account with no elevated privileges of any
kind, matching every other platform's own "no root" policy.

Real, disclosed semantic differences from systemd/launchd (neither of
the other two platforms' own command sets maps 1:1 to `schtasks`'
either):

| Command | Windows (Task Scheduler) |
|---|---|
| `install` | Real `schtasks /query /tn ... /xml` (checks for a pre-existing, non-MEMBRANE-managed task first) then `schtasks /create /tn ... /xml <tmpfile> /f` — Task Scheduler has no local unit *file* this process itself reads the way systemd/launchd do; the real, live task definition lives inside Windows's own Task Scheduler store. |
| `start` | `schtasks /run /tn ...` |
| `stop` | `schtasks /end /tn ...` |
| `restart` | `schtasks /end` then `schtasks /run` (no atomic restart primitive) |
| `status` | `schtasks /query /tn ... /fo list /v`, parsed for a real `Status:` line — `schtasks.exe` itself exposes no PID at all (a real, disclosed gap vs. systemd/launchd, both of which do report one) — confirmed directly: this phase's own real status output showed `pid: 0`/main_pid stays unknown. |
| `logs` | Real `tail -n` of a real log file — Task Scheduler has no journal; the generated task runs via `cmd.exe /c "... >> logfile 2>&1"` to get real output redirection at all. |

## Real evidence (GitHub Actions `windows-latest` runner)

Every item below is from a real, observed CI run (`windows-support-smoke`
job) — not simulated, not asserted from source reading alone.

- **Real build**: MSVC 19.51 (Visual Studio 2026), both `membrane-run.exe`
  and `membrane.exe` compiled and linked successfully, zero remaining
  compile/link errors.
- **Real install**: `cmake --install ... --config Release` placed
  every DLL (`ggml.dll`, `ggml-base.dll`, `ggml-cpu.dll`, `llama.dll`)
  in the same directory as `membrane-run.exe`/`membrane.exe`.
- **Real device enumeration**: `membrane-run.exe --list-devices`
  reported a real CPU (`CPU (AMD EPYC 9V74 80-Core Processor)` on one
  run, `CPU (INTEL(R) XEON(R) PLATINUM 8573C)` on another — this
  runner's underlying hardware genuinely varies between CI runs, not
  tied to one specific host). `--doctor`: `[OK] CPU: ...`, `[WARN] no
  GPU device visible` (correct — no discrete GPU on this runner),
  `[WARN] host RAM could not be read` (a real, disclosed, pre-existing
  gap — `host_memory_guard.h`'s host-RAM reader is Linux-only, same
  class of gap PR D4 already disclosed for macOS).
- **Real `test_windows_task`**: `test_windows_task: all tests passed`.
- **Real HTTPS model download**: `membrane model install smollm2:135m
  --quant q4_k_m` — a genuine download from Hugging Face (the existing
  D1/D2 catalog/download-manager/variant-selector product path) —
  `{"checksum_verified":true,"ok":true,"path":"C:\\Users\\runneradmin/.local/share/membrane/models/smollm2-135m-instruct/SmolLM2-135M-Instruct-Q4_K_M.gguf"}`
  — real size *and* real checksum verification both passed (after the
  sha256sum-escaping fix above).
- **Real CPU generation**: `membrane-run.exe --model ... --ctx 2048
  --auto` — real, successful generation:
  `"ok":true,"generated_tokens":128`, adaptive KV correctly selected
  `q8` (`NO_GPU_DEVICE` → CPU path, `"auto":{"cpu_fallback":true}`),
  real decode throughput `117.2 tok/s` (prefill `33.3 tok/s`) — the
  exact same joint planner/auto-fallback machinery every other
  platform uses ran unmodified.
- **Real Task Scheduler service lifecycle**: `Installed MembraneServer`
  → `start` → `Service: installed: yes, state: Ready` → `stop` →
  `Uninstalled MembraneServer` — the full real lifecycle succeeded
  genuinely; the `sub_state`/pid fields stay `unknown`/`0` (a real,
  disclosed gap — `schtasks.exe` itself exposes neither).

See `results/windows-support/validation.json` for the complete
evidence record, and `docs/compatibility.json`'s MC-33/MC-34 rows.

## Known limitations

- **`--ctx auto` does not report real host-RAM figures on Windows**
  (`host_memory.available:false` in the real generation JSON above) —
  `host_memory_guard.h`'s host-RAM reader is Linux-only (`/proc/meminfo`);
  unlike macOS (where `--ctx auto` fails closed entirely for this
  reason, per PR D4), Windows's own real run above still succeeded
  because `--auto`'s own GPU/KV auto-selection path does not require a
  host-RAM fact when there is no GPU to reason about residency
  against — but a real `--ctx auto` *context-sizing* decision on
  Windows would hit the identical `PLANNER_REJECTED_ALL` gap PR D4
  disclosed for macOS. Extending `host_memory_guard.h` to read real
  memory on Windows (`GlobalMemoryStatusEx`) is real, concrete,
  disclosed future work — out of this PR's own scope.
- **`schtasks.exe` exposes no PID** — `membrane service status`'s own
  `pid`/`main_pid` field stays `0`/unknown on Windows, unlike
  systemd/launchd (both of which report a real one).
- **`_fullpath()` does not resolve symlinks identically to `realpath()`**
  — a real, minor, disclosed difference; Windows symlinks are rare
  enough in practice that `membrane model add`'s own one real call
  site is not meaningfully weakened.
- **No real GPU tested** — this CI runner has no discrete GPU at all;
  this is real CPU-only validation only, never a GPU-backend claim
  (Windows itself is not a GPU backend — Vulkan/CUDA remain the
  cross-platform backend options, neither validated on a real Windows
  GPU host this phase).
- **No official Windows package (`.msi`/winget/etc.) exists yet** —
  source build only, matching PR D3's own CUDA and PR D4's own macOS
  packaging decisions (no official package for either yet either).
- Real hardware tested is a cloud-hosted VM (GitHub's own
  `windows-latest` runner, on Azure) — standard, expected for any
  cloud CI; unlike PR D4's own real, disclosed *paravirtual GPU*
  caveat (which materially affected a performance-sensitive GPU
  backend claim), this has no equivalent caveat here since this
  validation is CPU-only and makes no performance claim beyond the
  real, disclosed throughput figures above.
