Platform / backend support matrix — v0.8.0

Mega Phase D, PR D8. Evidence states used below:

- **VALIDATED_REAL** — a real, physical (or real cloud-VM) device was
  used to produce this evidence directly (a human/agent-run command, or
  a CI job running on real hardware, e.g. a real per-PR generation).
- **VALIDATED_CI** — the SAME real validation runs automatically on
  every PR via a named CI job (so it is continuously re-verified, not a
  one-time snapshot) — a subset of VALIDATED_REAL, called out
  separately here specifically because "ran once, in the past" and
  "runs on every single change" are different strengths of claim.
- **CONFIGURATION_ONLY** — compiles/configures correctly; no real
  runtime execution evidence exists.
- **NOT_VALIDATED** — not attempted, reason given.
- **UNSUPPORTED** — a real, confirmed incompatibility.

Never write "supported" where only compilation succeeded — see
`scripts/verify-release-v0.8.0.py`'s own overclaim checks.

## Platform × backend matrix

| Platform | Backend | State | Evidence |
|---|---|---|---|
| Linux | CPU | VALIDATED_CI | Every CI job in this project (`build-and-test`, sanitizers, `packaging-smoke`, etc.) — the default backend, exercised on every single commit. |
| Linux | Vulkan | VALIDATED_REAL | This project's own real dev host (AMD Ryzen 5 5600H w/ integrated Radeon Graphics + a real discrete NVIDIA GTX 1650, both enumerated via Vulkan) — `results/v0.3/gpu-vulkan-validation.json`, `results/v0.4/adaptive_vulkan_matrix.jsonl`, reconfirmed this phase (`results/release-v0.8.0/readiness.json`). `packaging-smoke`'s own Vulkan leg additionally compiles/installs it on every PR (CI has no real GPU there — build/install only). |
| Linux | CUDA | VALIDATED_REAL | This project's own real dev host's real NVIDIA GTX 1650 (Turing, compute 7.5) with a real, locally-installed CUDA 12.6 toolkit — `results/cuda-backend/validation.json` (Mega Phase D, PR D3), reconfirmed this phase (`results/release-v0.8.0/readiness.json`). `cuda-backend-build-smoke` additionally compiles it on every PR against a real CUDA toolkit on `ubuntu-latest` — **compile-only, no GPU on that runner** (explicit in the job's own name and its `Built binary reports CUDA compiled in (no device present)` step). |
| macOS | CPU | VALIDATED_CI (bundled with the Metal job) | `macos-metal-smoke` (`macos-14` hosted runner) runs `--doctor`/`--list-devices` and real launchd lifecycle on every PR; the job's own real generation step passes `--gpu-layers auto`, so it is not a CPU-*only*-forced configuration — see the Metal row below for the actual generation evidence. |
| macOS | Metal | VALIDATED_CI | `macos-metal-smoke` — real device enumeration, real HTTPS model download, real generation (`--gpu-layers auto`) on every PR. **Real but paravirtualized**: the device is GitHub's own `MTL0 (Apple Paravirtual device)`, real Apple Silicon cloud infrastructure but not bare-metal GPU access (confirmed by the device's own self-reported name) — no performance claim should be drawn from its throughput, only functional correctness. `results/macos-metal/validation.json` (Mega Phase D, PR D4). |
| Windows | CPU | VALIDATED_CI | `windows-support-smoke` (`windows-latest` hosted runner) — real install, real device enumeration, real HTTPS model download, real generation, real Task Scheduler service lifecycle, on every PR. `results/windows-support/validation.json` (Mega Phase D, PR D5). |
| Windows | CUDA | NOT_VALIDATED | No discrete GPU exists on the `windows-latest` hosted runner (confirmed directly, not inferred) and no other real Windows+NVIDIA environment is available to this project. Never claimed. |
| Windows | Vulkan | NOT_VALIDATED | Same reasoning as Windows CUDA — no discrete GPU on the CI runner, no other real Windows GPU environment available. |

## Known, still-true platform gaps (re-verified this phase)

- **`--ctx auto` has no real host-RAM reading on Windows or macOS** —
  `host_memory_guard.c`'s real memory reader is `/proc/meminfo`-based
  (Linux-only); re-confirmed by source read this phase (no `_WIN32`/
  `__APPLE__` branch exists in that file). On macOS this makes `--ctx
  auto` fail closed entirely (`PLANNER_REJECTED_ALL`); on Windows the
  GPU/KV auto-selection path still works (it doesn't need a host-RAM
  fact when there's no GPU to reason about), but a real context-SIZING
  decision would hit the identical gap. Extending `host_memory_guard.c`
  to real platform-native memory APIs (`GlobalMemoryStatusEx`/
  `sysctl hw.memsize`) is real, disclosed future work, not attempted
  this phase (Section 15 of the D8 task: this phase's own long-context
  fix is a different, narrower issue — see `docs/server.md`'s new
  `CTX_TOO_SMALL_FOR_PROMPT` row — not a general host-RAM-probing fix).
- **No official Windows or macOS package exists** — see "Package
  policy" below; source build only on both platforms.
- **Checksum verification degrades gracefully on stock macOS** — no
  `sha256sum` preinstalled; real size/GGUF validation still happens,
  real checksum verification needs `brew install coreutils` separately
  (unchanged from PR D4).
- **CUDA/Vulkan real hardware evidence is single-GPU** — one real
  device each (GTX 1650 for both), not a claim about any other NVIDIA/
  AMD/Intel GPU generation.

## Package policy (v0.8.0)

Only Linux ships an official package this release, matching the
platform evidence above exactly:

- `membrane_0.8.0_amd64.deb` — Vulkan-enabled, the official asset.
- `membrane-cpu_0.8.0_amd64.deb` — CPU-only variant, also official
  (unchanged policy since v0.3.0/v0.4.0, `results/release-artifacts/
  manifest.json`).

No Windows or macOS package this release — real per-PR CI validation
exists for both platforms (see the matrix above), but no packaging
pipeline (`.zip`/`.msi`/`.pkg`/Homebrew) has been built and validated
from an actual release artifact yet (Sections 20/21 of the D8 task:
"do not publish unless the exact artifact itself was validated"). Both
remain source-build-only, exactly as disclosed since PR D4/D5.
