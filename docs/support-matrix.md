# Platform support matrix

Evidence labels, strictest first:

- **real hardware**: a physical device (or a real cloud VM) ran it, by a
  person or a CI job on that hardware.
- **CI**: runs automatically on GitHub-hosted runners on every pull request.
  Real execution, but on virtualized infrastructure.
- **compile only**: builds and links; nothing was run on a device.
- **not validated**: not attempted. Never claimed.
- **unsupported**: known not to work.

"Supported" is not a single column here, on purpose. A platform is *packaged*
when the release ships an artifact that was extracted and run by the
acceptance test (`scripts/package-acceptance.sh`) in CI; a backend is
*validated* only at the strength written below.

## Platforms and packages

| Platform | Official artifact | Architecture | Signed | Package acceptance test |
|---|---|---|---|---|
| Linux | `membrane_<version>_amd64.deb` (Vulkan-enabled, runs CPU-only) and `membrane-cpu_<version>_amd64.deb` | amd64 | n/a (checksums in `SHA256SUMS`) | CI: the extracted `.deb` |
| macOS | `membrane_<version>_macos_arm64.tar.gz` | arm64 (Apple Silicon) | **no** (unsigned, not notarized) | CI: the extracted archive |
| Windows | `membrane_<version>_windows_x86_64.zip` | x86_64 | **no** (unsigned) | CI: the extracted ZIP |

No Intel-Mac artifact (CI has no x86_64 macOS runner in use), no Windows on Arm,
no `.pkg`, `.msi`, Homebrew, winget or Chocolatey package: none was built and
tested, so none is offered.

## Platform x backend matrix

| Platform | Backend | Evidence | Notes |
|---|---|---|---|
| Linux | CPU | CI, on every job | The default backend. |
| Linux | Vulkan | **real hardware** | Real dev host with an integrated AMD GPU and a discrete NVIDIA GTX 1650 (`results/v0.3/gpu-vulkan-validation.json`, `results/v0.4/adaptive_vulkan_matrix.jsonl`). One machine, not a claim about every GPU. |
| Linux | CUDA | **real hardware** (earlier release) and compile only in CI | GTX 1650 with CUDA 12.6 (`results/cuda-backend/validation.json`). CI compiles it against a real toolkit but has no GPU. Source-build option, not in the official package. |
| macOS | CPU | CI | Real generation from the extracted archive. |
| macOS | Metal | CI | Real generation (`--gpu-layers auto`) on GitHub's macOS runner; the device is `Apple Paravirtual device`, so this proves function, not performance, and is not bare-metal GPU validation. |
| Windows | CPU | CI | Real model download, generation, server, API and Task Scheduler lifecycle from the extracted ZIP. |
| Windows | CUDA | **not validated** | No Windows NVIDIA hardware available. Not shipped, not claimed. |
| Windows | Vulkan | **not validated** | Same. Not shipped, not claimed. |

## Known platform limits

- GPU evidence is single-device per backend (GTX 1650 for Vulkan/CUDA; one
  virtualized Metal device).
- macOS and Windows artifacts are unsigned: Gatekeeper and SmartScreen will
  warn. Signing and notarization can be added to the release workflow later;
  none is faked now. See [install.md](install.md).
- There is one real-hardware environment for Linux; macOS and Windows are
  validated on hosted CI only.
- Only Linux has a real-hardware GPU result. Do not read "validated" on macOS
  or Windows as a GPU-performance statement.

## Package policy

`membrane` (the Vulkan-enabled `.deb`) is the official Linux package; it runs
correctly CPU-only and uses a GPU only when asked. `membrane-cpu` is a
CPU-only variant for people who want to avoid the `libvulkan1` dependency.
The Windows ZIP and macOS archive are portable: extract, run, delete to
remove. They contain only `bin/` (and `lib/` on macOS), `LICENSE`, the
third-party license text, `README.txt` and `PACKAGE-INFO.json`. See
[results/release-artifacts/manifest.json](../results/release-artifacts/manifest.json).

## Runtime support matrix (Milestone L)

What MEMBRANE can do with each inference **runtime** (not backend):

| Runtime | Inspect / list models | Observe | Advise | Apply (change state) | Adaptive | Orchestration (`membrane run`) |
|---|---|---|---|---|---|---|
| membrane-native | yes | yes | yes | yes: context, KV precision/placement, GPU layers (via `server.json` overrides, next load), unload, pin/unpin, activate | yes (opt-in) | yes, with a Planner v2 plan |
| Ollama (v0.34.3 audited) | yes | yes (`/api/ps`) | yes | **unload only** (explicit `--unload`; `keep_alive=0`); no persistent context/GPU/KV control exists | observe / recommend only | yes, from reported facts (no Planner-grade accuracy) |
| vLLM (v0.31.0 audited) | yes | inspect only (health, models, load) | not implemented | **none** (model/context/KV fixed at launch) | no | yes, as a resident, read-only candidate |

Evidence: Ollama and vLLM are tested against in-process mock servers that
follow the audited response shapes; no real Ollama or vLLM daemon was run for
this milestone. Multi-GPU: topology and device visibility only; no
distributed-execution claim ([orchestration.md](orchestration.md)).

**Runtime availability depends on the platform, and MEMBRANE does not hide
that.** The adapters are generic, and each only ever talks HTTP to
`127.0.0.1`; whether a given runtime exists on your machine is up to that
runtime:

| Runtime | Linux | macOS | Windows |
|---|---|---|---|
| membrane-native | packaged (`.deb`) | packaged (archive) | packaged (ZIP, CPU) |
| Ollama | adapter tested against mocks; the daemon is separate software | same | same |
| vLLM | adapter tested against mocks; vLLM's own support is Linux-first (NVIDIA/AMD GPUs) | not a supported vLLM platform upstream for GPU serving; the adapter simply reports "not reachable" | same |

`membrane run` and `doctor` detect runtimes by trying their loopback API, not
by looking for a binary in a platform-specific location, so there is no
Linux-path assumption. MEMBRANE never installs, launches or controls Ollama or
vLLM.

## Cross-platform behavior (same code, different facts)

Planning, observation, advice, `apply` and the adaptive controller have one
implementation. Platform differences enter only through measured facts and
runtime capabilities:

| Area | Linux | macOS | Windows |
|---|---|---|---|
| Host RAM total/available | `/proc/meminfo` (`MemTotal`, `MemAvailable`) | `sysctl hw.memsize`; available = free + inactive pages from `host_statistics64` | `GlobalMemoryStatusEx` (`ullTotalPhys`, `ullAvailPhys`) |
| Swap reported | yes | no (`unknown`) | no (`unknown`) |
| Unreadable memory | `unknown`, never `0` | same | same |
| SHA-256 for downloads | built in | built in | built in |
| Service manager | `systemd --user` | `launchd` | Task Scheduler |
| `apply`: native controls | context, KV precision/placement, GPU layers, unload, pin, activate | same (capability-gated by the runtime) | same (capability-gated by the runtime) |
| GPU facts | Vulkan/CUDA when built in | Metal | none claimed |

The macOS "available" figure counts free plus inactive pages. It is a
reasonable estimate of memory that can be reclaimed without swapping, but it
is not identical to Linux `MemAvailable`; observations are labeled with their
source (`mach_host_statistics64`) so the difference is visible.
