# MEMBRANE v1.2.0

Hardware-aware memory intelligence and control for local LLM inference.

v1.1.0 made MEMBRANE something you can understand and steer, but it was a
Linux product that happened to compile elsewhere. v1.2.0 makes macOS and
Windows real install targets: official portable archives, native data
locations, product-owned checksums, and an acceptance test that runs the
**extracted artifact** on each OS. Everything is **additive**: the
OpenAI-compatible API (v1), the `--json` shapes and the Linux data locations
are unchanged ([upgrade-v1.1-to-v1.2.md](upgrade-v1.1-to-v1.2.md)).

## What is new

| OS | Artifact | Architecture | Validated backend | Signed |
|---|---|---|---|---|
| Linux (Debian family) | `membrane_1.2.0_amd64.deb` (Vulkan-enabled, runs CPU-only), `membrane-cpu_1.2.0_amd64.deb` | amd64 | CPU, Vulkan (real hardware) | n/a |
| macOS | `membrane_1.2.0_macos_arm64.tar.gz` | arm64 (Apple Silicon) | CPU, Metal (CI; paravirtualized GPU) | **no** |
| Windows | `membrane_1.2.0_windows_x86_64.zip` | x86_64 | CPU (CI) | **no** |

`SHA256SUMS` covers every artifact; the `.deb` files have SBOMs. The macOS and
Windows archives are **unsigned and not notarized**: Gatekeeper and SmartScreen
will warn, and [install.md](install.md) says how to proceed safely or build
from source instead.

* **Package acceptance.** `scripts/package-acceptance.sh` runs a fresh-user
  sequence against the extracted artifact on all three OSes: version, status
  screen, `doctor`, `models available`, a real tiny-model download over HTTPS
  with checksum verification, `plan` with measured host memory, `serve`,
  `/health`, `/v1/models`, a chat completion, no CORS header, `observe`,
  `advise`, `apply` (dry run), `adaptive status`, `run --explain`, shutdown,
  and the official Python `openai` SDK harness (models, chat, streaming,
  rejected tool calls). Windows also runs the Task Scheduler lifecycle from the
  extracted ZIP. The same script found three real bugs before release (below).
* **Native locations.** macOS: `~/Library/Application Support/membrane`.
  Windows: `%LOCALAPPDATA%\membrane`. Linux: unchanged XDG paths. `XDG_*`
  overrides win everywhere, and an existing legacy directory keeps being used.
  Nothing is moved.
* **Product-owned SHA-256.** Downloads are verified by MEMBRANE itself, not by
  `sha256sum`, which stock macOS and Windows do not have.
* **`doctor`** gains a `platform` check: OS, architecture, the resolved
  directories, the service manager, and what evidence backs this platform.
* **Fixes found by running the real packages:** Windows `mkdir` of drive-rooted
  paths failed (`C:`); the installed macOS `membrane` could not find its
  libraries (`$ORIGIN` is ELF-only); the Windows ZIP lacked the curl/zlib DLLs;
  Windows console color is enabled only after virtual-terminal processing is
  accepted.
* Host RAM was already read natively on all three OSes (`/proc/meminfo`,
  `sysctl`+`host_statistics64`, `GlobalMemoryStatusEx`) since v1.0.0; stale
  documentation that called it a gap is corrected.

## What did not change

The OpenAI-compatible chat API subset, the loopback-only unauthenticated
security boundary (no CORS headers; installing opens no port and changes no
firewall), planning/observe/advise/apply/adaptive logic (one implementation;
platforms differ only through measured facts and runtime capabilities), and the
Linux `.deb` packages and data locations.

## Why 1.2.0 and not 1.1.1 or 2.0.0

New official artifacts for two operating systems and a new doctor check are a
substantial additive capability, so more than a patch. No frozen contract
broke: API v1 and every `--json` shape are unchanged (the doctor JSON gains
one check; the CLI contract permits that within a major version). Data
locations changed only for macOS/Windows users who had no package, and their
existing registry is kept in place. Hence a minor release.

## Evidence and honest limits

* CI on the merged milestone PR: all 76 checks green, including the
  extracted-artifact acceptance on Linux, macOS and Windows.
* Real hardware: Linux only (CPU and Vulkan on the maintainer's machine). macOS
  and Windows are validated on GitHub-hosted runners; the macOS GPU there is a
  paravirtualized Metal device, so this proves function, not performance.
* **Windows GPU backends (CUDA, Vulkan) are not validated on real hardware and
  not shipped.** The Windows ZIP is a CPU build.
* No Intel-Mac artifact, no Windows on Arm, no `.pkg`/`.msi`, no Homebrew,
  winget or Chocolatey package: none was built and tested.
* macOS "available memory" is free plus inactive pages, an estimate of
  reclaimable memory, labeled with its source.
* Signing and notarization are not done; the pipeline can add them later.
* Unchanged from v1.1.0: adaptive behavior validated on a synthetic timeline;
  Ollama and vLLM tested against mock servers; no multi-GPU execution; CXL and
  FPGA remain research only.

Details: [results/release-v1.2.0/readiness.json](../results/release-v1.2.0/readiness.json),
[support-matrix.md](support-matrix.md).
