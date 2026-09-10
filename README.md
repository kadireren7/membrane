# MEMBRANE

MEMBRANE automatically chooses, fits, installs, runs, and serves local
AI models on your hardware — install it, tell it which model you want,
and talk to it over a real OpenAI-compatible API. No manual GPU-layer
math, no llama.cpp flags to learn first.

[![CI](https://github.com/kadireren7/membrane/actions/workflows/ci.yml/badge.svg)](https://github.com/kadireren7/membrane/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kadireren7/membrane/actions/workflows/codeql.yml/badge.svg)](https://github.com/kadireren7/membrane/actions/workflows/codeql.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

![MEMBRANE resolves a GGUF model and context into GPU layer count, KV precision, and KV placement, then hands the plan to llama.cpp for generation](docs/assets/membrane-hero.svg)

## Install

Easiest path on Ubuntu/Debian/Pop!_OS: download the latest `.deb` from
[the Releases page](https://github.com/kadireren7/membrane/releases)
and `sudo apt install ./membrane_<version>_amd64.deb` — no manual
CMake flags. That one package is Vulkan-enabled but runs correctly
CPU-only (GPU offload stays opt-in); a CPU-only
`membrane-cpu_<version>_amd64.deb` also exists as a CI-validation/
build-your-own artifact, not a release asset. Building from source, or
on Windows/macOS (real per-PR CI validation, no official package yet):
[`docs/install.md`](docs/install.md), [`docs/support-matrix.md`](docs/support-matrix.md).

## Say which model you want

```bash
membrane use qwen2.5:7b
```

If `qwen2.5:7b` isn't installed yet, this previews the download (size,
hardware fit, recommended variant), asks to confirm, installs it, and
selects it — activating it live if `membrane serve`/the background
service is already running. Already installed? Same command just
selects/activates it, no restart needed. `membrane setup` runs the
same flow for a guided first-run (register a local `.gguf`, or a
catalog name, and stand up the background service in one pass). See
[`docs/model-lifecycle.md`](docs/model-lifecycle.md).

## Talk to it from your app

```bash
curl http://127.0.0.1:8642/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5:7b", "messages": [{"role": "user", "content": "Hello"}]}'
```

Or point any OpenAI-compatible client at `http://127.0.0.1:8642/v1`
with any placeholder API key (this server has no authentication of its
own — loopback-only is its real security boundary). Real, tested this
project: the official Python and Node.js `openai` SDKs
(`client.models.list()`, non-streaming and streaming
`chat.completions.create()`, real `stop`-sequence early termination,
real tool-calling/`response_format` rejection). Streaming
(`stream: true`, real Server-Sent Events) and a live model switch
(just send a different `"model"` — no restart, no admin call) both
work the same way. See
[`docs/client-compatibility.md`](docs/client-compatibility.md) for the
exact, honestly-labeled client matrix — never "should work," always
what was actually run.

`membrane doctor` checks installation, hardware, the model registry,
config, the service, and the HTTP endpoint in one pass, and tells you
exactly what (if anything) needs attention.

## What MEMBRANE adds on top of llama.cpp

llama.cpp is the inference engine underneath — the real GGUF model
loader, the real backend kernels (CPU/Vulkan/CUDA/Metal), the real
decode loop. MEMBRANE doesn't replace or reimplement any of that; it's
the product/runtime layer on top that a raw inference engine doesn't
provide on its own:

| Layer | Who owns it |
|---|---|
| Model loading, backend kernels, decode loop | llama.cpp (unmodified) |
| **Model catalog + real HTTPS download + checksum verification** | MEMBRANE |
| **Hardware-aware variant selection** ("which quant actually fits this machine") | MEMBRANE |
| **Memory-aware KV-cache/GPU-layer planning** (the `--auto` joint planner) | MEMBRANE |
| **Model lifecycle**: install, select, live-switch, uninstall guards | MEMBRANE |
| **Background service** (systemd/launchd/Task Scheduler abstraction) | MEMBRANE |
| **OpenAI-compatible HTTP API** (streaming, stop sequences, bounded admission) | MEMBRANE |
| **Client integration** (validated against real SDKs) | MEMBRANE |

If you already know exactly which GGUF file and which llama.cpp flags
you want, `membrane-run` (below) is still the direct, low-level CLI —
nothing about it changes underneath MEMBRANE's own product layer.

## Known limitations

Read before you rely on this in production:

- **OpenAI-compatible chat API subset, not the full OpenAI API** — no
  tool calling (`tools`/`tool_choice` are explicitly rejected, never
  silently dropped), no `/v1/embeddings`, no `response_format` beyond
  the default. See [`docs/api-contract.md`](docs/api-contract.md).
- **Bounded concurrency, not unlimited throughput** — up to 2 resident
  models at once (LRU eviction, pinning), up to 2 concurrent decodes
  per process (both env-overridable for testing only, not documented/
  supported knobs) — deliberately conservative, real-memory-motivated,
  never marketed as a performance/throughput feature. See
  [`docs/soak-and-concurrency-testing.md`](docs/soak-and-concurrency-testing.md).
- **Open WebUI and Continue are not both independently validated
  end-to-end yet** — real for two SDKs and curl; see
  [`docs/client-compatibility.md`](docs/client-compatibility.md).
- **No official Windows or macOS package yet** — real per-PR CI
  validation exists for both (including real `--ctx auto` host-memory
  sizing as of v1.0.0), source build only. See
  [`docs/support-matrix.md`](docs/support-matrix.md).
- **Backend/platform evidence is real but scoped**: one real CUDA
  device, one real (paravirtualized) Metal device, real Windows/macOS
  CPU on hosted CI — not a claim about every GPU/OS combination.
- **Model-family compatibility evidence is scoped to the exact tested
  fixtures** (SmolLM2-135M/360M, Qwen2.5-1.5B) — see
  [`docs/model-compatibility.md`](docs/model-compatibility.md); a
  shared architecture name is not itself evidence.
- **Independent, external multi-host validation remains limited** —
  most real evidence is the maintainer's own hardware plus GitHub-
  hosted CI, disclosed throughout, not hidden. Real external
  contributions do exist (see
  [`docs/external-validation.md`](docs/external-validation.md)) but
  are reviewed on their own timeline, not folded into a release
  automatically.
- **No authentication on the local server** — loopback-only binding is
  the real security boundary; remote exposure is your own
  responsibility (`--allow-non-loopback` warns loudly).

## Advanced: build from source / direct CLI

CPU-only:

```bash
cmake -S . -B build-llama -DMEMBRANE_ENABLE_LLAMA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-llama -j --target membrane-run membrane
```

Vulkan (needs Vulkan development headers, `glslc`, and SPIR-V headers
already on the system):

```bash
cmake -S . -B build-vulkan -DMEMBRANE_ENABLE_LLAMA=ON -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j --target membrane-run membrane
```

CUDA (needs the CUDA toolkit already installed;
[`docs/cuda-backend.md`](docs/cuda-backend.md)):

```bash
cmake -S . -B build-cuda -DMEMBRANE_ENABLE_LLAMA=ON -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j --target membrane-run membrane
```

Full walkthrough (install/uninstall, troubleshooting):
[`docs/install.md`](docs/install.md). Reproduction guide (llama-free
core library, sanitizers, CI): `docs/reproduction.md`.

```bash
cmake --install build-vulkan --prefix "$HOME/.local"
```

Installs `membrane-run`/`membrane` plus their shared-library
dependencies, each with an `$ORIGIN`-relative RPATH, so the installed
binaries run standalone — no build tree, no `LD_LIBRARY_PATH`.
`cmake --build build-vulkan --target uninstall` removes exactly what
that install run recorded.

## Quick Start (`membrane-run`, the direct, non-HTTP CLI)

Prefer the CLI directly, or want the exact plan before you run it? No
`Q8_0` block internals, no planner internals required to get here —
each command below is optional past the first:

```bash
# Check what MEMBRANE sees on this host (no model needed)
membrane-run --doctor

# Point it at your model and see if it supports compressed KV -- no
# generation, cheap
membrane-run --model model.gguf --inspect-model

# Run it -- CPU inference, full-precision KV, no flags to learn first
membrane-run --model model.gguf --prompt "Hello"

# Once that works, preview what --auto would choose (needs an explicit
# --ctx -- memory planning depends on knowing the context size up front)
membrane-run --model model.gguf --ctx 2048 --auto --plan-only

# Then actually run it with --auto managing GPU offload and KV memory
membrane-run --model model.gguf --prompt "Hello" --ctx 2048 --auto
```

![--auto fans out into GPU layers, KV type, and KV residency, all automatically managed; an explicit --kv q8 override fixes only KV type, leaving the other two on auto](docs/assets/membrane-auto.svg)

Explicit flags override only the field they name — `--auto --kv q8`
keeps GPU layers and KV placement on auto while pinning precision to
`q8`. `membrane-run --list-devices` lists every backend device MEMBRANE
can see (no model needed). Full flag reference and exit codes:
`membrane-run --help`.

`--plan-only` resolves the exact same planner a real run uses and
prints the result without generating a token:

```bash
./build-vulkan/tools/membrane-run/membrane-run \
  --model model.gguf --prompt "Hello" --ctx 32768 --auto --plan-only
```

![Terminal transcript of membrane-run --auto --plan-only showing the real MEMBRANE plan block: device Vulkan1, gpu layers 30 of 30, kv precision Q8_0, kv placement auto, reason GPU_FULL_FIT](docs/assets/membrane-terminal.svg)

Real output — captured verbatim, transcript and command in
[`docs/assets/source/plan-example.txt`](docs/assets/source/plan-example.txt).
Add `--json` for the same plan as one machine-readable object
(`schema_version: 1`) instead of this text block; add `--verbose` for
the full requested/resolved breakdown and reason trace.

### How the planner works

![GGUF model and context flow into model metadata, then into the MEMBRANE planner, which branches into GPU layers, KV type, and KV placement, converging into one resolved plan handed to llama.cpp for generation](docs/assets/membrane-flow.svg)

One logical planning pipeline, not three independent tools. GPU layer
count comes from a pre-load memory estimate; KV precision and KV
placement finish resolving once real model shape is available, before
the KV cache/context itself is constructed. There is no runtime KV
migration and no per-layer mixed precision — see "Current scope"
below.

![KV cache has two independent axes: representation (native, q8, q5, adaptive) and residency (default, gpu, cpu, auto); changing one never changes the other](docs/assets/membrane-precision-placement.svg)

`--kv` never changes where the cache lives. `--kv-placement` never
changes how it's encoded.

### Measured results

MEMBRANE plans against whatever memory is actually available on the
machine it runs on. The numbers below are two specific tested
configurations, not a general hardware claim — a different GPU or
model will measure differently.

![Bar chart: default all-GPU KV placement succeeds at context 26,500 and fails at context 26,800 with a real Vulkan out-of-device-memory error; MEMBRANE auto and cpu KV placement both succeed at context 28,500 in the same tested configuration](docs/assets/membrane-capacity.svg)

Source: `results/v0.3/kv-residency-productization/capacity_uplift.json`,
re-verified by `scripts/verify-results.py`: the default all-GPU-KV path
succeeds at `ctx=26500` and fails at `ctx=26800` with a real Vulkan
out-of-device-memory error; `--kv-placement auto`/`cpu` both succeed at
`ctx=28500` in the same test. This Qwen2.5 result validates KV
**placement** at **native** precision — the current `qwen2` architecture
compatibility check does not validate `q8`/`q5`/adaptive KV
**compression** for that model family (only `LLM_ARCH_LLAMA` models are
validated for compressed KV) — see
[`docs/model-compatibility.md`](docs/model-compatibility.md).

![Bar chart across six tested contexts on SmolLM2-135M: q8 KV VRAM reduction ranges from about 2 percent at small contexts to about 25 percent at context 16384, while generation throughput is about 7 to 18 percent lower than native across the same sweep](docs/assets/membrane-q8-tradeoff.svg)

Source: `results/v0.3/gpu-vulkan-validation.json`, SmolLM2-135M on the
tested GTX 1650: VRAM reduction ran from roughly 2% at small contexts up to
roughly 25% at `ctx=16384`, while generation speed measured roughly 7-18%
lower than native across the same sweep. This is a tradeoff, not a speedup —
`q8` reduces VRAM at every tested context at a real throughput cost. At the
same tested point, 5 generated-text outputs across every `q8`/`q5`/placement
combination were byte-identical (md5-verified, `quality.json`) — generated
text only, not a token-ID or numeric quantization-error claim.

## Current capabilities

| Supported | Evidence-scoped (real, but narrow) | Not a product path | Research only |
|---|---|---|---|
| CPU inference | CUDA (one real device, source build only) | Tool calling / `/v1/embeddings` | Dynamic/runtime KV migration |
| Vulkan GPU offload | Metal (one real, paravirtualized device) | Structured JSON response mode | Per-layer mixed `q8`/`q5` precision |
| KV precision: `q8`, `q5`, adaptive | Windows (real CPU CI, real `--ctx auto`) | | FPGA/CXL (simulation/synthesis-tool evidence only) |
| Static CPU/GPU KV residency | macOS (real CI, paravirtual Metal, real `--ctx auto`) | | |
| `--auto` planning, `membrane use`/`setup`/`doctor` | | | |
| OpenAI-compatible chat API (streaming, stop, bounded admission) | | | |
| Bounded concurrent decode (multiple simultaneous requests) | | | |
| Bounded multi-model residency (up to 2, LRU eviction, pinning) | | | |
| `GET /membrane/v1/capabilities` (real, live discovery) | | | |
| JSON diagnostics (`schema_version: 1`) | | | |

See [`docs/support-matrix.md`](docs/support-matrix.md) for the exact,
evidence-state-labeled platform/backend matrix and
[`docs/model-compatibility.md`](docs/model-compatibility.md) for the
exact validated model families.

## Current scope

- Real evidence spans Linux (CPU/Vulkan/CUDA, on real local hardware),
  Windows (CPU, real per-PR CI), and macOS (real per-PR CI, a
  paravirtualized Metal device) — see `docs/support-matrix.md` for the
  exact evidence state per platform/backend; no GPU evidence exists on
  Windows.
- KV placement is decided once, before context construction — no
  runtime migration, promotion, or demotion.
- Adaptive KV selection is whole-cache (`q8` or `q5` for the entire
  context), never per-layer or per-block.
- Memory figures in a plan are pre-load estimates and point-in-time
  snapshots, not measured peak VRAM or an OOM guarantee.
- `q8`/`q5`/adaptive KV precision is validated only for `LLM_ARCH_LLAMA`
  models — checked and rejected for other architectures before use.
- Up to 2 resident models at once (deterministic LRU eviction,
  optional pinning); a live switch (`membrane use`, or simply a
  different `"model"` field in a chat request) reuses an already-
  resident model instantly, or evicts the least-recently-used
  unpinned/idle one to make room — never unlimited, never silent.
  `docs/server.md`'s own "Multi-model residency" section has the full
  policy.

Mechanism detail: `docs/live-runtime.md` (KV precision) and
`docs/kv-residency.md` (KV placement). Full architecture/backend/
precision/placement compatibility matrix: `docs/compatibility.md`.
`--auto`'s joint GPU-layers/precision/placement planner:
`docs/joint-planner.md`. Bounded apply-time fallback if the primary
plan can't be instantiated: `docs/auto-fallback.md`.

## Research & provenance

This repository intentionally keeps no experiment branches or phase-by-phase
research history. Full experiment records, negative results, and FPGA/CXL
research (simulation and synthesis-tool proxies only, no physical hardware)
live in
**[kadireren7/membrane-research](https://github.com/kadireren7/membrane-research)**,
with SHA256-verified provenance back to this repository.

**Release status**: latest stable tag `v1.0.0` (supersedes `v0.8.0`,
now historical). This release is a stability/hardening milestone, not
a features-stopped one: bounded concurrent decode, bounded multi-model
residency (up to 2 resident models, LRU eviction, pinning), a frozen
API v1 contract (`docs/api-v1-stability.md`) and CLI contract
(`docs/cli-stability-contract.md`), real Windows/macOS host-memory-
aware `--ctx auto`, and a real, tested v0.8→v1.0 upgrade path with zero
registry/config disruption — see
[docs/release-v1.0.0.md](docs/release-v1.0.0.md) for the full release
notes and [docs/upgrade-v0.8-to-v1.0.md](docs/upgrade-v0.8-to-v1.0.md)
if you're upgrading from v0.8.0. See "Known limitations" above for what
this release does not claim. CPU-only default behavior (no flags, or
`--gpu-layers 0`) is unchanged by any of this.

## AI-assisted development

Kadir Eren Altıntaş leads project architecture, experiment selection,
validation criteria, and release decisions. AI coding agents have assisted
implementation, analysis automation, documentation, and review. Full
disclosure:
[kadireren7/membrane-research](https://github.com/kadireren7/membrane-research)'s
`outreach/ai-assistance-disclosure.md`.

## License

Apache License 2.0 for MEMBRANE's own code — see [LICENSE](LICENSE).
The `third_party/llama.cpp` submodule and any model artifacts are under
their own separate terms — see [docs/licensing.md](docs/licensing.md).
Citation: [CITATION.cff](CITATION.cff).

---

Contributing: [CONTRIBUTING.md](CONTRIBUTING.md). Security: [SECURITY.md](SECURITY.md).
Support: [SUPPORT.md](SUPPORT.md). Community standards: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
