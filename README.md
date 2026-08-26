# MEMBRANE

[![CI](https://github.com/kadireren7/membrane/actions/workflows/ci.yml/badge.svg)](https://github.com/kadireren7/membrane/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kadireren7/membrane/actions/workflows/codeql.yml/badge.svg)](https://github.com/kadireren7/membrane/actions/workflows/codeql.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

MEMBRANE is a llama.cpp runtime layer for planning KV-cache precision
and CPU/GPU residency under constrained memory. It sits in front of an
unmodified `ggml`/llama.cpp inference path and decides, before a model
loads, how many layers to offload to GPU, what precision the KV cache
should be stored in, and which device should hold it — then runs a
normal llama.cpp decode on top of that plan.

## Why MEMBRANE

KV-cache memory grows with context length, and on a memory-constrained
GPU it is often the thing that runs out before compute does. MEMBRANE's
planner looks at real device memory and real model metadata before
load, and picks a GPU-layer count, KV precision, and KV residency that
fit — instead of a fixed `-ngl` guess that either wastes headroom or
fails with an out-of-memory error partway through a run.

## Quick Start

```bash
git clone --recursive https://github.com/kadireren7/membrane
cd membrane
cmake -S . -B build-llama -DMEMBRANE_ENABLE_LLAMA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-llama -j --target membrane-run

./build-llama/tools/membrane-run/membrane-run \
  --model model.gguf \
  --prompt "Hello" \
  --ctx 32768 \
  --auto
```

`--auto` manages three things through the same planner a manual run
uses:

- GPU layer selection (`--gpu-layers auto`)
- adaptive KV precision (`--kv adaptive`, resolves to `q8` or `q5`)
- KV placement (`--kv-placement auto`)

Any explicit flag overrides the corresponding auto-managed field —
`--auto` fills in the rest:

```bash
./build-llama/tools/membrane-run/membrane-run \
  --model model.gguf \
  --prompt "Hello" \
  --ctx 32768 \
  --auto \
  --kv q8 \
  --kv-placement cpu
```

This keeps GPU layer selection on auto-pilot while pinning KV precision
to `q8` and forcing the KV cache onto CPU RAM regardless of what the
planner would otherwise pick.

## Inspecting a plan before you run it

`--plan-only` resolves the exact same planner a real run uses — model
load, shape, GPU/adaptive/placement policy — and prints the result
without generating any tokens:

```bash
./build-llama/tools/membrane-run/membrane-run \
  --model model.gguf \
  --prompt "Hello" \
  --ctx 32768 \
  --auto \
  --plan-only
```

Real output, captured on the tested GTX 1650 host (Vulkan build,
SmolLM2-135M, `--ctx 4096 --auto --plan-only`; `Vulkan1` is the backend
device id llama.cpp assigned to the GPU on that host):

```text
MEMBRANE plan
  device: Vulkan1
  gpu layers: 30/30
  kv precision: Q8_0
  kv placement: auto
  kv layers: 30 GPU / 0 CPU
  estimated weight bytes: 202.6 MiB
  estimated GPU KV: 47.8 MiB, estimated CPU KV: 0.0 MiB
  reason: GPU_FULL_FIT
```

Add `--json` for a machine-readable plan (`"mode": "plan"`) instead of
the text block above — intended for tooling and future UI integration:

```bash
./build-llama/tools/membrane-run/membrane-run \
  --model model.gguf \
  --prompt "Hello" \
  --ctx 32768 \
  --auto \
  --plan-only \
  --json
```

`--verbose` adds a second block (requested vs. resolved config side by
side, the full reason trace, and any warnings) on top of the concise
plan — always on stderr, so `--json` output on stdout is unaffected
either way.

## Precision vs. placement

These are two separate dimensions. Mixing them up is the most common
way to misread a plan.

**Precision** — `--kv` — controls how each KV value is *represented*:

| Value | Meaning |
|---|---|
| `native` | Unmodified llama.cpp KV cache (default) |
| `q8` | Genuinely `Q8_0`-typed KV tensors, ~50% of native's memory |
| `q5` | Genuinely `Q5_1`-typed KV tensors, ~37.5% of native's memory |
| `adaptive` | Planner picks `q8` or `q5` for the whole cache based on memory pressure |

**Placement** — `--kv-placement` — controls which device *holds* the KV
cache, independent of precision:

| Value | Meaning |
|---|---|
| `default` | No-op — same device the model weights use (unchanged behavior) |
| `gpu` | Force KV cache onto GPU |
| `cpu` | Force KV cache onto CPU RAM |
| `auto` | Planner splits GPU/CPU KV layers to fit a safe memory budget |

`--kv` never changes where the cache lives; `--kv-placement` never
changes how it's encoded.

## How it works

```text
GGUF model
    |
    v
model metadata (layer count, tensor shapes, hparams)
    |
    v
MEMBRANE planner
    |-- GPU layer policy       (--gpu-layers auto|all|N|0)
    |-- adaptive KV precision  (--kv native|q8|q5|adaptive)
    `-- KV residency policy    (--kv-placement default|gpu|cpu|auto)
    |
    v
resolved llama.cpp runtime configuration
    |
    v
generation (unmodified ggml/llama.cpp decode)
```

The plan is resolved once, before context construction. There is no
runtime KV migration, no promotion/demotion, and no per-layer mixed
precision — see "Limitations."

## `--auto` semantics

`--auto` is a preset that feeds the same planner explicit flags use —
not a second planner. A few worked examples:

- `--auto --kv q8` — auto manages GPU layers and placement; KV
  precision is pinned to `q8`.
- `--auto --gpu-layers 0` — GPU offload is explicitly disabled; auto
  still manages KV precision on CPU (defaults to `q8` unless a budget
  rules it out).
- `--auto --kv-placement cpu` — auto manages layer count and
  precision; KV residency is forced to CPU.

Bare `--auto` on a build/host with no GPU backend or device resolves to
CPU-only automatically (`auto fallback: no GPU available, resolved to
CPU-only`, reason `NO_GPU_DEVICE`) — this is the one case where `--auto`
behaves differently from an explicit `--gpu-layers auto`, which still
fails closed (exit 5) if a GPU was explicitly requested and isn't
there.

## Measured results

**Qwen2.5-1.5B-Instruct, GTX 1650 (Vulkan), 28/28 GPU weight layers,
native KV precision** — the central Phase 12H capacity experiment
(`results/v0.3/kv-residency-productization/capacity_uplift.json`,
re-verified by `scripts/verify-results.py`):

| `--kv-placement` | ctx 26500 | ctx 26800 | ctx 28500 |
|---|---|---|---|
| default (all-GPU KV) | succeeds | **fails** — real `ggml_vulkan` out-of-device-memory error | fails (same error) |
| `auto` (26/28 GPU KV layers) | — | — | succeeds |
| `cpu` (0/28 GPU KV layers) | — | — | succeeds |

At this tested configuration, the default path's working ceiling sits
between ctx 26500 and ctx 26800; `--kv-placement auto` reached ctx
28500 in the same test — roughly a 6.4% narrow context uplift at this
specific boundary, reproduced 2/2 (default failure) and 3/3 (`auto`
success). GPU weight-layer selection and pre-load estimated weight
bytes were identical across all three placement modes at every row
(`est_weights=2499.5 MiB`, `gpu_layers_selected=28`) — this is a
pre-load estimate/selection match, not a post-load VRAM measurement.

This is a single measured configuration on one model, one GPU, one
host — not a general "run bigger contexts" claim. MEMBRANE's current
static residency planner is a **capacity** feature: it closes the gap
between what the pre-load memory estimate says should fit and what the
real allocator can actually deliver. It is not marketed as a speed
optimization.

**SmolLM2-135M, same GTX 1650, Vulkan `--gpu-bench`** (Phase 9A/9B,
`results/v0.3/gpu-vulkan-validation.json`): across the tested context
sweep, `q8` KV genuinely allocated in GPU VRAM measured lower than
native F16 KV at every tested context — from roughly 2% at small
contexts up to roughly 25% at `ctx=16384` — with a real throughput
cost: q8 generation speed measured roughly 7-18% lower than native
across the same sweep. One model family, one GPU, one host — not a
general VRAM or throughput claim.

At the same tested configuration, 5 generated outputs across
`q8`-gpu/`q8`-cpu/`q8`-default/`q5`-cpu/native-cpu placements were
byte-identical (`quality.json`, independently re-hashed by the
verifier). This shows precision/placement choices didn't change output
at this tested point — it is not a universal token-identity or
zero-quantization-error claim.

## CLI reference

| Flag | Purpose |
|---|---|
| `--auto` | Preset: GPU layers + adaptive KV + KV placement, all auto-managed |
| `--gpu-layers all\|auto\|N\|0` | GPU layer offload (default: `0`, CPU-only) |
| `--kv native\|q8\|q5\|adaptive` | KV cache precision |
| `--kv-placement default\|gpu\|cpu\|auto` | KV cache device residency |
| `--plan-only` | Resolve and print the plan; no generation |
| `--verbose` | Detailed requested/resolved/memory/reason-trace breakdown (stderr) |
| `--json` | Machine-readable output on stdout |

Full flag reference, exit codes, and every option's exact semantics:
`membrane-run --help`.

## JSON output

`--json` emits exactly one JSON object on stdout (`schema_version: 1`);
all diagnostics go to stderr, so stdout stays pure JSON either way.
Structure (a `--plan-only --json` run shown; a normal run's object adds
generation fields in the same shape):

```json
{
  "schema_version": 1,
  "mode": "plan",
  "requested": { "auto": true, "gpu_layers": "auto", "kv": "adaptive", "kv_placement": "auto" },
  "resolved": { "backend": "Vulkan", "device": "Vulkan1", "gpu_layers": 30, "kv": "q8", "kv_placement": "auto" },
  "memory_plan": { "estimated_kv_bytes": 50135040, "kv_gpu_layers": 30, "kv_cpu_layers": 0 },
  "reason_codes": { "primary": "GPU_FULL_FIT", "trace": ["AUTO_REQUESTED", "GPU_DEVICE_FOUND", "GPU_FULL_FIT"] },
  "warnings": []
}
```

## Capability status

| Capability | Status |
|---|---|
| CPU inference | Supported |
| Vulkan GPU offload | Supported |
| KV precision: `q8` (`Q8_0`) | Supported |
| KV precision: `q5` (`Q5_1`) | Supported |
| Adaptive whole-cache `q8`/`q5` selection | Supported |
| Static CPU/GPU KV residency (`--kv-placement`) | Supported |
| `--auto` planning | Supported |
| `--plan-only` / `--verbose` diagnostics | Supported |
| JSON diagnostics (`schema_version: 1`) | Supported |
| CUDA | Not a supported product path |
| Dynamic/runtime KV migration | Research only |
| Per-layer mixed `q8`/`q5` precision | Research only |

## Limitations

- Development and testing are Linux-focused; other platforms are
  untested.
- Vulkan is the only product GPU backend; CUDA is not currently
  supported as a product path.
- KV placement is decided once, before context construction — there is
  no runtime migration, promotion, or demotion.
- Memory figures in a plan are pre-load **estimates**, not measured
  peak VRAM or an OOM guarantee — actual `ggml` allocator overhead and
  other processes claiming memory after the plan resolves aren't
  visible to it.
- Device free-memory values are point-in-time snapshots and can go
  stale before a run finishes loading.
- Adaptive KV selection is whole-cache (`q8` or `q5` for the entire
  context) — there is no per-layer or per-block mixed-precision product
  policy.
- Hardware and model coverage is limited: validation so far covers
  `LLM_ARCH_LLAMA`/`qwen2`-family models on CPU and one GTX 1650 Vulkan
  host.

## Build

CPU-only:

```bash
cmake -S . -B build-llama -DMEMBRANE_ENABLE_LLAMA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-llama -j --target membrane-run
```

Vulkan (adds `--gpu-layers`/`--kv-placement gpu`/`auto` support; needs
Vulkan development headers, `glslc`, and SPIR-V headers already on the
system — MEMBRANE never installs system packages automatically):

```bash
cmake -S . -B build-vulkan \
  -DMEMBRANE_ENABLE_LLAMA=ON \
  -DGGML_VULKAN=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j --target membrane-run
```

Building the llama-free core library and test suite (no model, no
`third_party/llama.cpp` submodule needed):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Release status

Latest stable tag: `v0.2.0`. Release candidate: `v0.3.0-rc1`. Current
`main` contains product hardening completed after `v0.3.0-rc1` — Q5 KV,
adaptive KV precision, static KV residency planning, and the `--auto`/
`--plan-only`/`--verbose`/JSON diagnostics described above are all on
`main` but not yet part of a tagged release.

## Research & provenance

This repository intentionally does not keep experiment branches or a
phase-by-phase research history. Full experiment records, negative
results, benchmark evidence, FPGA/CXL research, and promotion
provenance live in
**[kadireren7/membrane-research](https://github.com/kadireren7/membrane-research)**,
with SHA256-verified provenance back to this repository.

Deeper docs: [`docs/live-runtime.md`](docs/live-runtime.md) (KV
precision mechanism), [`docs/kv-residency.md`](docs/kv-residency.md)
(KV placement design and limitations),
[`docs/reproduction.md`](docs/reproduction.md) (full build/test/CI
reproduction guide), [`docs/licensing.md`](docs/licensing.md) (license
boundaries).

## AI-assisted development

Kadir Eren Altıntaş leads project architecture, experiment selection,
validation criteria, release decisions, and repository direction. AI
coding agents have assisted implementation, analysis automation,
documentation, and review. Results promoted by the project are
validated through tests, CI, reproducible experiments, or explicitly
classified as estimates/simulation. Full disclosure:
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
