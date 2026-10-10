# Runtime orchestration (Milestone L)

`membrane run MODEL` answers "which runtime should serve this model on this
machine, and why?" -- and then uses it. You do not need to know which backend
is involved.

```bash
membrane run qwen2.5-1.5b-instruct              # choose, explain, then use
membrane run qwen2.5-1.5b-instruct --explain    # choose and explain only
membrane run MODEL --runtime ollama             # force one runtime (still explained)
membrane run MODEL --intent memory_saving
membrane run MODEL --json
```

## What it considers

Every runtime in the adapter table -- `membrane-native`, `ollama`, `vllm` --
through the same interface (`runtime_adapter.h`): availability and health, the
capability matrix, whether the model exists there (matched by exact id, or by
a normalized name: case, `-`/`_`/`:`/`/` and a trailing `:latest` ignored --
nothing fuzzier), whether it is loaded right now, its reported size and maximum
context. For `membrane-native` it also runs **Planner v2** on this host.

The decision (`tools/membrane-run/orchestrator.[ch]`) is a pure function of
those facts and the hardware topology. No ML, no opaque score, no benchmark
claim, no "MEMBRANE owns it, so prefer it".

## Candidates and honesty

A candidate describes the runtime, model, context, GPU layers, KV precision,
KV placement, estimated memory (with provenance), controllability and
observation quality. **A dimension a runtime does not expose is left empty,
never invented.** An external runtime never gets Planner-v2-grade accuracy:

| Evidence | Meaning | Source |
|---|---|---|
| `planned` | Planner v2 found a feasible plan | native only |
| `resident` | the runtime reports the model loaded now | native / Ollama / vLLM |
| `estimate` | reported size (x1.1) fits measured free RAM+VRAM; weights only, no KV accounting | external |
| `unknown` | nothing proves it fits | any |
| `infeasible` | excluded | exceeds all memory, or Planner v2 says no |

UMA/integrated GPU memory is the host's RAM and is never added to it a second
time. Unknown host facts give `unknown`, not `estimate`.

## Intents

| Intent | Ranking (first difference wins; ends in a stable runtime-id order) |
|---|---|
| `general` (default) | evidence, already-resident, observation quality |
| `interactive` | already-resident (no load latency), evidence, observation quality |
| `memory_saving` | smallest **known** memory, most memory-reducing controls, evidence |
| `long_context` | largest **known** maximum context, evidence, already-resident |

`throughput` is deliberately not offered: MEMBRANE has no capability or
measurement that distinguishes runtimes by throughput without inventing a
benchmark claim.

## Confidence

`high`: a Planner v2 plan or a runtime-reported loaded model, and a real
difference decided it. `medium`: only a size-vs-free-memory estimate. `low`:
nothing proves it fits, or the candidates tied and the stable name order
decided -- it says so.

## Acting on the choice

* `membrane-native` chosen: exactly `membrane use MODEL` (activate in the
  running service).
* An external runtime chosen: MEMBRANE does **not** launch or control it; it
  tells you the endpoint and model id to point a client at.
* Nothing is downloaded or installed by `run`. A model that exists nowhere
  gets the next step (`membrane use MODEL`, with your consent).

## Hardware topology

All enumerated GPU devices are kept (`gpu0`, `gpu1`, ...) with backend, name,
kind (`discrete` / `uma`), total and measured free memory; host RAM is its own
tier. `membrane_topology_pick_device()` picks, deterministically, the discrete
GPU with the most measured free memory that covers a footprint, or none.

### Multi-GPU: what is and is not claimed

The native runtime selects **one** device per model (`mp.main_gpu`, and
`--device` on `membrane-run`); the long-running service selects automatically.
llama.cpp itself has layer/row split modes across several devices, but MEMBRANE
does not wire them, and there is no multi-GPU hardware on the development
machine to test them. So MEMBRANE offers **topology-aware visibility and
device selection where the native runtime supports it, and does not claim
distributed execution.** CPU, discrete VRAM and UMA are the memory tiers the
decision model knows; CXL/FPGA are research-only and not part of it.
