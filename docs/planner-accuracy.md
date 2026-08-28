# Planner memory accuracy

Before making `--auto` smarter (a later phase), this measures how closely
MEMBRANE's existing memory *estimates* correspond to *observed* memory
behavior. It changes no planner policy — see "What this is not" below.

## What MEMBRANE estimates today

Two independent pre-load/pre-run figures, both computed before any generation
happens:

- **GPU weight + KV bytes** (`gpu_policy.estimated_model_bytes` /
  `estimated_kv_bytes` in `--json` output): weight bytes come from an average
  bytes-per-layer figure read directly from GGUF tensor metadata (no model
  load) — specifically, only tensors named `blk.N.*` (see
  `tools/membrane-run/gpu_device.h`'s own doc comment on
  `membrane_gpu_estimate_model()`); KV bytes come from the exact real
  `ggml_row_size()` formula for the requested KV type — not a heuristic, the
  same arithmetic llama.cpp's own allocator uses.
  `gpu_policy.device_free_bytes`/`device_total_bytes` are a single
  **pre-load** query of the target device.
- **Host-side KV bytes** (`storage.kv_allocated_bytes`): the identical exact
  formula, whenever the KV cache lands in system RAM (CPU backend, or
  `--kv-placement cpu`).

Neither figure is a measured peak or a guarantee — see `gpu_policy.
safety_reserve_bytes` (15% of total, or 512 MiB, whichever is larger) for the
margin the planner already keeps.

## What is now observable (Phase 19 additions)

An **additive**, non-behavior-changing instrumentation addition to
`membrane-run`'s normal single-pass mode:

- **`gpu_memory_observed`** (new JSON block): re-queries the exact same
  device via the exact same API (`ggml_backend_dev_get_props()`, through
  `membrane_gpu_list_devices()`) a second time, after `run_kv_store_pass()`
  (`decode_loop.cpp`) returns, and reports `device_free_bytes_before`/
  `_after`/`observed_delta_bytes`. `available: false` on any CPU-only run.

Host RSS **already existed** before this phase
(`memory.rss_after_model_load_kb`/`rss_after_context_kb`/`peak_rss_kb`, real
`/proc/self/status` VmRSS/VmHWM checkpoints, read *while the context is still
alive* — see `decode_loop.cpp`) — Phase 19 didn't need to add anything here,
only use what was already reported.

Neither is a continuous peak sample — both are single point-in-time reads, so
neither can catch a transient spike that both grows and shrinks between reads.

## Measurement semantics — read this before trusting a number

- **`gpu_memory_observed` measures WEIGHTS ONLY, never KV.**
  `run_kv_store_pass()` calls `llama_free(ctx)` — destroying the KV
  cache/context — *before it returns* (`decode_loop.cpp`). The post-run GPU
  query in `main.cpp` runs after that return, so it structurally cannot
  include the KV cache: only the model weights (freed separately, later, by
  `run_normal_mode()`'s own caller) are still GPU-resident at that point.
  **This was a real bug in this phase's first draft** — an earlier version of
  this document compared the observed delta against
  `estimated_model_bytes + estimated_kv_bytes` and, from four Vulkan runs
  whose observed delta looked suspiciously identical across native/q8/q5/
  adaptive, concluded the *driver's* free-memory reporting was too coarse to
  resolve KV-size differences. That reasoning was wrong: the deltas were
  identical because **none of the four runs had any KV memory left to
  observe by the time the query ran** — the model weights are identical
  across those four runs, so of course the delta was identical too. Caught by
  CodeRabbit review; see git history for the incorrect version. The
  corrected comparison basis is `estimated_model_bytes` alone (`planner.
  estimated_total_gpu_bytes_basis` in `measurements.json` says exactly this
  for every row).
- **Host RSS** (`/proc/self/status`) is a real per-process measurement, read
  while the context is still alive, and behaved as expected at every tested
  point.
- `results/planner-accuracy/measurements.json`'s `errors.*` fields are
  computed and *mechanically recomputed-and-verified*
  (`scripts/verify-planner-accuracy.py`) from `planner`/`observed` — never
  by `membrane-run` itself, which makes no pass/fail decision from them.

## Controlled points

8 points, `scripts/measure-planner-accuracy.py` (reproducible; real local
GGUF fixtures, no downloads), generated on this Phase 19 head on the one real
device this repository's evidence has always used (GTX 1650, Vulkan):
SmolLM2-135M × {native, q8, q5, adaptive} on Vulkan, SmolLM2-135M × {native,
q8} on CPU, SmolLM2-360M native on Vulkan, and Qwen2.5-1.5B native with
`--kv-placement cpu` on Vulkan (weights on GPU, KV forced to host RAM —
isolates the two estimate components onto two different, independently
checkable observations). All at `ctx=2048`, 8 generated tokens — no context
sweep. Every point `FIT` (no failures at this deliberately small, non-boundary
set of points).

## Findings

### GPU weight-byte estimate under-counts real usage by ~12-21%, one direction only

With the corrected weights-only basis:

| Point | Estimated (weights only) | Observed | Error |
|---|---|---|---|
| SmolLM2-135M (native/q8/q5/adaptive — same weights every time) | 212.5 MB | 270.1 MB | +21.3% |
| SmolLM2-360M (native) | 629.4 MB | 712.1 MB | +11.6% |
| Qwen2.5-1.5B (native, weights on GPU) | 2621.0 MB | 3000.9 MB | +12.7% |

Real usage exceeds the estimate at every point, by a wide and non-constant
margin. This is a genuine estimate gap, not measurement noise — the four
SmolLM2-135M rows (identical weights, only KV mode differs, which this
measurement can't see per the note above) reproduce the exact same 21.3%
figure four times.

**A source-verifiable, plausible explanation** (not fully quantified by this
phase's committed evidence — see caveat below): `gpu_policy.
estimated_model_bytes` is `selected_layers × bytes_per_layer`, and
`bytes_per_layer` is an average over only `blk.N.*`-prefixed tensors
(`tools/membrane-run/gpu_device.h`'s own doc comment on
`membrane_gpu_estimate_model()` is explicit about this). A GGUF model also
has non-layer tensors — token embeddings, the output/`lm_head` projection,
the final norm — that a full/near-full `--gpu-layers all` offload also puts
on the GPU, but that this estimate never counts.
`membrane_gpu_estimate_model()` already computes a separate `total_bytes`
field covering *every* tensor (layer and non-layer), but nothing in
`gpu_policy.c` or `main.cpp` currently reads it — it's computed and silently
discarded. This structurally explains an under-count in the observed
direction, and a smaller model (where the fixed-size embedding/output
tensors are a larger fraction of the total) plausibly explains why 135M's
gap (21.3%) is larger than 360M's (11.6%) or Qwen2.5-1.5B's (12.7%).

**Caveat:** this explanation was checked with a one-off, uncommitted local
instrumentation call to `membrane_gpu_estimate_model()` during this phase's
review-fix pass, confirming `total_bytes` is meaningfully larger than
`selected_layers × bytes_per_layer` for all three models (roughly
13-26% larger, in the same direction and rough scale as the observed gap) —
but that number isn't wired into any committed, re-verifiable artifact, so
it's reported here as a plausible, code-verifiable *mechanism*, not a
quantified, mechanically-checked finding the way the table above is. Whether
to expose `total_bytes` in `gpu_policy`'s JSON and/or use it in the budget
math is exactly the kind of concrete, evidence-backed input Phase 20's joint
planner should consume — not something this measurement-only phase changes.

### Host-side KV byte formula consistently under-estimates real RSS delta

Where KV genuinely lives in host RAM (CPU backend, or GPU weights +
`--kv-placement cpu`), the exact `ggml_row_size()`-based formula against the
real RSS delta after context creation (while the context is still alive):

| Point | Estimated KV | Observed RSS delta | Error |
|---|---|---|---|
| SmolLM2-135M CPU native | 46.1 MB | 47.9 MB | +3.9% |
| SmolLM2-135M CPU q8 | 24.5 MB | 26.4 MB | +7.5% |
| Qwen2.5-1.5B GPU weights, CPU KV | 57.3 MB | 58.6 MB | +2.2% |

Every point over-observes relative to the formula, by a small (2-8%)
one-directional margin. Plausible explanation: allocator/ggml-context
bookkeeping overhead the pure `n_layer × kv_size × bytes_per_token` formula
doesn't (and isn't meant to) account for. This is a real, small,
one-directional bias — a natural future input to Phase 20's safety-margin
reasoning, not something this phase changes. Unlike the GPU-side finding
above, this one has no known confound (RSS was read while the context was
still alive, and the KV formula/measurement pairing is exact) — treat it as
solid, not merely plausible.

### Reserve sizing (informational only)

At every tested point, `safety_reserve_bytes` (15%/512 MiB) was larger than
the corresponding estimate error in absolute bytes (e.g. SmolLM2-135M's
largest gap, +57.6 MB, against a reserve of 683 MB on that device — see
`results/planner-accuracy/measurements.json`'s `planner.reserved_gpu_bytes`).
This is a single-point observation on one device, not a general claim; no
reserve-sizing change is made by this phase. Given the GPU-side finding
above is a real, structural under-count (not noise), a model whose non-layer
tensor fraction is large enough, combined with a tight enough reserve on a
different device, could plausibly erode this margin — worth Phase 20
checking explicitly rather than assuming the margin always absorbs it.

## What this is not

- Not a planner policy change — `gpu_policy.c`, `kv_residency_policy.c`,
  `adaptive_kv_policy.c` are untouched.
- Not a benchmark campaign — 8 fixed points, `ctx=2048` throughout, no
  boundary/OOM sweep.
- Not a general hardware claim — every GPU number above is from one real
  device (GTX 1650).
- Not a claim that `gpu_memory_observed` is broken — it reports exactly what
  it says it reports (a driver-level free-heap re-query, taken after context
  teardown). The finding is about what the *planner's estimate* leaves out,
  not a flaw in this phase's observation mechanism.

## How this is verified

`scripts/verify-planner-accuracy.py` checks `results/planner-accuracy/
measurements.json`'s schema; that every populated `errors.*` field is
**recomputed from its `planner`/`observed` inputs and matches exactly** (not
merely "inputs present"); that the specific error percentages this very
document's Findings tables cite **actually appear, formatted the same way,
in the committed `measurements.json`** (a mismatch here means this file has
drifted from the data it claims to summarize); that no byte/percent field is
a nonsensical-negative-impossible value; that non-finite JSON constants
(`NaN`/`Infinity`) are rejected outright rather than silently parsed; and
that `gpu_memory_observed`-derived fields are never present when
`backend: "cpu"` (no GPU device to observe). Run via
`python3 scripts/verify-results.py` (chained, same pattern Phase 18
established for `verify-compatibility.py`).
