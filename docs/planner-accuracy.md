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
  load); KV bytes come from the exact real `ggml_row_size()` formula for the
  requested KV type — not a heuristic, the same arithmetic llama.cpp's own
  allocator uses. `gpu_policy.device_free_bytes`/`device_total_bytes` are a
  single **pre-load** query of the target device.
- **Host-side KV bytes** (`storage.kv_allocated_bytes`): the identical exact
  formula, whenever the KV cache lands in system RAM (CPU backend, or
  `--kv-placement cpu`).

Neither figure is a measured peak or a guarantee — see `gpu_policy.
safety_reserve_bytes` (15% of total, or 512 MiB, whichever is larger) for the
margin the planner already keeps.

## What is now observable (Phase 19 additions)

Two **additive**, non-behavior-changing instrumentation additions to
`membrane-run`'s normal single-pass mode:

1. **`gpu_memory_observed`** (new JSON block): re-queries the exact same
   device via the exact same API (`ggml_backend_dev_get_props()`, through
   `membrane_gpu_list_devices()`) a second time, after model load + context
   creation + generation are all done, and reports `device_free_bytes_before`/
   `_after`/`observed_delta_bytes`. `available: false` on any CPU-only run.
2. Host RSS: **already existed** before this phase
   (`memory.rss_after_model_load_kb`/`rss_after_context_kb`/`peak_rss_kb`,
   real `/proc/self/status` VmRSS/VmHWM checkpoints) — Phase 19 didn't need to
   add anything here, only use what was already reported.

Neither is a continuous peak sample — both are single point-in-time reads, so
neither can catch a transient spike that both grows and shrinks between reads.

## Measurement semantics — read this before trusting a number

- **GPU free bytes** come from the Vulkan backend's device-memory-budget
  query. This is a **driver-reported heap figure**, not `nvidia-smi`'s
  process-level VRAM accounting, and — a finding from this phase, not an
  assumption — **it does not appear to update at fine granularity on the
  tested hardware**. See "Finding: GPU delta granularity" below.
- **Host RSS** (`/proc/self/status`) is a real per-process measurement and
  behaved as expected at every tested point.
- `results/planner-accuracy/measurements.json`'s `errors.*` fields are
  computed by `scripts/measure-planner-accuracy.py`, not by `membrane-run`
  itself — the product never makes a pass/fail decision from them.

## Controlled points

8 points, `scripts/measure-planner-accuracy.py` (reproducible; real local
GGUF fixtures, no downloads), generated on `add0302`+Phase 19 head on the
one real device this repository's evidence has always used (GTX 1650,
Vulkan): SmolLM2-135M × {native, q8, q5, adaptive} on Vulkan, SmolLM2-135M ×
{native, q8} on CPU, SmolLM2-360M native on Vulkan, and Qwen2.5-1.5B native
with `--kv-placement cpu` on Vulkan (weights on GPU, KV forced to host RAM —
isolates the two estimate components onto two different, independently
checkable observations). All at `ctx=2048`, 8 generated tokens — no context
sweep. Every point `FIT` (no failures at this deliberately small, non-boundary
set of points).

## Findings

### GPU delta granularity is coarser than KV-precision-level differences

The four SmolLM2-135M Vulkan points (native/q8/q5/adaptive) have real,
different KV-byte estimates (47.2 MB / 25.1 MB / 17.7 MB / 25.1 MB) — a ~30 MB
spread. Their `gpu_memory_observed.observed_delta_bytes` were **effectively
identical across all four runs** (~270 MB, varying only by driver-run-to-run
noise far smaller than the real KV-size spread). Cross-checked independently
with `nvidia-smi` polled before/during/after one run: it reported a flat
2 MiB used the entire time (its own polling missed the transient allocation
entirely — these runs complete in well under a second).

**Conclusion:** on this hardware/driver, this observation method has enough
resolution to validate *model-scale* estimates (tens to hundreds of MB — see
below) but not to distinguish *KV-precision-scale* differences (tens of MB).
Do not read a specific KV-mode's `gpu_estimate_error_percent` in
`measurements.json` as reflecting that mode's real accuracy in isolation —
report and compare only at a scale this method can actually resolve.

### Model-scale weight-loading estimate is accurate

At the scale this method CAN resolve, agreement is good:

| Point | Estimated | Observed | Error |
|---|---|---|---|
| SmolLM2-135M (native, ctx 2048) | 259.7 MB | 270.1 MB | +3.9% |
| SmolLM2-360M (native, ctx 2048) | 713.3 MB | 712.2 MB | **-0.15%** |
| Qwen2.5-1.5B (native, KV forced to CPU) | 2621.0 MB (weights only) | 3001.0 MB | +12.7% |

The 360M point's near-exact agreement is the strongest single data point here.
The Qwen2.5-1.5B gap is larger — plausibly non-`blk.N.` tensors (embeddings,
output projection) being a much larger fraction of a 1.5B model's real GPU
footprint than the average-bytes-per-`blk.N.`-layer estimate accounts for;
this phase reports the gap, it does not diagnose or fix it.

### Host-side KV byte formula consistently under-estimates real RSS delta

Where KV genuinely lives in host RAM (CPU backend, or GPU weights +
`--kv-placement cpu`), the exact `ggml_row_size()`-based formula against the
real RSS delta after context creation:

| Point | Estimated KV | Observed RSS delta | Error |
|---|---|---|---|
| SmolLM2-135M CPU native | 46.1 MB | 47.9 MB | +3.9% |
| SmolLM2-135M CPU q8 | 24.5 MB | 26.3 MB | +7.1% |
| Qwen2.5-1.5B GPU weights, CPU KV | 57.3 MB | 61.3 MB (**60x** the KV estimate's own scale — see note) | +6.4% |

Every point over-observes relative to the formula, consistently, by roughly
4-7%. Plausible explanation: allocator/ggml-context bookkeeping overhead the
pure `n_layer × kv_size × bytes_per_token` formula doesn't (and isn't meant
to) account for. This is a real, small, one-directional bias — a natural
future input to Phase 20's safety-margin reasoning, not something this phase
changes.

(Qwen2.5-1.5B note: its `host_rss_after_context_delta_kb` figure landed at
essentially the same ~61 MB scale as its own KV estimate purely because
`ctx=2048` is small relative to the model — no methodology change from the
SmolLM2 rows.)

### Reserve sizing (informational only)

At every tested point, `safety_reserve_bytes` (15%/512 MiB) was far larger
than any observed estimate error, including the largest one found
(Qwen2.5-1.5B's +12.7%, ~380 MB, against a reserve of 683 MB — see
`results/planner-accuracy/measurements.json`'s `planner.reserved_gpu_bytes`
for that point). This is a single-point observation on one device, not a
general claim; no reserve-sizing change is made by this phase.

## What this is not

- Not a planner policy change — `gpu_policy.c`, `kv_residency_policy.c`,
  `adaptive_kv_policy.c` are untouched.
- Not a benchmark campaign — 8 fixed points, `ctx=2048` throughout, no
  boundary/OOM sweep.
- Not a general hardware claim — every GPU number above is from one real
  device (GTX 1650). A different GPU/driver may report free-memory deltas at
  a different granularity entirely.
- Not a claim that `gpu_memory_observed` is broken — it reports exactly what
  it says it reports (a driver-level free-heap re-query); the finding is that
  *this driver's* reporting granularity is coarser than KV-mode-level
  differences, which is itself useful information for later phases (e.g. "if
  you need per-mode GPU validation, use a different device/instrument, not
  this field").

## How this is verified

`scripts/verify-planner-accuracy.py` checks `results/planner-accuracy/
measurements.json`'s schema, that `errors.*` fields are only populated when
their corresponding `planner`/`observed` inputs are both present, that no
byte/percent field is a nonsensical negative-impossible value, and that
`gpu_memory_observed`-derived fields are never present when `backend: "cpu"`
(no GPU device to observe). Run via `python3 scripts/verify-results.py`
(chained, same pattern Phase 18 established for `verify-compatibility.py`).
