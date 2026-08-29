# Performance optimization (Phase 25)

**This phase's own conclusion is that zero performance-changing code
optimizations were safe to implement** for the two highest-priority
targets Phase 24 measured (model load, decode throughput) — both are,
on this host and in this codebase, dominated by third-party (llama.cpp/
ggml-backend) cost with no MEMBRANE-owned code inside the timed
boundary. This is an explicitly sanctioned Phase 25 outcome, not a
shortfall (see the Phase 25 task's own "A successful Phase 25 may
legitimately conclude... MEMBRANE has no safe product-local
optimization"). Two low-risk, additive **instrumentation/observability**
changes were implemented instead (OPT-03, OPT-06) — neither changes
runtime behavior or counts against the "at most two optimizations" cap.

Every claim below is backed by `results/performance-optimization/
validation.json` (checked by `scripts/verify-performance-optimization.py`)
and/or Phase 24's own immutable `results/performance-profiling/
measurements.json` (never rewritten this phase). Real host throughout:
same Pop!_OS 24.04 dev machine as every prior phase (AMD Ryzen 5 5600H,
real NVIDIA GTX 1650 Vulkan device `Vulkan1`, real AMD Radeon RADV
RENOIR integrated `Vulkan0`, 5.6 GiB RAM).

## Methodology

1. Read Phase 24's committed evidence first (`docs/performance-
   profiling.md`, `results/performance-profiling/measurements.json`) and
   recomputed every cited share/percentage directly from the JSON before
   trusting the prior phase's own prose — this session's own starting
   master prompt included a correction to an earlier (incorrect) verbal
   handoff, underscoring why re-deriving from the committed artifact,
   not from a summary, is the right default.
2. Source-level ownership investigation for the two highest-priority
   targets (OPT-01, OPT-02) — read the exact code inside each timed
   boundary, not just the boundary's name.
3. New, additive stage-boundary instrumentation (OPT-03) to split
   `planner_ms` into its three real sub-costs, verified by direct
   re-measurement (sum ≈ `planner_ms`), not assumed correct by
   construction.
4. A larger, more carefully controlled precision-stability experiment
   (OPT-04): 8 warm repeats per KV mode, interleaved, vs. Phase 24's
   original 3-repeats-per-session design.
5. OPT-05 (the two unmeasured Qwen2.5-1.5B GPU points) deliberately NOT
   re-attempted — the Phase 25 task explicitly forbids retrying that
   6/6-failure campaign.
6. OPT-06 (fallback trace missing from the error-JSON schema) — a real,
   low-risk, non-performance gap Phase 24 found — fixed additively.

## OPT-01: model load time — INVESTIGATED_NOT_ACTIONABLE

**Measured cost** (Phase 24, unchanged): 53.7% of total on
`smollm2-135m_vulkan_native`, 38.8% on `qwen2.5-1.5b_cpu_native`.

**Ownership: LLAMA_CPP_OWNED / BACKEND_OWNED / DRIVER_OWNED.**
`model_load_ms` times exactly one call in `tools/membrane-run/main.cpp`:

```cpp
model = llama_model_load_from_file(o.model_path, mp);
```

No MEMBRANE-authored code executes inside this boundary. MEMBRANE only
sets three fields on the `llama_model_params` struct passed in
(`n_gpu_layers`, `main_gpu`, `devices`) — every other field, including
`load_mode` (defaults to `LLAMA_LOAD_MODE_MMAP`), is llama.cpp's own
default, untouched by this product.

**Cold-start root cause, now source-confirmed** (Phase 24 called this
"plausible, not measured" — this phase verified it directly):
`third_party/llama.cpp/src/llama-model-loader.cpp:1500` calls
`ggml_backend_dev_init(dev, nullptr)` — the real Vulkan device creation
and shader/pipeline compilation step — from *inside* the model-loading
path `llama_model_load_from_file()` triggers.
`third_party/llama.cpp/src/llama.cpp`'s `llama_backend_init()` (called
once, early, well before `resolve_gpu_config()`) only calls
`ggml_backend_load_all()` — loading backend *registries*, never
initializing a device. MEMBRANE's own device-enumeration call
(`membrane_gpu_list_devices()`) only calls `ggml_backend_dev_get_props()`
and never pays this cost either. `ggml-vulkan.cpp` contains dozens of
`ggml_vk_create_pipeline()` call sites reachable from that same
device-init path — consistent with the large, real, one-time cold-start
cost Phase 24 measured (1362.7 ms cold vs. 475.1/467.4 ms warm for
`smollm2-135m_vulkan_native`).

**Considered and rejected:** setting `load_mode` to `MLOCK` or
`DIRECT_IO` (a real, MEMBRANE-controllable knob llama.cpp exposes,
found during this investigation) — not attempted. `MLOCK` risks real
instability on this host's 5.6 GiB RAM for any model larger than the
smallest test fixture; neither option changes the two real cost
drivers for a Vulkan point (GPU weight upload, one-time shader/pipeline
compilation), which happen regardless of how the file is read from
disk.

**Conclusion:** no MEMBRANE-owned code exists inside `model_load_ms` to
optimize. Forking/patching llama.cpp or ggml-vulkan merely because they
are slow is out of this phase's scope (Section 3 of the Phase 25 task).

## OPT-02: decode throughput — INVESTIGATED_NOT_ACTIONABLE

**Measured cost** (Phase 24, unchanged): 22.6% of total on the
small-model point, 30.4% on the large-model point; CPU decode as low
as 2.2–3.7 tok/s for Qwen2.5-1.5B.

**Ownership: LLAMA_CPP_OWNED / BACKEND_OWNED.** `decode_ms` times
`run_generation()`'s loop, which calls `timed_decode()` once per
generated token. `timed_decode()` (`tools/membrane-llama-runtime/
decode_loop.cpp`) wraps exactly one real inference call,
`llama_decode(ctx, batch)`, plus MEMBRANE's own per-step bookkeeping.

**MEMBRANE's own per-token overhead, quantified by direct source
read** — every one of these runs on `membrane-run`'s real product-CLI
path (`hook_ctx` is always `NULL` there):

- `membrane_llama_hook_set_step_context()` / `membrane_llama_hook_flush_step()`
  (`llama_hook.cpp`) — both are `if (ctx == NULL) return;` no-ops, since
  `run_kv_store_pass()` always passes `hook_ctx = NULL` on this path
  (only `membrane-llama-run`'s own research/comparison tooling ever
  passes a real hook context).
- `membrane_runtime_begin_step()` / `membrane_runtime_end_step()` /
  `membrane_runtime_add_inference_seconds()` (`runtime_core.c`) — two
  integer increments, one true no-op, one double addition. O(1),
  unmeasurable at millisecond precision.
- Token-piece conversion / the optional `--stream` callback only run
  when text output or streaming is actually requested, are O(piece
  length), and are existing, unavoidable CLI-output work — not hidden
  research instrumentation, and not inside `decode_ms`'s own timed span
  in the first place.

**Conclusion:** MEMBRANE adds no meaningful per-token overhead on the
real product path. The entire `decode_ms` cost is inside
`llama_decode()` itself. Per Section 10 of the Phase 25 task ("If
profiling shows llama.cpp matmul dominates: do not reimplement
llama.cpp"), no safe product-local optimization exists here, and no
SIMD/threading work was attempted (there is no MEMBRANE-owned hot loop
to speed up).

## OPT-03: planner-stage cost attribution — OPTIMIZED (instrumentation)

New additive `planner_stages` JSON object (`device_enumeration_ms`,
`gguf_prescan_ms`, `joint_planner_core_ms`), timed independently inside
`resolve_gpu_config()` around its three real sub-steps. Each field
defaults to `0.0` when its own call site doesn't run on a given
invocation (e.g. explicit CPU-only never enumerates a device).

**Correctness check** (5 repeats, `smollm2-135m_vulkan_native`): the
three sub-costs sum to approximately the already-measured `planner_ms`
every time:

| Repeat | `device_enumeration_ms` | `gguf_prescan_ms` | `joint_planner_core_ms` | sum | `planner_ms` |
|---|---|---|---|---|---|
| 1 | 0.288 | 10.956 | 0.009 | 11.253 | 11.278 |
| 2 | 0.255 | 8.573 | 0.001 | 8.829 | 8.840 |
| 3 | 0.311 | 9.155 | 0.002 | 9.468 | 9.483 |
| 4 | 0.268 | 8.449 | 0.001 | 8.718 | 8.730 |
| 5 | 0.299 | 8.275 | 0.001 | 8.575 | 8.584 |

CPU-only: all three sub-costs measured exactly `0.0`, matching
`planner_ms`'s own already-tiny `0.007` ms.

**Real finding, correcting Phase 24's own attribution:**
`gguf_prescan_ms` (8.3–9.2 ms) dominates the GPU-requested planner
window — **not** device enumeration (0.25–0.31 ms). Phase 24's own doc
described this as "device-enumeration-plus-GGUF-scan cost" as if both
mattered comparably; this phase's direct measurement shows the GGUF
metadata pre-scan (`membrane_gpu_estimate_model()`, which calls
`gguf_init_from_file()` with `no_alloc=true`) is responsible for nearly
all of it. `joint_planner_core_ms` stays at 0.001–0.009 ms throughout,
confirming Phase 24's own hypothesis that the joint planner's own
arithmetic is genuinely negligible (the success condition Section 13 of
the Phase 25 task named).

**GGUF prescan optimization — considered, not implemented.** Section 14
of the Phase 25 task explicitly forbids breaking pre-load planning
semantics to avoid this scan: the joint planner needs GGUF metadata
*before* model load, to decide `gpu_layers` before calling
`llama_model_load_from_file()`. `membrane_gpu_estimate_model()` already
uses the minimal-cost API llama.cpp itself exposes for this
(`no_alloc=true`, metadata only). No further reduction identified
without patching ggml's own `gguf.cpp` (out of scope).

**Device enumeration caching — considered, not implemented.** It does
not dominate (0.25–0.31 ms of an 8.6–12.0 ms window), so the absolute
upside is negligible, and every real call site
(`resolve_gpu_config()`'s planning-time query,
`observe_gpu_memory_after_run()`'s post-run report, the fallback
controller's `real_refresh_fn()`) needs a fresh `memory_free` reading —
Section 15 of the Phase 25 task's own rule: "Never cache dynamic free
VRAM as immutable."

## OPT-04: native vs Q8 vs Q5 ordering stability — INVESTIGATED_NOT_ACTIONABLE (`NO_REPRODUCIBLE_PRECISION_SPEED_ORDERING`)

**Methodology:** SmolLM2-135M, Vulkan, `ctx=2048`, all 30 GPU layers, 8
warm repeats per mode (24 runs total), interleaved round-robin (native,
q8, q5) × 8, preceded by one separate discarded cold warm-up run — a
substantially larger, more carefully controlled sample than Phase 24's
original 3-repeats-per-session design.

**Cross-session comparison** (decode tokens/second, median):

| Session | native | q8 | q5 | Ordering |
|---|---|---|---|---|
| A — Phase 24 original (3 repeats) | 298.3 | 267.3 | 267.3 | native fastest |
| B — Phase 24 regenerated (3 repeats) | (slowest) | (fastest) | — | q8 fastest, native slowest |
| C — Phase 25, this investigation (8 repeats, rounds 2–8) | 309.6 | 267.2 | 258.7 | native fastest |

Round 1 of session C's own 8-repeat run was a clear outlier across all
three modes simultaneously (native 212.9 vs. 298–315 for rounds 2–8;
q8 232.0 vs. 262–274 for rounds 2–8) — most plausibly a residual
warm-up/host-scheduling effect despite the separate discarded warm-up
run beforehand, disclosed here rather than silently excluded without
explanation (full raw data: `results/performance-optimization/
validation.json`'s OPT-04 entry, all 24 samples).

**Conclusion:** excluding round 1, session C's *within-session*
ordering is tight and consistent (native clearly fastest, q8 second, q5
third) — but this does not resolve the *cross-session* question Phase
24 raised, since only one additional independent session was added: 2
of 3 sessions (A, C) show native fastest; 1 (B) shows the opposite.
**No code change made.** Precision-throughput ordering remains
genuinely unresolved across sessions on this host — now confirmed with
a larger, cleaner sample rather than assumed unstable from a 3-repeat
sample alone.

## OPT-05: the two unmeasured Qwen2.5-1.5B GPU points — DEFERRED_NEEDS_MORE_DATA

Per Section 29 of the Phase 25 task, **not re-attempted this phase** —
real host VRAM pressure was already established twice in Phase 24
(6/6 failures across two sessions); retrying would repeat risk to host
stability without producing new evidence. Remains deferred exactly as
Phase 24 scoped it: would need either a host with more available VRAM
headroom, or a dedicated investigation into `gpu_policy.h`'s
safety-reserve conservatism for larger models, neither of which is a
Phase 25 task.

## OPT-06: fallback trace missing from the error JSON schema — OPTIMIZED (additive fix, not performance)

**The gap** (found during Phase 24's own OPT-05 investigation): a fully
exhausted `AUTO_FALLBACK_EXHAUSTED` run's JSON (`print_error_json()`)
carried no `"fallback"` object at all, unlike the success schema's
detailed per-attempt trace — so a caller parsing JSON alone could not
see *why* the fallback was exhausted on a fully-failed run.

**The fix:** extracted the existing attempts-array/`attempted`/
`final_status`/`reason_code` emission out of `print_fallback_json()`
into a shared `print_fallback_trace_json(trace, trailing_comma)` helper
— byte-identical output on the already-working success path (pure
extraction, one new bool parameter to keep JSON comma-valid at two
different call sites) — then added an optional trailing
`fallback_trace` parameter to `print_error_json()` (default `NULL`,
every pre-existing call site unaffected). Only the one real
`AUTO_FALLBACK_EXHAUSTED` call site (`run_normal_mode()`, after
`membrane_fallback_run()` actually ran) passes `&gs.fallback_trace`.

**Verification:**

- Live CLI checks (real `membrane-run` binary): a normal success run
  with no fallback engagement still emits
  `"fallback":{"attempted":false,...}` unchanged; a real CLI
  model-load-failure error path (which does not pass a fallback trace)
  correctly omits the `"fallback"` key entirely — confirming the
  additive-only contract for every pre-existing error call site.
- `test_auto_fallback.c`'s existing Phase 21 test suite
  (`test_all_retryable_fail_exhausted`,
  `test_memory_shrink_skips_then_next_applies`,
  `test_same_candidate_never_attempted_twice`,
  `test_first_retryable_fails_second_succeeds`) already independently
  verifies every `membrane_fallback_trace_t` structural property this
  fix depends on — attempt count, exhausted reason/status, candidate
  indices never repeated and preserved in order, failure classes
  preserved per entry, and a memory-refit SKIP entry visible
  (`fit_after_refresh=0`/`apply_started=0`) — all passing under ASan.

**Known limitation, disclosed rather than silently assumed away:** a
full *live* end-to-end reproduction of `AUTO_FALLBACK_EXHAUSTED`
through the real CLI (real GPU, real driver-reported memory divergence)
was **not** attempted this phase — Section 29 explicitly forbids
retrying the Qwen2.5 GPU VRAM-pressure campaign that is this project's
only known real trigger for this failure mode, and no smaller, safe,
deterministic way to force it was found (a small model has enough real
headroom on this host's GPUs that forcing it would require the same
kind of memory-pressure probing Section 29 discourages). The new
JSON-wiring is therefore verified by direct code review plus the
layered evidence above, not by a fresh live top-to-bottom reproduction.

## Rejected / not-attempted approaches (kept, not hidden)

- **SIMD/threading changes to decode** — never attempted; OPT-02 found
  no MEMBRANE-owned hot loop to speed up in the first place (Section 10
  of the Phase 25 task).
- **`load_mode` change for model load** (MLOCK/DIRECT_IO) — considered
  under OPT-01, rejected: real instability risk on this 5.6 GiB RAM
  host, no evidence it would change the two real cost drivers anyway.
- **Device-enumeration caching** — considered under OPT-03, rejected:
  negligible absolute upside (0.25–0.31 ms), and every real call site
  needs a fresh dynamic free-memory reading by design.
- **A persistent model-cache daemon / resident service** — explicitly
  out of scope (Section 7 of the Phase 25 task); never considered.
- **Re-attempting the Qwen2.5 GPU VRAM campaign** (OPT-05) — explicitly
  forbidden (Section 29); not attempted.
- **A live `AUTO_FALLBACK_EXHAUSTED` reproduction for OPT-06** — not
  attempted, for the same Section 29 reason; disclosed as a known
  testing limitation above rather than silently skipped.

## Limitations

- Single host, single pair of Vulkan devices — every number above is
  device-scoped, never generalized (Section 19 of the Phase 24 task,
  carried forward).
- OPT-04's larger (n=8) sample still comes from a single additional
  session; a fully confident cross-session stability verdict would need
  several independent sessions run at different times/thermal/host-load
  states, not attempted this phase (time-boxed investigation, not a
  full statistical study).
- OPT-06's JSON-schema fix has no live end-to-end reproduction this
  phase, only structural/unit-level and direct-code-review verification
  (see above).
- `results/performance-profiling/measurements.json` (Phase 24) remains
  the authoritative, immutable baseline evidence — this phase never
  rewrites it; all Phase 25 evidence lives in
  `results/performance-optimization/validation.json`.

## Phase 26+ candidates (not implemented here)

- Re-run OPT-04's precision-stability experiment across several
  independent sessions (different times of day / thermal states) if a
  confident native-vs-q8-vs-q5 directional claim is ever needed.
- If a host with more VRAM headroom becomes available, revisit OPT-05
  (the two unmeasured Qwen2.5-1.5B GPU points) for real, rather than
  continuing to defer.
- If llama.cpp/ggml-vulkan's own upstream ever exposes a
  lower-cost model-load or backend-init path, re-evaluate OPT-01 against
  it — nothing in this phase's own code prevents that, since no
  MEMBRANE code was changed for OPT-01/OPT-02.
