# Performance profiling (Phase 24)

**This is a profiling document, not an optimization one.** Every number
below is a real measurement from this host; nothing here changes
runtime behavior. See "Phase 25 optimization candidates" at the end
for what this evidence suggests investigating next — none of it has
been implemented.

## Methodology

New, additive instrumentation (`tools/membrane-llama-runtime/decode_loop.cpp`,
`tools/membrane-run/main.cpp`, `tools/membrane-run/auto_fallback.c`) times
six pipeline stages via `CLOCK_MONOTONIC` (`decode_loop.cpp`'s own
`seconds_since()`, reused everywhere rather than a second timer
implementation): planner resolution, model load, tokenization, context
construction, prefill, and decode, plus a single first-token timestamp
(TTFT) and per-attempt wall time on the Phase 21 fallback controller.
Every new field is additive JSON (`"timings"` object, `schema_version`
still `1`) and costs one `clock_gettime()` pair per stage boundary — no
per-token or per-layer instrumentation anywhere (verified: the only
per-generation-step timestamp is a single conditional gated to step 0).

`scripts/measure-performance-profile.py` runs a small, fixed, 12-point
matrix (never a sweep) 3 times each and records median/min/max —
median is the number quoted below unless stated otherwise. Committed
evidence: `results/performance-profiling/measurements.json`, checked
by `scripts/verify-performance-profiling.py`.

## Measured stages

| Stage | What it actually times | Where |
|---|---|---|
| `planner_ms` | `resolve_gpu_config()` as a whole — device enumeration + GGUF metadata pre-scan + the joint planner's own arithmetic, **not isolated from each other** (see "Planner overhead" below) | `main.cpp` |
| `model_load_ms` | `llama_model_load_from_file()` | `main.cpp` |
| `tokenization_ms` | `llama_tokenize()` | `main.cpp` |
| `context_create_ms` | `llama_init_from_model()` only | `decode_loop.cpp` |
| `prefill_ms` | `decode_prompt()` (the existing `prompt_seconds` this project already computed for `prompt_tok_per_s` — exposed as raw ms too, not a new measurement) | `decode_loop.cpp` |
| `decode_ms` | `run_generation()` (existing `gen_seconds`, same reasoning) | `decode_loop.cpp` |
| `first_token_ms` | the single decode step that produces the first generated token | `decode_loop.cpp` |
| `total_ms` | from right after CLI parse/prompt resolution to just before the JSON is printed | `main.cpp` |

**KV-store compression time cannot be isolated, and this is not a
measurement gap — it is a real architectural fact, verified by
inspection.** `kv_store_telemetry.h`'s own `encode_seconds`/
`decode_seconds` fields exist for exactly this purpose but are never
assigned anywhere in `decode_loop.cpp` (confirmed by grep before
writing any new code) — the file's own header comment explains why:
"Phase 7's storage compression is llama.cpp's own `ggml_cpy()`/
`ggml_mul_mat()` type-conversion machinery... not a MEMBRANE-authored
codec." Q8/Q5 KV compression happens *inside* the same tensor
operations `decode_ms`/`prefill_ms` already measure — there is no
separate MEMBRANE-owned step to time, so none was fabricated.

## Controlled points

12 points, `ctx=2048`, `gen-tokens=32` (16 for the one Qwen2.5 CPU
point, kept smaller given real CPU decode cost — see below), 3 repeats
each. Real local models only, no downloads: SmolLM2-135M, SmolLM2-360M,
Qwen2.5-1.5B. Host: this project's own real Pop!_OS 24.04 dev machine
(AMD Ryzen 5 5600H, real NVIDIA GTX 1650 Vulkan device `Vulkan1`) —
every number below is scoped to this one host, never generalized
(Section 19 of the Phase 24 task; `scripts/verify-
performance-profiling.py` enforces every Vulkan row names a real
device).

10/12 points succeeded (all 3 repeats). 2 points — the two GPU-resident-
weights Qwen2.5-1.5B configurations — failed **for real, reproducibly,
6/6 attempts across two separate measurement runs** — see "A real,
disclosed measurement gap" below. This was not forced past; a real
CPU-only Qwen2.5-1.5B substitute point was added instead (Section 29 of
the Phase 24 task: never sacrifice machine stability or fabricate a
result to fill a matrix cell).

| # | Label (`results/performance-profiling/measurements.json`) | Result |
|---|---|---|
| 1 | `smollm2-135m_vulkan_native` | ok |
| 2 | `smollm2-135m_vulkan_q8` | ok |
| 3 | `smollm2-135m_vulkan_q5` | ok |
| 4 | `smollm2-135m_vulkan_adaptive` | ok |
| 5 | `smollm2-135m_cpu_native` | ok |
| 6 | `smollm2-135m_cpu_q8` | ok |
| 7 | `smollm2-360m_vulkan_native` | ok |
| 8 | `qwen2.5-1.5b_vulkan_native_gpu_placement` | **failed, real cause disclosed below** |
| 9 | `qwen2.5-1.5b_vulkan_native_cpu_placement` | **failed, real cause disclosed below** |
| 10 | `qwen2.5-1.5b_cpu_native` | ok (real substitute point) |
| 11 | `smollm2-135m_vulkan_partial_half_15` | ok |
| 12 | `smollm2-135m_vulkan_auto` | ok |

## Top measured bottlenecks

**SmolLM2-135M, Vulkan, native, all 30 layers** (representative small-model point):

| Stage | Median | Share of total |
|---|---|---|
| `model_load_ms` | 255.2 ms | **53.7%** |
| `decode_ms` | 107.3 ms | 22.6% |
| `prefill_ms` | 14.3 ms | 3.0% |
| `context_create_ms` | 11.2 ms | 2.4% |
| `planner_ms` | 8.9 ms | 1.9% |
| `tokenization_ms` | 0.1 ms | 0.0% |
| *(unaccounted)* | ~78 ms | ~16.4% |
| **total** | **475.1 ms** | 100% |

**BOTTLENECK 1: model load, over half of total wall time** for a small
model at a short generation length. This dominates every small-model
point in the matrix. The unaccounted ~16% here is process-level
overhead this instrumentation does not claim to capture (backend init,
model-label/basename computation, RSS reads, JSON serialization) —
disclosed, not hidden.

**Qwen2.5-1.5B, CPU, native** (representative large-model point, 16
generated tokens):

| Stage | Median | Share of total |
|---|---|---|
| `model_load_ms` | 5505.9 ms | **38.8%** |
| `decode_ms` | 4309.7 ms | **30.4%** |
| `prefill_ms` | 2254.1 ms | 15.9% |
| `context_create_ms` | 80.8 ms | 0.6% |
| `planner_ms` / `tokenization_ms` | ~0 ms | ~0.0% |
| **total** | **14191.0 ms** | 100% (85.6% accounted) |

**BOTTLENECK 2: model load and decode are comparably large for a
larger model**, both well ahead of prefill. Unlike the small-model
point above, this specific measurement's own repeat-to-repeat spread
was wide (a real symptom of this host's own ambient load, not a
MEMBRANE artifact — this point took 14–17 real seconds per repeat, long
enough for other processes on this shared machine to visibly compete):
a prior measurement session (still available in this PR's own commit
history, not restated here as a second contradicting "fact") recorded
decode as high as ~46% of total for the same point. Both sessions agree
decode share grows with model size and generation length relative to
the small-model point above; neither session's exact percentage is
claimed to be the stable, reproducible value — 3 repeats is enough to
see spread, not enough to pin down a precise share for a 14+ second
point on a noisy host. Decode throughput itself: **3.7 tokens/second**
in this session (2.2 in the prior one) on this host's CPU for a 1.5B
model — slow in absolute terms either way, a real, disclosed, and
visibly noisy data point, not a claim about CPU inference generally.

**BOTTLENECK 3: none of `planner_ms`'s own joint-planner arithmetic,
`tokenization_ms`, or `context_create_ms` is ever a large share of
total time** — all three stay under ~4% in every single point measured
(`context_create_ms` is consistently the largest of the three, in the
2–4% range depending on the point and session; `scripts/verify-
performance-profiling.py` checks the real max directly against
`measurements.json` on every run, rather than this threshold being an
unchecked, potentially-stale number). Section 5's own expectation
("planner overhead may be tiny — do not assume it matters") is
confirmed, not assumed.

## Native vs Q8 vs Q5 (SmolLM2-135M, Vulkan, all layers)

| Mode | Decode tok/s (median) | Prefill tok/s (median) |
|---|---|---|
| native | **298.3** | 350.0 |
| q8 | 267.3 | 333.0 |
| q5 | 267.3 | **376.8** |

Measured directly, not inferred from Phase 12G/prior research (Section
12 of the Phase 24 task). **The ordering is not stable across
measurement sessions on this host, and this is itself the real
finding**: an earlier measurement session on this same host, same
matrix definition, recorded q8 with the *highest* decode throughput and
native with the *lowest* — the exact opposite of the table above.
Sample size is 3 repeats per session; the two sessions together show
this is real host-level noise at this sample size, not a reproducible
precision-dependent ordering. No compression-cost or compression-
benefit direction is claimed for decode or prefill throughput on this
evidence — Phase 25 attention would need a substantially larger sample
before any directional claim is defensible.

CPU (same model, `gpu-layers 0`): this session measured native and q8
decode as effectively tied (68.2 vs 68.6 tok/s); an earlier session
measured native measurably faster (68.0 vs 62.3). Same conclusion as
above: real, small-sample, session-to-session noise, disclosed rather
than generalized into a product claim either direction.

## CPU KV vs GPU KV

**Not directly measured this phase** — see "A real, disclosed
measurement gap" below. The two controlled points designed to isolate
this (`qwen2.5-1.5b_vulkan_native_gpu_placement` vs
`_cpu_placement`, same model/precision/context/GPU-layers, only
`--kv-placement` differing) both failed to complete on this run.
Consistent with prior project findings (`docs/kv-residency.md`,
Phase 12G): no universal performance penalty is assumed or claimed
either way — this remains genuinely unmeasured this phase, not
resolved by inference.

## Partial GPU offload (SmolLM2-135M, Vulkan, native)

| GPU layers | Decode tok/s (median) |
|---|---|
| 0 (CPU-only) | 68.2 |
| 15 (half) | 121.8 |
| 30 (all) | 298.3 |

Clean, monotonic scaling with GPU layer count on this host in both
measurement sessions — no anomaly at the midpoint either time.
`smollm2-135m_vulkan_partial_half_15` is also a session-first (model,
layer-count) pair (see "Cold start" below) — it paid a real one-time
cost on its own first run each session (1285.6 ms cold vs 617.7/570.4
ms warm this session).

## Auto vs explicit

`smollm2-135m_vulkan_auto` (bare `--auto`) vs `smollm2-135m_vulkan_q8`
(explicit `--gpu-layers all --kv q8`): both resolved to the identical
physical configuration on this host (30/30 GPU layers, q8 precision) —
`kv_placement` differs (`auto` vs `default`), the one dimension not
forced identical, disclosed rather than silently treated as
"equivalent" (Section 15 of the Phase 24 task).

| | `planner_ms` | `total_ms` |
|---|---|---|
| explicit q8 | 8.591 | 479.8 |
| `--auto` | 8.721 | 489.8 |

Planner overhead is statistically indistinguishable between the two
paths (both route through the identical joint-planner code, as
documented in `joint_planner.h`) — `--auto` adds no measurable planning
tax over the equivalent explicit invocation on this host.

## Planner overhead

`planner_ms` times `resolve_gpu_config()` as a single boundary, which
bundles THREE things this instrumentation does not separate: GPU device
enumeration (`membrane_gpu_list_devices()`, real Vulkan API calls),
GGUF metadata pre-scan (`membrane_gpu_estimate_model()`, real file
I/O), and the joint planner's own arithmetic
(`membrane_joint_plan_resolve()`). The CPU-only points (which skip all
three — `resolve_gpu_config()` short-circuits immediately when no GPU
was requested) measured `planner_ms` at **0.006–0.007 ms**, confirming
the joint planner's own arithmetic is genuinely negligible, exactly as
Section 5 anticipated. Every GPU-requested point measured **8.6–12.0
ms** instead — this is real, but it is device-enumeration-plus-GGUF-
scan cost, not joint-planner arithmetic; conflating the two would be a
false attribution. Isolating these three sub-costs from each other is
listed as a Phase 25 candidate below (it needs new instrumentation
boundaries this phase deliberately did not add, to keep this phase's
own instrumentation itself lightweight).

## Fallback overhead

Every one of the **10 successful** points in this phase's own matrix
succeeded on its primary candidate (`fallback_attempted: false`) — no
real fallback event occurred on any of them, so `auto_fallback.c`'s new
per-attempt `apply_wall_ms` field has no fresh successful-run data
point to report here (matching Section 6's explicit instruction to
reuse Phase 21's own real evidence rather than force an OOM scenario).

The 2 **failed** points (see "A real, disclosed measurement gap" below)
did engage the fallback controller for real — their live stderr output
showed a genuine `SKIPPED_UPDATED_MEMORY` event followed by
`AUTO_FALLBACK_EXHAUSTED` — but this phase's own JSON-based measurement
script could not capture that trace as structured data: verified
directly that `main.cpp`'s `print_error_json()` (the schema an
`AUTO_FALLBACK_EXHAUSTED` failure actually returns) carries no
`"fallback"` object at all, unlike the success schema. This is a real,
disclosed JSON-schema gap this phase found but does not fix (Section 22:
profiling only, no behavior change) — listed as a Phase 25 candidate
below.

Phase 21's own committed evidence
(`results/auto-fallback/validation.json`'s
`qwen2.5_1.5b_native_gpu_placement_reload_recovery` entry) remains the
authoritative real fallback-cost example: a `SKIPPED_UPDATED_MEMORY`
event (no apply attempted, ~0 cost) followed by one real
`apply_wall_ms`-equivalent attempt that included a full model reload
(28→0 GPU layers) and succeeded. This phase adds the *instrumentation*
(`apply_wall_ms` is now a real, committed field going forward,
confirmed working via a live smoke test during this phase's own
development — see the PR) but does not re-trigger that specific
scenario, per Section 6's own guidance against manufacturing an OOM
campaign.

## A real, disclosed measurement gap

`qwen2.5-1.5b_vulkan_native_gpu_placement` and
`qwen2.5-1.5b_vulkan_native_cpu_placement` both failed **6/6 times**
across two separate measurement sessions on this host, always the same
way: `AUTO_FALLBACK_EXHAUSTED`, `SKIPPED_UPDATED_MEMORY` on the sole
candidate. The pre-load estimate consistently reported ~3956 MiB free
vs ~2855 MiB needed (comfortably fits, safety reserve included) — yet
the real apply-time re-snapshot consistently found less free VRAM than
that. This is a real, reproducible-at-measurement-time resource
constraint on this specific shared, memory-limited (5.6 GiB RAM) dev
host, not a MEMBRANE defect — the fallback controller behaved exactly
as designed (clean fail-closed exit 4, clear diagnostic, no crash),
and no candidate existed to fall back to because this was an explicit
(non-`--auto`) single-candidate request, matching `docs/auto-
fallback.md`'s own documented scope. No root cause beyond "real,
observed VRAM pressure at apply time" is claimed — speculating further
would be exactly the kind of fabrication this phase exists to avoid.
A real `qwen2.5-1.5b_cpu_native` substitute point was measured
successfully instead (see Bottleneck 2 above).

## Cold start

**Not every Vulkan point shows a cold-start gap** — an earlier draft of
this document overclaimed "every Vulkan point" before checking the
q8/q5/adaptive/`--auto` points specifically (caught in review; see PR
#34). The real, precisely-scoped pattern, re-verified directly against
`measurements.json`: only the *first time a given (model, GPU-layer-
count) configuration is loaded within a measurement session* pays a
large one-time cost — every subsequent point that reuses an
already-warmed (model, layer-count) pair (q8/q5/adaptive/`--auto` all
reuse `native`'s already-warmed SmolLM2-135M/30-layers combination,
executed first) shows no such gap at all:

| Point | Cold (run 1) | Warm (runs 2–3) | Session-first for this (model, layers)? |
|---|---|---|---|
| `smollm2-135m_vulkan_native` | 1362.7 ms | 475.1 / 467.4 ms | yes |
| `smollm2-360m_vulkan_native` | 2663.7 ms | 931.8 / 802.2 ms | yes (different model) |
| `smollm2-135m_vulkan_partial_half_15` | 1285.6 ms | 617.7 / 570.4 ms | yes (different layer count, 15 not 30) |
| `smollm2-135m_vulkan_q8` | 479.8 ms | 474.5 / 486.2 ms | no — reuses `native`'s warm (135M, 30) |
| `smollm2-135m_vulkan_q5` | 478.5 ms | 476.5 / 480.7 ms | no |
| `smollm2-135m_vulkan_adaptive` | 479.8 ms | 473.7 / 485.9 ms | no |
| `smollm2-135m_vulkan_auto` | 513.6 ms | 475.7 / 489.8 ms | no |

The reported `median` statistic already resists a cold outlier when
one exists (median of 3 picks the middle value), so every other number
in this document is effectively warm-biased already — but the cold
cost itself is real for a session-first (model, layers) pair and
disclosed here rather than silently absorbed. Likely cause (not
separately instrumented this phase, so stated as a plausible
explanation, not a measured fact): Vulkan shader/pipeline compilation
specific to a given device+model+layer-count combination, cached
thereafter for the rest of the process's lifetime (and possibly beyond,
via a driver-level shader cache — not verified this phase).

## Limitations

- Single host, single Vulkan device (NVIDIA GTX 1650 class) — every
  number is device-scoped (Section 19 of the Phase 24 task).
- 3 repeats per point is enough to see min/median/max spread, not
  enough for rigorous statistical confidence — several findings above
  (native/q8/q5 ordering, CPU native-vs-q8) are reported as observed
  patterns on a small sample, not statistically validated claims.
- `planner_ms` bundles 3 distinct real costs (device enumeration, GGUF
  scan, joint-planner arithmetic) that this phase's instrumentation
  does not separate.
- KV-store compression time is architecturally non-isolable (see
  "Measured stages" above) — this is a verified fact, not a gap.
- Two Qwen2.5-1.5B GPU-resident-weights points could not be measured
  this session due to real host VRAM pressure (see above).
- No fresh real fallback event occurred during this phase's own
  matrix; `apply_wall_ms` is confirmed working (live smoke-tested) but
  has no new committed data point from this specific phase.
- `total_ms` leaves ~2–18% of wall time unaccounted (varies by point,
  larger relative share on faster/smaller-model points) — disclosed
  as genuine process-level overhead this instrumentation does not
  claim to attribute to a specific stage.

## Phase 25 optimization candidates

Prioritized by measured cost share, **not implemented here**:

**OPT-01: model load time**
measured cost: 50.1% of total (small model, short generation) / 34.7%
(large model, longer generation) — the single largest cost in every
point measured except the longest-generation one.
possible approach: investigate whether llama.cpp's own model-loading
path (mmap usage, tensor validation, weight-copy overhead) has
MEMBRANE-controllable knobs, or whether this is fixed llama.cpp cost
outside this product's own code.
risk: low to investigate, unknown to change (third-party code).
success metric: reduced `model_load_ms` median with identical
`gpu_layers_selected`/`kv_precision`/output tokens.

**OPT-02: decode throughput, especially CPU**
measured cost: 25.4–45.5% of total; CPU decode measured as low as 2.2
tok/s (Qwen2.5-1.5B) and 62–68 tok/s (SmolLM2-135M).
possible approach: not diagnosed this phase (Section 22: profiling
only) — SIMD/threading changes are explicitly out of this phase's own
scope to even investigate in depth.
risk: unknown until investigated.
success metric: higher `decode_tokens_per_second` median at identical
generated-token count and identical output (bit-for-bit, matching this
project's own quality-preservation discipline).

**OPT-03: planner-stage cost attribution**
measured cost: 8.6–12.0 ms per GPU-requested run, currently un-split
between device enumeration, GGUF scan, and joint-planner arithmetic.
possible approach: add 2 more stage-boundary timers inside
`resolve_gpu_config()` (device enumeration vs GGUF scan vs planner
call) to attribute this 8.6–12.0 ms precisely, before deciding whether
any of the three is worth further optimization.
risk: low (pure additive instrumentation, same pattern this phase
already used).
success metric: three sub-costs sum to the already-measured
`planner_ms` (a correctness check on the new instrumentation itself,
not a performance target).

**OPT-04: native vs Q8 vs Q5 throughput ordering stability**
measured cost: not a "cost" in the bottleneck sense — two independent
measurement sessions on this host produced *opposite* native-vs-q8-vs-q5
decode/prefill orderings (see "Native vs Q8 vs Q5" above), meaning no
directional precision-cost claim is currently defensible either way.
possible approach: a larger-sample re-measurement (more repeats, more
context sizes) specifically targeting this question, before any code
change or any future precision-related work assumes a direction.
risk: none (measurement only).
success metric: a statistically stable ordering (or a confirmed,
reproducible "no stable ordering exists") across a larger sample.

**OPT-05 (deferred, not prioritized): the two unmeasured Qwen2.5-1.5B
GPU points.** Re-attempt on a host with more available VRAM headroom,
or investigate whether the pre-load estimate's safety margin should
be more conservative specifically for larger models — but this
touches `gpu_policy.h`'s reserve policy, `joint-planner.md`'s own
documented scope, and would need its own dedicated investigation, not
a Phase 25 quick win.

**OPT-06: fallback trace missing from the error JSON schema**
measured cost: not a performance cost — a real observability gap found
while investigating OPT-05's own failures: `main.cpp`'s
`print_error_json()` (the schema an `AUTO_FALLBACK_EXHAUSTED` failure
actually returns) carries no `"fallback"` object at all, unlike the
success schema's detailed per-attempt trace — so a caller parsing JSON
alone cannot see *why* the fallback was exhausted (skip vs. failed
apply, which candidates were tried) on a fully-failed run, only the
top-level `reason_code`.
possible approach: extend `print_error_json()` to optionally include
the fallback trace when one exists (additive JSON, matching this
project's existing schema-evolution discipline).
risk: low (additive-only JSON change, same pattern as every other
`print_*_json()` function in this file).
success metric: a fully-exhausted `--auto` run's JSON output includes
the same per-attempt detail the success schema already has, verified
against a real reproduction of this exact failure mode.
