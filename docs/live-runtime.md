# Live llama.cpp shadow/injection runtime (Product Phase 5/6)

The live runtime integration: MEMBRANE observes (Phase 5, **Level B**)
and, for `inject-*` modes (Phase 6, **Level C2**), authoritatively
writes back reconstructed K/V projection values *while* `llama.cpp` is
actually generating tokens, in memory, with no `.kvdump`/`.memkv`
round-trip. In every SHADOW mode, native llama.cpp KV remains the sole
authoritative store for attention. In every INJECT mode, native KV
cache *allocation* is still unchanged (never a process-memory claim),
but the tensor that actually feeds the cache write is replaced with
MEMBRANE's reconstruction, within the scope requested.

## Architecture

Three files, deliberately separated:

- `tools/membrane-llama-runtime/runtime_core.h/.c` — llama-free.
  Block packing/tail accounting, telemetry aggregation, weighted-error
  math, and the baseline/shadow token-sequence comparison. Zero
  ggml/llama includes; built and tested unconditionally
  (`test_runtime_core`, part of the default, llama-free CI suite).
- `tools/membrane-llama-runtime/llama_hook.cpp` — the only file that
  touches a `ggml_tensor`. Installs a
  `ggml_backend_sched_eval_callback` (`llama_context_params.cb_eval`)
  — a genuinely public, documented API
  (`ggml/include/ggml-backend.h`), unmodified on the pinned commit.
  Extracts data via `ggml_backend_tensor_get()` (never retains a
  pointer into ggml/llama-managed memory) and hands it to
  `runtime_core`.
- `tools/membrane-llama-runtime/main.cpp` — CLI, model load, greedy
  decode loop, mode dispatch, telemetry printing. Only builds with
  `-DMEMBRANE_ENABLE_LLAMA=ON`; the default llama-free build is
  unaffected.

## Why this hook, not the existing `kv_type_override` patch

This repository already has a working Level-C mechanism
(`patches/llama.cpp-membrane-kv-type-override.patch`, see
`docs/phase4-runtime-policy.md`) that lets a caller allocate each
layer's KV cache tensor at a different precision. This phase
deliberately does **not** use it: that mechanism requires patching
`third_party/llama.cpp`'s working tree, and its own real-runtime
validation already found that offline-predicted quality margins did
not reliably transfer to real execution (see `docs/phase4-runtime-
policy.md` for the full result). Phase 5 needs no submodule
modification at all: the eval callback is a stable, public API on the
unmodified pinned commit
(`c0bc8591e8815c63cb01dd3f051a8b0df02501c9`).

## What gets observed

`llm_graph_context::build_qkv()` (`src/llama-graph.cpp`) tags each
layer's K/V projection output `"Kcur-%d"` / `"Vcur-%d"`
(`ggml_format_name`) — these are exactly the tensors that flow into
`cpy_k`/`cpy_v`, the actual KV-cache write. Every other graph node
returns `false` on `ask` and is never touched.

**Important wrinkle, found by reading the source, not assumed (and
re-verified, more precisely, in Phase 6 — see below):** on this pinned
commit, for `LLM_ARCH_LLAMA` with no K bias (true for every model this
project validates against), `"Kcur-%d"` is tagged **three** times per
layer per step — twice inside `build_qkv()` (`src/llama-graph.cpp`:
once right after the raw linear projection, once after the final
reshape, both still pre-RoPE), and a third time by the caller
(`src/models/llama.cpp`), after `ggml_rope_ext` — three distinct tensor
objects, since each transformation produces a new node. Only that
third, post-RoPE value is what actually reaches the cache. (A layer
with a K bias would add one more internal tag.) `"Vcur-%d"` is tagged
twice, but both tags reference the *same* object (V is never RoPE'd on
this architecture) — one real graph node either way. The hook therefore
does not process a tensor immediately: it buffers the extracted value
per `(layer, is_v)`, overwriting on every observation, and only runs
MEMBRANE's select/encode/decode/validate pipeline once per key, in
`membrane_llama_hook_flush_step()`, after `llama_decode()` returns —
dependency-ordered graph execution guarantees the buffered value at
that point is the last one computed, i.e. the one the cache actually
holds. (This "last write wins" buffering is what makes SHADOW mode's
observation correct independent of the exact occurrence count; INJECT
mode's write-back, added in Phase 6, cannot defer this way — see
"Injection" below for how it identifies the correct occurrence
instead.)

## Modes

- `baseline` — native path. `cb_eval` is never installed at all (not
  merely a no-op callback) — zero MEMBRANE overhead, byte-identical to
  unpatched llama.cpp.
- `shadow-q8` — every observed full 128-element block runs through the
  maintained Q8_0 encode/decode.
- `shadow-adaptive` — the maintained content-driven Q4/Q8 selector,
  threshold unchanged from Phase 1-4
  (`MEMBRANE_QUANT_SELECT_DEFAULT_MAX_Q4_REL_L2_ERROR = 0.05`). Not
  tuned in this phase.
- `inject-q8` / `inject-adaptive` (Phase 6) — same encode/decode policy
  as their shadow counterparts, but the reconstructed values are written
  back via `ggml_backend_tensor_set()` into the tensor that actually
  feeds the KV-cache write, before the scheduler proceeds to compute the
  node(s) that consume it. See "Injection" below.

## Injection (Product Phase 6, Level C2)

**Mechanism.** No `third_party/llama.cpp` modification: the same public
`cb_eval` callback Phase 5 uses for observation is, for `inject-*`
modes, also used to *write*. `ggml-backend.h`'s scheduler computes a
node, calls back with `ask=false` (tensor materialized), and only then
proceeds to compute whatever consumes it — so mutating `t->data` inside
that call, via `ggml_backend_tensor_set()`, is visible to the
downstream node (`cpy_k`/`cpy_v`, the actual cache write) before it
runs. This is Level C2: the *source feeding* the cache write is
replaced, not the cache tensor itself after the fact.

**K's post-RoPE occurrence.** Because `"Kcur-%d"` is tagged three times
per layer per step for a K-bias-free `LLM_ARCH_LLAMA` layer (twice
pre-RoPE inside `build_qkv`, once post-RoPE by the caller; see above),
injection targets *only* that third occurrence — writing into an
earlier one would feed our reconstruction through a bias-add, reshape,
or RoPE a second time, corrupting it rather than replacing it cleanly.
The hook tracks an occurrence counter per layer, reset at the start of
every decode step (`membrane_llama_hook_set_step_context()`, called
before every `llama_decode()`, prompt chunks included), and additionally
verifies every previous step's final count for every in-model layer was
either 0 (never touched) or exactly the expected authoritative
occurrence number — any other value (a K-biased layer, a different
architecture, a llama.cpp revision that tags differently) is recorded
as a hard injection failure rather than silently targeting the wrong
tensor or never injecting K at all. This is a real, previously-latent
bug this project caught on itself, via this exact check, during Phase
6's own review cycle — see the introducing PR's history for detail.
`"Vcur-%d"`'s single real node is always authoritative.

**Scope.** `--inject-layer N` (repeatable; default every layer),
`--inject-tensor k|v|both` (default both), `--inject-token-start/-end N`
(inclusive absolute position range; default every token) — all
llama-free, unit-tested logic in `runtime_core.h`
(`membrane_runtime_scope_*`). Reconstruction only ever touches *full*
128-element blocks within an in-scope token's slice of the tensor; a
block never straddles a token boundary, and any trailing remainder
(`elements_per_token % 128`) is left as native, untouched, and counted
in `native_tail_values` — never silently dropped.

**Failure semantics.** A reconstruction failure (hard infrastructure
error or a soft decode failure — both treated as fatal for injection,
unlike the measurement-only contract in `precision_policy.h`) stops at
the first failing block; every value from that point on in the buffer
is guaranteed untouched, and the tensor is **not** written back — native
values are retained for it. The callback always returns `true` (never
aborts the graph), but the failure is recorded on the collector
(`membrane_runtime_injection_has_failed()`) and the whole run exits
nonzero, reporting `injection_succeeded: false` — never silent fallback
while still claiming inject mode.

**Behavior comparison.** Every `inject-*` run performs three decode
passes over the same prompt (see `main.cpp`'s module comment for detail:
pass A, pure baseline/free-running, is the reference; pass B, injection/
free-running, is what's reported as canonical telemetry, and its
free-running token sequence is compared against A's via
`membrane_runtime_detect_divergence()`; pass C, injection/teacher-forced
on A's own reference tokens, isolates KV-perturbation effects from
autoregressive cascade and feeds the aligned `logit_abs_diff`/
`top1_preservation_rate`/`topk_overlap`/NLL summary). Never more than
one `llama_context` is alive at a time.

**Debug-only correctness proof.** `--debug-perturb-injection`
deliberately corrupts every successfully-reconstructed value before
write-back, so a local run can directly observe that downstream logits
change in response (proof write-back is consumed, not a silent no-op).
Never set for a reported run.

## Block geometry

Same 128-element blocks as Phase 1-4. A per-observation remainder
(`n_elems % 128`) is never processed or padded — it is counted and
reported as `tail_values_excluded`, alongside `total_values_observed`
and the full block count, so nothing is silently dropped from the
accounting.

## Timing semantics

`inference_seconds` is the *outer* measurement for a step: it spans
`llama_decode()` plus this step's MEMBRANE flush. `membrane_seconds`
(extraction inside the callback + the flush's quantize/encode/decode
work) is always a **subset** of it, never additive — "overhead ratio"
is the fraction of that wall time MEMBRANE actually spent in, not
extra time layered on top of an unrelated baseline figure.

## Validation

Because shadow mode does not replace native llama KV state,
deterministic `baseline` and `shadow-q8`/`shadow-adaptive` runs (same
model, prompt, and greedy decoding) are expected to produce identical
`token_ids` arrays. The CLI's `--json` output includes `token_ids`
specifically so a local real-model run can check this directly.
`membrane_runtime_tokens_equal()` (`runtime_core.h`) is the reusable,
unit-tested comparison helper for that check.

## Live-interleaving check

`--debug-runtime` prints, after each `llama_decode()` call returns,
the number of KV blocks MEMBRANE processed strictly within that step
(`gen step N: llama decode; membrane processed K KV blocks`) — this is
how a local run can check that MEMBRANE processing happens interleaved
with generation, not as a post-run pass, rather than a claim recorded
here as an already-verified fixed result.

## Limitations

- SHADOW modes never replace native llama KV; INJECT modes replace only
  the tensor feeding the cache write — native FP16/F32 cache
  *allocation* is unchanged in every mode. "Encoded payload reduction
  for observed/injected KV blocks" is never an actual process-memory or
  RAM claim, and INJECT modes never claim actual RAM savings either.
- No end-to-end LLM speedup claim; MEMBRANE processing (shadow or
  inject) adds real wall-clock overhead by design, reported as measured.
- No physical FPGA/CXL claim anywhere in this tool.
- Verified against `LLM_ARCH_LLAMA` (the architecture `build_qkv`'s
  Kcur/Vcur tagging was read from) — other architectures' graph
  builders were not audited and may tag tensors differently.
- CPU-only, single host; no GPU/accelerator backend exercised.
- INJECT modes' behavior/quality results are local, single-model,
  single-prompt measurements. No committed CSV/JSONL/log artifact
  exists yet for these numbers, so none are reproduced here — this
  document states only the mechanism and semantics, never a specific
  figure this repository cannot check via
  `scripts/verify-results.py`. This is not a general claim about other
  models, prompts, or context lengths.
- The aligned/teacher-forced evaluation (pass C) isolates KV-
  perturbation effects from autoregressive cascade for *logit/NLL*
  comparison, but the free-running divergence (pass B vs. pass A) can
  still cascade once token identity itself diverges — both are
  reported, neither is presented as the other.
