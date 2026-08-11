# Live llama.cpp shadow runtime (Product Phase 5)

The first live runtime integration: MEMBRANE observes and quantizes/
dequantizes real K/V projection values *while* `llama.cpp` is actually
generating tokens, in memory, with no `.kvdump`/`.memkv` round-trip.
This is **Level B**: a controlled shadow-validation path. Native
llama.cpp KV remains the sole authoritative store for attention in
every mode here.

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

**Important wrinkle, found by reading the source, not assumed:** on
this pinned commit, `"Kcur-%d"` is tagged *twice* per layer — once
before RoPE (inside `build_qkv`), once after (by the caller) — two
distinct tensor objects, since RoPE produces a new node. Only the
post-RoPE value is what actually reaches the cache. `"Vcur-%d"` is
also tagged twice, but both tags reference the *same* object (V is
never RoPE'd on this architecture). The hook therefore does not
process a tensor immediately: it buffers the extracted value per
`(layer, is_v)`, overwriting on every observation, and only runs
MEMBRANE's select/encode/decode/validate pipeline once per key, in
`membrane_llama_hook_flush_step()`, after `llama_decode()` returns —
dependency-ordered graph execution guarantees the buffered value at
that point is the last one computed, i.e. the one the cache actually
holds.

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

- Shadow only: native llama KV remains authoritative. "Encoded payload
  reduction for observed KV blocks" is never an actual process-memory
  or RAM claim.
- No end-to-end LLM speedup claim; MEMBRANE shadow processing adds
  real wall-clock overhead by design, reported as measured.
- No physical FPGA/CXL claim anywhere in this tool.
- Verified against `LLM_ARCH_LLAMA` (the architecture `build_qkv`'s
  Kcur/Vcur tagging was read from) — other architectures' graph
  builders were not audited and may tag tensors differently.
- CPU-only, single host; no GPU/accelerator backend exercised.
