# Compatibility expansion (Phase 26)

**Decision: `EXPAND_QWEN2_COMPRESSION`.** Compressed KV (`q8`/`q5`/
`adaptive`) is now supported for Qwen2 (specifically: proven against the
already-committed `Qwen2.5-1.5B-Instruct` fixture), in addition to
`llama`. This is a real, evidence-backed expansion — two independent
layers of proof were required before any code changed, and a real
CPU+Vulkan experiment confirmed it end to end. See
[`docs/compatibility.json`](compatibility.json)'s `MC-17`/`MC-18`/`MC-19`
rows and [`results/compat-expansion/validation.json`](../results/compat-expansion/validation.json)
for the machine-checked evidence this page summarizes.

## Why Qwen2, and why now

Phase 25 (`docs/performance-optimization.md`) concluded there is no
MEMBRANE-owned model-load/decode hot loop worth optimizing further. This
phase's own task explicitly redirected effort toward product value
instead: which one compatibility/backend/model expansion is both
high-value and provably safe? Qwen2 was the strongest candidate because
it is *already* a real, tested MEMBRANE product model (`native`
precision has been `SUPPORTED` and exercised on real hardware since
Phase 18) — the only open question was whether `compat_check.c`'s
`arch_name == "llama"` gate protected a real incompatibility or was
simply a conservative validation gate never revisited since Phase 8.

## Architecture inventory (focused, not exhaustive)

Per Section 2 of the Phase 26 task, this is not an exhaustive survey of
every architecture llama.cpp implements — just the two relevant to this
decision, both confirmed against the pinned `third_party/llama.cpp/src/
llama-arch.cpp`'s own `LLM_ARCH_*` name table:

| | `llama` (`LLM_ARCH_LLAMA`) | `qwen2` (`LLM_ARCH_QWEN2`) |
|---|---|---|
| Real tested model | SmolLM2-135M/360M-Instruct | Qwen2.5-1.5B-Instruct |
| GQA | yes (varies by model) | yes |
| Sliding-window attention | no | no (`hparams.swa_type` stays `LLAMA_SWA_TYPE_NONE` — confirmed: `qwen2.cpp`'s `load_arch_hparams()` never touches it) |
| KV-cache class (pinned llama.cpp) | `llama_kv_cache` (generic, non-SWA/non-hybrid path) | same |
| Attention/KV build helpers | `build_attn_inp_kv()` / `build_attn()` | same (verified: identical call sites in `models/llama.cpp` and `models/qwen2.cpp`) |

## Qwen2 source-level investigation (Sections 3–6)

**Question:** is `compat_check.c`'s `strcmp(arch_name, "llama") != 0`
gate protecting a real incompatibility, or is it a conservative gate from
when only `llama` had been proven?

**Answer, source-proven:** it was the latter. `llama-model.cpp`'s
`llama_model::create_memory()` dispatches every architecture through a
`switch (arch)` — a short list of architectures needs a *specific*
memory-class instantiation (`LLM_ARCH_DEEPSEEK32` → `llama_kv_cache_dsa`,
`LLM_ARCH_DEEPSEEK4` → `llama_kv_cache_dsv4`, recurrent/hybrid
architectures → `llama_memory_recurrent`/`llama_memory_hybrid`, SWA
architectures → `llama_kv_cache_iswa`), and **`LLM_ARCH_QWEN2` is in
none of those lists** — it falls through to the final `default:` →
`else` branch:

```cpp
// llama-model.cpp, create_memory()'s final branch
} else {
    GGML_ASSERT(!hparams.is_swa_any());
    res = new llama_kv_cache(
            *this, hparams, params.type_k, params.type_v,
            !cparams.flash_attn, cparams.offload_kqv, cparams.kv_unified,
            cparams.n_ctx_seq, cparams.n_seq_max, 1,
            hparams.n_swa, hparams.swa_type, nullptr, filter,
            nullptr, nullptr,
            params.kv_type_override, params.kv_type_override_ud,
            params.kv_dev_override, params.kv_dev_override_ud);
}
```

This is the **exact same branch `llama` itself uses**, and it explicitly
passes `params.kv_type_override`/`kv_dev_override` — the two fields
MEMBRANE's own product patches (`patches/llama.cpp-membrane-kv-type-
override.patch`, `patches/llama.cpp-membrane-kv-device-override.patch`)
add to `llama_context_params`. Both patches modify only generic,
architecture-independent files (`llama-context.cpp`, `llama-kv-cache.cpp`)
— never any per-architecture model file.

`llama-kv-cache.cpp`'s own KV tensor creation confirms the override is
consumed identically regardless of architecture:

```cpp
// llama-kv-cache.cpp
const ggml_type layer_type_k = kv_type_override
    ? kv_type_override(il, false, kv_type_override_ud) : type_k;
const ggml_type layer_type_v = kv_type_override
    ? kv_type_override(il, true, kv_type_override_ud) : type_v;
ggml_tensor * k = has_k ? ggml_new_tensor_3d(ctx, layer_type_k,
    n_embd_k_gqa, kv_size, n_stream) : nullptr;
```

No architecture parameter appears anywhere in this function. `RoPE`
treatment, GQA ratio, and attention layout (Section 5 of the task) are
all irrelevant to this specific question: `kv_type_override` fires
*after* Q/K/V are computed (RoPE already applied), at the point the
cache tensors are allocated and written — it changes storage dtype only,
never attention semantics.

## Real Qwen2.5-1.5B shape check

Qwen2.5-1.5B-Instruct's real hparams (dumped via a throwaway program
linking the already-built `membrane_gpu_estimate_model()`, not guessed):

```text
arch_name=qwen2  n_embd=1536  n_head=12  n_head_kv=2  n_layer=28
head_dim=128  (1536 / 12)
```

`head_dim=128` is evenly divisible by both `Q8_0`'s and `Q5_1`'s
32-element block size (`128 / 32 = 4`) — every existing shape check in
`compat_check.c` (the same checks that already gate `llama` models)
passes for this model's real shape. No new shape-check code was needed.

## Real experiment (Sections 7–12)

A temporary local override (never committed) first proved the concept
safely; the same 8 checks were re-run against the final, committed
allowlist-based gate for the evidence below (`results/compat-expansion/
validation.json`, commit `3b8d87ebcb429d8f651fec2d2b88f94e14f2c293`).
`ctx=512`, `gen-tokens=16`, real local `Qwen2.5-1.5B-Instruct` fixture,
no downloads:

| Row | Backend | Precision | Placement | Result |
|---|---|---|---|---|
| CE-01 | CPU | native | default | baseline: `" Paris. The capital of France is also the capital of which other country?\nA"` |
| CE-02 | CPU | q8 | default | ok — coherent text, KV bytes exactly 53.125% of native |
| CE-03 | CPU | q5 | default | ok — coherent text, KV bytes exactly 37.5% of native |
| CE-04 | CPU | adaptive | default | resolved to q8 (`CPU_ADAPTIVE_Q8_DEFAULT`) — byte-identical to CE-02 |
| CE-05 | Vulkan | q8 | default | ok — byte-identical text/KV to CE-02 |
| CE-06 | Vulkan | q5 | default | ok — byte-identical text/KV to CE-03 |
| CE-07 | Vulkan | q8 | **cpu** | ok — same text as CE-05, KV correctly placed off-GPU (`PLACEMENT_CPU_FULL`) |
| CE-08 | Vulkan | adaptive | default | resolved to q8 (`Q8_FITS`) — byte-identical to CE-05 |

**Quality:** every compressed row produced coherent, plausible text
starting with the same first token as native (`" Paris."`) — the same
quality discipline the existing `llama` q8/q5 evidence uses (never
required to be byte-identical to native, since real quantization is
expected to shift generation; here it didn't even need to, at this
short a generation length).

**Memory:** `q8` KV bytes were **exactly** 53.125% of native's, `q5`
**exactly** 37.5% — matching the real `ggml` `Q8_0`/`Q5_1` block-format
formulas (34 and 24 bytes per 32-element block, vs. `F16`'s 2
bytes/element) to the byte, not approximately. `scripts/verify-
compat-expansion.py` recomputes and checks this ratio independently on
every run, not just at evidence-authoring time.

**Placement independence (CE-07):** `q8` + `--kv-placement cpu` resolved
correctly and independently of precision — the same two-axis design
`llama` already has, no special-casing.

**Adaptive (CE-04, CE-08):** resolved to `q8` on both backends via the
*unchanged* joint planner — `membrane_joint_plan_resolve()` already
delegated its architecture gate to `membrane_check_kv_compat()`
(`joint_planner.c`'s own `build_candidate()`), so widening that one
function's allowlist was sufficient. Zero Qwen2-specific code exists
anywhere in `joint_planner.c` (Section 11 of the Phase 26 task).

**Why `--gpu-layers 4`, not `all`, for the Vulkan rows:** a full
28-layer Qwen2.5-1.5B GPU-resident weight load already has a real,
disclosed VRAM-pressure problem on this host, independent of this
phase's own work — see `docs/performance-profiling.md`'s "A real,
disclosed measurement gap" (Phase 24) and `docs/performance-
optimization.md`'s OPT-05 (Phase 25, deliberately deferred, not
re-attempted here per that phase's own Section 29). Partial offload
(4 layers) isolates the KV-storage-*type* question this phase is
actually investigating from that unrelated, already-known capacity
constraint — Section 28 of this phase's own task: no deliberate OOM.

## Compatibility matrix changes (Section 14)

`docs/compatibility.json`'s `MC-17`/`MC-18`/`MC-19` reclassified from
`UNSUPPORTED` to `SUPPORTED`, each citing the specific `results/
compat-expansion/validation.json` row(s) that prove it. **Scope stays
exact**: this is a claim about `Qwen2.5-1.5B-Instruct` and the `qwen2`
architecture string, not "all Qwen2 models" or "all llama.cpp
architectures." `MC-26` (`NOT_YET_VALIDATED` for every architecture
besides `llama`/`qwen2`) is unchanged in meaning — Qwen2 simply moved
out of that bucket into direct evidence, everything else llama.cpp
implements is still there.

## compat_check.c design (Section 15)

`if arch != llama -> reject` became an explicit allowlist array
(`MEMBRANE_COMPRESSED_KV_ARCH_ALLOWLIST = {"llama", "qwen2"}`) plus a
`membrane_arch_supports_compressed_kv()` helper, not a widened `&&`
chain on the original `if`. Adding a future third architecture is one
array entry, not a restructured conditional. Existing shape checks
(head-dimension/block-alignment, degenerate-shape rejection) are
unchanged and still run after the allowlist check, for every allowlisted
architecture — proven by `test_compat_check.c`'s
`test_q8_qwen2_bad_shape_rejected`.

## Historical evidence (Section 13)

`docs/joint-planner.md`'s and `docs/auto-fallback.md`'s own "Qwen2.5
regression" sections describe **real, immutable evidence** captured at
the Phase 20/21 commits (`results/joint-planner/estimate-
correction.json`, `results/auto-fallback/validation.json` — neither file
is rewritten by this phase). Both doc sections were edited to state
plainly that they describe historical, since-superseded behavior, with
a pointer to this page for the current, real evidence — the underlying
JSON evidence files themselves are untouched.

`test_joint_planner.c`'s two "qwen2 is the incompatible-architecture
example" unit tests (`test_incompatible_architecture_adaptive_fails_
closed`, `test_explicit_q8_incompatible_architecture`) were switched to
use `gemma3` instead — these are live, current-behavior unit tests (not
evidence artifacts), so leaving them asserting a now-false claim about
`qwen2` would have been a real regression, not evidence preservation.

## CUDA decision (Section 19)

**`DEFERRED_NO_VALIDATION_HARDWARE`.** This dev host has no NVIDIA CUDA
toolkit installed and no way to validate a CUDA backend build or runtime
behavior — llama.cpp does provide a CUDA backend upstream (analogous
plumbing to the Vulkan backend this product already uses), so a future
CUDA expansion is plausible in principle, but MEMBRANE assumptions that
are currently Vulkan-specific (`gpu_device.cpp`'s device enumeration via
`ggml_backend_dev_*`, generic and already backend-agnostic; the
`--device` substring-match UX, backend-agnostic) were not audited in
depth this phase because there is no hardware to validate any of it
against. Packaging/CI complexity was not assessed in depth for the same
reason — auditing a decision this phase cannot validate at all would
itself risk becoming unfounded speculation. Confirmed by inspection, not
assumed: `git grep -i cuda` across this repo's own (non-`third_party`)
CMake files returns nothing (`scripts/verify-compatibility.py`'s own
`check_cuda_absence` already enforces this — MC-23).

## AMD/Intel Vulkan decision (Section 20)

No new hardware was available this phase beyond what Phase 23 already
used (one real AMD RADV RENOIR integrated GPU, `Vulkan0` on this host).
No broad AMD/Intel Vulkan support claim is made or implied by this
phase's work — the real signal Phase 23 already recorded stands
unchanged, not reclassified into a blanket claim.

## What was *not* done

- No CUDA backend implementation (deferred, no hardware — see above).
- No other architecture's compressed-KV support investigated in depth
  (Qwen2 was clearly the strongest candidate per Section 18's own
  scoring criteria: already a real, tested MEMBRANE product model, a
  local fixture already committed, and — as this investigation
  confirmed — genuinely lower structural risk than any unproven
  architecture would have been).
- No broader Vulkan hardware validation campaign (would need
  independent hardware this session does not have — same constraint
  Phase 23 already disclosed).
- No planner or fallback *code* changes — both already worked correctly
  for Qwen2 once `compat_check.c`'s gate widened, proven by new tests
  rather than new production code.

## Limitations

- Single host, single pair of real Vulkan devices — every row above is
  device-scoped, never generalized.
- Vulkan rows used partial offload (4 layers), not full 28-layer
  residency, to avoid this host's already-known, unrelated VRAM-pressure
  issue with this specific model at "all" GPU layers (OPT-05, Phase 25)
  — the KV-storage-type question this phase investigates does not
  depend on full weight residency, but a claim specifically about
  Qwen2.5-1.5B fully GPU-resident + compressed KV together remains
  unmeasured on this host.
- `ctx=512`, 16 generated tokens per row — small, controlled points
  (Section 28 of the Phase 26 task), not a capacity/throughput sweep;
  no claim is made about compressed-KV behavior at large context for
  this architecture.
- "Qwen2.5-1.5B tested" is not "all Qwen2 models supported" — see
  `docs/compatibility.md`'s own "Do not generalize" framing.
