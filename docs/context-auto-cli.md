# `--ctx auto` -- hardware-aware automatic context sizing

Phase 35 exposes Phase 33's context recommendation core and Phase 34's
host-memory guard through the real product CLI, as a new `--ctx auto`
value. This is the first public-facing piece of MEMBRANE's v0.4 thesis:
removing the need to guess GPU layers, KV precision, KV placement, and
context size.

## Usage

```
membrane-run \
  --model model.gguf \
  --prompt "Hello" \
  --ctx auto \
  --auto
```

`--ctx auto` requires a real prompt (`--prompt`/`--prompt-file`/
`--prompt -`) -- the recommendation needs to know how much context this
exact run needs. It does **not** require `--auto`: `--ctx auto` alone
recommends a context under whatever other flags (explicit or default)
are already in force --

```
membrane-run --model model.gguf --prompt "Hello" --ctx auto
```

recommends a context for a plain CPU-only, native-KV run (the ordinary
defaults), no different in spirit from typing `--ctx 2048` yourself
except the number is chosen for you. Explicit `--kv`/`--gpu-layers`/
`--kv-placement`/`--device` all stay hard constraints either way --
recommendation never overrides them (see "Explicit constraints" below).

`--plan-only` works with `--ctx auto` too, and is the recommended way
to see what would be chosen without generating anything:

```
membrane-run --model model.gguf --prompt "Hello" --ctx auto --auto --plan-only
```

`--ctx auto` is not yet supported together with `--compare-kv`/
`--gpu-bench` (those explicit benchmark modes were out of this phase's
scope) -- use an explicit `--ctx N` with them.

## How recommendation works

1. A cheap **vocab-only** model load (no weight tensors) tokenizes the
   real prompt with the exact tokenizer the real run will use.
2. The model's own GGUF metadata is read once (the same pre-load scan
   `--inspect-model`/`--gpu-layers auto` already use) for architecture,
   layer/KV shape, and the model's own context-length ceiling.
3. Real host memory (`/proc/meminfo`) and, when a GPU is in play, real
   device free/total VRAM are read once.
4. A bounded, deterministic set of candidate context sizes (Phase 33's
   own `membrane_ctxrec_generate_candidates()`) is evaluated against
   the **existing, unchanged** joint planner plus Phase 34's host-
   memory guard -- never a second planner.
5. The largest candidate that is compatible, fits real GPU memory (if
   any), and fits real host memory becomes the recommendation.
6. That exact context is fed into the normal runtime path -- the same
   `resolve_gpu_config()`/model-load/generation pipeline every other
   run already uses.

### The one case the joint planner itself cannot handle

Phase 34 found that the joint planner's own adaptive-precision path has
no CPU-only fallback at zero GPU budget (a real, disclosed, deferred
gap -- see `docs/host-memory-guard.md`). For a genuinely CPU-only run
requesting `--kv adaptive`, `--ctx auto` uses a small, separate
resolution path that composes the exact same underlying pure functions
(`membrane_adaptive_kv_resolve()` with the real CPU-only argument this
file's own `resolve_cpu_adaptive_kv()` already uses, plus Phase 34's
host-memory guard) in a bounded loop over the same candidate contexts
-- never a new ranking decision, never a second planner. Every other
combination (any GPU participation, or CPU-only with an explicit
`--kv`) goes through the unchanged joint planner directly.

## Prompt/generation minimum

`minimum_required_context = prompt_tokens + --gen-tokens + 8` -- the
exact same formula this project's own pre-existing "`--ctx` not given
at all" default already used, reused for consistency rather than
inventing a new margin. No candidate context below this is ever
generated.

## Explicit constraints

`--kv`, `--gpu-layers`, `--kv-placement`, and `--device` all remain
hard constraints under `--ctx auto`, exactly as they already are for
`--auto` and the joint planner in general -- recommendation searches
context size only, never re-decides a dimension you explicitly fixed.

## Human output

```
Context recommendation
  model max: 8192
  minimum required: 37
  hardware fit: 8192
  recommended: 8192
  policy: RECOMMENDATION_POLICY_MAX_ESTIMATED_FIT
  host memory: fits (required 352.3 MiB, reserve 256.0 MiB)
  estimated to fit under the current memory snapshot and MEMBRANE's
  safety reserve -- not a guaranteed-no-OOM promise
```

printed once, right before the existing `MEMBRANE plan` block, for
both `--ctx auto` normal runs and `--plan-only` runs. `--verbose` adds
a per-candidate breakdown (context, feasible/rejected, reason code,
selected GPU layers/precision, host requirement) -- normal output
stays concise by default.

If the applied plan ends up differing from the recommendation (only
possible if real memory conditions changed between the recommendation
snapshot and the actual, later apply step -- the existing Phase 21
fallback mechanism resolved a different, still-legal plan), an
explicit `NOTE:` line says so. **Context itself never changes once
recommended** -- fallback may vary GPU layers/precision/placement, but
never context (Section 17 of the Phase 35 task's own hard requirement).

## JSON

Additive only -- `schema_version` stays `1`. A successful `--ctx auto`
run's JSON gains one new top-level object:

```json
"context_recommendation": {
  "requested": "auto",
  "model_max_context": 8192,
  "minimum_required_context": 37,
  "hardware_fit_context": 8192,
  "recommended_context": 8192,
  "policy": "RECOMMENDATION_POLICY_MAX_ESTIMATED_FIT",
  "host_memory_checked": true,
  "host_memory_fit": true,
  "host_required_bytes": 369370368,
  "host_available_bytes": 983916544,
  "host_reserve_bytes": 268435456
}
```

Top-level `ctx_size` always equals `recommended_context`. A plain
`--ctx N` run's JSON is byte-for-byte unaffected -- this object is
never present for that case (confirmed directly: the field is simply
absent, not present-and-empty). `--plan-only --json`'s own, distinct
`mode:"plan"` shape gains the same object; the two schemas' intentional
distinction (Phase 13.2) is preserved.

## Failure modes

Every `--ctx auto` failure is reported with `ok:false`, a stable
`reason_code` (the underlying `membrane_ctxrec_result_t` status, or a
Phase-35-specific code below), a human `message`, and -- where a real,
legal fix exists -- a bounded list of `suggestions`:

| reason_code | Meaning | Typical suggestions |
|---|---|---|
| `MODEL_MAX_CONTEXT_UNKNOWN` / `INVALID_MODEL_MAX_CONTEXT` | The model's GGUF metadata doesn't expose (or exposes an invalid) context ceiling. | Use an explicit `--ctx N` (an intentional trade-off, not risk-free). |
| `MINIMUM_EXCEEDS_MODEL_MAX` | The prompt + `--gen-tokens` needs more context than the model supports at all. | Shorten the prompt; reduce `--gen-tokens`; choose a model with a larger context window. |
| `NO_FEASIBLE_CONTEXT` / `PLANNER_REJECTED_ALL` | No candidate context both fits and is compatible (may be a real host/GPU memory shortfall). | Reduce `--gen-tokens`; try `--kv q5`; try `--kv-placement cpu`; close memory-heavy applications. |
| `CTX_AUTO_NEEDS_PROMPT` | `--ctx auto` given with no prompt at all (CLI-parse-time error). | Provide `--prompt`/`--prompt-file`/`--prompt -`. |
| `CTX_AUTO_TOKENIZE_FAILED` | The cheap vocab-only tokenization pass itself failed. | Verify the model file is a valid GGUF. |
| `CTX_AUTO_HOST_MEMORY_STALE_AT_APPLY` | Host memory that fit at recommendation time no longer fits, re-checked immediately before the real (expensive) model load. | Try again; use a smaller explicit `--ctx N`. |

Suggestions are never offered for a dimension you already fixed
explicitly (e.g. `--kv q5` is never suggested if `--kv q5` was already
the failing request) -- see `context_auto_cli.c`'s own tests. Exactly
one JSON object is ever printed on stdout; no prose is ever mixed in.

## Safety wording

MEMBRANE never claims a guaranteed fit. `--ctx auto`'s recommendation
is **an estimated fit under the current memory snapshot and MEMBRANE's
safety reserve** -- not a promise. Allocator fragmentation, other
processes, and driver-side allocations can all still vary independently
of anything estimated here (see `docs/context-recommendation.md` and
`docs/host-memory-guard.md` for the full, itemized residual uncertainty
this recommendation inherits).

## Known limitations

- `--ctx auto` is not supported with `--compare-kv`/`--gpu-bench` this
  phase (explicit CLI error) -- out of scope, not analyzed end-to-end.
- `--inspect-model` accepts `--ctx auto` without error but does not act
  on it (it stays a read-only, no-prompt-required inspection) -- a
  small note points users to `--ctx auto` with a real prompt instead.
- The host-memory re-check immediately before model load re-validates
  the SAME already-selected context's own requirement -- it is never a
  second context recommendation; if it fails, the run fails clearly
  rather than silently re-planning.
- No real, local GGUF fixture combines a below-4096 `model_max_context`
  with valid, readable hparams -- `stories15M.gguf` (the only local
  fixture with a below-floor ceiling) also lacks readable hparams
  entirely, so it exercises the metadata-unavailable failure path
  (`INVALID_INPUT`) rather than the pure core's own `NO_FEASIBLE_
  CONTEXT` below-floor path end-to-end through the real CLI. That exact
  path IS covered by Phase 33's own unit tests
  (`test_context_recommender.c`, using synthetic facts) -- disclosed
  here rather than silently claimed as covered by a real fixture it
  is not.
- Similarly, `MINIMUM_EXCEEDS_MODEL_MAX` and `HOST_MEMORY_UNKNOWN`
  were not independently exercised through the real CLI this phase
  (doing so would require either a deliberately-oversized real prompt
  or deliberately breaking `/proc/meminfo` access) -- both paths are
  covered by the underlying Phase 33/34 core's own unit tests, which
  this phase's new orchestration code calls unchanged.
