# Planner v2 joint variant/quant optimizer (Milestone G2)

Status: read-only joint planning correctness only. **No performance
claim of any kind is made or implied by this document or by
`membrane plan`'s own output.** Latency/throughput evidence, and any
performance-based selection policy, is Milestone G3's job — see
"Explicitly deferred to G3" below.

This builds directly on [`docs/planner-v2-foundation.md`](planner-v2-foundation.md)
(Milestone G1). Read that first — this document only covers what G2
*adds*: the model VARIANT/QUANT dimension, reasoned about jointly with
context × GPU-layers × KV-precision × KV-placement, instead of picked
independently beforehand.

## What G1 left out (and G2 closes)

G1's own "Explicitly out of scope" section named this gap directly:
`variant_selector.cpp` chose a quant *before* handing one fixed variant
to the context/GPU/KV joint pipeline (`context_recommender.h` +
`joint_planner.h`), using catalog `size_bytes` as a coarse proxy, never
reconciled with the joint planner's own real memory math. A user could
not ask "which quant, at which context, with how many GPU layers, is
actually feasible" as one question — only "which quant looks like it
fits" and, separately and later, "given that one fixed model, what's
the context/GPU/KV plan."

G2's architectural fix: **variant selection no longer happens before
the joint planner.** Every feasible candidate variant is run through
the *same* real `context_recommender.h`/`joint_planner.h` pipeline G1
already built, and the selected variant emerges from those real
results — not from a separate, earlier, coarser check.

## The new orchestration layer

`tools/membrane-run/plan_v2_resolver.h`/`.c` — the one function this
milestone was missing:

```c
int membrane_plan_v2_resolve(const membrane_plan_v2_request_t *req,
        membrane_plan_v2_result_t *out);
```

Pure C, llama-free, no GGUF/device/I/O access — same testable-without-
a-model/GPU pattern as every `*_policy.h`/`joint_planner.h`/
`context_recommender.h` module in this project. It does **not**
reimplement GPU-layer, KV-precision, KV-placement, or memory-fit
arithmetic; for each candidate variant it is given real (or scaled-
estimate) hparams for, it calls the exact same, unchanged
`membrane_ctxrec_resolve()` + `membrane_plan_assemble()` G1 already
uses for one model. Its only new logic is running that pipeline once
per candidate variant and selecting among the results by one small,
fixed, documented policy (below).

The C++ orchestration layer (`tools/membrane/plan_cmd.cpp`) builds the
per-variant candidate list from real registry/catalog data and renders
the result — schema-version-2 JSON or human text.

## The full candidate space

For a catalog family with multiple real variants:

```
Q8_0
  -> ctx candidates (membrane_ctxrec_generate_candidates())
     -> GPU layers x KV precision x placement (membrane_joint_plan_resolve(),
        called once per ctx candidate)
Q5_K_M
  -> ... same pipeline ...
Q4_K_M
  -> ... same pipeline ...
```

Quant names are never hardcoded — every variant comes from the real,
built-in catalog (`model_catalog.cpp`) or, for an installed model,
`best_effort`-matched against it.

### Real hparams vs. disclosed estimate

Quantized GGUF variants of the *same* model family share identical
layer count / embedding / head-count hparams — quantization changes
per-tensor byte width, never tensor shapes/counts or the training
context-length ceiling. So:

- **A real installed sibling exists** (the registered file's basename
  exactly matches one catalog variant's real filename, resolved via
  `find_installed_catalog_variant()`): every *other* variant of that
  same family is evaluated through the real joint pipeline too, using
  the installed sibling's real GGUF hparams (`n_layer`/`arch`/`n_embd`/
  `n_head`/`n_head_kv`/`model_max_context`, all copied verbatim) plus
  that *other* variant's own real, catalog-recorded `size_bytes` to
  rescale only the byte-count fields (`bytes_per_layer`/
  `output_role_bytes`/`total_weight_bytes`) by the real ratio of two
  verified download sizes (`plan_cmd.cpp`'s `scale_hparams()`). Marked
  `estimate_only: true`, `source: catalog_metadata` — **never** claimed
  exact. Different quant methods do not necessarily scale every tensor
  uniformly (K-quants may keep some tensors at higher precision than
  others), so this is a real, useful, but disclosed approximation.
- **No real installed sibling exists at all** for that family: there is
  no real hparams basis for *any* variant. Every variant falls back to
  the same coarse, catalog-size-only host-memory check
  `variant_selector.h`'s own `membrane_select_variant()` already
  performs (reused, never re-derived) — no context/GPU-layer/
  KV-precision decision is made at all, disclosed via the automatic
  `MODEL_METADATA_UNAVAILABLE` reason `membrane_plan_assemble()` already
  emits for a `NULL` ctxrec result (unchanged from G1).

Nothing here invents tensor-level exactness for a file that was never
read.

## Hard constraints vs. policy preferences

**Hard constraints** (all reused, unchanged, from the existing pure
modules — never re-implemented in the resolver):

- host RAM safety (`host_memory_guard.h`, inside `context_recommender.c`
  for the real path; `membrane_select_variant()`'s own call for the
  coarse path)
- VRAM/device capacity (`gpu_policy.h`, inside `joint_planner.c`)
- model/KV-mode compatibility (`compat_check.h`, inside `joint_planner.c`)
- minimum valid context (`context_recommender.h`'s own
  `MEMBRANE_CTXREC_MIN_CANDIDATE` floor)
- explicit user overrides (`--quant`/`--ctx`/`--kv`/`--gpu-layers`) — see
  below

**Policy preferences** (the *only* new decision logic G2 adds, in
`membrane_plan_v2_resolve()`):

1. A candidate with an **actual evaluated decision**
   (`decisions.has_decisions == 1`, i.e. real/scaled hparams existed and
   the joint pipeline found it feasible) is always preferred over one
   with only a **coarse estimate and no real decision**
   (`decisions.has_decisions == 0`) — a plan this system can actually
   stand behind beats a merely-estimated-to-fit one.
2. Within the same tier, prefer the higher-quality variant — the first
   feasible one in the caller-supplied quality order. Quality order is
   the real, verified catalog `size_bytes`, **descending** — never an
   invented or weighted score. A larger quantized file uses more bits
   per weight in every quant naming scheme this catalog uses (F16 >
   Q8_0 > Q5_K_M > Q4_K_M), so this is a real physical proxy, not a
   guess.
3. Within one variant, context/GPU-layers/KV-precision/KV-placement are
   entirely `context_recommender.c`/`joint_planner.c`'s own existing,
   unchanged policy (largest feasible context, then `joint-auto-v1`'s
   own GPU/KV/placement order) — never re-decided here.
4. Deterministic tie-break: input order (already the quality order,
   itself already deterministic) — no random/unstable criterion is ever
   consulted.

No ML/AI scoring, no opaque weighted formula.

## Explicit user override semantics

An explicit dimension is a **hard constraint**, never silently relaxed:

| Flag | Effect |
| --- | --- |
| `--quant Q8_0` | Only `Q8_0` is evaluated at all (the C++ layer filters `variants[]` down to that one entry before calling the resolver — the resolver itself has no override logic of its own). If infeasible: `"feasible": false`, never a silent fallback to `Q5_K_M`/`Q4_K_M`. |
| `--ctx N` | Every candidate variant's ctxrec request becomes a single-candidate request for exactly `N` — never reduced. The variant dimension may still adapt (a different quant may be the one that makes `N` feasible). |
| `--gpu-layers N` / `--kv MODE` | Forwarded as a hard constraint to *every* variant's own request unchanged. |
| `--quant X --ctx N` | Both fixed at once. If no feasible plan exists for that exact pair: `"feasible": false`, with the real limiting reason — never silently relaxed. |

Automatic (non-explicit) dimensions may adapt freely around an explicit
one — e.g. explicit quant, auto context: quant is fixed, context may be
whatever the joint pipeline finds feasible for it.

## Deterministic selection policy — summary

`plan-v2-variant-joint-v1` (`MEMBRANE_PLAN_V2_POLICY_VERSION`, echoed in
every JSON response as `policy_version`):

1. Hard constraints (unchanged reused modules + explicit overrides).
2. An evaluated decision beats a coarse estimate.
3. Higher quality (real catalog size, descending) wins ties within a
   tier.
4. Context/GPU/KV/placement: the existing joint pipeline's own order.
5. Deterministic input-order tie-break.

Never described as "optimal", "best", or "fastest" — only "selected",
"preferred by policy", "recommended", or "feasible". No timing/
throughput evidence exists yet.

## Objective modes: deferred, not implemented

The task considered `balanced`/`context`/`quality` selectable modes.
**G2 implements exactly one fixed policy and does not expose a
`--objective` flag.** Working through the natural definitions:

- `balanced` (current default product policy, extended to the real
  joint pipeline): prefer the higher-quality feasible variant, then the
  largest feasible context within it.
- `quality`: prefer the higher-fidelity variant, then maximize feasible
  context within it.

These two are **the same policy** given the evidence currently
available — there is no product evidence today that trades quality
against context in a different direction for one mode than the other
(that would require throughput/latency/quality-metric evidence, which
is explicitly G3's job, not G2's). Introducing three selectable modes
now would let a user pick a mode whose real consequence this system
cannot yet honestly quantify. So G2 keeps the one policy above and
defers real, evidence-backed objective modes (including a genuine
`context`-maximizing mode, and any `speed` mode) to G3, once real
performance data exists to make the modes mean something concrete.

## Bounded search

- `MEMBRANE_PLAN_V2_MAX_VARIANTS` = 8 — a real catalog family in this
  project never lists more than 4 quants today (`smollm2-135m-instruct`).
  8 is a generous, still-bounded ceiling.
- Each variant with real/scaled hparams runs the existing
  `MEMBRANE_CTXREC_MAX_CANDIDATES` (20) context ladder, each internally
  bounded to `MEMBRANE_JOINT_MAX_CANDIDATES` (8) joint-planner
  candidates.
- Worst case: 8 variants × 20 ctx candidates × ≤8 internal joint
  candidates — a few hundred pure-arithmetic evaluations, no I/O, no
  model load. `test_plan_v2_resolver.c`'s own search-bound test (test J)
  exercises an over-sized fixture and asserts `candidate_count` is
  clamped, never overflowed.
- Alternatives shown are a bounded shortlist
  (`MEMBRANE_PLAN_V2_MAX_ALTERNATIVES` = 5), never every internal
  candidate dumped verbatim.

## Infeasibility across variants

When nothing is feasible, every variant's own real reason is surfaced
(not just one aggregate message):

```
Feasible: no

Why
  - Q8_0: infeasible (HOST_MEMORY_LIMIT)
  - Q5_K_M: infeasible (HOST_MEMORY_LIMIT)
  - Q4_K_M: infeasible (CONTEXT_TOO_LARGE)
```

using the real, existing `MEMBRANE_PLAN_REASON_*` vocabulary
(`membrane_plan.h`) — never a new, parallel reason taxonomy. The
detailed, subsystem-specific code (e.g. `host_memory_guard.h`'s own
`HOST_MEMORY_INSUFFICIENT`) is still available verbatim in each
candidate's own `reasons[]` array.

## JSON schema — version 2

G1 shipped `schema_version: 1`. G2's output is materially different
(a structured multi-candidate shape, not one plan) — bumped to
`schema_version: 2`. The top-level `identity`/`workload`/`hardware`/
`decisions`/`feasibility`/`reasons`/`explanation` fields are still
present and now mirror the **selected** (or, if none is feasible, the
first evaluated) candidate — existing consumers reading only those
top-level fields keep working. New fields:

```json
{
  "schema_version": 2,
  "policy_version": "plan-v2-variant-joint-v1",
  "variants_evaluated": [
    { "variant": "Q8_0", "source": "catalog_metadata", "estimate_only": true,
      "evaluated": true, "feasible": true, "limiting_resource": "",
      "selected": true, "plan": { /* the same schema-1 shape, for this candidate */ } },
    { "variant": "Q5_K_M", "...": "..." }
  ],
  "selected_variant": "Q8_0",
  "alternatives": ["Q5_K_M", "Q4_K_M"],
  "feasible": true
}
```

## `membrane plan` — no product integration change

`membrane use`/`membrane serve` are **not** touched by this milestone.
`membrane plan` remains strictly read-only (same guarantee as G1,
re-verified: `test_plan_cmd.cpp`'s registry-hash-unchanged tests still
pass, and Milestone G2 adds no new mutation path). Execution
integration is explicitly deferred — see Part 15 of the G2 task and
"Explicitly deferred to G3" below.

## Real smoke evidence

`results/planner-v2-joint-variant/validation.json` — real `membrane
plan` runs against the repository's own real `SmolLM2-135M-Instruct`
F16 fixture (registered under the catalog's own exact filename so the
real cross-variant join activates), a never-installed catalog family
(coarse-only fallback), an explicit `--quant` override, an explicit
impossible `--ctx`, and the read-only-guarantee check (registry hash
unchanged across all of the above). Includes a real, live example of
host-memory pressure on this dev host changing which variant the joint
pipeline selects between two runs minutes apart — not a fabricated or
smoothed-over result.

## Explicitly deferred to G3

- Any latency/throughput/quality measurement or claim.
- Any performance-based selection policy or `--objective` flag.
- Connecting `membrane use`/`membrane serve` to Planner v2 by default.
- Ollama/vLLM adapters, external runtime discovery, active runtime
  monitoring, dynamic memory/KV migration, CXL, FPGA, remote nodes,
  fleet management, a cloud control plane, benchmark comparison, a new
  inference backend, pricing/business features, or an "AI optimizer" —
  none of this is implemented, started, or implied here (same list G1
  already excluded, carried forward unchanged).

## Confirmed

- `v1.0.0` tag/release untouched.
- No Ollama/vLLM/external-runtime-abstraction work started.
- `membrane use`/`membrane serve` unmodified.
