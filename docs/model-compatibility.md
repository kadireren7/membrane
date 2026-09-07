Model family compatibility

Mega Phase D, PR D8. This document exists so a reader never has to
infer "does my model work?" from a single fixture's own name. The
authoritative, row-level detail lives in `docs/compatibility.json`
(34 rows as of this release) — this page is a family-level summary of
it, never a superset of what that file actually says.

**One real fixture never implies its whole family.** A model that
happens to share an architecture name with a validated fixture (e.g.
"some other Qwen2 checkpoint") is NOT thereby validated — only the
exact family/size/quant combinations listed below have real evidence.

## Real evidence by family

| Model family | Architecture | Backend(s) validated | Status | Evidence |
|---|---|---|---|---|
| SmolLM2-135M-Instruct | `llama` | CPU, Vulkan | SUPPORTED | `docs/compatibility.json` MC-01..MC-04, MC-05..MC-15 (various KV/placement combinations); `results/v0.3/kv-residency-productization/`, `results/phase18-compatibility-smoke.json` |
| SmolLM2-360M-Instruct | `llama` | Vulkan | SUPPORTED | `docs/compatibility.json` (grouped with SmolLM2-135M's own Vulkan rows) |
| Qwen2.5-1.5B-Instruct | `qwen2` | Vulkan | SUPPORTED | `docs/compatibility.json` MC-17/MC-18/MC-19 (Mega Phase D, PR D6's own compat-expansion work) |
| Any other `llama`-arch model (Mistral, Gemma2/3, Phi-3, TinyLlama, etc.) | `llama` (shared GGUF arch tag) | — | NOT_YET_VALIDATED | Sharing llama.cpp's own `llama` arch tag with a validated fixture is NOT evidence of compatibility — MEMBRANE's own KV/placement/planner code has never been exercised against these specific model families. Expected to work (the planner code is architecture-generic), but not claimed. |
| Any other `qwen2`-arch model | `qwen2` | — | NOT_YET_VALIDATED | Same reasoning — only the exact Qwen2.5-1.5B-Instruct fixture above has real evidence. |
| Any architecture llama.cpp itself does not support | n/a | — | UNSUPPORTED | Bounded entirely by upstream llama.cpp/GGUF support — MEMBRANE adds no model-loading code of its own (`compat_check.c`'s own architecture allowlist, `docs/compatibility.json` MC-20/MC-21/MC-22/MC-24/MC-25). |

## What "SUPPORTED" means here, exactly

A `SUPPORTED` row means: this exact model family, at this exact real
fixture size, was loaded and generated from on the named backend, with
the specific KV precision/placement/context-class combination that row
records — see `docs/compatibility.json`'s own `reason_code`/`notes`
fields for the exact scope of each row. It does NOT mean:

- every quantization of that model family works (only the quants
  actually installed and tested do — see `docs/model-catalog.md` for
  which quants the built-in catalog actually offers per family);
- every size in that family's own lineup works (a 1.5B fixture's real
  evidence says nothing about a hypothetical 72B checkpoint of the same
  architecture, beyond "the same generic planner code path applies");
- performance/quality claims — compatibility rows are pass/fail on
  "does it produce output correctly," not a benchmark.

## Built-in catalog vs. compatibility matrix

The built-in model catalog (`membrane model search`, `docs/model-
catalog.md`) currently offers exactly the three families above —
`smollm2-135m-instruct`, `smollm2-360m-instruct`, `qwen2.5-1.5b-
instruct` — deliberately kept in lockstep with this real compatibility
evidence (Section 8 of the D8 task: "no stale/mirror-only entries").
Every catalog entry's own `compatibility_status`/`compatibility_
evidence` fields point back to the exact `docs/compatibility.json` row
above, so the two documents can never silently drift apart (see
`scripts/verify-model-catalog.py`, which enforces this).

## Backend × family matrix (real evidence, condensed)

| | CPU | Vulkan | CUDA | Metal |
|---|---|---|---|---|
| SmolLM2-135M/360M-Instruct (`llama`) | ✅ real | ✅ real | Backend-level only (MC-23/MC-27/MC-28 use a synthetic/generic model scope, not this specific family — see `docs/support-matrix.md`) | ✅ real (this release's own D8 capstone, see `docs/support-matrix.md`) |
| Qwen2.5-1.5B-Instruct (`qwen2`) | Expected (shares the same generic code path as the CPU rows above) but not separately real-tested this exact family/backend pair | ✅ real | Not real-tested this exact family | Not real-tested this exact family |

See `docs/support-matrix.md` for the platform/backend evidence states
(VALIDATED_REAL/VALIDATED_CI/etc.) this table's own cells reference.
