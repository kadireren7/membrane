# Compatibility

What combinations of model architecture, backend, KV precision, and KV
placement does MEMBRANE actually support, intentionally reject, or simply
not have direct evidence for yet? The machine-readable answer is
[`docs/compatibility.json`](compatibility.json) (validated by
`scripts/verify-compatibility.py`); this page is its summary.

This is a compatibility classification, not a benchmark report. It answers
*works / intentionally rejected / not yet validated*, not *how fast* or
*how much memory*. For measured performance/memory numbers see README's
"Measured results" and the evidence files each row cites.

## What the statuses mean

- **SUPPORTED** — direct product-path evidence exists: a unit/integration
  test, a real-model smoke run, or evidence already committed to the repo.
  Never inferred just because code compiles, llama.cpp implements the
  architecture, or a similar model works.
- **UNSUPPORTED** — MEMBRANE intentionally rejects or has not implemented
  this combination. Every UNSUPPORTED row states a concrete `reason_code`
  and cites the code path or test that enforces the rejection.
- **NOT_YET_VALIDATED** — plausible, but there is insufficient direct
  MEMBRANE evidence. This is a distinct, first-class state: "we haven't
  checked" is not the same claim as "it's rejected," and neither is the
  same claim as "it's proven to work."

## Validated model families

| Family | Architecture (`general.architecture` / llama.cpp enum) | Rows |
|---|---|---|
| SmolLM2-135M-Instruct, SmolLM2-360M-Instruct | `llama` / `LLM_ARCH_LLAMA` | MC-01..MC-12 |
| Qwen2.5-1.5B-Instruct | `qwen2` / `LLM_ARCH_QWEN2` | MC-13..MC-19 |
| Everything else llama.cpp implements (gemma3, mistral, phi3, gemma2, ...) | not exercised by any MEMBRANE evidence | MC-26 (`NOT_YET_VALIDATED`) |

Architecture strings were confirmed against
`third_party/llama.cpp/src/llama-arch.cpp`'s `LLM_ARCH_*` name table, not
inferred from a model's filename.

## Precision compatibility

`native` KV has no MEMBRANE-side architecture check at all — it's
unmodified llama.cpp behavior (`compat_check.c` runs zero checks for it,
confirmed by `test_native_always_ok`). `q8`/`q5`/`adaptive` all go through
`membrane_check_kv_compat()` in `tools/membrane-run/compat_check.c`, which
accepts **only** an architecture string on an explicit allowlist
(`MEMBRANE_COMPRESSED_KV_ARCH_ALLOWLIST` — currently `llama` and, since
Phase 26, `qwen2`; an exact string match against each entry, never a
family heuristic) plus a per-head-dimension block-alignment check.
`adaptive` is not a separate gate: it resolves to a concrete candidate
(`q8` or `q5`) first, then that candidate goes through the same check —
so an unsupported architecture fails `adaptive` closed with the identical
`KV_COMPAT_UNSUPPORTED` reason, never a silent fallback to `native`.

Qwen2.5 was the concrete counter-example this repo used to have direct
evidence *against* through the Phase 25 commit: validated at `native`
precision (MC-13..MC-16) but explicitly rejected for `q8`/`q5`/`adaptive`.
Phase 26 closed that gap with real evidence (source-level architectural
equivalence between `LLM_ARCH_LLAMA` and `LLM_ARCH_QWEN2` in the pinned
llama.cpp's own KV-cache construction code, plus a real CPU+Vulkan
q8/q5/adaptive experiment against this exact model) — see
[`docs/compat-expansion.md`](compat-expansion.md) for the full narrative.
Qwen2.5-1.5B-Instruct is now `SUPPORTED` on **every** row in this matrix
(MC-13..MC-19): the same real model, one compatibility answer across both
axes, not two anymore. This is a claim about *this one architecture
string and this one tested model*, not a generalization to every
Qwen2-family variant or every architecture llama.cpp implements — see
"Not yet validated" below.

## Placement compatibility

`--kv-placement` (`default`/`gpu`/`cpu`/`auto`) is validated independently
of `--kv` precision — MC-09/MC-10/MC-11 hold precision at `q8`/`q5` while
varying placement, and MC-01..MC-04 vs MC-13..MC-16 hold placement at
`default` while varying precision and architecture. A model being
`SUPPORTED` for `native` + `cpu` placement says nothing about its
`q8` compatibility, and vice versa — the two axes are never collapsed into
one model-level label.

`gpu`/`auto` placement requires `--gpu-layers all|auto|N`; requesting it
without that fails at CLI-parse time, before any model load (MC-20). On a
build with no GPU backend compiled in at all, `--gpu-layers all|auto`
itself fails closed before model load (MC-21) — `--kv-placement cpu` and
`default` remain valid on such a build regardless.

## Backend compatibility

- **CPU** — supported as a backend; individual combinations remain
  subject to the same CLI, device, placement, and architecture/precision
  gates as any other backend (e.g. MC-20's CLI-level placement rejection
  applies regardless of backend, and MC-21 is specifically a CPU-only
  build rejecting a GPU placement request).
- **Vulkan** — a product GPU backend. Precision/placement rows tested so
  far were run on one real device (a GTX 1650) — see "Hardware scope"
  below.
- **CUDA** — a product GPU backend as of Mega Phase D, PR D3
  (`-DGGML_CUDA=ON`, `docs/cuda-backend.md`). Real, tested against one
  device (a real GTX 1650, same device used for the Vulkan rows above)
  — native and adaptive/q8 KV precision confirmed (MC-23, MC-27, MC-28);
  q5 not yet separately exercised (MC-29). No official CUDA-enabled
  `.deb` package exists yet — source build only, see
  `docs/cuda-backend.md`'s own "Packaging" section for why.

## Hardware scope

Each row in `docs/compatibility.json` carries a `hardware_scope`:

- `tested` — validated on one specific, named device (currently: a real
  GTX 1650 for every row in this matrix with `hardware_scope: "tested"`,
  including MC-17/MC-18/MC-19 since Phase 26's real Vulkan experiment —
  `results/compat-expansion/validation.json`'s CE-05/CE-06/CE-08 rows).
  This is a claim about that exact device, not "all Vulkan GPUs" or "all
  NVIDIA GPUs."
- `backend-level` — the claim is about the backend/build configuration
  itself (e.g. "no GPU backend compiled in" fails closed), not tied to
  specific hardware.
- `not-hardware-specific` — CPU-only rows, and CLI-level checks that never
  touch a device at all.

## Known intentional rejections

- Compressed KV (`q8`/`q5`/`adaptive`) on any architecture not on
  `compat_check.c`'s explicit allowlist (currently `llama`, `qwen2`) — no
  row currently in this matrix demonstrates this directly against a real
  third architecture, but `test_compat_check.c`'s
  `test_q8_gemma3_still_rejected` and `test_joint_planner.c`'s
  `test_incompatible_architecture_adaptive_fails_closed` both prove it
  against a real, still-unsupported architecture string (`gemma3`).
- `--kv-placement gpu`/`auto` without `--gpu-layers` (MC-20).
- `--gpu-layers all|auto` on a build with no GPU backend compiled in
  (MC-21).
- `--kv-placement` (non-default) together with `--compare-kv`/`--gpu-bench`
  (MC-22).
- Runtime KV migration/relocation/retirement — KV placement is decided
  once, before context construction; this exists only as research-only,
  simulation-based prior-phase evidence, never as a `membrane-run` product
  path (MC-24).
- Per-layer or per-block mixed `q8`/`q5` precision — adaptive selection is
  always whole-cache (MC-25).

## Not yet validated

Native-precision inference on any llama.cpp-supported architecture other
than `llama`/`qwen2` (MC-26). `compat_check.c` places no restriction on
`native` mode for any architecture string — but that permissiveness is a
fact about the *gate*, not a MEMBRANE support claim: no MEMBRANE evidence
file or smoke has ever exercised any architecture besides the two above.

## Compatibility != performance

This matrix intentionally excludes tok/s, latency, and VRAM-percentage
numbers from its primary rows — those live in the cited evidence files and
in README's "Measured results." A capacity/OOM boundary at a specific,
extreme context (e.g. SmolLM2-360M `q8` on Vulkan failing at `ctx=180000`
on a 4GB-class GPU while `q5` keeps succeeding through `ctx=220000`, see
MC-06/MC-07's notes) is a measured *capacity* limit at one tested
device, not a compatibility rejection — it is recorded as a note on an
otherwise-`SUPPORTED` row, not as its own `UNSUPPORTED` row.

## How compatibility is verified

`python3 scripts/verify-results.py` runs `scripts/verify-compatibility.py`
as an unnumbered final step, so every claim in this file is checkable by
the same one command CONTRIBUTING.md points to for every other `docs/`
claim — the two scripts stay separate files with separate concerns
(research-evidence integrity vs. product-compatibility-claim integrity),
neither one's checks counted into the other's total.

- `scripts/verify-compatibility.py` — schema/required-field validation,
  unique row IDs, valid status/hardware_scope enums, every evidence path
  resolves to a real file, `SUPPORTED` rows carry evidence,
  `UNSUPPORTED` rows carry a concrete reason, `NOT_YET_VALIDATED` rows
  never cite a `results/` evidence file (that would read as direct
  validation proof), plus two source-invariant checks tying the matrix
  back to the actual `compat_check.c` gate and the actual absence of a
  CUDA build option.
- `tools/membrane-run/test_compat_check.c`,
  `tools/membrane-run/test_product_cli.cpp` — llama-free unit tests
  exercising the compatibility gate and CLI-level rejections directly,
  cited as evidence throughout the matrix above.
- `results/phase18-compatibility-smoke.json` — a handful of small, real
  `--plan-only --json`/CLI smokes against local GGUF fixtures already
  committed to `models/`, run to fill specific gaps this phase's audit
  found in existing evidence (not a benchmark sweep).
- `results/compat-expansion/validation.json` (Phase 26) — 8 real,
  live `membrane-run` invocations validating Qwen2 compressed KV
  (native/q8/q5/adaptive, CPU and Vulkan, default and `cpu` placement),
  checked by `scripts/verify-compat-expansion.py`. See
  [`docs/compat-expansion.md`](compat-expansion.md) for the full
  investigation this evidence backs.
