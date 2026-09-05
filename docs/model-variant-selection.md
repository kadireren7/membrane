# Hardware-aware variant selection

Mega Phase D, PR D2. `membrane model install NAME` (no `--quant`) picks
a sensible GGUF variant automatically instead of requiring the user to
already know what `Q4_K_M` means.

## Inputs

- Real available host RAM (`membrane_read_host_meminfo()`,
  `runtime_session.h` — the same real `/proc/meminfo` reader every
  other host-memory-aware feature in this project already uses, never
  a second implementation).
- The catalog's own real, verified variant sizes (`model_catalog.h`).

No VRAM/GPU-layer planning happens here — that is the existing joint
planner's own, separate, already-solved job (`docs/joint-planner.md`),
which runs once the model is actually loaded. This module only answers
"is downloading this file at all a reasonable idea," using the whole
file's real size as a proxy for its host-resident footprint.

## The policy (deterministic, Section 10 of the task)

1. Evaluate every variant via `host_memory_guard.h`'s own, already-
   documented `membrane_host_memory_guard_resolve()` — the exact same
   256 MiB fixed + 10% reserve policy every other real-memory-fit
   decision in this project already trusts (`docs/host-memory-guard.md`),
   treating the variant's catalog `size_bytes` as the host-resident
   weight-byte estimate.
2. Among variants that fit, return the **largest** (highest precision)
   — never "smallest that fits," which would leave real available
   headroom on the table for no reason.
3. If none fit, refuse (`NO_FEASIBLE_VARIANT`) and report every
   variant's own real reason — never silently picks an oversized one,
   never downloads something guaranteed to fail.

Identical inputs always produce an identical result — no randomness,
no time-of-day dependence beyond the real, current host memory state
at the moment `install` runs.

## User overrides

`--quant Q4_K_M` (or `--variant`, an alias) is always honored — an
explicit choice is hard, never second-guessed. If the real fit
estimate says that exact variant won't fit, `install` **warns** (never
blocks) and proceeds anyway, since the user asked for it explicitly
(Section 12 of the task).

## `--dry-run`

Reports the real selection decision (or every real reason nothing
fits) without downloading anything — the "before downloading a 10+ GB
model, estimate if it can reasonably fit" check from Section 11 of the
task, made available as a real, standalone step.

## Real, disclosed limitation

The catalog's recorded `size_bytes` is the real, verified **download**
size — a reasonable proxy for host-resident footprint at a modest
context size, but not a byte-exact runtime estimate the way the
existing joint planner is once a model is actually loaded and its real
GGUF metadata (layer count, hidden dims, KV shape) is known. For these
small/mid catalog models this proxy is conservative enough to be
useful; it has not been validated against very large (10+ GB) models
where KV-cache-at-a-large-context could meaningfully change the real
picture.

## Real evidence

See `results/model-variant-selection/validation.json` — real dry-run
selections against this dev host's own actual, fluctuating available
memory: `SmolLM2-135M-Instruct` (all 4 variants fit, `F16` selected),
`SmolLM2-360M-Instruct` (`F16` correctly rejected as too large, `Q4_K_M`
selected), and `Qwen2.5-1.5B-Instruct` (correctly refused outright when
real available memory was too tight for any variant, then an explicit
`--quant` override correctly warned-but-proceeded).
