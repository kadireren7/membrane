# MEMBRANE v0.2.0-rc1 — release notes

Product Phase 8: productization. Turns the Phase 5-7 research runtime
into a single, clean, user-facing entry point (`membrane-run`) with a
real, committed, machine-verified memory result.

## What works

- `membrane-run --model MODEL.gguf --prompt "..." [--ctx N] [--kv native|q8]`
  — one normal decode pass (model load, one context, generate, print,
  exit). No hidden native-reference or teacher-forced comparison pass,
  no per-step logit retention, in normal mode.
- `--kv q8`: the KV cache tensor is allocated as genuine Q8_0, not a
  shadow copy — real, measured process-RSS reduction (see "Verified
  result" below). Default is `native` — nothing about model behavior
  changes unless you opt in.
- Compatibility is checked *before* context creation (architecture,
  head-dimension/Q8_0-block-size divisibility, context size) and fails
  with a clear message, never a silent fallback to native.
- `--compare-kv`: explicit benchmark mode, reproduces the Phase 7
  native-vs-q8 memory/quality/performance comparison. Clearly separate
  from normal mode — never implied by `--kv q8` alone.
- `--json`, `--include-text`, `--quiet`, `--verbose`, `--threads`,
  `--prompt`/`--prompt-file`/`--prompt -` (stdin), streaming text
  output in normal human mode, stable exit codes (0/2/3/4/5).
- `cmake --install` produces a working, standalone `membrane-run`
  binary (shared-library RPATH handled — this was a real gap found and
  fixed during this phase, not assumed to work).

## Quick start

```bash
git clone --recursive https://github.com/kadireren7/membrane
cd membrane
cmake -S . -B build-llama -DMEMBRANE_ENABLE_LLAMA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-llama -j --target membrane-run

./build-llama/tools/membrane-run/membrane-run \
  --model model.gguf --prompt "Hello" --ctx 4096 --kv q8
```

No `third_party/llama.cpp` patch is required for this. See
`docs/live-runtime.md` for the mechanism.

## Verified result

`results/v0.2/smollm2-q8-memory.json` (committed; `scripts/verify-results.py`
checks it — schema, git SHA format, context sequence, q8-bytes-less-
than-native, ratio arithmetic, positive and monotonically-increasing
RSS reduction with context size, no absolute paths/prompt text/NaN,
valid quality ranges). SmolLM2-135M, one prompt, greedy decoding, CPU,
single host:

| context | native KV | q8 KV | native RSS after ctx | q8 RSS after ctx | reduction |
|---|---|---|---|---|---|
| 512 | 11.25 MiB | 5.98 MiB | 301,252 kB | 296,252 kB | 1.66% |
| 1024 | 22.50 MiB | 11.95 MiB | 312,908 kB | 302,472 kB | 3.34% |
| 2048 | 45.00 MiB | 23.91 MiB | 335,964 kB | 314,644 kB | 6.35% |
| 4096 | 90.00 MiB | 47.81 MiB | 382,028 kB | 339,172 kB | 11.22% |
| 8192 | 180.00 MiB | 95.62 MiB | 474,480 kB | 388,268 kB | 18.17% |

Quality at every context: token identity preserved, top1 preservation
100%, logit rel-L2 ≈ 0.0087. Reproduce with
`scripts/benchmark-v0.2.sh MODEL.gguf`.

## Known limitations

- One model (SmolLM2-135M), one prompt, CPU-only, single host — not a
  general claim about other models, prompts, or hardware.
- Only `LLM_ARCH_LLAMA` models are supported for `--kv q8`; other
  architectures are rejected at the pre-flight compatibility check
  (verified end-to-end against a real local model with an incompatible
  head dimension, not just synthetic test inputs).
- No adaptive KV-store mode in this release — every real run measured
  so far selects 100% Q8 anyway (established in Phase 4-7), so adaptive
  would be identical to `q8` here.
- No speedup claim: generation throughput was measured in both
  directions (q8 faster in a short run, essentially tied in a longer
  run) across local testing — reported as measured, not asserted.
- `membrane-llama-run` (Phase 5/6 shadow/injection diagnostics) remains
  available as the research/diagnostic tool; `membrane-run` is the
  product entry point.

## Model/backend scope

CPU backend only. `membrane-run` itself was verified in this phase
against two local `LLM_ARCH_LLAMA` models: SmolLM2-135M (the full
normal-mode/compare-mode/memory sweep this document's numbers come
from) and `stories15M.gguf` (head dimension 48, used specifically to
exercise the compatibility-rejection path end-to-end, confirming it
fails closed rather than crashing or silently falling back — not a
memory/quality claim for that model). SmolLM2-360M was exercised by
earlier phases' diagnostic tooling (Phase 4-6), not by `membrane-run`
in this phase -- not claimed as verified here.

This is a release-candidate tag preparation document only — no tag has
been created yet.
