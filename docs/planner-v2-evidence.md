# Planner v2 real-workload evidence (Milestone G3)

Status: evidence-gathering only. **No code in the planner, the
resolver, or the selection policy changed this milestone.** This
document answers one question with real, disclosed evidence: does
Planner v2 (G1 + G2) produce useful, defensible decisions in real
workloads on a real host? See
[`results/planner-v2-evidence/validation.json`](../results/planner-v2-evidence/validation.json)
for the full machine-readable artifact this document narrates, and
[`docs/planner-v2-joint-variant.md`](planner-v2-joint-variant.md) /
[`docs/planner-v2-foundation.md`](planner-v2-foundation.md) for what
G1/G2 actually built (read those first).

## What was tested

- **`membrane plan`'s real decisions** against `smollm2-135m-instruct`
  (3 real quant files: F16, Q8_0, Q4_K_M -- see "Models used" below)
  and `smollm2-360m-instruct` (1 real quant file: F16), on this
  project's real, shared, 5.6 GiB-RAM dev host, CPU (`build-cpu-rc1`)
  and Vulkan (`build-vulkan`, rebuilt fresh this milestone -- it was
  stale at `membrane 0.8.0` before that).
- **Real `membrane-run` loads and generations** at the exact
  context/GPU-layers/KV-precision dimensions the planner selected, for
  a naive-default baseline, a manual high-quality baseline, the
  planner-selected configuration, and a context-focused alternative
  (Part 3's A/B/C/D baseline framework) -- all on the same underlying
  llama.cpp runtime, isolating the value of *planning*, not comparing
  unrelated products.
- **Sibling byte-ratio scaling error** (G2's `scale_hparams()`),
  measured against two real installed GGUF quant files of the same
  family, not merely estimated.
- **A small, bounded Vulkan/GPU evidence set** (1 family, 3 configs, 2
  reps each -- the Part 10 ceiling).
- **A bounded real infeasibility check**, safety-bounded with
  `ulimit -v` so a genuine OOM could not destabilize the shared host.

## What was NOT tested

- `qwen2.5-1.5b-instruct-fp16.gguf` (3.5 GiB) -- deliberately not
  real-load-tested. Real available RAM on this host during this
  session ranged roughly 200 MiB-1.3 GiB (other real desktop processes
  competing for it); loading a 3.5 GiB model alongside them was judged
  an unacceptable host-destabilization risk for this milestone's
  evidence bar. Disclosed, not silently skipped.
- CUDA. Only Vulkan was exercised (no CUDA toolkit on this host,
  consistent with the still-open Mega Phase D3 item).
- Output *quality* (perplexity/task accuracy) at any quant level.
  "Quality" throughout this document means catalog file-size ordering
  only, exactly as G2 defined it -- this milestone did not
  independently verify that ordering against a quality metric.
- True time-to-first-token. `membrane-run`'s decode path is a single
  shot (load, one context, prompt, generation, exit) with no
  streaming/token-by-token timestamps; adding real TTFT instrumentation
  would be invasive runtime work Part 9 of the task explicitly says to
  skip rather than widen scope for.
- Any multi-host validation. Everything here is one shared dev
  machine, one GPU.
- Any Ollama/vLLM/external-runtime-adapter work. None started.

## Hypotheses and verdicts

Defined in full, before any measurement, in
`results/planner-v2-evidence/validation.json`'s own `hypotheses`
object. Summary:

| # | Question | Verdict |
| --- | --- | --- |
| H1 | Feasible-labeled candidates actually load/run | Supported when plan+run share a memory snapshot: 3/3 TRUE_FEASIBLE |
| H2 | Infeasible-labeled candidates fail or have thin real headroom | Mixed: 1 clean TRUE_REJECT, 2 FALSE_REJECT/UNVERIFIED_REJECT (see below) |
| H3 | A lower-memory quant/KV plan can unlock real larger context | Confirmed on CPU (4096 -> 8192); not observed on GPU for the same family (VRAM was not the binding constraint there) |
| H4 | Context/memory gains must not come with a throughput floor violation | Floor never triggered -- the one real expansion case was *faster*, not slower |
| H5 | Sibling scaling-error must be measured, not assumed | Measured: small for Q8_0 (<1%), up to 12.4%/-26.7% for Q4_K_M's per-layer/output-role fields |
| H6 | Current policy must be checked against measured outcomes | Checked; not changed this milestone (see Policy conclusion) |

## Sibling scaling-error (H5)

G2's `scale_hparams()` predicts an uninstalled sibling's
`bytes_per_layer`/`output_role_bytes`/`total_weight_bytes` by rescaling
a real installed sibling's real values by the ratio of two
catalog-recorded, verified download sizes. Measured against real
`SmolLM2-135M-Instruct` F16, Q8_0, and Q4_K_M GGUF files (Q8_0 and
Q4_K_M were downloaded this milestone from the catalog's own verified
URLs, sha256-checked against the catalog's own recorded hashes --
`scripts/planner-v2-evidence/measure_sibling_scaling_error.py`):

| Base -> target | `total_weight_bytes` error | `bytes_per_layer` error | `output_role_bytes` error |
| --- | --- | --- | --- |
| F16 -> Q8_0 | 1.25% | 0.57% | 0.62% |
| F16 -> Q4_K_M | 1.72% | 12.41% | **-26.72%** |

The large `output_role_bytes` error for Q4_K_M is not noise: K-quants
keep the output/embedding tensors at higher precision than the rest of
the model, so a whole-file size ratio over-predicts how much *that
specific tensor group* shrinks. This is exactly the caveat
`docs/planner-v2-joint-variant.md` already disclosed ("K-quants may
keep some tensors at higher precision than others") -- this milestone
quantifies it instead of only asserting it. **Conclusion: the
approximation is accurate enough to drive feasibility for
`total_weight_bytes` (the field the host-memory guard's coarse check
actually uses), but not accurate enough to treat `output_role_bytes`
as a real per-tensor prediction for a K-quant target.** No formula
change was made this milestone (Part 5 of the task: measure first,
never silently adjust).

## Feasibility accuracy (H1/H2)

Full confusion matrix in the artifact's `feasibility_confusion_matrix`.
Classifications used: `TRUE_FEASIBLE`, `TRUE_REJECT`,
`FALSE_REJECT`, `UNVERIFIED_REJECT` (no `FALSE_FEASIBLE` was observed
-- every feasible-labeled candidate that was runtime-tested actually
succeeded).

- **TRUE_FEASIBLE x3**: `smollm2-135m-instruct` Q8_0@ctx=4096
  (planner-selected), Q4_K_M@ctx=8192 (context alternative), and
  F16@ctx=4096 under a plan+run pair taken within seconds of each
  other (~1.2-1.3 GiB available both times).
- **TRUE_REJECT x1**: F16@ctx=200000 -- deliberately, robustly
  infeasible regardless of ambient host-memory drift. The real bounded
  load (under a 1.5 GiB `ulimit -v` safety cap) genuinely failed:
  `ggml_aligned_malloc: insufficient memory (attempted to allocate
  4398.75 MB)`, clean exit code 4, no host impact.
- **FALSE_REJECT x1**: `smollm2-360m-instruct`'s single installed
  variant @ ctx=2048, plan+run paired within seconds (832-838 MiB
  available both times). The plan said infeasible (`HOST_MEMORY_LIMIT`)
  but the real load+generate succeeded, with real peak RSS (~797 MiB)
  landing inside real available memory (~818-838 MiB) by only ~20-40
  MiB. This is evidence of **deliberate conservatism** in the
  host-memory guard's reserve margin, not of an unsafe or broken
  check -- the config genuinely just barely fit.
- **UNVERIFIED_REJECT x1**: the same family's F16 default-ctx-ladder
  rejection at an original 576 MiB-available snapshot, re-tested
  minutes later at a different (~1019 MiB) snapshot, where it
  succeeded. Not a same-condition test -- host-available memory on
  this shared desktop genuinely changed between the two calls; this
  cannot be resolved into TRUE_REJECT or FALSE_REJECT without
  re-testing under the exact original condition, which a live shared
  host cannot reliably reproduce on demand.

**Reading**: Planner v2's feasibility calls are accurate when host
memory is held roughly constant between the decision and the attempt.
Its inaccuracies in this evidence set all run in the *safe* direction
(rejecting things that would have fit), consistent with a
conservative, not reckless, reserve-margin policy -- appropriate for a
read-only advisor, see the execution-readiness decision below.

## Context-capacity finding (H3)

Real, on CPU, `smollm2-135m-instruct`:

```
Q8_0 (selected, quality-first):
  safe ctx = 4096
  avg generation tok/s = 102.47

Q4_K_M (alternative):
  safe ctx = 8192
  avg generation tok/s = 140.26

Q4_K_M @ 8192:
  real load/run PASS (2/2 reps)
  36.9% FASTER than Q8_0 @ 4096, not slower
```

This is the strong form of evidence Part 7 of the task asked for: a
real load and real generation succeeded at the larger context, not
just a feasibility claim on paper. It did **not** reproduce on the
Vulkan/GPU backend for the same family: VRAM on this host's GTX 1650
(4 GiB) was abundant enough relative to a 135M model that every
variant, including F16, already got the maximum ctx=8192 -- the
CPU-side host-RAM pressure that produces the quality-vs-context
tradeoff is a property of this host's live RAM condition for a
CPU-bound plan, not a universal property of the model family.

## Performance floor (H4)

Threshold, defined **before** any throughput number was computed
(Part 8 of the task):

> A configuration that gains material context/memory capacity (>=1.5x
> a comparable same-backend baseline's feasible context) is classified
> `TECHNICAL_FEASIBILITY_ONLY`, not a product win, if its measured
> generation tok/s drops by more than 85% relative to that baseline.

Measured (CPU, `smollm2-135m-instruct`, 2-rep averages, 24 generated
tokens, 5-token prompt):

| Config | Variant | ctx | avg generation tok/s |
| --- | --- | --- | --- |
| A: naive default | F16 | ~40 (auto) | 60.89 |
| B: manual high-quality | F16 | 4096 | 62.44 |
| C: planner-selected | Q8_0 | 4096 | 102.47 |
| D: context alternative | Q4_K_M | 8192 | 140.26 |

D vs C: **+36.9%** (faster, not slower) -- the floor was never
approached. C vs B: planner's variant choice (Q8_0) beat the manual
high-quality pick (F16) by **+64.1%** at the *same* context, a second,
independent point in the planner's favor for this host/family. **No
configuration tested in this evidence set exhibited a throughput
collapse anywhere near the 85% floor** -- a genuine limitation (this
small, bounded evidence set happened not to contain a bad trade), not
evidence the floor can never be crossed.

## CPU/GPU scope

CPU: the primary, most-validated backend this milestone (9 real
`membrane-run` invocations across 5 distinct configurations, 2
`membrane plan` snapshots, 2 paired plan+run checks, 1 bounded
infeasibility check).

GPU (Vulkan, NVIDIA GTX 1650): 1 family, 3 configs, 2 reps each -- the
Part 10 ceiling, no sweep. All 6 runs succeeded; generation throughput
(250-285 tok/s) ran roughly 1.9-2x the best real CPU config measured
(D, ~140 tok/s). VRAM is reported only as a pre-run free-memory
snapshot -- the measurement method (an `nvidia-smi` query before and
after the subprocess) cannot see peak VRAM while the process is still
running, since VRAM releases on process exit; disclosed as unmeasured,
not fabricated. **GPU policy remains less validated than CPU policy
after this milestone** -- one device, one small model family, six
total runs.

## Policy conclusion (H6, Part 12)

**Policy unchanged.** `plan-v2-variant-joint-v1` stays exactly as G2
shipped it.

Answering the task's five policy questions (full answers in the
artifact's `policy_evaluation` object): the highest-quality feasible
variant had reasonable throughput in absolute terms, but was not
always the fastest or highest-context option on this host (Q4_K_M beat
the selected Q8_0 on both axes, once, in this real CPU evidence); a
lower quant did unlock materially more context, confirmed by a real
run; `size_bytes` descending stayed accurate as a real physical
proxy for bits-per-weight (not independently re-verified as an output
*quality* proxy, since no quality metric was measured); "quality
first, then max context" remains defensible as a **default**, but is
demonstrably not the only reasonable choice on this exact host for
this exact family.

This is real, evidence-backed, and narrow: one tiny model family, one
shared dev host, one GPU. That is enough to *document* the real
tradeoff, not enough to *change default product behavior* from.

## Objective modes (Part 13)

**Not added.** The task explicitly permits `--objective
quality`/`--objective context` if evidence shows they select
meaningfully different plans -- and this milestone's real CPU evidence
does show exactly that divergence (quality-first picks Q8_0@4096; a
context-first mode would pick Q4_K_M or Q5_K_M@8192, a real,
materially different plan). The deliberate decision is to defer
implementing the flag anyway: the divergence is demonstrated on one
tiny model family, one host, one backend, and did **not** reproduce on
the GPU backend for the same family (VRAM was not the binding
constraint there). That is narrower than the bar the task itself
repeats throughout ("G3 should normally be conservative"). Recorded
here as a well-evidenced candidate for a future milestone once broader
evidence (more families, more hosts) exists.

## Execution-integration readiness (Part 17)

**`READY_FOR_READ_ONLY_ONLY`.** `membrane plan` remains advisory only;
`membrane use`/`membrane serve` are untouched, exactly as G1/G2 left
them.

Reasoning: real evidence this milestone covers one tiny model family
with real multi-quant validation, one single-quant sanity point, one
shared resource-constrained host, and one GPU -- no mid-size or large
real model was load-tested. The real confusion-matrix results above
also show the host-memory guard can reject configurations that truly
would have fit (`FALSE_REJECT`, once, plus one unresolved
`UNVERIFIED_REJECT`) -- a safe direction for a read-only advisor to
err in, but not yet the standard of accuracy `membrane use`/`membrane
serve` would need before acting on Planner v2's decisions by default,
or even under an opt-in flag.

## Models used

| Family | Variant | Source this milestone |
| --- | --- | --- |
| smollm2-135m-instruct | F16 | pre-existing repo fixture |
| smollm2-135m-instruct | Q8_0 | downloaded from the catalog's own verified URL; sha256 verified against the catalog's own recorded hash |
| smollm2-135m-instruct | Q4_K_M | downloaded from the catalog's own verified URL; sha256 verified against the catalog's own recorded hash |
| smollm2-360m-instruct | F16 | pre-existing repo fixture |

Exact hashes recorded in `results/planner-v2-evidence/validation.json`'s
`models_used` object. `llama-quantize` was deliberately not built (would
have required a full llama.cpp reconfigure/rebuild on this
memory-constrained host); downloading the two small, already
sha256-verified catalog files was the lower-risk path to a real
multi-quant sibling comparison.

## Reproducing this evidence

```
# 1. Build both binaries at the commit you want to measure:
make -C build-cpu-rc1 -j1 membrane membrane-run
make -C build-vulkan  -j1 membrane membrane-run   # optional, GPU evidence

# 2. Sibling scaling error (Part 5):
PYTHONPATH=third_party/llama.cpp/gguf-py \
  scripts/planner-v2-evidence/measure_sibling_scaling_error.py \
  scratch/sibling_scaling_error.json

# 3. Planner decisions + real runtime validation (Parts 6/7/8/9/10):
scripts/planner-v2-evidence/run_evidence_harness.py

# 4. Re-assemble the committed artifact from the two scripts' raw output:
scripts/planner-v2-evidence/build_validation_artifact.py

# 5. Validate the committed artifact's schema/invariants (also runs in CI):
python3 scripts/verify-planner-v2-evidence.py
```

`scratch/` (gitignored) holds the harness's raw, uncurated JSON; the
paired-snapshot infeasibility checks in this document were run as
one-off manual commands alongside the harness -- see
`build_validation_artifact.py`'s own `load()` calls for the exact
filenames it expects if reproducing those by hand.

## Limitations (not exhaustive elsewhere in this document)

See `results/planner-v2-evidence/validation.json`'s `limitations`
array for the full, disclosed list -- summarized: no mid/large real
model tested; only one real GPU; no VRAM peak measurement; no true
TTFT; two feasibility mismatches trace to genuine live host-memory
drift on a shared desktop, not fully reproducible on demand; no output
quality metric at any quant level; all throughput figures are 2-rep
averages of a short prompt and short generation, a deliberately small,
bounded point set, not a statistically powered benchmark.

## Confirmed

- `v1.0.0` tag/release untouched.
- No Ollama/vLLM/external-runtime-adapter work started.
- `membrane use`/`membrane serve` unmodified.
- No selection-policy code changed.
- No `--objective` flag added (deferred, see above).
