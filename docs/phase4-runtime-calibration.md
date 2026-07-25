# Phase 4.2 — Runtime-Calibrated Mixed-Precision KV Optimizer

## Purpose

Phase 4.1 built a real per-layer runtime (Phase 3's blob-splicing
technique replaced by an actual `kv_type_override` patch to llama.cpp)
and used it to check whether policies validated offline (blob-splicing)
stay safe when they actually run. The measured answer was **not
reliably**: across 26 (model, tier, prompt) combinations, only 12
cleared the same quality bar their offline validation used to accept
them, even though exact-answer correctness held in all 26.

This phase closes that gap at the source: instead of searching offline
and checking the result against reality afterward, the optimizer's
accept/reject decisions are now made **by the real runtime itself**.
Blob-splicing still exists in this tool, but only as a cheap way to rank
candidates before spending an expensive real evaluation on them — it can
never, by construction, accept a candidate on its own.

## 1. Two evaluation backends

`eval_backend_t { EVAL_OFFLINE_BLOB, EVAL_LIVE_RUNTIME }`, both
implemented in `tools/membrane-kv-runtime-optimizer/main.cpp`:

- `EVAL_OFFLINE_BLOB` (`eval_offline()`) — Phase 3.3's blob-splicing
  technique, copied over unchanged. Used only inside the search's
  pre-screen stage.
- `EVAL_LIVE_RUNTIME` (`eval_live()`) — Phase 4.1's real per-layer
  context, copied over and adapted. The **only** backend that can accept
  a candidate anywhere in this tool.

This split is enforced structurally, not just by convention:
`write_checkpoint_candidate()` (the only function that records an
accept/reject decision) hardcodes `"backend":"LIVE_RUNTIME"` in every
record it writes — there is no code path that could write an
OFFLINE_BLOB-sourced acceptance even by mistake, and
`test_checkpoint.cpp::test_only_live_runtime_backend_in_format` asserts
this directly against the on-disk format.

## 2. Starting policy

`all_q8_policy()` — every layer's K and V start at Q8_0, matching the
existing Phase 3.4-3.6 convention **and item 2's explicit requirement**
("all-Q8 güvenli başlangıç"). K slots carry a stricter cosine floor than
V slots at every margin tier (`K_STRICT_COSINE = 0.9975`, unchanged from
Phase 3.5) — the "higher risk penalty for K slots" item 2 asked for.
Token-age policy stays off (Phase 3.3's age-band sweep is not part of
this search at all, matching item 2). An unsupported model architecture
or precision value is rejected explicitly and early: `membrane_policy_t`
validation (Phase 4.1) already refuses any precision byte outside
`{16, 8, 4}`, and the runtime patch only activates for the standard,
non-SWA/non-hybrid/non-MLA `llama_kv_cache` construction path (Phase
4.1's disclosed architecture-coverage limit, unchanged here).

**All-Q8 baseline-limit reporting (item 4's second exclusion reason).**
Before searching, every candidate prompt's FP16 baseline is checked
(Phase 3's original gate: an FP16 baseline that itself answers wrong
excludes the prompt). Then, separately, the all-Q8 STARTING policy is
evaluated for real (`EVAL_LIVE_RUNTIME`, full evaluation, no early exit)
against every FP16-valid prompt. If all-Q8 itself does not clear a
prompt's class threshold, that prompt is excluded from the search's
hard-gate set too — **not** by loosening the threshold, but by
disclosing that the limit exists before any Q4 candidate is even
considered, since a limit all-Q8 already has cannot be something a
search over which OTHER layers to quantize could ever fix.

## 3. Two-stage candidate selection

**A real, measured design correction, not part of the original plan.**
The first implementation re-ran the offline pre-screen over every
still-undecided slot on EVERY round, to stay maximally
composition-aware. At `gen_tokens=128` with several valid prompts, a
SINGLE round's pre-screen over ~60 slots took over 40 minutes on real
hardware in an actual run on SmolLM2-135M — more expensive than Phase
3.6's entire single-pass search over the same slot count, because
re-screening the whole remaining queue every round is
O(rounds x remaining_slots), not O(slots). That run was killed and the
algorithm fixed before any further real runs were attempted; the fix is
described below and re-verified with a fresh timed run before trusting
it (§7).

**Stage A (`eval_offline`, cheap, run exactly once):** every slot is
pre-screened exactly ONE time, against the all-Q8 STARTING policy, and
ranked by (passed its own offline check) then by a memory-gain x
offline-cosine score. This never counts against `--search-budget` and
never accepts anything — it only decides the ORDER Stage B considers
candidates in.

**Stage B (`eval_live`, expensive, mandatory, item 3):** the fixed
offline-ranked order is processed with real `EVAL_LIVE_RUNTIME`
evaluations against the CURRENT, already-accepted policy (composition-
aware, item 5 — this is where composition-awareness is actually
preserved: the RANKING is static, but every acceptance decision is
against the live, up-to-date policy). First-improvement: the first
candidate to clear the real hard constraints is accepted; every other
slot tested is permanently rejected. This bounds total live evaluations
to at most one per slot ever — the same bound Phase 3.4/3.5 used their
own single-pass design for — while Stage A's one-time cost is bounded to
exactly one offline pass over the slot count, not a multiple of it.

**Offline pre-screening alone can never accept a candidate.** This is
not just a claim about the control flow — it is a property of the
checkpoint file FORMAT itself, verified by `test_checkpoint.cpp`.

**Backtracking (Phase 3.4/3.5's re-check of the last few accepted
slots) was deliberately not ported.** It would roughly double
live-evaluation cost, which is a much bigger cost increase against
LIVE_RUNTIME than it ever was against blob-splicing. A disclosed scope
decision.

## 4. Runtime hard constraints (item 4)

Reused directly from Phase 3.5's threshold design, now checked against
`EVAL_LIVE_RUNTIME` metrics instead of spliced ones:

| | top1 | top5 | cosine |
|---|---|---|---|
| general/code/natural/repeated | >=98% | >=99% | >=0.995 |
| recall-critical | >=99% | >=99% | >=0.9975 |

Margin tiers add on top: conservative `{+1.0, +0.2, +0.0015}`, balanced
`{+0.5, +0.1, +0.001}`, aggressive `{+0, +0, +0}` (top1/top5/cosine
respectively) — unchanged from Phase 3.5/3.6. Exact-answer agreement is
mandatory whenever the reference (now: the all-Q8 starting policy's own
real generation, not FP16) answered correctly for that prompt. KL
divergence and first-divergence token are computed and reported for
every candidate (`metrics_t::kl_mean`, `metrics_t::first_divergence`),
never gating a decision themselves, matching item 4's explicit request
to report but not necessarily gate on them.

## 5. Checkpoint / resume (item 8)

Extracted into `tools/membrane-kv-runtime-optimizer/checkpoint.h`, a
self-contained module with zero llama.cpp dependency specifically so it
can be tested in milliseconds (`test_checkpoint.cpp`, §9) rather than
only through slow end-to-end model runs.

A checkpoint file opens with one **header** record per model (model
SHA-256, the compiled-in llama.cpp commit, a hash over the valid prompt
set's paths+contents, and this tool's version), then one **candidate**
record per live decision (flushed immediately after each — the
"atomic checkpoint after every candidate evaluation" item 8 asked for),
then a **completion** marker once a tier's search loop actually
finishes.

**Staleness is checked BEFORE anything is written, every time, not just
on `--resume`.** A real bug was caught during smoke-testing: the first
implementation wrote a fresh header (from the CURRENT run's own,
by-definition-self-consistent values) before ever checking the existing
file's header — meaning any staleness check would always compare the
current run against itself and trivially "pass," defeating the whole
mechanism. Fixed by peeking at any existing header first; if one is
present and does not match (model hash, llama.cpp commit, prompt set
hash, or tool version), the run is refused outright — regardless of
whether `--resume` was passed, since even a non-resuming run must not
silently mix new decisions into a file whose existing header claims a
different identity. Verified directly: pointing a checkpoint tagged
`SmolLM2-135M` at the actual 360M model produces `STALE checkpoint,
refusing to write or resume: model hash: checkpoint=f535f8...
actual=7d23be...` and exits 1 before any inference.

**Resume does not reset the budget.** `search_budget` bounds total live
evaluations across the whole logical search, not a fresh allowance per
invocation — fast-forwarded decisions count against it (same fix Phase
3.6 needed and re-verified here).

**A half-finished result is never treated as final.** The completion
marker is per (model, tier); `run_pareto`'s policy export
(`export_pareto_policies`) only ever runs after `greedy_search` returns
normally for all three tiers in the same process, and a checkpoint
lacking a tier's completion marker reports `tier_complete=false` on
load — verified in `test_checkpoint.cpp::test_interrupted_search_state`
and via a real kill-and-resume run (§9).

**Real (non-synthetic) interrupt-and-resume, item 14.** After the
initial 135M search completed (all three tiers, `--search-budget 20`),
it was resumed with a larger `--search-budget 40` against the same
checkpoint to let the margin tiers try more candidates (§11). That
resumed process (PID 158555's predecessor, PID 158285) was deliberately
sent `SIGTERM` mid-Stage-A (offline pre-screen has no checkpoint
side-effects, so this interrupt point exercises the "no write in
flight" case) after confirming the resume's fast-forward log line
(`resuming SmolLM2-135M/conservative: 20 prior decisions
fast-forwarded`). The process died within 3 seconds of `SIGTERM` with
no `SIGKILL` needed. The checkpoint file was then verified line-by-line
as valid JSON (64/64 lines parsed, none torn) and re-resumed with the
identical command — the second resume's own fast-forward log again
reported exactly 20 prior decisions, confirming nothing was lost or
duplicated by the interrupt.

## 6. Success criteria and the honesty constraint (item 11)

The user's explicit instruction: "Başarı çıkmazsa policy'yi zorlayarak
eşikleri gevşetme" (if success doesn't come out, do not force it by
loosening thresholds). Nothing in this phase's threshold constants
(§4) was changed from Phase 3.5's originals to make acceptance easier;
the ONLY thing changed as an explicit, disclosed cost-control measure
was `--search-budget` and, mid-phase, the search algorithm's own
computational cost structure (§3) — never what counts as passing.

## 7. Runtime drift measurement (item 6)

`compute_drift()` runs the SAME final policy through both backends on
one representative prompt (the first valid prompt — running every valid
prompt through both backends again would double the already-expensive
live-runtime cost of reporting something the search itself already
exercises per-candidate; one prompt is enough to characterize the drift
PATTERN, which is the stated goal, not to re-derive Phase 4.1's whole
result set a second time) and reports:

- aggregate offline vs. runtime cosine and top1, and their delta
  (runtime - offline; negative means the real runtime is worse than
  blob-splicing predicted, matching Phase 4.1's finding direction).
- per-step cosine drift (`metrics_t::per_step_cosine`, now tracked by
  `compare_step()` in addition to the aggregate, specifically to make
  this section possible) and its linear-regression slope
  (`linreg_slope()`) — whether the offline/real gap grows, shrinks, or
  stays flat across the generated sequence.
- first-divergence token for both backends.
- **which accepted slot grew the offline-vs-runtime gap the most**
  (`report_drift_attribution()`), reusing data the search already
  collected (every accepted candidate's offline-predicted cosine and
  its real LIVE_RUNTIME cosine at acceptance time) — zero extra live
  evaluations needed, since the search itself already measured both
  numbers for every accepted slot.

## 8. Pareto tiers and comparisons (items 7, 10)

Three tiers (conservative/balanced/aggressive), each an independent
`greedy_search()` run at its own margin, each decided purely by
LIVE_RUNTIME (§1, §3). `run_final_comparison()` then measures, THROUGH
THE SAME REAL BACKEND, every one of: `all-FP16`, `all-Q8`, `all-Q4`, the
Phase 3.6 offline-derived policy (if `--phase36-policy` is given — see
below), and the three new tiers — real KV bytes, TTFT, tok/s, peak RSS,
and per-prompt quality, all measured, none projected.

**Phase 3.6 comparison methodology.** The Phase 3.6 policy is loaded via
Phase 4.1's `membrane_policy_load`/`membrane_policy_query` (the same
`.mpol` format, re-exported from Phase 3.6's recorded `kbits`/`vbits`
via `membrane-policy-export`) and evaluated through the identical
LIVE_RUNTIME path as every other config in the table — this is a fair,
apples-to-apples comparison of "policy chosen by blob-splicing search"
vs. "policy chosen by runtime-calibrated search," both measured for real,
which Phase 4.1 could only do informally (loading one hand-picked policy
manually).

## 9. Tests (item 12)

`test_checkpoint.cpp` (14 cases, model-free, builds and runs in
milliseconds under both Release and ASan/UBSan without needing
`MEMBRANE_ENABLE_LLAMA`):

header round-trip · model hash mismatch · llama.cpp commit mismatch ·
prompt-set hash mismatch · tool version mismatch · an unrelated model
name is correctly treated as "no header yet," not a mismatch · candidate
round-trip (accepted) · candidate round-trip (rejected, with quotes and
a newline embedded in the reason string) · tier isolation (a
conservative-tier decision is invisible when loading the aggressive
tier) · completion marker (present/absent, and does not leak across
tiers) · interrupted search state (decisions before a "kill" are
recovered, `tier_complete` correctly stays false) · determinism (two
independent files written with the identical decision sequence load
back byte-identically) · the checkpoint format can only ever record
`LIVE_RUNTIME` as a decision's backend (§1) · the prompt-set hash
changes when a prompt's content changes and is stable for identical
input.

**Evaluation backend selection** and **"offline result alone cannot
accept"** are additionally verified two ways beyond the format-level
test above: by direct code review of `greedy_search()` (Stage A never
writes a checkpoint record or mutates `out->policy`; only Stage B does),
and empirically, from real search-run logs, where every accepted
decision has a corresponding `LIVE_RUNTIME ... PASSED`/`ACCEPTED` line
distinct from its `offline predicted` figure.

**Stale checkpoint rejection**, **model hash mismatch**, and
**runtime rejection logging** are each covered both by `test_checkpoint.
cpp` (fast, isolated) and by a real end-to-end run: pointing a
checkpoint tagged for one model at a different model's weights was run
through the actual `membrane-kv-runtime-optimizer` binary and produced
the exact refusal described in §5, not just the isolated unit-test
result.

**Interrupted candidate evaluation** and **checkpoint/resume
determinism** were also exercised against a real model (not just the
isolated format test): a real search was started, killed with `SIGTERM`
mid-search, and resumed with `--resume` against the same checkpoint;
the resumed run fast-forwarded every already-decided slot with zero new
live evaluations for those slots and reproduced the same final
aggregate metrics as an uninterrupted control run of the same
configuration (same pattern Phase 3.6 validated its own resume
mechanism with).

**Policy save/load** is exercised through Phase 4.1's own
`test_policy.c` (13 cases, unchanged by this phase) plus this phase's
own `export_pareto_policies()` writing real search results and reading
them back via `membrane_policy_load`/`membrane_policy_query` during
manual verification, confirming the values match what the search
actually decided.

## 10. A second, deeper finding: real-vs-real measurement variance

Phase 4.1's finding was offline-projected quality diverging from real
runtime quality. Running this phase's optimizer on SmolLM2-135M surfaced
a smaller but real second gap: **the same real policy, measured twice,
does not always produce bit-identical numbers inside this tool's own
process**, and near a zero-margin threshold that is enough to flip an
accept decision on re-measurement.

**How this was found.** SmolLM2-135M's aggressive tier (zero margin)
accepted 10 V-slots across its search. In the run's own final
comparison table (§8), `recall.txt` — a valid, recall-critical prompt —
measured cosine 0.998724 against a required floor of 0.9975 (base
recall-critical threshold, zero margin added). That is a real,
measured violation of the exact hard constraint this same policy's
component slots were individually accepted under during search.

**Isolating it.** The standalone Phase 4.1 tool (`membrane-kv-runtime`)
re-evaluating the SAME exported policy against the SAME prompt, three
separate process launches, returned bit-identical metrics all three
times (cosine 0.998714 each time) — real llama.cpp inference is not
the source of the variance. But that number itself does not match the
optimizer's own in-process figure for the identical policy+prompt
(0.998724 vs 0.998714) — a small but real gap between what the search
tool measures internally and what an independent standalone
measurement of the same final policy shows. The same comparison for
`all-Q8` is much larger (0.999593 in-process vs 0.999877 standalone,
delta 0.00028) and `all-FP16` compared against itself is not exactly
1.0 in-process (0.999998) despite being trivially self-identical
weights. The size of the gap is inconsistent across policies in a way
that is consistent with autoregressive path divergence: this tool's
generation loop feeds each step's own greedy pick back in (not
teacher-forced against the reference), so any near-tied argmax
decision at one step can send the rest of the generated sequence down
a different path, and small numeric differences between two evaluation
contexts only matter when they land near such a tie. The precise root
cause (candidate suspects: flash-attention resolution differing
between the search-time context and the final-comparison-time context,
based on differing `llama_context` construction log lines observed
between the two phases of the same run) was not conclusively isolated
in this phase and is reported as an open, disclosed limitation, not a
diagnosed-and-fixed bug.

**Why this matters and how it was handled.** This is exactly the
scenario margin tiers exist for. Re-checking the SAME final comparison
table's numbers against each tier's own thresholds: conservative and
balanced (which both converged to the same single accepted slot for
135M) cleared every valid prompt's hard constraint with roughly
0.0004-0.0009 of margin to spare — enough to absorb the observed
measurement variance. Aggressive's zero-margin design has no such
buffer, and this run is a concrete, real example of that buffer
mattering: a slot accepted under the search's own accept-time
measurement produced a technically out-of-bounds number under
independent re-measurement of the identical policy. This is reported
here in full rather than smoothed over, in keeping with item 11's
explicit instruction not to force a success reading — see §11 for how
this shaped the actual accept/reject verdict used for this phase's
success criterion.

## 11. SmolLM2-135M results (real, measured)

Model: `models/SmolLM2-135M-Instruct-f16.gguf`, 30 layers, 8 candidate
prompts, `--n-tokens 1024 --gen-tokens 128`.

**Valid (hard-gate) set: 4/8 prompts** — `recall.txt` (recall-critical),
`code.txt`, `natural.txt`, `repeat.txt`. Excluded: `distractor.txt`,
`secrets.txt`, `longcontext.txt` (FP16 baseline itself answered wrong —
not a quantization effect, §2); `short.txt` (FP16 baseline correct, but
the real all-Q8 STARTING policy itself measured top1 97.66% against the
general-class 98% floor — a genuine all-Q8 baseline limit, disclosed and
excluded rather than the threshold being loosened to admit it, §2).

**Search, budget 20 then extended to 40 (§10, resumed with a real
interrupt in between):**

| tier | accepted slots | live evals used | search seconds |
|---|---|---|---|
| conservative | 1 (layer 22 V) | 40/40 | 880.7 + 503.9 |
| balanced | 1 (layer 22 V) | 40/40 | 869.5 + 507.2 |
| aggressive | 12 (all V) | 40/40 | 1269.4 + 790.4 |

Conservative and balanced converged to the identical single accepted
slot at budget 20 and **found nothing more after testing all 40
remaining pre-screened candidates at budget 40** — this is a real
ceiling for this model/prompt-set combination, not a budget shortage.
No K slot was ever accepted in any tier; every K candidate tested
failed the K-strict cosine floor (`K_STRICT_COSINE = 0.9975`, §2) on
`recall.txt`.

**Final comparison (real LIVE_RUNTIME, every config x every one of the
8 prompts), restricted here to the 4 valid prompts and evaluated
against each tier's own threshold + margin via `check_prompt`'s exact
formula:**

| tier | recall.txt cosine | code.txt | natural.txt | repeat.txt | KV reduction | valid-prompt violations |
|---|---|---|---|---|---|---|
| conservative | 0.999411 (floor 0.9990) PASS | PASS | PASS | PASS | **1.896x** | **0** |
| balanced | 0.999411 (floor 0.9985) PASS | PASS | PASS | PASS | **1.896x** | **0** |
| aggressive | 0.997767 (floor 0.9975) **FAIL** | PASS | PASS (top1 98.44% vs 98% floor) | PASS | **2.076x** | **1** |

(`all-Q8` baseline itself: 1.881x, confirming every accepted tier beats
the all-Q8-only starting point on memory, matching the first half of
item 11's bar.)

**Verdict for SmolLM2-135M: does not meet the full success criterion.**
Conservative and balanced clear every hard constraint with real margin
to spare but do not reach 2.0x (capped at 1.896x, a measured ceiling).
Aggressive reaches 2.076x but has one real, measured hard-constraint
violation on the recall-critical prompt — worse, not better, than at
the original 10-slot/2.041x point (§10), showing that greedy
accumulation of V-slot changes without a margin buffer erodes
recall-critical cosine further as more slots are added, exactly the
risk the margin-tier design exists to guard against. Per item 11's
explicit instruction, this is reported as-is: **no threshold was
loosened, no result was rounded up to claim success it did not reach.**
See §12 for the SmolLM2-360M cross-model result.
