# MEMBRANE product direction

This is a **planning document** (Phase 32). It defines what MEMBRANE
actually is today, who it is for, and the one-sentence thesis the v0.4
roadmap (`docs/v0.4-roadmap.md`) is built around. No runtime behavior
changed to produce this document — every claim below is sourced from
the current `main` tree (README.md, `docs/*.md`,
`results/release-v0.3.0/readiness.json`, `results/first-run/
validation.json`, `results/release-artifacts/manifest.json`) as of
`v0.3.0`.

## 1. What problem does MEMBRANE solve today?

Separated honestly, per the audit this phase requires:

### A. Capabilities (real, shipped, in `v0.3.0`)

- CPU inference and Vulkan GPU offload (opt-in), on top of unmodified
  `llama.cpp`/`ggml`.
- `--auto`: one joint planner resolves GPU layer offload, KV precision
  (`native`/`q8`/`q5`/`adaptive`), and KV placement
  (`default`/`gpu`/`cpu`/`auto`) together, before generation starts.
- Bounded, transparent apply-time fallback if the primary plan can't
  actually be instantiated (never a silent guess, never unbounded
  retry).
- `--plan-only`: resolves the exact same plan a real run would, prints
  it, generates nothing.
- `--doctor`, `--list-devices`, `--inspect-model`: first-run/pre-run
  diagnostics, all read-only, none require a full model load except
  the real run itself.
- Structured JSON diagnostics (`schema_version: 1`, additive-only
  across every phase so far).
- An official Debian-family amd64 `.deb` package.
- Compressed KV (`q8`/`q5`/`adaptive`) validated for `LLM_ARCH_LLAMA`
  and, since `v0.3.0-rc3`, `Qwen2.5-1.5B-Instruct` specifically (not
  "all Qwen2 models" — see `docs/compatibility.md`).

### B. User-facing value (what a user gets that manual `llama.cpp` CLI
usage doesn't hand them)

- Not having to hand-tune `-ngl` by trial-and-error crash-and-retry —
  `--auto` computes a real, memory-aware layer count from actual
  device/model bytes before ever loading the model.
- Not having to independently research which KV quantization type is
  even *safe* for a given model's architecture/head shape —
  `compat_check.c` gates it before any attempt, with a specific reason
  if it can't.
- A genuine dry-run (`--plan-only`) — you can see the exact
  resolved configuration a real run *would* use, including a
  machine-readable reason, without spending the time/memory to
  actually run it.
- When a plan can't actually be instantiated (memory changed since
  planning, etc.), MEMBRANE retries other already-ranked, already-legal
  plans instead of just crashing — real resilience, not just a nicer
  error message.
- Actionable, specific human error text and a JSON error contract
  instead of raw `ggml`/`llama.cpp` assertion failures or opaque exit
  codes.

### C. Technical differentiators (vs. driving `llama.cpp`'s own CLI
directly)

- **KV placement as an independent axis from KV precision** — MEMBRANE
  can put the KV cache on a *different* device residency decision than
  where the model weights are offloaded, and reason about the two
  separately. Plain `llama.cpp` CLI usage ties KV residency to
  wherever the relevant layers already are; it has no equivalent
  first-class "KV placement" concept.
- **Joint, pre-load, cross-dimension planning.** GPU layers, KV
  precision, and KV placement are resolved as one linked decision
  (`docs/joint-planner.md`), not three independent flags a user reasons
  about separately.
- **Bounded apply-time fallback** (`docs/auto-fallback.md`) — genuine
  automation beyond "try once, fail if it doesn't fit," which is
  `llama.cpp`'s own default behavior.
- **Compatibility gating before attempt** — a compressed-KV request for
  an unsupported architecture/shape fails closed with a specific reason
  *before* touching a context, rather than either silently proceeding
  or failing deep inside an unrelated-sounding internal error.

### D. Limitations (real, current)

- Compressed-KV compatibility is architecture-narrow: `llama` +
  `qwen2` only (native KV has no architecture restriction).
- All real backend validation evidence (CPU, NVIDIA Vulkan, AMD Vulkan
  RADV) comes from **one** physical developer host, plus
  container-based packaging checks on that same host.
- Packaging is Debian-family amd64 only.
- No CUDA.
- `--auto` still requires an explicit `--ctx` — the single largest
  remaining "you must already know a memory-relevant number" gap (see
  `docs/v0.4-roadmap.md` §6).
- Memory estimates are real formulas over real GGUF/device metadata,
  but are pre-load estimates and point-in-time snapshots — never an
  absolute OOM guarantee, and (see the roadmap's feasibility section)
  do not currently account for `ggml`'s own transient compute-buffer
  allocation at context-creation time.

### E. Things `llama.cpp` already does itself (do not re-claim these
as MEMBRANE differentiation)

- The raw mechanisms MEMBRANE decides *how to use* are not MEMBRANE
  inventions: `llama.cpp`'s own CLI already exposes quantized KV cache
  types (`--cache-type-k`/`--cache-type-v`) and GPU layer offload
  (`-ngl`/`--n-gpu-layers`). MEMBRANE does not invent quantized KV
  storage or GPU offload — it decides *whether/how many/which type* is
  safe for the current model and hardware, automatically, and reports
  why.
- Model loading, tokenization, generation, and flash attention are
  100% `llama.cpp`/`ggml` — MEMBRANE never touches or reimplements this
  path (see `tools/membrane-run/main.cpp`'s own header comment: a
  single decode pass through the real, unmodified inference path).
- GGUF metadata parsing is `ggml`'s own `gguf.h` API; MEMBRANE only
  calls it.

## 2. Positioning vs. raw `llama.cpp`

**MEMBRANE is not "a better `llama.cpp`."** It is a planning and
safety layer on top of unmodified `llama.cpp` — it does not replace or
outperform `llama.cpp`'s own inference (Phase 24/25 found zero
MEMBRANE-owned speed-changing opportunity; see
`docs/performance-optimization.md`). Its value is deciding *how to
configure* `llama.cpp`'s already-existing offload/quantization knobs
so a user doesn't have to guess, with an inspectable plan and
compatibility gating instead of trial-and-error crashes.

Ranking the candidate "value areas" by real user value vs. mostly
implementation detail a typical user never directly reasons about:

| Area | User value |
|---|---|
| Joint memory planning (GPU layers + KV precision + KV placement together) | **HIGH** — directly answers "will this fit, and how" |
| Constrained-memory planning generally | **HIGH** — same as above, the core of the product |
| Compatibility gating | **HIGH** — prevents silent garbage output or an opaque crash |
| Inspectable preflight (`--plan-only`) | MEDIUM-HIGH — builds trust, not part of the fastest happy path |
| First-run diagnostics (`--doctor`/`--list-devices`/`--inspect-model`) | MEDIUM-HIGH — reduces onboarding friction directly |
| Explicit KV precision selection (`--kv q8`/`q5`) | MEDIUM — mostly invisible once `adaptive`/`--auto` picks for you |
| Independent KV placement | MEDIUM — real value for constrained/advanced users, most users should never need to touch it directly |
| Bounded apply-time fallback | MEDIUM — valuable resilience, usually invisible unless something nearly fails |

## 3. v0.4 product thesis

Candidate under evaluation: *"You give MEMBRANE a GGUF. MEMBRANE
figures out how to fit and run it safely."*

**Verdict: aspirational for v0.4, not accurate today.** Today the user
must still supply `--ctx` explicitly for `--auto` to do anything at
all — the single most consequential memory-relevant number is still
the user's own responsibility. The candidate thesis becomes accurate
only once a context recommendation exists (the v0.4 primary feature,
§8 of the roadmap). It also needs a qualifier: "safely" cannot mean an
absolute guarantee — this project's own conventions never claim that,
and a real, disclosed gap (unmodeled compute-buffer overhead, see the
roadmap's feasibility section) means it shouldn't start now either.

**Accurate thesis, today (`v0.3.0`):** Given a local GGUF model and an
explicit context size, MEMBRANE resolves a memory-aware plan — GPU
offload, KV precision, KV placement — that is inspectable before you
run it and fails closed rather than silently guessing when it can't
fit.

**Strongest accurate v0.4 thesis** (the one sentence the v0.4 roadmap
is built to make true):

> MEMBRANE takes a local GGUF model and your host's real memory limits
> and turns them into one inspectable, safe run plan — instead of you
> guessing GPU layers, KV type, KV placement, and context size by
> trial and error.

## 4. v0.4 primary user

**Persona:** a technically capable Linux user who wants to run GGUF
LLMs locally, but does not want to manually reason about GPU layers,
KV type, KV placement, context memory, or RAM/VRAM fit. Not "everyone"
— specifically someone comfortable with a terminal and a `.deb`
install or a source build.

**What they already know:** how to use a terminal; roughly what a
"model" and "context length" mean; that GPUs have limited VRAM; how to
find/download a `.gguf` file themselves (MEMBRANE never downloads
models on a user's behalf).

**What they should NOT need to know:** `ggml` tensor/block-format
byte math; `llama.cpp` internals; how to hand-compute VRAM headroom;
the joint planner's own internal ranking logic; which architectures
are on the compressed-KV allowlist (that should just fail closed with
a clear reason if it doesn't apply).

**What success looks like for them:** they run `membrane-run --doctor`
(sanity-check the environment), point `--inspect-model` at their GGUF
(learn what it supports), then get a working generation via `--auto`
on the first or second try — and, if they care to look, can always see
*why* a given configuration was chosen (`--plan-only`, `--verbose`,
JSON) without reading source code.

## 5. Core job-to-be-done

> "When I have a GGUF model and limited local hardware, I want
> MEMBRANE to tell me a configuration — including context size — that
> will actually run, without me manually testing several combinations
> myself."

**Measurable success:** a user can point MEMBRANE at a GGUF and,
within one command, get a plan that is legal — it fits real host/GPU
memory under the existing safety reserve — and that plan is
reproducible: running `--plan-only` against the exact same inputs
always shows the exact same resolved configuration a real run would
use.

## 6. Product copy

Two distinct pieces of copy, deliberately not merged:

- **Product tagline (v0.4-target — becomes accurate once the v0.4
  primary feature ships, §8 of the roadmap; do not adopt in README.md
  before then):** *"You give it a GGUF. MEMBRANE figures out how it
  fits."*
- **Technical subtitle (accurate today, safe to use now):**
  *"Hardware-aware KV and GPU planning for local `llama.cpp`
  inference."* (Naming `llama.cpp` explicitly keeps this precise —
  see §2's "not a better `llama.cpp`" framing.)
- **v0.4 theme (one short phrase):** *"From configuration to
  recommendation."*

No README changes are made in this phase (Section 31 of the Phase 32
task: doc/product-direction is its own deliverable, not a README
rewrite) — these are proposals for when the v0.4 primary feature
actually ships.
