# MEMBRANE v1.1.0

Hardware-aware memory intelligence and control for local LLM inference.

v1.0.0 made MEMBRANE a reliable local model runtime. v1.1.0 makes it
something you can understand and steer: it shows what memory is doing,
explains it, and -- only when you ask -- changes it, across more than one
local runtime. Everything below is **additive**: the OpenAI-compatible API
(v1) and the `--json` shapes are unchanged, and a v1.0.0 registry and
`server.json` keep working untouched ([upgrade-v1.0-to-v1.1.md](upgrade-v1.0-to-v1.1.md)).

## What is new

**plan -> observe -> advise -> apply -> adapt**

* `membrane observe` / `advise` (read-only): provenance-labeled facts, and
  explainable, capability-aware recommendations. (Milestone I)
* `membrane apply` (Milestone J): the first command that changes anything.
  **Dry run by default**; `--yes` or a terminal confirmation to apply. A pure
  action compiler, stale-plan protection, ordered execution with rollback, a
  local audit log, and no guessed target values. [active-control.md](active-control.md)
* `membrane adaptive` (Milestone K): **off by default**, opt-in, bounded
  adaptation with hysteresis, cooldown, a rate limit, anti-thrashing and
  manual-override adoption; one controller thread inside `membrane serve`
  whose only mutation path is `apply`. [adaptive-runtime.md](adaptive-runtime.md)
* `membrane run MODEL` (Milestone L): chooses among membrane-native, Ollama
  and vLLM from facts, **prints why**, then uses it; a read-only vLLM adapter
  (audited against vLLM v0.31.0); a hardware topology that keeps every GPU.
  [orchestration.md](orchestration.md), [runtime-vllm.md](runtime-vllm.md)

**Product polish**

* Bare `membrane` is a status screen with only-working next steps.
* `membrane models` / `models available` / `models recommend`; a catalog of 9
  families (tiny, general, multilingual, coding, reasoning, long-context) with
  purpose, license and an honest evidence label.
* `membrane use` / `chat`: a numbered picker; partial and misspelled names are
  suggested and confirmed -- never chosen silently.
* Concise default output for `plan` / `observe` / `advise` / `apply` /
  `doctor`; `--verbose` keeps the full report; `--json` is unchanged. Color
  only on a terminal, honoring `NO_COLOR` / `--no-color`.
* `membrane doctor` detects a stale or different `membrane` binary, names a
  missing GPU runtime library (e.g. `libcudart.so.12`) as an environment
  problem, and explains Vulkan vs. CUDA.
* Fixes: blank `Backend:`/`Device:` lines in `plan`; `use` printing
  `Backend: ?`.

## What did not change

The OpenAI-compatible chat API subset (no tool calling, no `/v1/embeddings`,
no `response_format` beyond text -- rejected explicitly), the loopback-only,
unauthenticated security boundary (no CORS headers; none added), the
Debian-family `.deb` as the one official package.

## Why 1.1.0 and not 2.0.0

No frozen contract is broken. The API v1 and the `--json` outputs are
unchanged or extended; the CLI contract allows new subcommands, flags and
fields within a major version, and states that human-readable wording is not
frozen. The new `server.json` blocks (`overrides`, `adaptive`) are optional,
and absent blocks mean the old behavior. A new capability set with no breaking
change is a minor release.

## Evidence and honest limits

* CI: every check on the merged milestone PRs was green (Linux CPU/ASan/
  Debug, CodeQL, packaging, CUDA build, Windows and macOS smoke).
* Local, on the maintainer's 5.6 GiB development host: the new unit suites
  under ASan/UBSan (and the adaptive controller under TSan), the repository's
  Python verifiers, the official Python and Node `openai` SDK harnesses, and a
  real service + chat + `apply` + `adaptive` run with a small model. The full
  local `ctest` was not run on that host (memory); CI ran it.
* **Adaptive pressure behavior is validated by a deterministic synthetic
  timeline, not by inducing real memory pressure on hardware.**
* **Ollama and vLLM support was tested against in-process mock servers that
  follow the audited API shapes; no live Ollama or vLLM daemon was run.**
* **No multi-GPU execution and no distributed orchestration.** Topology and
  device visibility only.
* Model-family evidence stays scoped to the exact validated fixtures; new
  catalog families are `metadata-only` or not yet run by MEMBRANE.
* No official Windows or macOS package (source build; per-PR CI).
* CXL / FPGA remain research-only.

Details: [results/release-v1.1.0/readiness.json](../results/release-v1.1.0/readiness.json).
