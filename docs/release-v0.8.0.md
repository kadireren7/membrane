# MEMBRANE v0.8.0

Mega Phase D consolidation release. This is a **release hardening**
phase: no new model families, no embeddings, no tool execution, no
continuous batching, no multi-model residency — see `README.md`'s
"Known limitations" for the complete, current scope.

## What changed since v0.4.0

v0.4.0 shipped one-command onboarding and release-supply-chain
maturity on top of the CPU/Vulkan planner. Mega Phase D (this release)
turns MEMBRANE into a product that chooses, fits, installs, runs, and
serves models on its own:

- **Model catalog + real HTTPS download + checksum verification**
  (D1) — `membrane model search/info/install`, a curated built-in
  catalog, resumable downloads, atomic finalization.
- **Hardware-aware variant selection** (D2) — the largest quant a
  real host is estimated to fit, deterministic, explicit override
  always honored.
- **Real CUDA backend** (D3) and **real macOS/Metal backend** (D4) —
  see "Platform/backend support" below for the exact, current evidence
  scope of each.
- **Real Windows support** (D5) — MSVC build, install, CPU generation,
  model download/checksum, Task Scheduler service lifecycle, all
  continuously validated on real per-PR CI.
- **`membrane use MODEL`** (D6) — select an installed model or install
  a catalog model with explicit consent, live-switching a running
  server with no restart, no admin-endpoint call needed from a normal
  client.
- **Real OpenAI-compatible client validation** (D7) — the official
  Python and Node.js `openai` SDKs, real streaming, real `stop`-
  sequence support, explicit `tools`/`response_format` rejection. A
  real concurrency bug (a stop-sequence match could hang a streaming
  response forever) was found and fixed.
- **Release hardening** (D8, this release) — a real, disclosed
  long-context fix (see below), a real, current support matrix, a real
  model-family compatibility matrix, real CUDA/Vulkan re-validation on
  this exact release candidate, a real v0.4→v0.8 upgrade test, and the
  v0.8.0 stable release itself.

## Install once, use it

```bash
sudo apt install ./membrane_0.8.0_amd64.deb
membrane use qwen2.5:7b
```

Two commands. `membrane use` resolves the name, previews the download,
asks for consent, installs it, and selects/activates it. See
`docs/model-lifecycle.md`.

## Model catalog and download

Three real, curated model families as of this release
(`docs/model-catalog.md`, `docs/model-compatibility.md`):
SmolLM2-135M-Instruct, SmolLM2-360M-Instruct, Qwen2.5-1.5B-Instruct.
Every catalog entry's URL, license, and checksum was individually
verified against the real upstream Hugging Face repository. Download
security (HTTPS-only, atomic finalization, resumable, checksum-
verified, safe filenames, no path traversal, no shell/command
injection) is unchanged since D1 — re-audited by source read this
phase, no regression found.

## Variant selection

Deterministic: the highest-priority variant that is estimated to fit
the current host's real available memory, under the currently
documented policy (`docs/model-variant-selection.md`) — not a "best
quality" claim, since quality is never measured by this selection
logic. An explicit `--quant`/`--variant` override is always honored,
never silently replaced.

## Platform/backend support

See `docs/support-matrix.md` for the full, evidence-state-labeled
matrix. Summary:

- **Linux**: CPU (VALIDATED_CI, every commit), Vulkan (VALIDATED_REAL,
  real local dual-GPU hardware, re-confirmed this release), CUDA
  (VALIDATED_REAL, one real local GPU, re-confirmed this release —
  real generation, real throughput, real GPU memory delta; CI only
  compiles it, no GPU on that runner).
- **Windows**: CPU (VALIDATED_CI, real per-PR generation + Task
  Scheduler lifecycle on `windows-latest`). No GPU evidence. No
  official package yet.
- **macOS**: Metal (VALIDATED_CI, real per-PR generation + launchd
  lifecycle on `macos-14`, a real but paravirtualized GPU device — no
  performance claim). No official package yet.

## Client compatibility

`docs/client-compatibility.md`. Real (VALIDATED_REAL): the official
Python (3.8.0) and Node.js (7.10.0) `openai` SDKs, and curl. Open
WebUI: not validated this release (real, disclosed host-memory-
pressure/shared-Docker-container constraint on the maintainer's dev
host, not a MEMBRANE limitation). Continue: configuration-format
checked, not run end-to-end (no VS Code environment available).

## API

An OpenAI-compatible **chat API subset** — `docs/api-contract.md` has
the exact field table. Supported: `model`, `messages`, `stream`,
`stream_options.include_usage`, `max_tokens`/`max_completion_tokens`,
`stop`. Explicitly rejected (400, never silently dropped):
`tools`/`tool_choice`, non-default `response_format`. New this release:
`CTX_TOO_SMALL_FOR_PROMPT` (see "Long-context fix" below) and a real
`created` field on every `GET /v1/models` entry (fixes a genuine Python
SDK compatibility bug — `openai.types.model.Model` declares it
required).

## Streaming, stop sequences, and cancellation

Real Server-Sent Events streaming, real `stop`-sequence early
termination (a genuine halt of generation, reusing the same mechanism
real client-disconnect cancellation already used — not a post-hoc
truncation of output already fully generated), real client-disconnect
cancellation. A real bug in the interaction between these two was found
and fixed this Mega Phase (D7): the stop-sequence implementation
initially reused the same flag the response queue uses to decide "the
client is gone, stop delivering," which silently dropped the real
terminal frame a still-connected client was waiting for. Fixed and
re-verified this release.

## Model lifecycle: `membrane use`/`setup`/`doctor`/uninstall

`membrane use MODEL` — installed, not-installed (consent-gated),
declined consent, `--yes`, service stopped, service running, already
active, switch failure (with real recovery), missing file: every case
re-verified this release, output coherent in each. A normal chat
request naming a different, installed model also live-switches the
server — no `membrane use`/admin-endpoint call required from the
client at all. `membrane model uninstall` refuses to delete the
currently active model and clears a dangling default. `membrane
doctor` flags a default/active-model mismatch or a missing selected
model file.

## Long-context fix (Section 15)

A real PR D7 finding: an extremely oversized prompt could make the
server unresponsive for
minutes on a memory-constrained host, with no fast, structured
rejection. Root-caused this release: prompt tokenization (a real,
per-request vocab-only model load) happened before the existing,
already-correct "does this fit the model's real max context" check,
so the expensive part ran even for a prompt that could never possibly
fit. Fixed with a cheap, pre-tokenization byte-length guard (real byte-
level BPE/SentencePiece tokenizers can never produce more tokens than
input bytes, so a generous 6x safety margin against the model's real,
cached `model_max_context` catches only the unambiguous, extreme case,
with zero realistic false-rejection risk) — real, re-verified this
release: a ~150 KB prompt now fails in 0.16s with a structured
`CTX_TOO_SMALL_FOR_PROMPT` error instead of hanging; a legitimate
~4,000-token prompt is unaffected.

## Upgrade from v0.4.0

No breaking change, no schema migration. See
`docs/upgrade-v0.4-to-v0.8.md`. Real, tested this release: a real
`apt install` of the actual v0.4.0 GitHub release asset, register a
real model, upgrade in-place to v0.8.0, confirm the registry and
server config are byte-for-byte unchanged, confirm the new `membrane
use` command works immediately against the pre-existing registry
entry, clean removal.

## Security/privacy

Unchanged posture: loopback-only HTTP by default, no authentication of
its own (a real `Authorization` header is tolerated, never treated as
auth), no telemetry, HTTPS-only downloads, no shell/`system()` calls
anywhere in the download/service code paths (re-audited this release).

## Validation

91 real ctest cases (one, `test_mem_guard`, is a long-documented,
real host-memory-pressure-driven flake on the maintainer's own dev
host, unrelated to this release — confirmed transient by immediate
re-run). ASan/UBSan and TSan clean on the server/streaming/lifecycle
modules. Real CI: Linux (multiple sanitizer/build configurations),
Windows (`windows-latest`, real generation + service), macOS
(`macos-14`, real generation + service), CUDA (compile-only on CI, real
generation on the maintainer's own local hardware), CodeQL. See
`results/release-v0.8.0/readiness.json` for the complete evidence
breakdown.

## Known limitations

See `README.md`'s own "Known limitations" section — kept in one place
rather than duplicated, so it can never silently drift out of sync
with what ships.
