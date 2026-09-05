# MEMBRANE v0.4.0 — release notes

`v0.4.0` is the first stable release since `v0.3.0`. It carries three
PRs of productization/distribution/hardening work (Mega Phase C,
"Productization, Distribution, Hardening, and v0.4 Stable"): **no
planner or runtime policy change** — `v0.3.0`'s KV precision/placement/
joint-planning/fallback behavior is completely unchanged. This release
is about turning an already-correct runtime into a coherent,
installable, diagnosable, upgradeable product.

## Highlights

- **One-command onboarding**: `membrane setup --model model.gguf`
  registers the model, sets it as default, installs and starts the
  background service, and verifies the endpoint answers — replacing a
  previously-required three-command sequence (`model add`, `service
  install`, `service start`) new users had no guidance through. Driven
  by a real, fresh-container first-product audit, not invented ahead of
  evidence (`results/product-onboarding/validation.json`).
- **`membrane doctor`**: a unified OK/WARN/ERROR diagnostic spanning
  hardware, model registry, config, service, and live HTTP health —
  the first command that answers "is everything about my install
  actually working," not just one narrow slice of it.
- **Release-supply-chain maturity**: reproducible `.deb` builds
  (`SOURCE_DATE_EPOCH`-pinned, confirmed bit-identical across repeated
  builds from the same commit), a real CycloneDX-shaped SBOM
  (`scripts/generate-sbom.py`), a release automation script
  (`scripts/build-release.sh`) that never touches git, and a real
  registry schema-versioning gap found and fixed (the model registry
  now fails closed on an unrecognized `schema_version`, matching the
  server config's own pre-existing guarantee — `docs/schema-
  versioning.md`).
- **API contract frozen and doc-checked**: `docs/api-contract.md`
  states the "compatible subset, never complete" scope explicitly, a
  `/v1/` versioning policy, and a completeness check
  (`scripts/verify-api-contract.py`) that keeps `docs/server.md`'s
  error-code table and the real server code from silently drifting
  apart.
- **Real soak/concurrency testing + a second real client**: bounded
  resource-stability soak testing, a real external-process concurrency
  soak around the bounded-admission limit, `test_server`'s existing
  16-thread concurrency test reconfirmed clean under a real
  `-DMEMBRANE_ENABLE_TSAN=ON` build, and a real second client (Node's
  built-in `fetch`) validating protocol compliance beyond the existing
  Python `openai` SDK evidence (`docs/client-compatibility.md`).
- **Privacy/security source audit**: a real, grep-based confirmation
  (not a restatement of intent) that no prompt/message content is ever
  logged, no external network call exists anywhere in this codebase,
  and no error response leaks a filesystem path
  (`docs/privacy-security-audit.md`).

## What's new in v0.4 (vs. v0.3.0)

Everything below is additive to `v0.3.0`'s behavior — no existing flag
was renamed or removed, no exit code changed meaning, no JSON field was
removed, no `/v1/` endpoint's existing contract changed.

### `membrane setup` / `membrane doctor`

See `docs/product-onboarding` evidence and the README's own "As a local
OpenAI-compatible backend" section, which now leads with `membrane
setup` as the recommended path (the manual three-command sequence
still works, documented as the "under the hood" explanation).

### Release supply chain

`docs/release-supply-chain.md` — package-content audit (no content
changes needed, already minimal), a real container test proving a
package upgrade/removal never touches user data (byte-identical
registry/config before and after), the reproducible-build fix, the
SBOM, and the disclosed signed-tag policy (no signing identity is
configured on this project's own release host, so this tag is real,
annotated, and **unsigned** — disclosed here, never faked).

### API contract, hardening, and privacy audit

`docs/api-contract.md`, `docs/soak-and-concurrency-testing.md`,
`docs/client-compatibility.md`, `docs/privacy-security-audit.md` — see
each for full detail and real evidence.

## Installation

```bash
# Option A: official .deb (Debian-family, amd64)
sudo apt install ./membrane_0.4.0_amd64.deb
membrane setup --model /path/to/model.gguf
membrane status

# Option B: build from source, system install
cmake -S . -B build-vulkan -DMEMBRANE_ENABLE_LLAMA=ON -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j2 --target membrane-run membrane
sudo cmake --install build-vulkan

# Option C: build from source, user-local (no root)
cmake --install build-vulkan --prefix "$HOME/.local"
```

See `docs/upgrade-v0.3-to-v0.4.md` if you already have `v0.3.0`
installed.

## Compatibility scope

Unchanged since `v0.3.0`: current `docs/compatibility.json` still
covers the same 26 rows (19 SUPPORTED, 6 UNSUPPORTED, 1
NOT_YET_VALIDATED) — no new architecture/model was validated this
release, since this phase was product/API hardening, not compatibility
expansion. `Qwen2.5-1.5B-Instruct` remains the precise, model-scoped
`SUPPORTED` claim for Qwen2 compressed KV — still not "all Qwen2
models."

## Performance claims

**None.** No performance-changing code has shipped in this release —
consistent with `v0.3.0`'s own position (Phase 25). This release's own
soak/concurrency testing (`docs/soak-and-concurrency-testing.md`)
measures resource *stability* (RSS/thread/FD bounds), explicitly never
throughput, and is not a performance claim of any kind.

## Validation

Committed, `scripts/verify-release-v0.4.0.py`-checked evidence:
`results/release-v0.4.0/readiness.json` — see that file for exact
commit, CI run IDs, and per-suite pass counts. Summary:

- `scripts/verify-product-onboarding.py`,
  `scripts/verify-release-supply-chain.py`,
  `scripts/verify-api-contract.py`: real evidence for each of PRs
  C1/C2/C3, all currently passing.
- Full `ctest --output-on-failure` across build configurations (see
  each PR's own test-plan section for exact counts).
- Real `.deb` package (`membrane_0.4.0_amd64.deb`) built, installed,
  exercised, and cleanly removed inside a fresh, isolated container —
  real end-to-end CPU generation confirmed, real upgrade/data-
  preservation test passed.
- GitHub CI and CodeQL: SUCCESS — exact run IDs in the readiness
  evidence file.

## Known limitations

- **Independent multi-host validation remains limited** — unchanged
  disclosure from `v0.3.0`, still true: all real backend evidence
  comes from the maintainer's own development host.
- Under this project's own severe, shared-host memory pressure, a
  normally-fast host-memory-guard rejection can be slow enough to time
  out a client — a real, disclosed environmental finding from this
  release's own soak testing, not addressed with an architecture
  change this phase (`docs/soak-and-concurrency-testing.md`).
- A non-instruct model with no chat template is only caught at
  chat-completion time, not proactively warned about at `model add`/
  `setup` time (Mega Phase C's own first-product audit finding,
  `results/product-onboarding/validation.json`).
- Open WebUI and OpenAI-compatible editor plugins remain disclosed
  configuration guidance, not independently tested this release either
  (`docs/client-compatibility.md`).
- Release tags are real and annotated but **unsigned** — no GPG signing
  identity is configured on this project's release host, disclosed
  rather than faked.
- Every `v0.3.0` known limitation not explicitly superseded above still
  applies unchanged (Qwen2 scope, no CUDA product path, narrow AMD/
  Intel GPU validation, no FPGA/CXL hardware claim, Debian-family
  amd64-only official package) — see `docs/release-v0.3.0.md`.

## External validation status

Unchanged from `v0.3.0` — still limited, still disclosed explicitly,
still not a release blocker. See `docs/release-v0.3.0.md`'s own
"External validation status" section; nothing about this changed this
release.

## Upgrade notes

See `docs/upgrade-v0.3-to-v0.4.md`. No breaking CLI or JSON-schema
change since `v0.3.0`; every new command (`membrane setup`, `membrane
doctor`) is additive, and every existing command's behavior is
unchanged.

## Research / advanced notes

Full experiment records, negative results, and FPGA/CXL research
(simulation and synthesis-tool proxies only, no physical hardware) live
in the companion repository
[kadireren7/membrane-research](https://github.com/kadireren7/membrane-research),
with SHA256-verified provenance back to this repository.
