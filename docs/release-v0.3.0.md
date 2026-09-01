# MEMBRANE v0.3.0 — release notes

`v0.3.0` is the first stable release since `v0.2.0`. It carries
`v0.3.0-rc3`'s real product-behavior change (Qwen2 compressed-KV
support), plus three post-RC3 phases of CLI/first-run/packaging polish
(Phases 28–30) that shipped **no planner or runtime policy change** —
presentation, diagnostics, and distribution only. `v0.2.0`'s CPU-only
default behavior is unchanged throughout.

This is a **stable-readiness phase** (Phase 31): feature-frozen, release
metadata/docs/evidence/packaging only, no new product capability added
in this phase itself.

## Highlights

- **`--auto` joint planning**: one policy pipeline resolves GPU layer
  offload, KV precision (`native`/`q8`/`q5`/`adaptive`), and KV
  placement (`default`/`gpu`/`cpu`/`auto`) together, before generation
  starts, with an inspectable plan (`--plan-only`) and machine-readable
  JSON (`schema_version: 1`).
- **Bounded apply-time fallback**: if the primary plan can't actually
  be instantiated at runtime (e.g. GPU memory changed since planning),
  MEMBRANE retries a bounded number of other already-ranked, already-
  legal plans — never a fabricated candidate, never silent
  (`docs/auto-fallback.md`).
- **Qwen2 compressed KV** (carried from RC3, the one real behavior
  change since v0.2.0): `q8`/`q5`/`adaptive` KV precision, previously
  `llama`-only, validated end to end against the real, local
  `Qwen2.5-1.5B-Instruct` fixture. See "Compatibility scope" below for
  the exact, scoped claim — this is not "all Qwen2 models."
- **First-run CLI UX** (Phases 28–30): `--doctor` (first-run
  diagnostics, ending in a concrete next command), `--list-devices` (no
  model needed), `--inspect-model` (GGUF-metadata-only compatibility
  check, no model load, no generation), and a reorganized `--help`
  (QUICK START/EXAMPLES sections).
- **Official Debian-family amd64 `.deb` package** (Phase 29–30):
  `sudo apt install ./membrane_0.3.0_amd64.deb` — Vulkan-enabled,
  proven to run correctly CPU-only, real runtime dependencies
  auto-detected via `dpkg-shlibdeps` (never hand-maintained).
- **JSON diagnostics contract hardened**: a CLI-parse-time failure now
  emits a single JSON error object on stdout whenever `--json` appears
  anywhere in argv (Phase 28 fix), not just once parsing had already
  progressed far enough to notice it.

## What's new in v0.3 (vs. v0.2.0)

Everything below is additive to `v0.2.0`'s CPU-only, native-KV-only,
single-pass behavior — no existing flag was renamed or removed, no
exit code changed meaning, no JSON field was removed.

### Automatic memory planning

`--auto` resolves GPU layer offload, KV precision, and KV placement
together via a joint planner (`docs/joint-planner.md`), instead of
requiring three separately-reasoned-about flags. Explicit flags always
override the field they name (`--auto --kv q8` keeps GPU layers and
placement on auto while pinning precision to `q8`).

### KV precision

`native` (unmodified llama.cpp), `q8` (genuinely `Q8_0`-typed, not a
shadow copy, ≈53.125% of native KV bytes), `q5` (genuinely `Q5_1`-typed,
≈37.5% of native KV bytes), and `adaptive` (MEMBRANE picks exactly one
of `q8`/`q5` for the entire context based on real memory pressure,
never a per-layer mix, never picks `q5` merely because it's smaller).
See `docs/live-runtime.md`.

### KV placement

A separate axis from precision: `default` (unmodified device
placement), `gpu` (every KV layer on the offloaded model's GPU, or fail
closed), `cpu` (every KV layer in system RAM), `auto` (as many layers
GPU-resident as safely fit, rest in system RAM). Decided once, before
context construction — no runtime migration. See `docs/kv-residency.md`.

### Qwen2 support

See "Compatibility scope" below.

### Fallback behavior

`docs/auto-fallback.md` — bounded, transparent, reported in both human
output (`--verbose`) and JSON (the `fallback` object, including on a
fully-exhausted error response).

### CLI / first-run UX

`--doctor`, `--list-devices`, `--inspect-model`, reorganized `--help`,
actionable errors (e.g. `--auto` without `--ctx` names the actual
missing requirement instead of a generic flag-based message). A real
pre-existing CLI bug was also fixed this cycle: `--plan-only` with an
explicit `--ctx` no longer requires a throwaway `--prompt` it never
reads (auto-sizing, the only real use of prompt content on that path,
never applies once `--ctx` is explicit).

### Installation

Three paths, easiest first — see `docs/install.md`:

```bash
# Option A: official .deb (Debian-family, amd64)
sudo apt install ./membrane_0.3.0_amd64.deb
membrane-run --doctor

# Option B: build from source, system install
cmake -S . -B build-vulkan -DMEMBRANE_ENABLE_LLAMA=ON -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j2 --target membrane-run
sudo cmake --install build-vulkan

# Option C: build from source, user-local (no root)
cmake --install build-vulkan --prefix "$HOME/.local"
```

### Official `.deb` package

Policy (Phase 30, `results/release-artifacts/manifest.json`): the ONE
official package is `membrane_<version>_amd64.deb` — Vulkan-enabled,
proven to run correctly CPU-only with real container tests (no GPU
passthrough): `--doctor`/`--list-devices` handle zero visible devices
cleanly, generation still works, GPU offload stays fully opt-in. A
`membrane-cpu_<version>_amd64.deb` variant also exists but is a
CI-validation/build-your-own artifact, not an official release asset.

### Diagnostics / JSON

Every mode (`run`, `plan`, `list-devices`, `doctor`, `inspect-model`,
and CLI-parse-time failures) emits exactly one JSON object on stdout
when `--json` is given, `schema_version: 1` throughout, no field
removed since `v0.2.0`.

## Compatibility scope

Current `docs/compatibility.json` (verified by
`scripts/verify-compatibility.py`, checked as part of this release's
own readiness evidence): **26 total rows — 19 SUPPORTED, 6 UNSUPPORTED,
1 NOT_YET_VALIDATED.**

Precise, model-scoped Qwen2 claim — do not generalize beyond this:
**`Qwen2.5-1.5B-Instruct`**, `q8`/`q5`/`adaptive` KV precision, on
**CPU** and **Vulkan**, with independently-resolved KV placement, is
`SUPPORTED` (`docs/compatibility.json` MC-17/MC-18/MC-19; real
evidence: `results/compat-expansion/validation.json`, 8 REAL rows).
This is **not** "all Qwen2 models supported" (only
`Qwen2.5-1.5B-Instruct` was tested) or "all llama.cpp architectures
supported" (the compressed-KV allowlist, `compat_check.c`, is exactly
`{llama, qwen2}`).

## Performance claims

**None.** No performance-changing code has shipped in any phase behind
this release (Phase 25 investigated the two highest-priority measured
bottlenecks — model load, decode throughput — and found both fully
third-party cost, no MEMBRANE-owned hot loop; `docs/performance-
optimization.md`). This document, and every other release-facing doc,
does not say "faster," "performance optimized," or "improved inference
speed" — the profiling/attribution work (Phase 24–25) added
observability, not speed.

## Validation

Committed, `scripts/verify-release-v0.3.0.py`-checked evidence:
`results/release-v0.3.0/readiness.json` — see that file for exact
commit, CI run IDs, and per-suite pass counts. Summary:

- `scripts/verify-results.py`: core + every chained validator (exact
  counts in the readiness evidence file).
- Full `ctest --output-on-failure` across the llama-free, llama-enabled
  (ASan/UBSan), and Vulkan build configurations.
- Official `.deb` package (`membrane_0.3.0_amd64.deb`) built, installed,
  exercised, and cleanly removed inside a fresh, isolated container
  (no GPU passthrough) — real end-to-end CPU generation confirmed.
- Real local Vulkan smoke on the maintainer's own development host
  (real AMD Radeon RADV integrated GPU + NVIDIA GTX 1650).
- GitHub CI and CodeQL: SUCCESS — exact run IDs in the readiness
  evidence file.

## Known limitations

- **Independent multi-host validation remains limited.** All real-run
  evidence (CPU, NVIDIA Vulkan, AMD Vulkan RADV) comes from the
  maintainer's own development host, plus container-based packaging
  checks (a different environment, but not a different physical
  machine/GPU). This is disclosed honestly, not hidden, and does not
  block this release — see "External validation status" below.
- Qwen2 support is scoped to `Qwen2.5-1.5B-Instruct` specifically, not
  "all Qwen2 models."
- CUDA is not a MEMBRANE product path (evaluated, deferred — no
  validation hardware available, not assumed to be a bad fit).
- AMD/Intel GPU validation is real but narrow (one AMD RADV integrated
  device, one physical host).
- VRAM/estimate accuracy depends on model, context size, and hardware —
  never an absolute OOM guarantee.
- No custom GPU kernels — MEMBRANE wires up explicit, product-level
  control over the existing, unmodified `ggml`/llama.cpp Vulkan
  backend.
- No FPGA/CXL hardware claim (companion `kadireren7/membrane-research`
  repository only, simulation/synthesis-tool-proxy evidence).
- Debian-family Linux, amd64 only for the official package — no
  Fedora/RPM, Arch, Flatpak, Snap, AppImage, macOS, or Windows claim.

## External validation status

**Still limited — disclosed explicitly, not a release blocker for this
version.** All real backend evidence (CPU + NVIDIA-Vulkan +
AMD-Vulkan-RADV) comes from **one** physical developer host — three
real backends, one real machine — plus real, isolated container
validation of the packaged `.deb` (a different execution environment,
but the same physical host). Independent-host validation (a genuinely
different physical environment, ideally a different distro and GPU
vendor/driver stack) has not yet happened. The maintainer has made an
explicit, informed decision to ship `v0.3.0` stable now rather than
wait for it — this is a documented known limitation of the current
stable release, not a hidden gap.

## Upgrade notes

No breaking CLI or JSON-schema change since `v0.2.0`. Every new flag
(`--auto`, `--plan-only`, `--kv-placement`, `--doctor`,
`--list-devices`, `--inspect-model`) is additive; every new JSON field
is additive (`schema_version` has stayed `1` throughout). The one
narrow CLI validation relaxation this cycle (`--plan-only` no longer
requires `--prompt` when `--ctx` is explicit) only ever turns a
previously-rejected invocation into a working one — no previously-valid
invocation is affected.

## Research / advanced notes

Full experiment records, negative results, and FPGA/CXL research
(simulation and synthesis-tool proxies only, no physical hardware) live
in the companion repository
[kadireren7/membrane-research](https://github.com/kadireren7/membrane-research),
with SHA256-verified provenance back to this repository. Mechanism
detail: `docs/live-runtime.md` (KV precision), `docs/kv-residency.md`
(KV placement), `docs/joint-planner.md` (`--auto`'s planner),
`docs/auto-fallback.md` (bounded fallback), `docs/compatibility.md`
(full compatibility matrix).
