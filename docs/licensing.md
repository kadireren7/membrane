# Licensing

This document maps the license boundaries within this repository. It is
written by the project maintainer, not a lawyer — where something is
genuinely unclear, that is stated explicitly rather than asserted with
false confidence. If you need legal certainty for a specific use case,
consult your own counsel; do not treat this document as legal advice.

## MEMBRANE's own code

Everything under `include/`, `src/`, `tools/`, `tests/`, `rtl/`,
`scripts/`, `docs/`, and `paper/` that is not explicitly called out below
is original code/prose written for this project and is licensed under the
**Apache License 2.0** — see [LICENSE](../LICENSE) at the repository root.
This is the license that applies if you fork, reuse, or build on
MEMBRANE's own source.

## `third_party/llama.cpp` (git submodule)

Vendored as a git submodule pointing at the upstream
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) repository
(see [.gitmodules](../.gitmodules)). Its own `LICENSE` file (present in
the submodule checkout) is the **MIT License**, copyright the ggml
authors. MEMBRANE does not relicense it, does not distribute pre-built
binaries of it, and links against it only when
`-DMEMBRANE_ENABLE_LLAMA=ON` is explicitly set (see
`CMakeLists.txt`'s `MEMBRANE_ENABLE_LLAMA` option and
`membrane_ggml_quant`/`test_ggml_quant_parity` targets).

The submodule itself vendors at least one further third-party license
(`third_party/llama.cpp/licenses/LICENSE-jsonhpp`, for a bundled JSON
library) — that license applies to that specific vendored file, not to
MEMBRANE's own code, and is upstream's responsibility to track, not
this project's.

**What is unverified here**: this project has not independently audited
every file inside the `llama.cpp` submodule for license consistency with
its stated MIT license — that is upstream's responsibility. If you
redistribute a build that links `third_party/llama.cpp`, you are
responsible for satisfying its MIT license's attribution requirement
yourself.

## Model artifacts (`models/*.gguf`)

**Not included in this repository** — `*.gguf` is gitignored (see
`.gitignore`). The SmolLM2-135M-Instruct and SmolLM2-360M-Instruct
checkpoints this project's docs reference are real, public models
published by HuggingFaceTB on Hugging Face; obtaining and using them is
governed by **their own upstream license terms on Hugging Face**, not by
this repository. MEMBRANE does not distribute model weights and makes no
license claim about them — see
[docs/reproduction.md](reproduction.md) Level 2 for how to obtain and
convert them yourself.

## Captured benchmark traces (`benchmarks/cxl-sim/traces/*.kvtrace`,
`*.attntrace`, `*.attntrace.manifest.json`)

These are committed (see the Phase 6.1 note in `.gitignore`) and contain
**derived numerical data** (quantized/raw KV-cache tensor values and
attention scores) produced by running the above models' forward pass on
short, non-sensitive prompts (`benchmarks/kv/prompts/*.txt`, all authored
for this project, not copied from any external corpus). These are not a
redistribution of the model weights themselves — no weight tensors are
present in these files, only the runtime activations they produced on
specific inputs.

**What is unverified here**: whether a captured activation trace is
legally "a derivative work" of the model that produced it is a genuinely
unsettled question in ML licensing generally, not something this project
resolves. These traces are committed under the same Apache 2.0 license as
the rest of the repository's own code/data on the position that runtime
activations on short authored prompts are the project's own measurement
data — but this is a stated position, not a verified legal conclusion.
If your use case depends on this distinction, seek your own legal advice.

## `benchmarks/results/` (Phase 2–4 local artifacts)

Most files under `benchmarks/results/` are **not committed** to this
repository (gitignored, with `README.md` and `phase1-store-summary.md` as
the only tracked exceptions — see `.gitignore`). Where present locally,
they are derived numerical output in the same category as the traces
above.

## Synthesizable RTL (`rtl/`)

Original SystemVerilog written for this project, licensed the same as
the rest of MEMBRANE's own code (Apache 2.0). It has not been
placed/routed on any FPGA vendor's toolchain, so no vendor IP-core
license question arises — see
[README.md's Technical limitations](../README.md#technical-limitations).

## Locally-extracted development tools (`tools/.local-*/`)

Not part of this repository (gitignored — see `.gitignore`'s Phase 5.3
note). These are Debian-packaged copies of Verilator/yosys/nextpnr/
Icarus Verilog obtained via `apt-get download` in an environment without
root access to install them system-wide, used only as local development
tooling. Each is under its own upstream open-source license (Verilator:
LGPL-3.0; yosys: ISC; nextpnr: MIT/ISC family; Icarus Verilog: GPL-2.0 /
LGPL-2.1) — none of their source or binaries are distributed by this
repository.

## Summary table

| Component | License | Distributed by this repo? |
|---|---|---|
| MEMBRANE's own code/docs/RTL | Apache 2.0 | Yes |
| `third_party/llama.cpp` | MIT (upstream) | As a git submodule reference only |
| Model weights (`.gguf`) | Upstream (Hugging Face) | No — gitignored |
| Captured traces (`.kvtrace`/`.attntrace`) | Apache 2.0 (stated position, see caveat above) | Yes |
| `tools/.local-*/` dev tools | Various upstream OSS | No — gitignored |
