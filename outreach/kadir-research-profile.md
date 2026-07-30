# Research profile — Kadir Eren Altıntaş

A short, factual introduction for anyone deciding whether to respond to
an outreach email or review the MEMBRANE repository. No academic degree,
institutional title, or publication is claimed here — none exist yet,
and this document is deliberately written to not imply otherwise.

## Background

- Student at **42 İstanbul**, part of the 42 network's project-based,
  peer-learning software engineering curriculum (no lectures, no
  professors — project- and peer-review-based, C/C++-heavy in its early
  curriculum).
- Primary technical background: **C/C++ and systems-level development**
  — memory management, concurrency, low-level performance work. MEMBRANE
  itself (the project this outreach is about) is written from scratch in
  C11 (core library) and C++17 (simulators/tools), plus SystemVerilog
  RTL for the FPGA datapath.
- Prior relevant experience: a brief background involving **İHA-1**
  (unmanned aerial vehicle) work — mentioned here only because it
  involved real-time/embedded systems experience relevant to this
  project's hardware ambitions, not elaborated further since it is not
  the focus of this research.
- **Sole author of MEMBRANE** — every line of code, every document, and
  every benchmark artifact in the repository was authored or directed by
  Kadir; there are no co-authors or additional contributors as of this
  writing.

## Research interests

- KV-cache memory management for LLM inference: precision tiering,
  exact vs. approximate retrieval, and where each approach's tradeoffs
  actually show up in measurement rather than intuition.
- The boundary between software simulation and physical hardware
  validation — specifically, being honest about which claims a
  simulator can and cannot support, and building the tooling
  (`scripts/verify-results.py`, `paper/scripts/verify-paper.py`) to
  enforce that boundary automatically rather than by discipline alone.
- Synthesizable RTL design for numerical/quantization datapaths, and
  bit-exact verification methodology between a CPU reference
  implementation and its hardware counterpart.
- Near-memory and CXL-based memory-tiering architectures for AI
  workloads, at the level of "what would actually bind first" rather
  than a general survey of the space.

## What kind of collaboration is being sought

Not funding, not a job, not co-authorship on unrelated work. Specifically:

1. **Hardware access** — FPGA board + synthesis toolchain
   (Vivado/Quartus), even time-boxed or remote, to run real
   place-and-route and, if that succeeds, a real board bring-up. See
   `outreach/membrane-technical-brief.md` and
   `docs/phase8-hardware-validation-plan.md`.
2. **CXL platform access** — a real CXL Type-3 memory device or an
   accessible emulation/prototyping platform, to check whether this
   project's near-memory simulation resembles real device behavior.
3. **Engineering input** on board-specific integration questions (DMA
   framework choice, AXI clocking constraints) from anyone with real
   board-bring-up experience — this project's RTL is deliberately kept
   interface-only and vendor-IP-free (`hardware/README.md`) so it can be
   adapted without redistributing anything proprietary.
4. **Critical review** — anyone willing to point out where a claim in
   `paper/main.md` or `docs/results-summary.md` is overstated, or where
   the related-work comparison (`paper/related-work-matrix.md`) is
   missing something relevant, is exactly the kind of engagement this
   project is looking for, independent of whether hardware access
   follows.

## Contact and links

- Repository: https://github.com/kadireren7/membrane
- Paper draft: `paper/main.md` / `paper/main.tex`
- Quick demo (no model download, ~25s): `scripts/demo.sh --quick`
- Contact: see the repository's `SUPPORT.md` / `SECURITY.md` for the
  current maintainer contact address.
