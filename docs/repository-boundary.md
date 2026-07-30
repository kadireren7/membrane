# Public/private repository boundary

This document defines the intended boundary between the public
`kadireren7/membrane` repository (this one) and a possible future
private companion repository, referred to below as `membrane-labs`.

**`membrane-labs` does not exist yet.** No private repository has been
created. This document is a plan for how the boundary *would* work if
and when one is, so that the decision (and its consequences for
reproducibility) is made deliberately rather than improvised later. Do
not read anything in this document as a claim that private, unreleased
work already exists somewhere else — it doesn't, as of this writing.

## Public `kadireren7/membrane` — what belongs here

- **Verified research implementation.** Every simulator, quantizer,
  RTL module, and CLI tool actually described in `docs/`, `paper/`, and
  `README.md` — the real, working, currently-verified codebase.
- **Committed benchmark artifacts.** Everything tracked in
  `benchmarks/MANIFEST.json` — real CSV/JSONL results, SHA-256-verified,
  with a generating command recorded for each.
- **Reproducible scripts.** `scripts/`, `paper/scripts/` — everything
  needed to regenerate figures, tables, and verification reports from
  the committed artifacts, runnable by anyone who clones the repo.
- **Released RTL.** `rtl/` and `hardware/vendor-wrapper/` — the
  cosimulation-verified, vendor-IP-free quantization datapath and its
  integration skeletons.
- **Paper and documentation.** `paper/`, `docs/` — the manuscript,
  phase-by-phase research record, and this project's disclosure
  discipline in full.
- **Negative findings.** Every null/negative result this project has
  measured stays here, with the same visibility as positive results —
  moving a negative finding to a private repo to "clean up" the public
  narrative would violate this project's own stated disclosure
  discipline and is explicitly not permitted (see Rules, below).
- **Public bug fixes and reproduction improvements.** Anything that
  makes the public repository more correct, more reproducible, or more
  honest belongs here, always — regardless of whether related
  unreleased work exists privately.

## Private `membrane-labs` (planned, not yet created) — what would belong there

- **Unreleased experiments.** Work that hasn't reached the verification
  bar this project holds its public claims to (see
  `docs/research-release-freeze.md`).
- **Vendor-specific FPGA/Vivado/Vitis projects.** Board-specific project
  files, IP integration state, and vendor toolchain artifacts that
  would either be too large to usefully commit publicly or would
  entangle this project with a specific vendor's redistribution terms.
- **Board constraint files.** Pin/timing constraint files (`.xdc`,
  `.sdc`, or equivalent) tied to a specific physical board a
  collaborating lab provides access to.
- **Raw hardware logs.** Full, unprocessed logs from real board runs
  (per `hardware/experiment-protocol.md`) before they've been reduced
  to a `hardware/results-schema.json`-conformant record and reviewed
  for public inclusion.
- **Experimental divider/pipeline variants.** Alternative
  `membrane_fp_divider` re-pipelining attempts, partial redesigns, or
  anything exploratory that hasn't yet passed the same bar the current
  public RTL has (cosimulation + yosys elaboration).
- **Unpublished predictors/codecs.** Any new retrieval predictor,
  quantization codec, or policy variant that hasn't been measured
  against this project's existing negative-result discipline yet.
- **Large-model experiments.** Evaluation at model scales beyond
  SmolLM2-135M/-360M, before the same quality/claim-audit rigor this
  project applies today has been applied to those results too.
- **Collaboration-sensitive materials.** Anything a specific lab or
  company shares under an understanding of limited distribution (e.g.
  board-specific configuration a vendor asked not to be made public).

## Rules

1. **Public sources are never deleted or degraded to make room for
   private work.** If a public artifact, script, or doc is superseded,
   it is replaced with something at least as reproducible, or the
   change is disclosed — reproduction must never break silently.
2. **`membrane-labs` must explicitly reference the public baseline
   commit SHA it started from.** Any private work building on this
   project's public state names the exact commit
   (e.g. `58ec90b`, or whatever the frozen baseline is at
   `docs/research-release-freeze.md`'s time of writing) so the
   relationship between public and private state is always traceable.
3. **No file silently diverges between the two repositories.** If the
   same file (e.g. a shared RTL module) exists in both, one of them is
   the authoritative source and the other explicitly says so — never
   two silently-drifting copies of the same claimed artifact.
4. **Verified, publishable results move from private to public in a
   controlled way**, not automatically: a private experiment that
   reaches this project's public verification bar (real measurement,
   sourced, claim-audited, negative results included honestly) is
   promoted to the public repo through the same process as any other
   change here — reviewed, verified, and committed under Kadir's
   direction, not silently synced.
5. **This document does not claim `membrane-labs` exists.** If and when
   it is created, this document should be updated to link it explicitly
   and to record the actual baseline commit used — until then, this is
   a plan, not a status report.
