# README media (placeholder)

No visual media exists yet. This directory documents what the top-level
`README.md` expects to eventually embed, so a future addition has a
clear bar to meet rather than being added ad hoc. This does not block
any pull request — the README works without it.

## Expected asset

- **A short demo clip or static screenshot** of `scripts/demo.sh --quick`
  running end to end (build, quant parity, FPGA Verilator cosimulation),
  or of `docs/architecture.md`'s diagram set.
- **Aspect ratio**: 16:9.
- **Size**: lightweight, under 8 MB preferred — this is a README asset
  loaded on every repository view, not a release artifact.
- **No third-party watermark** — anything generated with an external
  tool must be watermark-free before it's committed here.
- **No fake or implied hardware** — per this project's own limitations
  disclosure (`README.md` "Limitations"), no image or animation may
  depict or imply a physical FPGA board or CXL device that doesn't
  exist in this project's environment.
- **Animation is optional; a static fallback is required** — if an
  animated clip is added, a static image (first-frame or equivalent)
  must also exist for viewers/contexts that don't render video/GIF.

## Adding the real asset

1. Record or capture the real thing — a real terminal running
   `scripts/demo.sh --quick`, unedited, or a real render of an existing
   diagram in `docs/architecture.md`. Do not stage a fake or idealized
   run.
2. Check it against every constraint above.
3. Add it here, then reference it from `README.md` directly below the
   badges (per that file's own intended structure).
4. Update this document's "Expected asset" section to describe what was
   actually added, so this file never describes a stale placeholder next
   to a real, different asset.
