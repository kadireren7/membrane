# MEMBRANE brand

Positioning sentence:

> **Hardware-aware memory intelligence and control for local LLM inference.**

Supporting line: *Plan, observe, advise, apply, and adapt local model workloads across runtimes.*

MEMBRANE is infrastructure. The identity is meant to look precise and quiet, not like generic
AI branding. No brain, robot, sparkle, circuit or neon motifs.

## The mark

An **M** drawn as one continuous stroke, standing on a thin **boundary line that has an
opening** directly under the point where the strokes meet.

- The M is the product. The line is the membrane: a boundary between tiers of memory.
- The gap is the aperture: flow across the boundary is controlled, not blocked and not open.
- Teal marks the boundary; everything else is neutral ink. Only one accent carries meaning.

It is built from two straight strokes and two short bars, so it survives at 16 to 32 px
and does not depend on gradients or text. The wordmark is drawn as geometric paths, so no
font is needed.

## Files

All in `docs/assets/brand/`. Plain SVG, no raster, no external fonts, no metadata.

| File | Use |
|---|---|
| `membrane-logo-light.svg` | Mark plus wordmark for light backgrounds. |
| `membrane-logo-dark.svg` | Mark plus wordmark for dark backgrounds. |
| `membrane-logo.svg` | Same logo, switches by `prefers-color-scheme` (use where the page can honor it). |
| `membrane-mark.svg` | Mark alone, switches by `prefers-color-scheme`. |
| `membrane-mark-mono.svg` | Single-color (black) mark for print, stamps, one-color contexts. |
| `membrane-icon.svg` | Square app icon / social avatar / favicon source (dark tile). |
| `membrane-architecture.svg` | The one architecture diagram used in the README. |

The README uses a `<picture>` element so GitHub's own light/dark theme chooses between the
light and dark logo files.

## Palette

| Role | Light background | Dark background | Use |
|---|---|---|---|
| Ink | `#0D1117` | `#E6EDF3` | Mark body, wordmark, text |
| Accent (boundary) | `#0F766E` | `#2DD4BF` | The boundary bars, section accents |
| Tile / dark surface | `#0D1117` | `#0D1117` | App icon tile, dark diagrams |
| Muted text | `#57606a` | `#9da7b3` | Secondary diagram text |

Semantic colors for status, matching the terminal's meaning (the CLI's own colors are not
changed by this page):

| Meaning | Hex |
|---|---|
| Healthy / success | `#3FB950` |
| Warning | `#D29922` |
| Error | `#F85149` |
| Informational | `#58A6FF` |

Keep to one accent. Do not use the semantic colors as brand decoration.

## Usage

- **Minimum size:** mark 16 px (icon tile 24 px); logo 20 px tall. Below that, use the icon.
- **Clear space:** at least the stroke width of the M on every side.
- **Do** use the light logo on light and the dark logo on dark surfaces; use the mono mark
  when one color is all you have.
- **Do** keep the gap under the M visible.
- **Don't** recolor the accent, add gradients, glows or shadows, outline or rotate the mark,
  close the gap in the boundary, stretch it, or set the wordmark in a font.
- **Don't** pair the mark with brain, chip-circuit or robot imagery, and don't use it to
  imply endorsement by llama.cpp, Ollama or vLLM, or support for hardware that is not claimed
  in the [support matrix](support-matrix.md).
