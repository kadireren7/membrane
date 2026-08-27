# README assets

Every SVG here is embedded in the top-level `README.md`. This file records
where each one's content came from, so a number or claim in a diagram can
always be traced back to something real.

| Asset | Kind | Source |
|---|---|---|
| `membrane-hero.svg` | conceptual, hand-authored | Illustrative — labels the three decisions MEMBRANE resolves (GPU layers, KV precision, KV placement); not a captured run |
| `membrane-auto.svg` | conceptual, hand-authored | Illustrative — `--auto` override semantics from `tools/membrane-run/product_cli.cpp` / `main.cpp`, not a captured run |
| `membrane-flow.svg` | conceptual, hand-authored | Illustrative — planning pipeline shape from `tools/membrane-run/main.cpp` (`resolve_gpu_config`, `resolve_cpu_adaptive_kv`, `resolve_kv_placement`) |
| `membrane-precision-placement.svg` | conceptual, hand-authored | Illustrative — the `--kv` / `--kv-placement` value sets from `membrane-run --help` |
| `membrane-terminal.svg` | real captured output | `source/plan-example.txt` — exact transcript of `membrane-run --model SmolLM2-135M-Instruct-f16.gguf --prompt "Hello" --ctx 4096 --auto --plan-only`, captured on a real GTX 1650 (Vulkan) host at commit `635d191eeef4775325ead805ea55ceb769f3bf51` |
| `membrane-capacity.svg` | generated from committed evidence | `scripts/generate_readme_assets.py` reading `results/v0.3/kv-residency-productization/capacity_uplift.json` |
| `membrane-q8-tradeoff.svg` | generated from committed evidence | `scripts/generate_readme_assets.py` reading `results/v0.3/gpu-vulkan-validation.json` |

## Regenerating the data-driven charts

```bash
python3 scripts/generate_readme_assets.py
```

Reads only already-committed files under `results/`, writes
`membrane-capacity.svg` and `membrane-q8-tradeoff.svg` deterministically (no
network access, no modification of evidence). Running it twice produces
byte-identical output — verified before each commit that touches evidence or
the generator.

## Design system

Self-contained dark panels (`#0d1117` background, `#161b22` cards,
`#30363d` borders) so every asset renders identically in GitHub light and
dark mode — nothing depends on a transparent background or the surrounding
theme. One accent color (`#5ee6c8`) for MEMBRANE/success, muted red
(`#f85149`) only for an explicit failure state in the capacity chart. No
unicode arrow/check/cross glyphs (`→ ✓ ✕`) — they fall back to missing-glyph
boxes on some renderers; drawn paths or ASCII (`->`) are used instead.

## Provenance file

`source/plan-example.txt` holds the raw transcript behind
`membrane-terminal.svg`, plus the exact command and host/commit it was
captured from — so the rendered card can always be checked against the real
output it claims to show.
