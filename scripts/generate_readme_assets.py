#!/usr/bin/env python3
"""Generate README evidence-chart SVGs directly from committed result JSON.

Reads only already-committed evidence files under results/ -- never fetches
anything, never modifies evidence, never invents a number. Output is
deterministic: running this twice produces byte-identical SVGs.

Usage:
    python3 scripts/generate_readme_assets.py
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"

BG = "#0d1117"
PANEL = "#161b22"
BORDER = "#30363d"
TEXT = "#e6edf3"
MUTED = "#8b949e"
ACCENT = "#5ee6c8"
FAIL = "#f85149"
SANS = "-apple-system,'Segoe UI',Helvetica,Arial,sans-serif"
MONO = "ui-monospace,Menlo,Consolas,monospace"


def esc(s):
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def svg_wrap(width, height, title, alt, body):
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{esc(alt)}">\n'
        f"  <title>{esc(title)}</title>\n"
        f'  <rect width="{width}" height="{height}" fill="{BG}"/>\n'
        f"{body}"
        f"</svg>\n"
    )


def generate_capacity_chart():
    """membrane-capacity.svg from results/v0.3/kv-residency-productization/capacity_uplift.json"""
    data = json.loads(
        (
            ROOT
            / "results/v0.3/kv-residency-productization/capacity_uplift.json"
        ).read_text()
    )
    rows = data["rows"]

    def find(ctx, placement):
        for r in rows:
            if r["n_ctx"] == ctx and r["placement"] == placement:
                return r
        raise KeyError((ctx, placement))

    bars = [
        ("default", find(26500, "default"), "26,500"),
        ("default", find(26800, "default"), "26,800"),
        ("auto", find(28500, "auto"), "28,500"),
        ("cpu", find(28500, "cpu"), "28,500"),
    ]

    width, height = 760, 496
    max_ctx = 28500
    bar_area = 380
    left = 200
    row_h = 54
    section_gap = 44
    top = 116

    def mark_icon(cx, cy, ok):
        """Small drawn check/cross glyph -- avoids unicode font fallbacks."""
        color = ACCENT if ok else FAIL
        if ok:
            path = f"M{cx - 6},{cy} L{cx - 1.5},{cy + 5} L{cx + 6.5},{cy - 6}"
        else:
            path = (
                f"M{cx - 5.5},{cy - 5.5} L{cx + 5.5},{cy + 5.5} "
                f"M{cx + 5.5},{cy - 5.5} L{cx - 5.5},{cy + 5.5}"
            )
        return (
            f'<path d="{path}" fill="none" stroke="{color}" '
            f'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>'
        )

    body = []
    body.append(
        f'<text x="30" y="34" font-family="{SANS}" font-size="13" fill="{MUTED}" '
        f'letter-spacing="1.5">MEASURED — Qwen2.5-1.5B · GTX 1650 · Vulkan · '
        f'native KV · 28/28 GPU weight layers</text>'
    )
    body.append(
        f'<text x="30" y="58" font-family="{SANS}" font-size="15" fill="{TEXT}" '
        f'font-weight="700">KV placement, context capacity at fixed weight offload</text>'
    )
    body.append(
        f'<text x="30" y="{top - 24}" font-family="{MONO}" font-size="11.5" '
        f'fill="{MUTED}">DEFAULT (all-GPU KV)</text>'
    )
    body.append(
        f'<text x="30" y="{top + 2 * row_h + section_gap - 22}" '
        f'font-family="{MONO}" font-size="11.5" fill="{MUTED}">'
        f"MEMBRANE PLACEMENT (auto / cpu)</text>"
    )

    for i, (mode, row, label) in enumerate(bars):
        y = top + i * row_h + (section_gap if i >= 2 else 0)
        ok = row["result"] == "SUCCEEDS"
        color = ACCENT if ok else FAIL
        bw = bar_area * (row["n_ctx"] / max_ctx)
        body.append(
            f'<text x="{left - 16}" y="{y + 22}" text-anchor="end" '
            f'font-family="{MONO}" font-size="13" fill="{TEXT}">{mode}</text>'
        )
        body.append(
            f'<rect x="{left}" y="{y}" width="{bw:.1f}" height="26" rx="4" '
            f'fill="{color}" fill-opacity="0.85"/>'
        )
        body.append(
            f'<text x="{left + bw + 30:.1f}" y="{y + 19}" font-family="{MONO}" '
            f'font-size="13" fill="{color}">ctx {label}</text>'
        )
        body.append(mark_icon(left + bw + 14, y + 13, ok))

    bottom_row_y = top + 3 * row_h + section_gap
    rule_y = bottom_row_y + 40
    body.append(
        f'<line x1="30" y1="{rule_y}" x2="730" y2="{rule_y}" '
        f'stroke="{BORDER}" stroke-width="1"/>'
    )
    caption_lines = [
        "Default fails between ctx 26,500 and 26,800 with a real Vulkan",
        "out-of-device-memory error. --kv-placement auto/cpu succeed at ctx",
        "28,500 in the same test — one tested configuration; see README",
        "for the exact conservative-lower-bound uplift figure.",
    ]
    for i, line in enumerate(caption_lines):
        body.append(
            f'<text x="30" y="{rule_y + 26 + i * 20}" font-family="{SANS}" '
            f'font-size="12" fill="{MUTED}">{line}</text>'
        )

    svg = svg_wrap(
        width,
        height,
        "MEMBRANE KV-placement capacity result",
        "Bar chart: default all-GPU KV placement succeeds at context 26,500 "
        "and fails at context 26,800 with a real Vulkan out-of-device-memory "
        "error; MEMBRANE auto and cpu KV placement both succeed at context "
        "28,500 in the same tested configuration on Qwen2.5-1.5B, GTX 1650, "
        "Vulkan, native KV precision, 28 of 28 GPU weight layers.",
        "\n".join(body) + "\n",
    )
    out = ASSETS / "membrane-capacity.svg"
    out.write_text(svg)
    return out


def generate_q8_tradeoff_chart():
    """membrane-q8-tradeoff.svg from results/v0.3/gpu-vulkan-validation.json"""
    data = json.loads((ROOT / "results/v0.3/gpu-vulkan-validation.json").read_text())
    rows = data["models"]["SmolLM2-135M-Instruct-f16"]["context_matrix"]
    rows = sorted(rows, key=lambda r: r["ctx"])

    points = []
    for r in rows:
        vram_pct = r["vram_reduction_pct"]
        tp_delta = (
            100.0
            * (r["q8"]["generation_tok_per_s"] - r["native"]["generation_tok_per_s"])
            / r["native"]["generation_tok_per_s"]
        )
        points.append((r["ctx"], vram_pct, tp_delta))

    width, height = 760, 500
    left = 60
    right = 720
    n = len(points)
    col_w = (right - left) / n

    vram_top, vram_base = 92, 196
    tp_zero, tp_bottom = 268, 372
    max_vram = 30.0
    max_tp_abs = 20.0

    body = []
    body.append(
        f'<text x="30" y="30" font-family="{SANS}" font-size="13" fill="{MUTED}" '
        f'letter-spacing="1.5">MEASURED — SmolLM2-135M · GTX 1650 · Vulkan · '
        f'tested contexts only</text>'
    )
    body.append(
        f'<text x="30" y="54" font-family="{SANS}" font-size="15" fill="{TEXT}" '
        f'font-weight="700">q8 KV vs. native: VRAM and throughput tradeoff</text>'
    )

    body.append(
        f'<text x="30" y="{vram_top - 8}" font-family="{MONO}" font-size="11.5" '
        f'fill="{ACCENT}">VRAM reduction (%)</text>'
    )
    body.append(
        f'<line x1="{left}" y1="{vram_base}" x2="{right}" y2="{vram_base}" '
        f'stroke="{BORDER}" stroke-width="1"/>'
    )
    for i, (ctx, vram_pct, _tp) in enumerate(points):
        cx = left + col_w * (i + 0.5)
        bar_h = (vram_pct / max_vram) * (vram_base - vram_top)
        bw = col_w * 0.5
        body.append(
            f'<rect x="{cx - bw / 2:.1f}" y="{vram_base - bar_h:.1f}" '
            f'width="{bw:.1f}" height="{bar_h:.1f}" rx="3" fill="{ACCENT}" '
            f'fill-opacity="0.85"/>'
        )
        body.append(
            f'<text x="{cx:.1f}" y="{vram_base - bar_h - 6:.1f}" text-anchor="middle" '
            f'font-family="{MONO}" font-size="11" fill="{TEXT}">{vram_pct:.1f}%</text>'
        )

    body.append(
        f'<text x="30" y="{tp_zero - 8}" font-family="{MONO}" font-size="11.5" '
        f'fill="{FAIL}">generation throughput vs. native (%)</text>'
    )
    body.append(
        f'<line x1="{left}" y1="{tp_zero}" x2="{right}" y2="{tp_zero}" '
        f'stroke="{BORDER}" stroke-width="1"/>'
    )
    for i, (ctx, _vram, tp_delta) in enumerate(points):
        cx = left + col_w * (i + 0.5)
        bar_h = (abs(tp_delta) / max_tp_abs) * (tp_bottom - tp_zero)
        bw = col_w * 0.5
        body.append(
            f'<rect x="{cx - bw / 2:.1f}" y="{tp_zero:.1f}" width="{bw:.1f}" '
            f'height="{bar_h:.1f}" rx="3" fill="{FAIL}" fill-opacity="0.85"/>'
        )
        body.append(
            f'<text x="{cx:.1f}" y="{tp_zero + bar_h + 16:.1f}" text-anchor="middle" '
            f'font-family="{MONO}" font-size="11" fill="{TEXT}">{tp_delta:.1f}%</text>'
        )

    for i, (ctx, _v, _t) in enumerate(points):
        cx = left + col_w * (i + 0.5)
        body.append(
            f'<text x="{cx:.1f}" y="{tp_bottom + 34}" text-anchor="middle" '
            f'font-family="{MONO}" font-size="11.5" fill="{MUTED}">ctx {ctx}</text>'
        )

    caption_lines = [
        "One model family, one GPU, one host — not a general VRAM or",
        "throughput claim. q8 reduces VRAM at every tested context; the",
        "reduction and the throughput cost both vary by context.",
    ]
    for i, line in enumerate(caption_lines):
        body.append(
            f'<text x="30" y="{tp_bottom + 58 + i * 20}" font-family="{SANS}" '
            f'font-size="12" fill="{MUTED}">{line}</text>'
        )

    svg = svg_wrap(
        width,
        height,
        "MEMBRANE q8 KV VRAM/throughput tradeoff",
        "Bar chart across six tested contexts (512 to 16384) on SmolLM2-135M, "
        "GTX 1650, Vulkan: q8 KV VRAM reduction ranges from about 2 percent at "
        "small contexts to about 25 percent at context 16384, while generation "
        "throughput is about 7 to 18 percent lower than native across the same "
        "sweep.",
        "\n".join(body) + "\n",
    )
    out = ASSETS / "membrane-q8-tradeoff.svg"
    out.write_text(svg)
    return out


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    written = [generate_capacity_chart(), generate_q8_tradeoff_chart()]
    for path in written:
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
