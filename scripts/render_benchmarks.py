#!/usr/bin/env python3
"""Render the README benchmark charts from benchmarks/sonnet46/results.json.

Emits four static SVGs (two charts x light/dark) into docs/assets/, so the
README can theme them with a <picture> element. Deterministic and stdlib-only;
re-run after refreshing the benchmark:

    python scripts/render_benchmarks.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "sonnet46" / "results.json"
OUT = ROOT / "docs" / "assets"

FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"

#: Entity -> (display label, palette slot). Colors follow the entity in every
#: chart; "original" is the uncompressed reference, not a series.
#:
#: One classifier is plotted, not four. The retrained encoders span 0.560 to
#: 0.577 exact match -- about ten questions in six hundred -- so showing them
#: all implies a ranking the benchmark cannot support. mbert-control is the
#: best of them on both QA metrics.
ENTITIES = {
    "rules": ("grug rules", 0),
    "mbert-control": ("grug classifier", 1),
    "lingua2": ("LLMLingua-2", 2),
}

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "baseline": "#c3c2b7",
        "series": ("#2a78d6", "#eb6834", "#1baf7a"),
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "baseline": "#383835",
        "series": ("#3987e5", "#d95926", "#199e70"),
    },
}


def load_rows() -> dict[str, dict]:
    rows = json.loads(RESULTS.read_text(encoding="utf-8"))["rows"]
    return {row["backend"]: row for row in rows}


def text(x: float, y: float, s: str, size: int, fill: str, **attrs: str) -> str:
    extra = "".join(f' {k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}"{extra}>{s}</text>'


def quality_chart(rows: dict[str, dict], theme: dict) -> str:
    width, height = 760, 430
    left, right, top, bottom = 70, 34, 78, 56
    plot_w, plot_h = width - left - right, height - top - bottom

    x_min, x_max = 0.0, 100.0  # tokens sent, % of original
    y_min, y_max = 0.60, 0.80  # QA F1

    def px(v: float) -> float:
        return left + (v - x_min) / (x_max - x_min) * plot_w

    def py(v: float) -> float:
        return top + (y_max - v) / (y_max - y_min) * plot_h

    original = rows["original"]
    parts = [
        f'<rect width="{width}" height="{height}" rx="8" fill="{theme["surface"]}"/>',
        text(24, 34, "Keep the answers, drop the tokens", 16, theme["ink"], font_weight="700"),
        text(
            24,
            54,
            "MeetingBank QA · 600 questions over 694k tokens · requested rate 0.33 "
            "· answers judged by Sonnet 4.6",
            11.5,
            theme["secondary"],
        ),
    ]

    # y gridlines and ticks
    for value in (0.60, 0.65, 0.70, 0.75, 0.80):
        y = py(value)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
            f'stroke="{theme["grid"]}" stroke-width="1"/>'
        )
        parts.append(text(left - 8, y + 4, f"{value:.2f}", 11, theme["muted"], text_anchor="end"))
    # x baseline and ticks
    base_y = top + plot_h
    parts.append(
        f'<line x1="{left}" y1="{base_y}" x2="{left + plot_w}" y2="{base_y}" '
        f'stroke="{theme["baseline"]}" stroke-width="1"/>'
    )
    for value in (0, 25, 50, 75, 100):
        parts.append(
            text(px(value), base_y + 18, f"{value}%", 11, theme["muted"], text_anchor="middle")
        )
    parts.append(
        text(
            left + plot_w / 2,
            base_y + 38,
            "tokens sent to the LLM (% of the original document)",
            11,
            theme["muted"],
            text_anchor="middle",
        )
    )
    parts.append(text(left - 46, top - 12, "answer quality (F1)", 11, theme["muted"]))

    # reference: the uncompressed baseline
    ref_y = py(original["f1"])
    parts.append(
        f'<line x1="{left}" y1="{ref_y:.1f}" x2="{left + plot_w}" y2="{ref_y:.1f}" '
        f'stroke="{theme["muted"]}" stroke-width="1" stroke-dasharray="4 4"/>'
    )
    ox, oy = px(100.0), py(original["f1"])
    parts.append(
        f'<circle cx="{ox:.1f}" cy="{oy:.1f}" r="6" fill="none" '
        f'stroke="{theme["muted"]}" stroke-width="2"/>'
    )
    parts.append(
        text(ox - 12, oy + 22, "original, uncompressed", 12, theme["secondary"], text_anchor="end")
    )
    parts.append(
        text(
            ox - 12,
            oy + 37,
            f"F1 {original['f1']:.2f} · 100% tokens",
            11,
            theme["muted"],
            text_anchor="end",
        )
    )

    # (dx, dy) offsets keep the labels clear of each other and the marks.
    offsets = {"rules": (11, -16), "mbert-control": (11, -16), "lingua2": (11, 24)}
    for key, (label, slot) in ENTITIES.items():
        row = rows[key]
        x, y = px(row["ratio"] * 100), py(row["f1"])
        dx, dy = offsets[key]
        color = theme["series"][slot]
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}" '
            f'stroke="{theme["surface"]}" stroke-width="2"/>'
        )
        parts.append(text(x + dx, y + dy, label, 12.5, theme["ink"], font_weight="600"))
        parts.append(
            text(
                x + dx,
                y + dy + 15,
                f"F1 {row['f1']:.2f} · {row['ratio'] * 100:.0f}% tokens",
                11,
                theme["secondary"],
            )
        )
    parts.append(text(left + 4, ref_y - 8, "uncompressed quality", 10.5, theme["muted"]))

    body = "\n".join(parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'font-family="{FONT}" role="img" aria-label="Answer quality versus tokens sent: '
        f"grug rules beats the uncompressed baseline with 62% of the tokens; the grug "
        f'classifier keeps F1 0.70 at 37%; LLMLingua-2 sits lowest.">\n{body}\n</svg>\n'
    )


def negation_chart(rows: dict[str, dict], theme: dict) -> str:
    width, height = 760, 240
    left, right, top, bottom = 150, 70, 78, 34
    plot_w = width - left - right
    bar_h, gap = 22, 16

    x_max = 50.0  # percent
    order = ["lingua2", "mbert-control", "rules"]  # most losses first

    def bx(value: float) -> float:
        return left + value / x_max * plot_w

    parts = [
        f'<rect width="{width}" height="{height}" rx="8" fill="{theme["surface"]}"/>',
        text(24, 34, "Negations lost during compression", 16, theme["ink"], font_weight="700"),
        text(
            24,
            54,
            "share of load-bearing negations dropped at rate 0.33 · lower is better "
            "· losing one flips the meaning",
            11.5,
            theme["secondary"],
        ),
    ]

    for value in (10, 20, 30, 40, 50):
        x = bx(value)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{height - bottom}" '
            f'stroke="{theme["grid"]}" stroke-width="1"/>'
        )
        parts.append(
            text(x, height - bottom + 16, f"{value}%", 10.5, theme["muted"], text_anchor="middle")
        )
    parts.append(
        f'<line x1="{left}" y1="{top - 6}" x2="{left}" y2="{height - bottom}" '
        f'stroke="{theme["baseline"]}" stroke-width="1"/>'
    )

    for i, key in enumerate(order):
        label, slot = ENTITIES[key]
        value = rows[key]["negation_loss_rate"] * 100
        y = top + i * (bar_h + gap)
        color = theme["series"][slot]
        parts.append(
            text(left - 10, y + bar_h / 2 + 4, label, 12.5, theme["secondary"], text_anchor="end")
        )
        if value > 0:
            end = bx(value)
            parts.append(  # 4px rounded data end, square against the baseline
                f'<path d="M {left},{y} H {end - 4:.1f} Q {end:.1f},{y} {end:.1f},{y + 4} '
                f"V {y + bar_h - 4} Q {end:.1f},{y + bar_h} {end - 4:.1f},{y + bar_h} "
                f'H {left} Z" fill="{color}"/>'
            )
            label_x = end + 8
        else:
            label_x = left + 8
        parts.append(
            text(label_x, y + bar_h / 2 + 4, f"{value:.0f}%", 12.5, theme["ink"], font_weight="600")
        )

    body = "\n".join(parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'font-family="{FONT}" role="img" aria-label="Negations lost during compression: '
        f'LLMLingua-2 drops 43%; the grug classifier and grug rules lose none.">'
        f"\n{body}\n</svg>\n"
    )


def main() -> None:
    rows = load_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    for mode, theme in THEMES.items():
        (OUT / f"qa-quality-{mode}.svg").write_text(quality_chart(rows, theme), encoding="utf-8")
        (OUT / f"negation-loss-{mode}.svg").write_text(
            negation_chart(rows, theme), encoding="utf-8"
        )
    print(f"wrote 4 charts to {OUT}")


if __name__ == "__main__":
    main()
