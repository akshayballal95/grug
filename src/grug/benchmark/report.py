"""Turn benchmark rows into a chart.

The SVG is written by hand rather than with a plotting library: grug's whole
pitch is a light install, and a grouped bar chart is fifty lines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["to_csv", "to_svg"]

_PALETTE = ["#4c6ef5", "#f59f00", "#12b886", "#e64980", "#7950f2", "#868e96"]


def to_csv(rows: list[Any], path: str | Path) -> Path:
    """Write rows as CSV for a spreadsheet or an external plotter."""
    import csv
    from dataclasses import asdict, is_dataclass

    records = [asdict(r) if is_dataclass(r) else dict(r) for r in rows]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        for record in records:
            writer.writerow({k: v for k, v in record.items() if k in writer.fieldnames})
    return target


def to_svg(
    rows: list[Any],
    path: str | Path,
    *,
    metric: str = "exact_match",
    title: str = "QA accuracy vs compression",
) -> Path:
    """Scatter accuracy against achieved ratio: the trade-off in one picture.

    Ratio on x, quality on y. Up and to the left is better -- more compression
    for less loss. A horizontal line marks the uncompressed ceiling.
    """
    from dataclasses import asdict, is_dataclass

    records = [asdict(r) if is_dataclass(r) else dict(r) for r in rows]
    if not records:
        raise ValueError("no rows to plot")

    ceiling = next((r for r in records if r["backend"] == "original"), None)
    points = [r for r in records if r is not ceiling]
    width, height, pad = 720, 450, 64
    plot_w, plot_h = width - 2 * pad, height - 2 * pad - 30

    # Frame the data rather than the origin. Every backend scores between 0.55
    # and 0.63 here, so anchoring at zero spends three quarters of the plot on
    # empty space and hides the differences the chart exists to show. The
    # uncompressed line is drawn across it, which keeps a reference in view.
    ys = [r[metric] for r in records]
    y_lo, y_hi = min(ys), max(ys)
    margin = (y_hi - y_lo) * 0.35 or 0.05
    y_lo, y_hi = max(0.0, y_lo - margin), y_hi + margin

    def px(ratio: float) -> float:
        return pad + ratio * plot_w

    def py(value: float) -> float:
        return pad + plot_h - (value - y_lo) / (y_hi - y_lo) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="system-ui,sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{pad}" y="{pad - 28}" font-size="17" font-weight="600">{title}</text>',
        f'<text x="{pad}" y="{pad - 10}" font-size="12" fill="#666">'
        f"x: achieved ratio (lower = more compression) &#183; y: {metric}</text>",
        f'<line x1="{pad}" y1="{pad + plot_h}" x2="{width - pad}" y2="{pad + plot_h}" stroke="#ccc"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{pad + plot_h}" stroke="#ccc"/>',
    ]

    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = px(tick)
        parts.append(f'<line x1="{x}" y1="{pad}" x2="{x}" y2="{pad + plot_h}" stroke="#f1f3f5"/>')
        parts.append(
            f'<text x="{x}" y="{pad + plot_h + 18}" font-size="11" fill="#666" '
            f'text-anchor="middle">{tick:.2f}</text>'
        )
    for frac in range(5):
        value = y_lo + (y_hi - y_lo) * frac / 4
        y = py(value)
        parts.append(f'<line x1="{pad}" y1="{y}" x2="{width - pad}" y2="{y}" stroke="#f1f3f5"/>')
        parts.append(
            f'<text x="{pad - 8}" y="{y + 4}" font-size="11" fill="#666" '
            f'text-anchor="end">{value:.2f}</text>'
        )

    if ceiling:
        y = py(ceiling[metric])
        parts.append(
            f'<line x1="{pad}" y1="{y}" x2="{width - pad}" y2="{y}" stroke="#adb5bd" '
            f'stroke-dasharray="6 4"/>'
        )
        parts.append(
            f'<text x="{width - pad}" y="{y - 6}" font-size="11" fill="#868e96" '
            f'text-anchor="end">uncompressed ({ceiling[metric]:.3f})</text>'
        )

    colours: dict[str, str] = {}
    for record in points:
        name = record["backend"]
        colour = colours.setdefault(name, _PALETTE[len(colours) % len(_PALETTE)])
        x, y = px(record["ratio"]), py(record[metric])
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{colour}" opacity="0.9"/>')
        # The score, not the requested rate: the rate is identical across
        # points and says nothing, and next to a backend that missed its
        # target it actively misleads.
        parts.append(
            f'<text x="{x:.1f}" y="{y - 12:.1f}" font-size="11" fill="#495057" '
            f'text-anchor="middle">{record[metric]:.3f}</text>'
        )

    # Laid out along the bottom. In the plot it sat on top of the uncompressed
    # label, which is where that label has to go.
    legend_y = height - pad + 40
    cursor = pad
    for name, colour in colours.items():
        parts.append(f'<circle cx="{cursor + 5}" cy="{legend_y - 4}" r="5" fill="{colour}"/>')
        parts.append(
            f'<text x="{cursor + 16}" y="{legend_y}" font-size="12" fill="#343a40">{name}</text>'
        )
        cursor += 26 + len(name) * 7

    parts.append("</svg>")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(parts), encoding="utf-8")
    return target
