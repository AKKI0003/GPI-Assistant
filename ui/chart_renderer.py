"""
ui/chart_renderer.py

Deterministic matplotlib rendering of a core.chart_generator.ChartSpec into
a PNG file. No model involved here at all — by the time a ChartSpec
reaches this module its numbers have already been extracted and validated;
this module's only job is drawing them accurately and legibly.
"""

import os
import re
import time
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")  # headless — no display server needed on a server/CLI box
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from core.config import CHARTS_OUTPUT_DIR

if TYPE_CHECKING:
    from core.chart_generator import ChartSpec

# A calm, readable palette that stays distinguishable in grayscale printouts
# too — avoids matplotlib's default bright cycle which gets muddy with 4+
# series.
_PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug[:60] or "chart"


def _format_value(v: float) -> str:
    """Compact axis/label formatting: 1,234,567 -> '1.2M', 45000 -> '45.0K'."""
    if v is None:
        return ""
    av = abs(v)
    if av >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if av >= 1_000:
        return f"{v/1_000:.1f}K"
    if float(v).is_integer():
        return f"{int(v)}"
    return f"{v:.2f}"


def render_chart(spec: "ChartSpec", out_dir: str = None) -> str:
    """
    Render a validated ChartSpec to a PNG file and return its path.

    Raises ValueError if spec.insufficient_data is True — callers should
    check that before calling this, but this is a safety net so a broken
    chart is never silently produced from empty data.
    """
    if spec.insufficient_data:
        raise ValueError(f"Cannot render: {spec.reason or 'insufficient data'}")

    out_dir = out_dir or CHARTS_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    fig.patch.set_facecolor("white")

    n_categories = len(spec.categories)
    n_series = len(spec.series)

    if spec.chart_type == "pie":
        # Validation guarantees exactly one series for a pie chart.
        s = spec.series[0]
        values = [v if v is not None else 0 for v in s["values"]]
        labels = [f"{c}" for c in spec.categories]
        colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(values))]
        wedges, _texts, autotexts = ax.pie(
            values, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 10},
        )
        for t in autotexts:
            t.set_color("white")
            t.set_fontweight("bold")
        ax.axis("equal")

    elif spec.chart_type == "line":
        x = range(n_categories)
        for i, s in enumerate(spec.series):
            color = _PALETTE[i % len(_PALETTE)]
            values = s["values"]
            # Plot only the contiguous points we have; matplotlib handles
            # None as a gap if we mask it out rather than plotting 0.
            xs = [xi for xi, v in zip(x, values) if v is not None]
            ys = [v for v in values if v is not None]
            ax.plot(xs, ys, marker="o", linewidth=2.2, markersize=6,
                     color=color, label=s["name"])
            for xi, yi in zip(xs, ys):
                ax.annotate(_format_value(yi), (xi, yi),
                            textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=8, color=color)
        ax.set_xticks(list(x))
        ax.set_xticklabels(spec.categories, rotation=0 if n_categories <= 6 else 30,
                             ha="center" if n_categories <= 6 else "right")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        if n_series > 1:
            ax.legend(frameon=False, loc="best")

    else:  # bar (grouped bar when multiple series)
        x = list(range(n_categories))
        width = 0.8 / max(n_series, 1)
        for i, s in enumerate(spec.series):
            color = _PALETTE[i % len(_PALETTE)]
            offset = (i - (n_series - 1) / 2) * width
            xs = [xi + offset for xi in x]
            values = [v if v is not None else 0 for v in s["values"]]
            bars = ax.bar(xs, values, width=width * 0.92, color=color, label=s["name"])
            for rect, v, orig in zip(bars, values, s["values"]):
                if orig is None:
                    continue
                ax.annotate(_format_value(orig),
                            (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                            textcoords="offset points", xytext=(0, 3),
                            ha="center", fontsize=8, color="#1f2937")
        ax.set_xticks(x)
        ax.set_xticklabels(spec.categories, rotation=0 if n_categories <= 6 else 30,
                             ha="center" if n_categories <= 6 else "right")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        if n_series > 1:
            ax.legend(frameon=False, loc="best")

    if spec.chart_type != "pie":
        if spec.x_label:
            ax.set_xlabel(spec.x_label, fontsize=10)
        if spec.y_label:
            ax.set_ylabel(spec.y_label, fontsize=10)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _format_value(v)))
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    ax.set_title(spec.title, fontsize=13, fontweight="bold", pad=14)
    fig.tight_layout()

    filename = f"{_slugify(spec.title)}_{int(time.time())}.png"
    path = os.path.join(out_dir, filename)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return path
