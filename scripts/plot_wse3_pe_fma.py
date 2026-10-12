#!/usr/bin/env python3
"""Talk slide: WSE-3 PE FMA units as programmed in CSL."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT_PNG = ROOT / "results" / "wse3_pe_fma.png"
OUT_FIG = ROOT / "results" / "figures" / "wse3_pe_fma.png"

INK = "#1C2833"
MUTED = "#5D6D7E"
LINE = "#8A9399"
BG = "#FFFFFF"

F16 = "#D4E6F1"
MIX = "#E8DAEF"
F32 = "#D5F5E3"
LANE = "#1A5276"
ACC = "#6C3483"
SCALAR = "#196F3D"


def box(ax, x, y, w, h, fc, *, lw=0.8, ec=LINE):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="square,pad=0",
            linewidth=lw,
            edgecolor=ec,
            facecolor=fc,
        )
    )


def txt(ax, x, y, s, *, size=10, weight="normal", color=INK, ha="center", va="center"):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va)


def main() -> None:
    fig_w, fig_h = 9.4, 6.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    txt(ax, fig_w / 2, 6.08, "WSE-3 PE FMA units", size=18, weight="bold")
    txt(ax, fig_w / 2, 5.72, "One processing element  ·  1.4 GHz", size=10.5, color=MUTED)

    panel_x, panel_y, panel_w, panel_h = 0.45, 0.35, 8.5, 5.05
    box(ax, panel_x, panel_y, panel_w, panel_h, "#FBFCFC", lw=1.15, ec=LINE)
    cx = panel_x + panel_w / 2
    txt(ax, cx, 5.00, "One PE", size=14, weight="bold")
    txt(ax, cx, 4.64, "48 KiB SRAM  ·  8 × 6 KiB banks", size=10, color=MUTED)
    txt(ax, cx, 4.22, "1 cycle:  2 × 32-bit read  ·  1 × 32-bit write", size=11.5, weight="bold")

    rows = [
        (3.20, 8, 0.58, F16, LANE, "@fmach", "SIMD-8", "f16 × f16 + f16"),
        (2.05, 4, 1.24, MIX, ACC, "@fmachs", "SIMD-4", "f16 × f16 + f32"),
        (0.90, 1, 5.18, F32, SCALAR, "@fmacs", "scalar", "f32 × f32 + f32"),
    ]
    chip_left = 2.85
    for y, n, w, fc, accent, name, width, eq in rows:
        gap = 0.08
        txt(ax, 0.72, y + 0.22, name, size=12, weight="bold", ha="left")
        txt(ax, 0.72, y - 0.06, width, size=9.5, color=accent, ha="left")
        txt(ax, 0.72, y - 0.32, eq, size=8.6, color=MUTED, ha="left")
        for i in range(n):
            box(ax, chip_left + i * (w + gap), y - 0.02, w, 0.42, fc, lw=0.8, ec=accent)

    fig.tight_layout(pad=0.2)
    for dest in (OUT_PNG, OUT_FIG):
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(OUT_PNG)


if __name__ == "__main__":
    main()
