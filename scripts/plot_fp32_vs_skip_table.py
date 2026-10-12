#!/usr/bin/env python3
"""FP32 vs 2-slice skip vs 2-slice + TwoSum table (P=512, φ=1) as PNG."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT_PNG = ROOT / "results" / "fp32_vs_2slice_ozaki_twosum.png"
OUT_FIG = ROOT / "results" / "figures" / "fp32_vs_2slice_ozaki_twosum.png"

# Same run: P=512, φ=1, start-barrier kernel. Ratios are that row vs its FP32 cells.
ROWS = [
    (
        "2048",
        "4",
        "8.10e-7",
        "0.181 ms",
        "1.35e-6",
        "0.292 ms",
        "1.67×",
        "1.61×",
        "7.98e-8",
        "4.66 ms",
        "10.2×",
        "25.7×",
    ),
    (
        "8192",
        "16",
        "1.62e-6",
        "3.54 ms",
        "2.64e-6",
        "3.40 ms",
        "1.63×",
        "1.04×",
        "8.37e-8",
        "89.4 ms",
        "19.4×",
        "25.3×",
    ),
    (
        "16384",
        "32",
        "2.29e-6",
        "25.5 ms",
        "3.69e-6",
        "21.3 ms",
        "1.61×",
        "1.20×",
        "7.98e-8",
        "455 ms",
        "28.7×",
        "17.8×",
    ),
    (
        "20480",
        "40",
        "2.56e-6",
        "49.0 ms",
        "4.10e-6",
        "39.7 ms",
        "1.60×",
        "1.23×",
        "8.37e-8",
        "784 ms",
        "30.6×",
        "16.0×",
    ),
]

COL_W = [0.70, 0.46, 1.00, 0.92, 1.00, 0.92, 0.92, 0.98, 1.00, 0.92, 0.92, 0.98]
GROUPS = [
    (0, 2, "", "#F4F6F7"),
    (2, 4, "FP32  (baseline)", "#D6EAF8"),
    (4, 8, "2-slice Ozaki", "#E8DAEF"),
    (8, 12, "2-slice Ozaki + TwoSum", "#D5F5E3"),
]
SUBHEAD = [
    "N",
    "T",
    "Accuracy",
    "Time",
    "Accuracy",
    "Time",
    "Acc / FP32",
    "× faster",
    "Accuracy",
    "Time",
    "× better",
    "Time / FP32",
]
RATIO_ACC = {6}
BETTER_ACC = {10}
FASTER_TIME = {7}
RATIO_TIME = {11}


def _x_edges() -> list[float]:
    edges = [0.0]
    for width in COL_W:
        edges.append(edges[-1] + width)
    return edges


def _darken(hex_color: str, factor: float = 0.94) -> str:
    hex_color = hex_color.lstrip("#")
    rgb = [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]
    rgb = [max(0, int(c * factor)) for c in rgb]
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _mix(hex_color: str, white: float = 0.35) -> str:
    hex_color = hex_color.lstrip("#")
    rgb = [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]
    rgb = [min(255, int(c * (1 - white) + 255 * white)) for c in rgb]
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _ratio(val: str) -> float | None:
    if val in {"—", "-", ""}:
        return None
    return float(val.replace("×", "").replace("x", "").replace(",", ""))


def main() -> None:
    edges = _x_edges()
    total_w = edges[-1]
    n_data = len(ROWS)
    header_h = 0.42
    sub_h = 0.36
    row_h = 0.36
    pad = 0.08
    footer_h = 0.32
    total_h = pad + header_h + sub_h + n_data * row_h + footer_h + pad

    fig, ax = plt.subplots(figsize=(15.4, 2.75))
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_top = total_h - pad
    y_group = y_top - header_h
    y_sub = y_group - sub_h
    y_data0 = y_sub

    def cell(x0, x1, y0, y1, text, *, fc, fw="normal", fs=10, color="#1C2833"):
        ax.add_patch(
            FancyBboxPatch(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                boxstyle="square,pad=0",
                linewidth=0.6,
                edgecolor="#B0B6BA",
                facecolor=fc,
            )
        )
        if text:
            ax.text(
                (x0 + x1) / 2,
                (y0 + y1) / 2,
                text,
                ha="center",
                va="center",
                fontsize=fs,
                fontweight=fw,
                color=color,
            )

    for i0, i1, title, color in GROUPS:
        cell(edges[i0], edges[i1], y_group, y_top, title, fc=color, fw="bold", fs=10.5)

    for i, label in enumerate(SUBHEAD):
        group_color = next(c for a, b, _, c in GROUPS if a <= i < b)
        fc = (
            _mix(group_color, 0.28)
            if i in RATIO_ACC | BETTER_ACC | FASTER_TIME | RATIO_TIME
            else group_color
        )
        cell(edges[i], edges[i + 1], y_sub, y_group, label, fc=fc, fw="bold", fs=8.2)

    for r, row in enumerate(ROWS):
        y1 = y_data0 - r * row_h
        y0 = y1 - row_h
        bg = "#FBFCFC" if r % 2 == 0 else "#F4F6F7"
        for i, val in enumerate(row):
            group_color = next(c for a, b, _, c in GROUPS if a <= i < b)
            if i < 2:
                fc = bg
            elif i in RATIO_ACC | BETTER_ACC | FASTER_TIME | RATIO_TIME:
                base = _mix(group_color, 0.28)
                fc = base if r % 2 == 0 else _darken(base)
            else:
                fc = group_color if r % 2 == 0 else _darken(group_color)
            color = "#1C2833"
            fw = "normal"
            if i in BETTER_ACC and val not in {"—", "-"}:
                fw = "bold"
                color = "#1E8449"
            elif i in FASTER_TIME and val not in {"—", "-"}:
                fw = "bold"
                color = "#C0392B" if val == "1.61×" else "#1E8449"
            else:
                ratio = _ratio(val) if i in RATIO_ACC | RATIO_TIME else None
                if ratio is not None:
                    fw = "bold"
                    color = "#1E8449" if ratio < 1.0 else "#C0392B"
            cell(edges[i], edges[i + 1], y0, y1, val, fc=fc, fw=fw, color=color, fs=10)

    ax.text(
        total_w / 2,
        pad * 0.55,
        "WSE-3: 1.4 GHz, grid 512×512, φ=1",
        ha="center",
        va="center",
        fontsize=8.5,
        color="0.25",
    )
    fig.suptitle("Native FP32 vs 2-slice Ozaki", y=0.995, fontsize=14)
    fig.tight_layout(rect=(0.01, 0.0, 0.99, 0.92))
    for dest in (OUT_PNG, OUT_FIG):
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT_PNG)


if __name__ == "__main__":
    main()
