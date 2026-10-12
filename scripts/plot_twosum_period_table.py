#!/usr/bin/env python3
"""N=16384 blocked TwoSum period table (P=512, φ=1) as PNG.

Accuracy is against_float64_reference. Kernel time is the median
start-barrier device_kernel_s. × better = Rel_FP32 / Rel. Time / FP32
= t / t_FP32. Versus every product: × faster = t_1 / t, Acc / every 1
= Rel / Rel_1.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
MESH_JSONL = ROOT / "results" / "raw" / "gemm-mesh-direct.jsonl"
OZAKI_JSONL = ROOT / "results" / "raw" / "gemm-ozaki-direct.jsonl"
OUT_PNG = ROOT / "results" / "twosum_period_table_n16384.png"
OUT_FIG = ROOT / "results" / "figures" / "twosum_period_table_n16384.png"

FP32_RUN = "meshgemm-direct-ozimmu-phi1-p512-fp32-16384-16384-16384-1788888989770661897"
SKIP_RUN = "ozaki-fused-skip-a1b1-ozimmu-phi1-p512-16384-16384-16384-1788889240698722804"
PERIOD_RUNS = {
    1: "ozaki-fused-skip-a1b1-twosum-ozimmu-phi1-p512-16384-16384-16384-1790939854086001625",
    5: "ozaki-fused-skip-a1b1-twosum-every5-ozimmu-phi1-p512-16384-16384-16384-1791137086536717911",
    10: "ozaki-fused-skip-a1b1-twosum-every10-ozimmu-phi1-p512-16384-16384-16384-1791137970524798848",
    32: "ozaki-fused-skip-a1b1-twosum-every32-ozimmu-phi1-p512-16384-16384-16384-1791138811540100840",
    128: "ozaki-fused-skip-a1b1-twosum-every128-ozimmu-phi1-p512-16384-16384-16384-1791139593918182419",
}
# 3 products × K=16384. Remainder flushes once.
TWOSUMS = {1: 49152, 5: 9831, 10: 4916, 32: 1536, 128: 384}

COL_W = [1.20, 1.08, 0.98, 1.05, 1.05, 1.18, 1.05, 1.28]
GROUPS = [
    (0, 1, "", "#F4F6F7"),
    (1, 4, "Measured", "#D6EAF8"),
    (4, 6, "vs FP32", "#D5F5E3"),
    (6, 8, "vs every product", "#FDEBD0"),
]
SUBHEAD = [
    "Method",
    "Accuracy",
    "Time",
    "TwoSums",
    "× better",
    "Time / FP32",
    "× faster",
    "Acc / every 1",
]
BETTER = {4}
TIME_OVER = {5}
FASTER = {6}
ACC_OVER = {7}
GREEN = "#1E8449"
RED = "#C0392B"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def rel_and_kernel_s(records: list[dict[str, Any]], run_id: str) -> tuple[float, float]:
    rel = None
    times: list[float] = []
    for row in records:
        if row.get("run_id") != run_id:
            continue
        if row.get("kind") == "gemm_accuracy":
            rel = row["validation"]["against_float64_reference"]["relative_frobenius_error"]
        elif row.get("kind") == "gemm_direct_sample" and row.get("start_barrier"):
            times.append(float(row["timing"]["device_kernel_s"]))
    if rel is None or not times:
        raise SystemExit(f"missing accuracy or kernel samples for {run_id}")
    return float(rel), statistics.median(times)


def fmt_rel(value: float) -> str:
    mantissa, exp = f"{value:.2e}".split("e")
    return f"{mantissa}e{int(exp)}"


def fmt_ms(seconds: float) -> str:
    ms = seconds * 1e3
    if ms >= 100:
        return f"{ms:.0f} ms"
    return f"{ms:.1f} ms"


def fmt_ratio(value: float) -> str:
    if value >= 10:
        return f"{value:.1f}×"
    return f"{value:.2f}×"


def fmt_count(value: int) -> str:
    return f"{value:,}"


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


def build_rows() -> list[tuple[str, ...]]:
    mesh = load_jsonl(MESH_JSONL)
    ozaki = load_jsonl(OZAKI_JSONL)
    fp32_rel, fp32_s = rel_and_kernel_s(mesh, FP32_RUN)
    skip_rel, skip_s = rel_and_kernel_s(ozaki, SKIP_RUN)
    periods = [rel_and_kernel_s(ozaki, PERIOD_RUNS[p]) for p in (1, 5, 10, 32, 128)]
    every_rel, every_s = periods[0]
    rows = [
        ("FP32", fmt_rel(fp32_rel), fmt_ms(fp32_s), "—", "—", "—", "—", "—"),
        ("skip", fmt_rel(skip_rel), fmt_ms(skip_s), "—", "—", "—", "—", "—"),
    ]
    for period, (rel, seconds) in zip((1, 5, 10, 32, 128), periods):
        better = "—" if period == 0 else fmt_ratio(fp32_rel / rel)
        time_over = fmt_ratio(seconds / fp32_s)
        faster = "—" if period == 1 else fmt_ratio(every_s / seconds)
        acc_over = "—" if period == 1 else fmt_ratio(rel / every_rel)
        rows.append(
            (
                f"every {period}",
                fmt_rel(rel),
                fmt_ms(seconds),
                fmt_count(TWOSUMS[period]),
                better,
                time_over,
                faster,
                acc_over,
            )
        )
    return rows


def main() -> None:
    rows = build_rows()
    edges = _x_edges()
    total_w = edges[-1]
    n_data = len(rows)
    header_h = 0.42
    sub_h = 0.36
    row_h = 0.36
    pad = 0.08
    footer_h = 0.32
    total_h = pad + header_h + sub_h + n_data * row_h + footer_h + pad

    fig, ax = plt.subplots(figsize=(13.6, 4.15))
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

    ratio_cols = BETTER | TIME_OVER | FASTER | ACC_OVER
    for i, label in enumerate(SUBHEAD):
        group_color = next(c for a, b, _, c in GROUPS if a <= i < b)
        fc = _mix(group_color, 0.28) if i in ratio_cols else group_color
        cell(edges[i], edges[i + 1], y_sub, y_group, label, fc=fc, fw="bold", fs=8.4)

    for r, row in enumerate(rows):
        y1 = y_data0 - r * row_h
        y0 = y1 - row_h
        bg = "#FBFCFC" if r % 2 == 0 else "#F4F6F7"
        for i, val in enumerate(row):
            group_color = next(c for a, b, _, c in GROUPS if a <= i < b)
            if i == 0:
                fc = bg
            elif i in ratio_cols:
                base = _mix(group_color, 0.28)
                fc = base if r % 2 == 0 else _darken(base)
            else:
                fc = group_color if r % 2 == 0 else _darken(group_color)
            color = "#1C2833"
            fw = "normal"
            if val != "—":
                if i in BETTER or i in FASTER:
                    fw = "bold"
                    color = GREEN
                elif i in TIME_OVER or i in ACC_OVER:
                    fw = "bold"
                    color = RED
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
    fig.suptitle("Blocked TwoSum period, N = 16,384", y=0.995, fontsize=14)
    fig.tight_layout(rect=(0.01, 0.0, 0.99, 0.93))
    for dest in (OUT_PNG, OUT_FIG):
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT_PNG)
    for row in rows:
        print(" | ".join(row))


if __name__ == "__main__":
    main()
