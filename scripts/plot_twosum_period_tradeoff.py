#!/usr/bin/env python3
"""N=16384 blocked TwoSum period sweep: Rel Frob vs kernel time.

Start-barrier max-PE TSC at 1.4 GHz. Rel is
against_float64_reference.relative_frobenius_error. Period 1 is TwoSum
after every product; 5, 10, 32, and 128 bucket that many @fmachs first.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
MESH_JSONL = ROOT / "results" / "raw" / "gemm-mesh-direct.jsonl"
OZAKI_JSONL = ROOT / "results" / "raw" / "gemm-ozaki-direct.jsonl"
OUT_PNG = ROOT / "results" / "twosum_period_tradeoff_n16384.png"
OUT_FIG = ROOT / "results" / "figures" / "twosum_period_tradeoff_n16384.png"

# (jsonl, run_id). Period 1 has no twosum_period field in the record.
FP32_RUN = (
    MESH_JSONL,
    "meshgemm-direct-ozimmu-phi1-p512-fp32-16384-16384-16384-1788888989770661897",
)
SKIP_RUN = (
    OZAKI_JSONL,
    "ozaki-fused-skip-a1b1-ozimmu-phi1-p512-16384-16384-16384-1788889240698722804",
)
PERIOD_RUNS = {
    1: "ozaki-fused-skip-a1b1-twosum-ozimmu-phi1-p512-16384-16384-16384-1790939854086001625",
    5: "ozaki-fused-skip-a1b1-twosum-every5-ozimmu-phi1-p512-16384-16384-16384-1791137086536717911",
    10: "ozaki-fused-skip-a1b1-twosum-every10-ozimmu-phi1-p512-16384-16384-16384-1791137970524798848",
    32: "ozaki-fused-skip-a1b1-twosum-every32-ozimmu-phi1-p512-16384-16384-16384-1791138811540100840",
    128: "ozaki-fused-skip-a1b1-twosum-every128-ozimmu-phi1-p512-16384-16384-16384-1791139593918182419",
}

FP32_COLOR = "#0E6655"
SKIP_COLOR = "#6C3483"
TWOSUM_COLOR = "#B9770E"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def rel_and_kernel_ms(records: list[dict[str, Any]], run_id: str) -> tuple[float, float]:
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
    return float(rel), statistics.median(times) * 1e3


def main() -> None:
    mesh = load_jsonl(MESH_JSONL)
    ozaki = load_jsonl(OZAKI_JSONL)
    fp32 = rel_and_kernel_ms(mesh, FP32_RUN[1])
    skip = rel_and_kernel_ms(ozaki, SKIP_RUN[1])
    periods = sorted(PERIOD_RUNS)
    twosum = [rel_and_kernel_ms(ozaki, PERIOD_RUNS[p]) for p in periods]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "axes.linewidth": 0.8,
        }
    )
    fig, axis = plt.subplots(figsize=(8.4, 5.4))

    axis.plot(
        [t for _, t in twosum],
        [r for r, _ in twosum],
        color=TWOSUM_COLOR,
        linewidth=1.4,
        zorder=3,
    )
    axis.scatter(
        [t for _, t in twosum],
        [r for r, _ in twosum],
        s=110,
        c=TWOSUM_COLOR,
        marker="^",
        edgecolors="0.15",
        linewidths=0.7,
        zorder=5,
    )
    axis.scatter(
        [fp32[1]],
        [fp32[0]],
        s=100,
        c=FP32_COLOR,
        marker="s",
        edgecolors="0.15",
        linewidths=0.7,
        zorder=5,
    )
    axis.scatter(
        [skip[1]],
        [skip[0]],
        s=100,
        c=SKIP_COLOR,
        marker="D",
        edgecolors="0.15",
        linewidths=0.7,
        zorder=5,
    )

    # Offsets in points. Larger periods sit closer together on the log axes.
    offsets = {
        1: (10, -16),
        5: (10, 10),
        10: (10, 8),
        32: (8, 12),
        128: (-10, -18),
    }
    for period, (rel, ms) in zip(periods, twosum):
        dx, dy = offsets[period]
        axis.annotate(
            str(period),
            (ms, rel),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=10,
            color="0.15",
            ha="left" if dx >= 0 else "right",
        )
    axis.annotate(
        "FP32",
        fp32[::-1],
        textcoords="offset points",
        xytext=(8, -4),
        fontsize=10,
        color=FP32_COLOR,
    )
    axis.annotate(
        "skip",
        skip[::-1],
        textcoords="offset points",
        xytext=(-8, 8),
        fontsize=10,
        color=SKIP_COLOR,
        ha="right",
    )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(16, 700)
    axis.set_ylim(5.5e-8, 6.5e-6)
    axis.set_xlabel("Kernel time [ms]")
    axis.set_ylabel("Relative Frobenius error")
    axis.set_title("Blocked TwoSum period, N = 16,384", pad=8)
    axis.grid(True, which="major", color="0.88", linewidth=0.8)
    axis.grid(True, which="minor", color="0.94", linewidth=0.5)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color("0.55")

    axis.legend(
        [
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker="s",
                markersize=8,
                markerfacecolor=FP32_COLOR,
                markeredgecolor="0.15",
            ),
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker="D",
                markersize=8,
                markerfacecolor=SKIP_COLOR,
                markeredgecolor="0.15",
            ),
            Line2D(
                [0],
                [0],
                color=TWOSUM_COLOR,
                linewidth=1.4,
                marker="^",
                markersize=9,
                markerfacecolor=TWOSUM_COLOR,
                markeredgecolor="0.15",
            ),
        ],
        [
            "MeshGEMM FP32",
            "2-slice Ozaki skip",
            "skip + TwoSum (period labeled)",
        ],
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="0.75",
    )
    fig.text(
        0.5,
        0.01,
        "WSE-3: 1.4 GHz, grid 512×512, φ=1",
        ha="center",
        fontsize=9,
        color="0.25",
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    for dest in (OUT_PNG, OUT_FIG):
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(OUT_PNG)
    print(f"FP32  rel={fp32[0]:.4e}  {fp32[1]:.4g} ms")
    print(f"skip  rel={skip[0]:.4e}  {skip[1]:.4g} ms")
    for period, (rel, ms) in zip(periods, twosum):
        print(f"every {period:<3} rel={rel:.4e}  {ms:.4g} ms")


if __name__ == "__main__":
    main()
