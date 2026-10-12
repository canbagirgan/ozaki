#!/usr/bin/env python3
"""Latency vs relative-Frobenius scatter in the ozIMMU/GEMMul8 figure style."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "results" / "figures"

# Median of 3 start-barrier samples. Kernel time = max-PE TSC @ 1.4 GHz.
# Error = relative Frobenius vs float64(A) @ float64(B). N=K=M=16384, P=512,
# T=32, ozIMMU φ=1, seed 7.
POINTS = [
    {
        "label": "MeshGEMM FP16  @fmach",
        "short": "MeshGEMM FP16",
        "ms": 4.676734285714286,
        "rel": 1.694594341531862e-2,
        "color": "#c0392b",
        "z": 4,
    },
    {
        "label": "Skip A1B1  3×@fmachs",
        "short": "Skip A1B1",
        "ms": 21.289515714285717,
        "rel": 3.6872682803505268e-6,
        "color": "#1e8449",
        "z": 5,
    },
    {
        "label": "MeshGEMM FP32  @fmacs",
        "short": "MeshGEMM FP32",
        "ms": 25.508347142857144,
        "rel": 2.2922695634633718e-6,
        "color": "#5dade2",
        "z": 4,
    },
    {
        "label": "2-slice full  4×@fmachs",
        "short": "2-slice full",
        "ms": 27.994017142857142,
        "rel": 3.6872024967932648e-6,
        "color": "#a04000",
        "z": 3,
    },
]


def pareto_front(points: list[dict]) -> list[dict]:
    ordered = sorted(points, key=lambda p: p["ms"])
    front: list[dict] = []
    best = float("inf")
    for point in ordered:
        if point["rel"] < best:
            front.append(point)
            best = point["rel"]
    return front


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
        }
    )
    fig, axis = plt.subplots(figsize=(8.6, 5.4))

    front = pareto_front(POINTS)
    axis.plot(
        [p["ms"] for p in front],
        [p["rel"] for p in front],
        linestyle=(0, (5, 3.5)),
        color="0.15",
        linewidth=1.15,
        zorder=2,
        label="Pareto frontier",
    )

    for point in POINTS:
        axis.scatter(
            point["ms"],
            point["rel"],
            s=92,
            c=point["color"],
            edgecolors="none",
            zorder=point["z"],
            label=point["label"],
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(3.2, 42)
    axis.set_ylim(8e-7, 4.5e-2)
    axis.set_xlabel("Kernel latency [ms]")
    axis.set_ylabel("Rel. Frobenius error vs FP64")
    axis.set_title("N = 16,384", pad=10)
    axis.xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    axis.yaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.grid(True, which="major", color="0.88", linewidth=0.8)
    axis.grid(True, which="minor", color="0.94", linewidth=0.5)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color("0.55")

    handles, labels = axis.get_legend_handles_labels()
    # Scatter handles first (implementations), Pareto last like the paper.
    order = list(range(1, len(handles))) + [0]
    axis.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        title="Implementation / annotation",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fancybox=False,
        edgecolor="0.75",
        borderpad=0.7,
        handletextpad=0.6,
    )
    fig.text(
        0.5,
        -0.02,
        "CS-3 / WSE-3, P=512, T=32, ozIMMU φ=1, seed 7. "
        "Latency is median start-barrier max-PE TSC at 1.4 GHz.",
        ha="center",
        fontsize=8,
        color="0.35",
    )
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(
            FIGURES / f"latency-error-n16384.{ext}",
            dpi=180,
            bbox_inches="tight",
        )
    plt.close(fig)
    print(FIGURES / "latency-error-n16384.png")


if __name__ == "__main__":
    main()
