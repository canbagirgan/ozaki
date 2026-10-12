#!/usr/bin/env python3
"""GPU accuracy-latency trade-off (3 panels) with CS-3 P=512 ozIMMU φ=1 overlay.

The source GPU figure is ``error_latency_tradeoff.png`` (5 panels:
N=1024, 2048, 4096, 8192, 16384). No GPU CSV/JSON or plotting script
was in the repo, so GPU coordinates were digitized from that PNG and
stored in ``results/processed/gpu_error_latency_tradeoff.csv``.

This figure keeps the three N that have complete CS-3 accuracy+time
pairs: 2048, 8192, 16384 (FP32, skip, skip+TwoSum). CS-3 points are
start-barrier kernel time (max-PE TSC @ 1.4 GHz) and Rel Frob vs
float64(A) @ float64(B).
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
GPU_CSV = ROOT / "results" / "processed" / "gpu_error_latency_tradeoff.csv"
MESH_JSONL = ROOT / "results" / "raw" / "gemm-mesh-direct.jsonl"
OZAKI_JSONL = ROOT / "results" / "raw" / "gemm-ozaki-direct.jsonl"
OUT_PNG = ROOT / "results" / "error_latency_tradeoff_cs3.png"
OUT_FIG = ROOT / "results" / "figures" / "error_latency_tradeoff_cs3.png"

PANELS = (2048, 8192, 16384)

# Legend colors sampled from error_latency_tradeoff.png.
GPU_STYLE: dict[str, dict[str, Any]] = {
    "cuBLAS FP64": {"color": "#2F81B9", "marker": "o", "z": 4},
    "cuBLAS FP32": {"color": "#28C2D2", "marker": "o", "z": 5},
    "cuBLAS TF32": {"color": "#FF8B24", "marker": "o", "z": 5},
    "cuBLAS FP16/FP32": {"color": "#D93738", "marker": "o", "z": 5},
    "APE FP32T": {"color": "#9A6FC0", "marker": "o", "z": 4},
    "ozIMMU": {"color": "#38A538", "marker": "o", "z": 3},
    "ozIMMU EF": {"color": "#8F5A50", "marker": "o", "z": 3},
    "ozIMMU RN": {"color": "#E47EC5", "marker": "o", "z": 3},
    "ozIMMU H": {"color": "#828282", "marker": "o", "z": 3},
    "ozIMMU BLOCK": {"color": "#C1C132", "marker": "o", "z": 3},
    "GEMMul8": {"color": "#0E4A65", "marker": "o", "z": 5},
    "Zero vs. FP64 reference": {"color": "#1B4F72", "marker": "v", "z": 6},
}

CS3_STYLE: dict[str, dict[str, Any]] = {
    "CS-3 MeshGEMM FP32": {
        "color": "#0E6655",
        "marker": "s",
        "z": 8,
        "size": 88,
    },
    "CS-3 Ozaki skip A1B1": {
        "color": "#6C3483",
        "marker": "D",
        "z": 8,
        "size": 88,
    },
    "CS-3 Ozaki skip + TwoSum": {
        "color": "#B9770E",
        "marker": "^",
        "z": 9,
        "size": 100,
    },
}

# Fallbacks if JSONL is missing. Times are median start-barrier kernel seconds.
CS3_FALLBACK = {
    2048: {
        "CS-3 MeshGEMM FP32": (8.095107857388656e-07, 0.00018126357142857143),
        "CS-3 Ozaki skip A1B1": (1.3503718531476993e-06, 0.0002921),
        "CS-3 Ozaki skip + TwoSum": (8.374590385438746e-08, 0.004657724285714286),
    },
    8192: {
        "CS-3 MeshGEMM FP32": (1.6190687624042163e-06, 0.0035425671428571427),
        "CS-3 Ozaki skip A1B1": (2.643667357503105e-06, 0.0034036157142857144),
        "CS-3 Ozaki skip + TwoSum": (8.371847402047166e-08, 0.08941137785714286),
    },
    16384: {
        "CS-3 MeshGEMM FP32": (2.2922695634633718e-06, 0.025508347142857143),
        "CS-3 Ozaki skip A1B1": (3.6872682803505268e-06, 0.021289515714285716),
        "CS-3 Ozaki skip + TwoSum": (8.368138831789465e-08, 0.45468799),
    },
}

MESH_RUN = {
    2048: "meshgemm-direct-ozimmu-phi1-p512-fp32-2048-2048-2048-1790933156773046060",
    8192: "meshgemm-direct-ozimmu-phi1-p512-fp32-8192-8192-8192-1790933099079551473",
    16384: "meshgemm-direct-ozimmu-phi1-p512-fp32-16384-16384-16384-1788888989770661897",
}
SKIP_RUN = {
    2048: "ozaki-fused-skip-a1b1-ozimmu-phi1-p512-2048-2048-2048-1790933217626731063",
    8192: "ozaki-fused-skip-a1b1-ozimmu-phi1-p512-8192-8192-8192-1790933045547957289",
    16384: "ozaki-fused-skip-a1b1-ozimmu-phi1-p512-16384-16384-16384-1788889240698722804",
}
TWOSUM_RUN = {
    2048: "ozaki-fused-skip-a1b1-twosum-ozimmu-phi1-p512-2048-2048-2048-1790940297865946121",
    8192: "ozaki-fused-skip-a1b1-twosum-ozimmu-phi1-p512-8192-8192-8192-1790941144516276493",
    16384: "ozaki-fused-skip-a1b1-twosum-ozimmu-phi1-p512-16384-16384-16384-1790939854086001625",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def rel_frob_vs_f64(record: dict[str, Any]) -> float | None:
    validation = record.get("validation") or {}
    against = (
        validation.get("against_float64_reference")
        or validation.get("against_correctly_rounded_reference")
        or {}
    )
    value = against.get("relative_frobenius_error")
    return float(value) if value is not None else None


def median_kernel_s(records: list[dict[str, Any]], run_id: str) -> float | None:
    times = [
        row["timing"]["device_kernel_s"]
        for row in records
        if row.get("run_id") == run_id
        and row.get("kind") == "gemm_direct_sample"
        and row.get("start_barrier")
        and (row.get("timing") or {}).get("device_kernel_s") is not None
    ]
    return statistics.median(times) if times else None


def accuracy_rel(records: list[dict[str, Any]], run_id: str) -> float | None:
    for row in records:
        if row.get("run_id") == run_id and row.get("kind") == "gemm_accuracy":
            return rel_frob_vs_f64(row)
    return None


def load_cs3() -> dict[int, dict[str, tuple[float, float]]]:
    """Return {N: {label: (rel_frob, kernel_s)}}."""
    mesh = load_jsonl(MESH_JSONL)
    ozaki = load_jsonl(OZAKI_JSONL)
    out: dict[int, dict[str, tuple[float, float]]] = {}
    for n in PANELS:
        fp32_rel = accuracy_rel(mesh, MESH_RUN[n])
        fp32_t = median_kernel_s(mesh, MESH_RUN[n])
        skip_rel = accuracy_rel(ozaki, SKIP_RUN[n])
        skip_t = median_kernel_s(ozaki, SKIP_RUN[n])
        twosum_rel = accuracy_rel(ozaki, TWOSUM_RUN[n])
        twosum_t = median_kernel_s(ozaki, TWOSUM_RUN[n])
        fb = CS3_FALLBACK[n]
        out[n] = {
            "CS-3 MeshGEMM FP32": (
                fp32_rel if fp32_rel is not None else fb["CS-3 MeshGEMM FP32"][0],
                fp32_t if fp32_t is not None else fb["CS-3 MeshGEMM FP32"][1],
            ),
            "CS-3 Ozaki skip A1B1": (
                skip_rel if skip_rel is not None else fb["CS-3 Ozaki skip A1B1"][0],
                skip_t if skip_t is not None else fb["CS-3 Ozaki skip A1B1"][1],
            ),
            "CS-3 Ozaki skip + TwoSum": (
                twosum_rel if twosum_rel is not None else fb["CS-3 Ozaki skip + TwoSum"][0],
                twosum_t if twosum_t is not None else fb["CS-3 Ozaki skip + TwoSum"][1],
            ),
        }
    return out


def load_gpu() -> dict[int, list[dict[str, Any]]]:
    by_n: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with GPU_CSV.open() as handle:
        rows = csv.DictReader(
            line for line in handle if not line.startswith("#")
        )
        for row in rows:
            n = int(row["n"])
            by_n[n].append(
                {
                    "impl": row["impl"],
                    "ms": float(row["latency_ms"]),
                    "rel": float(row["rel_frob"]),
                    "family": "gpu",
                }
            )
    return by_n


def pareto_front(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(points, key=lambda p: p["ms"])
    front: list[dict[str, Any]] = []
    best = float("inf")
    for point in ordered:
        if point["rel"] < best:
            front.append(point)
            best = point["rel"]
    return front


def main() -> None:
    gpu = load_gpu()
    cs3 = load_cs3()

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.15), sharey=True)

    legend_order = [
        "cuBLAS FP64",
        "cuBLAS FP32",
        "cuBLAS TF32",
        "cuBLAS FP16/FP32",
        "APE FP32T",
        "ozIMMU",
        "ozIMMU EF",
        "ozIMMU RN",
        "ozIMMU H",
        "ozIMMU BLOCK",
        "GEMMul8",
        "CS-3 MeshGEMM FP32",
        "CS-3 Ozaki skip A1B1",
        "CS-3 Ozaki skip + TwoSum",
        "Pareto frontier",
        "Zero vs. FP64 reference",
    ]
    seen: set[str] = set()

    for axis, n in zip(axes, PANELS):
        points = list(gpu.get(n, []))
        for label, (rel, seconds) in cs3[n].items():
            points.append(
                {
                    "impl": label,
                    "ms": seconds * 1e3,
                    "rel": rel,
                    "family": "cs3",
                }
            )

        front = pareto_front(points)
        front_ids = {(p["impl"], p["ms"], p["rel"]) for p in front}
        axis.plot(
            [p["ms"] for p in front],
            [p["rel"] for p in front],
            linestyle=(0, (5, 3.5)),
            color="0.15",
            linewidth=1.15,
            zorder=2,
        )
        seen.add("Pareto frontier")

        for point in points:
            impl = point["impl"]
            on_front = (impl, point["ms"], point["rel"]) in front_ids
            if impl in CS3_STYLE:
                style = CS3_STYLE[impl]
                size = style["size"]
            else:
                style = GPU_STYLE[impl]
                size = 72 if style["marker"] != "v" else 78
            axis.scatter(
                point["ms"],
                point["rel"],
                s=size,
                c=style["color"],
                marker=style["marker"],
                edgecolors="black" if on_front else "none",
                linewidths=1.05 if on_front else 0.0,
                zorder=style["z"],
            )
            seen.add(impl)

        axis.set_xscale("log")
        axis.set_yscale("log")
        xs = [p["ms"] for p in points]
        axis.set_xlim(min(xs) / 1.45, max(xs) * 1.55)
        axis.set_ylim(6e-18, 5.5e-4)
        axis.set_title(f"N = {n:,}, T = {n // 512}", pad=8)
        axis.set_xlabel("Median latency [ms]")
        axis.xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        axis.yaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.grid(True, which="major", color="0.88", linewidth=0.8)
        axis.grid(True, which="minor", color="0.94", linewidth=0.5)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_color("0.55")

    axes[0].set_ylabel("Frobenius relative error")

    handles: list[Any] = []
    labels: list[str] = []
    for name in legend_order:
        if name not in seen:
            continue
        if name == "Pareto frontier":
            handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle=(0, (5, 3.5)),
                    color="0.15",
                    linewidth=1.15,
                )
            )
        elif name in CS3_STYLE:
            style = CS3_STYLE[name]
            handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="None",
                    marker=style["marker"],
                    markersize=8,
                    markerfacecolor=style["color"],
                    markeredgecolor="0.2",
                    markeredgewidth=0.8,
                )
            )
        else:
            style = GPU_STYLE[name]
            handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="None",
                    marker=style["marker"],
                    markersize=7.5,
                    markerfacecolor=style["color"],
                    markeredgecolor="none",
                )
            )
        labels.append(name)

    fig.legend(
        handles,
        labels,
        title="Implementation / annotation",
        loc="center left",
        bbox_to_anchor=(1.01, 0.52),
        frameon=True,
        fancybox=False,
        edgecolor="0.75",
        borderpad=0.7,
        handletextpad=0.55,
    )
    fig.suptitle("Accuracy-Latency Trade-off", y=1.02, fontsize=14)
    fig.text(
        0.5,
        -0.02,
        "WSE-3: 1.4 GHz, grid 512×512, φ=1",
        ha="center",
        fontsize=9,
        color="0.25",
    )
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.96))
    for dest in (OUT_PNG, OUT_FIG):
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(OUT_PNG)
    for n in PANELS:
        print(f"N={n}")
        for label, (rel, seconds) in cs3[n].items():
            print(f"  {label}: rel={rel:.4e}  {seconds*1e3:.4g} ms")
        print(f"  GPU impls: {', '.join(p['impl'] for p in gpu.get(n, []))}")


if __name__ == "__main__":
    main()
