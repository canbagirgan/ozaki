#!/usr/bin/env python3
"""Compare fixed-size GEMM measurements across 64x64 and 128x128 PE grids."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ozaki_roofline.fitting import summarize  # noqa: E402
from ozaki_roofline.model import (  # noqa: E402
    PRECISIONS,
    GemmShape,
    Grid,
    dense_compute_ceiling_flops,
    scaled_sram_bandwidth_Bps,
)
from ozaki_roofline.records import load_jsonl, write_csv  # noqa: E402
from scripts.analyze import (  # noqa: E402
    device_allocation_summaries,
    gemm_allocation_summaries,
    normalize_direct_gemm_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--p64",
        type=Path,
        default=ROOT / "results" / "raw" / "gemm-direct.jsonl",
    )
    parser.add_argument(
        "--p128",
        type=Path,
        default=ROOT / "results" / "raw" / "gemm-direct-p128.jsonl",
    )
    parser.add_argument(
        "--p64-user",
        type=Path,
        default=ROOT / "results" / "raw" / "gemm.jsonl",
    )
    parser.add_argument(
        "--p128-user",
        type=Path,
        default=ROOT / "results" / "raw" / "gemm-p128.jsonl",
    )
    parser.add_argument("--n", type=int, default=2560)
    parser.add_argument("--k", type=int, default=2560)
    parser.add_argument("--m", type=int, default=2560)
    parser.add_argument(
        "--processed",
        type=Path,
        default=ROOT / "results" / "processed",
    )
    parser.add_argument(
        "--figures",
        type=Path,
        default=ROOT / "results" / "figures",
    )
    return parser.parse_args()


def selected_records(
    path: Path,
    shape: GemmShape,
    grid_size: int,
    *,
    kind: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in load_jsonl(path)
        if row.get("kind") == kind
        and row.get("schema_version", 1) >= 2
        and row.get("shape") == {"n": shape.n, "k": shape.k, "m": shape.m}
        and row.get("grid", {}).get("width") == grid_size
        and row.get("grid", {}).get("height") == grid_size
    ]


def comparison_rows(
    allocations: list[dict[str, Any]],
    shape: GemmShape,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for allocation in allocations:
        key = (allocation["grid_width"], allocation["precision"])
        groups.setdefault(key, []).append(allocation)

    for (grid_size, precision), samples in sorted(groups.items()):
        grid = Grid(grid_size, grid_size)
        h2d = summarize(row["h2d_seconds_median"] for row in samples)
        kernel = summarize(row["kernel_device_seconds_median"] for row in samples)
        fastest_pe = summarize(row["kernel_min_pe_seconds_median"] for row in samples)
        median_pe = summarize(row["kernel_median_pe_seconds_median"] for row in samples)
        mean_pe_values = [
            row["kernel_mean_pe_seconds_median"]
            for row in samples
            if row["kernel_mean_pe_seconds_median"] is not None
        ]
        mean_pe = summarize(mean_pe_values) if mean_pe_values else None
        d2h = summarize(row["d2h_seconds_median"] for row in samples)
        e2e = summarize(row["e2e_seconds_median"] for row in samples)
        device_tput = summarize(row["device_kernel_flops"] for row in samples)
        e2e_tput = summarize(row["operational_e2e_flops"] for row in samples)
        imbalance = summarize(row["kernel_pe_makespan_imbalance"] for row in samples)
        residual = summarize(row["validation_relative_residual"] for row in samples)
        device_row = next(
            row for row in device_allocation_summaries(samples) if row["precision"] == precision
        )
        compute_roof = dense_compute_ceiling_flops(PRECISIONS[precision], grid)
        sram_roof = scaled_sram_bandwidth_Bps(grid)
        oi = device_row["device_sram_oi_dsd_flops_per_byte"]
        applicable_roof = min(compute_roof, sram_roof * oi)
        rows.append(
            {
                "grid": f"{grid_size}x{grid_size}",
                "grid_size": grid_size,
                "active_pes": grid.active_pes,
                "precision": precision,
                "n": shape.n,
                "k": shape.k,
                "m": shape.m,
                "local_tile": shape.n // grid_size,
                "allocations": len(samples),
                "samples": sum(row["samples"] for row in samples),
                "h2d_ms_median": 1e3 * h2d.median,
                "kernel_ms_median": 1e3 * kernel.median,
                "fastest_pe_ms_median": 1e3 * fastest_pe.median,
                "mean_pe_ms_median": 1e3 * mean_pe.median if mean_pe else None,
                "mean_pe_allocations": len(mean_pe_values),
                "median_pe_ms_median": 1e3 * median_pe.median,
                "worst_pe_ms_median": 1e3 * kernel.median,
                "kernel_ms_min_allocation": 1e3
                * min(row["kernel_device_seconds_median"] for row in samples),
                "kernel_ms_max_allocation": 1e3
                * max(row["kernel_device_seconds_median"] for row in samples),
                "d2h_ms_median": 1e3 * d2h.median,
                "e2e_ms_median": 1e3 * e2e.median,
                "device_tflops_median": device_tput.median / 1e12,
                "e2e_tflops_median": e2e_tput.median / 1e12,
                "makespan_to_median_pe": imbalance.median,
                "validation_relative_residual": residual.median,
                "implementation_sram_oi_flops_per_byte": oi,
                "effective_sram_TBps": device_tput.median / oi / 1e12,
                "scaled_sram_roof_TBps": sram_roof / 1e12,
                "compute_roof_TFLOPs": compute_roof / 1e12,
                "applicable_roof_TFLOPs": applicable_roof / 1e12,
                "applicable_roof_fraction": device_tput.median / applicable_roof,
            }
        )

    by_key = {(row["grid_size"], row["precision"]): row for row in rows}
    for precision in PRECISIONS:
        baseline = by_key.get((64, precision))
        scaled = by_key.get((128, precision))
        if baseline is None or scaled is None:
            continue
        kernel_speedup = baseline["kernel_ms_median"] / scaled["kernel_ms_median"]
        e2e_speedup = baseline["e2e_ms_median"] / scaled["e2e_ms_median"]
        scaled["kernel_speedup_vs_p64"] = kernel_speedup
        scaled["kernel_strong_scaling_efficiency"] = kernel_speedup / 4
        scaled["e2e_speedup_vs_p64"] = e2e_speedup
        scaled["e2e_strong_scaling_efficiency"] = e2e_speedup / 4
        baseline["kernel_speedup_vs_p64"] = 1.0
        baseline["kernel_strong_scaling_efficiency"] = 1.0
        baseline["e2e_speedup_vs_p64"] = 1.0
        baseline["e2e_strong_scaling_efficiency"] = 1.0
    return rows


def save_comparison_plot(
    rows: list[dict[str, Any]],
    figures: Path,
    *,
    name: str,
    host_scope: str,
    e2e_title: str,
) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    categories = ("fp16", "fp32")
    x = np.arange(len(categories))
    width = 0.34
    by_key = {(row["grid_size"], row["precision"]): row for row in rows}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for axis, metric, title in (
        (axes[0], "kernel_ms_median", "Device SUMMA makespan"),
        (axes[1], "e2e_ms_median", e2e_title),
    ):
        for offset, grid_size in ((-width / 2, 64), (width / 2, 128)):
            values = [by_key[(grid_size, precision)][metric] for precision in categories]
            axis.bar(x + offset, values, width, label=f"{grid_size}x{grid_size}")
        axis.set_xticks(x, [precision.upper() for precision in categories])
        axis.set_xlabel("Precision")
        axis.set_ylabel("Allocation-median time (ms)")
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend()
    fig.suptitle(f"Fixed 2560x2560x2560 GEMM strong scaling — {host_scope}")
    fig.text(
        0.5,
        -0.01,
        "Source: real WSE-3; 16 I/O channels; two warmups and nine samples per allocation",
        ha="center",
        fontsize=8,
    )
    fig.savefig(figures / f"{name}.svg", bbox_inches="tight")
    fig.savefig(figures / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_pe_spread_plot(
    rows: list[dict[str, Any]],
    figures: Path,
    *,
    name: str,
    host_scope: str,
) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    by_key = {(row["grid_size"], row["precision"]): row for row in rows}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True)
    stats = (
        ("fastest_pe_ms_median", "Fastest PE"),
        ("median_pe_ms_median", "Typical PE (median)"),
        ("worst_pe_ms_median", "Worst PE (reported)"),
    )
    x = np.arange(len(stats))
    width = 0.36
    for axis, grid_size in ((axes[0], 64), (axes[1], 128)):
        for offset, precision, color in (
            (-width / 2, "fp16", "#4C78A8"),
            (width / 2, "fp32", "#F58518"),
        ):
            values = [by_key[(grid_size, precision)][key] for key, _ in stats]
            axis.bar(x + offset, values, width, label=precision.upper(), color=color)
        axis.set_xticks(x, [label for _, label in stats])
        axis.set_title(f"{grid_size}x{grid_size} PE rectangle")
        axis.set_ylabel("Allocation-median time (ms)")
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend()
    fig.suptitle(f"Per-PE device time — {host_scope}")
    fig.text(
        0.5,
        -0.02,
        "Reported kernel time is the worst PE. Typical is the median PE. "
        "Arithmetic mean is omitted when the raw sample did not store it.",
        ha="center",
        fontsize=8,
    )
    fig.savefig(figures / f"{name}.svg", bbox_inches="tight")
    fig.savefig(figures / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    shape = GemmShape(args.n, args.k, args.m)
    p64 = selected_records(
        args.p64,
        shape,
        64,
        kind="gemm_direct_sample",
    )
    p128 = selected_records(
        args.p128,
        shape,
        128,
        kind="gemm_direct_sample",
    )
    if not p64 or not p128:
        raise ValueError("both 64x64 and 128x128 direct GEMM records are required")

    direct_allocations = gemm_allocation_summaries(
        normalize_direct_gemm_records(p64 + p128),
        host_scope="appliance_worker",
    )
    direct_allocations = [row for row in direct_allocations if row["samples"] >= 5]
    direct_rows = comparison_rows(direct_allocations, shape)

    p64_user = selected_records(
        args.p64_user,
        shape,
        64,
        kind="gemm_sample",
    )
    p128_user = selected_records(
        args.p128_user,
        shape,
        128,
        kind="gemm_sample",
    )
    user_allocations = gemm_allocation_summaries(
        p64_user + p128_user,
        host_scope="user_node",
    )
    user_allocations = [row for row in user_allocations if row["samples"] >= 5]
    user_rows = comparison_rows(user_allocations, shape)

    args.processed.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.processed / "grid-comparison-allocations.csv",
        direct_allocations,
    )
    write_csv(args.processed / "grid-comparison.csv", direct_rows)
    write_csv(
        args.processed / "grid-comparison-user-allocations.csv",
        user_allocations,
    )
    write_csv(args.processed / "grid-comparison-user.csv", user_rows)
    summary = {
        "shape": asdict(shape),
        "worker_comparison": direct_rows,
        "user_node_comparison": user_rows,
        "measurement_contract": {
            "channels": 16,
            "warmups_per_allocation": 2,
            "retained_samples_per_allocation": 9,
            "host_scope": "appliance_worker",
        },
    }
    (args.processed / "grid-comparison.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    save_comparison_plot(
        direct_rows,
        args.figures,
        name="grid-comparison",
        host_scope="appliance worker / WSE",
        e2e_title="Cold appliance-worker/WSE E2E",
    )
    save_comparison_plot(
        user_rows,
        args.figures,
        name="grid-comparison-user-node",
        host_scope="user-node appliance binding",
        e2e_title="Cold user-node E2E",
    )
    save_pe_spread_plot(
        direct_rows,
        args.figures,
        name="pe-time-spread",
        host_scope="appliance worker / WSE",
    )
    save_pe_spread_plot(
        user_rows,
        args.figures,
        name="pe-time-spread-user-node",
        host_scope="user-node appliance binding",
    )
    print(
        f"wrote comparison for {len(direct_allocations)} worker and "
        f"{len(user_allocations)} user-node allocations"
    )


if __name__ == "__main__":
    main()
