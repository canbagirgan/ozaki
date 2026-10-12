#!/usr/bin/env python3
"""Aggregate raw samples and generate the CS-3 roofline figures."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ozaki_roofline.fitting import (  # noqa: E402
    fit_summa_coefficients,
    fit_transfer,
    summarize,
)
from ozaki_roofline.model import (  # noqa: E402
    DEFAULT_GRID,
    NOMINAL_CLOCK_MHZ,
    PRECISIONS,
    GemmShape,
    Grid,
    dense_compute_ceiling_flops,
    packed_collective_words,
    effective_host_io_bandwidth_Bps,
    predict_cold_e2e_seconds,
    scaled_sram_bandwidth_Bps,
    summa_implementation_sram_oi,
)
from ozaki_roofline.records import load_jsonl, write_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bandwidth",
        type=Path,
        default=ROOT / "results" / "raw" / "bandwidth.jsonl",
    )
    parser.add_argument(
        "--gemm",
        type=Path,
        default=ROOT / "results" / "raw" / "gemm.jsonl",
    )
    parser.add_argument(
        "--bandwidth-direct",
        type=Path,
        default=ROOT / "results" / "raw" / "bandwidth-direct.jsonl",
    )
    parser.add_argument(
        "--gemm-direct",
        type=Path,
        default=ROOT / "results" / "raw" / "gemm-direct.jsonl",
    )
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
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "p64.json",
    )
    return parser.parse_args()


def group_by(
    records: Iterable[dict[str, Any]],
    key: Callable[[dict[str, Any]], tuple[Any, ...]],
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[key(record)].append(record)
    return groups


def bandwidth_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = group_by(records, lambda row: (row["precision"], row["direction"], row["bytes"]))
    for (precision, direction, byte_count), samples in sorted(groups.items()):
        container_bytes = samples[0].get("host_container_bytes", byte_count)
        device_time = summarize(row["device_seconds"] for row in samples)
        host_time = summarize(
            row.get("host_blocking_s", row.get("host_sequence_s")) for row in samples
        )
        device_bw = summarize(row["device_bandwidth_Bps"] for row in samples)
        host_bw = summarize(
            row.get("host_blocking_bandwidth_Bps", row.get("host_sequence_bandwidth_Bps"))
            for row in samples
        )
        rows.append(
            {
                "precision": precision,
                "direction": direction,
                "bytes": byte_count,
                "host_container_bytes": container_bytes,
                "channels": samples[0]["channels"],
                "elements_per_pe": samples[0]["elements_per_pe"],
                "samples": len(samples),
                "device_seconds_median": device_time.median,
                "device_seconds_p10": device_time.p10,
                "device_seconds_p90": device_time.p90,
                "host_blocking_seconds_median": host_time.median,
                "host_blocking_seconds_p10": host_time.p10,
                "host_blocking_seconds_p90": host_time.p90,
                "device_bandwidth_Bps_median": device_bw.median,
                "host_blocking_bandwidth_Bps_median": host_bw.median,
                "host_container_bandwidth_Bps_median": container_bytes / host_time.median,
            }
        )
    return rows


def transfer_fits(rows: list[dict[str, Any]]) -> dict[tuple[str, str], Any]:
    fits = {}
    groups = group_by(rows, lambda row: (row["precision"], row["direction"]))
    for key, samples in groups.items():
        ordered = sorted(samples, key=lambda row: row["bytes"])
        fits[key] = fit_transfer(
            [row["bytes"] for row in ordered],
            [row["host_blocking_seconds_median"] for row in ordered],
        )
    return fits


def direct_bandwidth_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = group_by(records, lambda row: (row["precision"], row["direction"], row["bytes"]))
    for (precision, direction, byte_count), samples in sorted(groups.items()):
        container_bytes = samples[0].get("host_container_bytes", byte_count)
        elapsed = summarize(row["worker_blocking_s"] for row in samples)
        bandwidth = summarize(row["worker_blocking_bandwidth_Bps"] for row in samples)
        rows.append(
            {
                "precision": precision,
                "direction": direction,
                "bytes": byte_count,
                "host_container_bytes": container_bytes,
                "channels": samples[0]["channels"],
                "elements_per_pe": samples[0]["elements_per_pe"],
                "samples": len(samples),
                "worker_blocking_seconds_median": elapsed.median,
                "worker_blocking_seconds_p10": elapsed.p10,
                "worker_blocking_seconds_p90": elapsed.p90,
                "worker_blocking_bandwidth_Bps_median": bandwidth.median,
                "host_container_bandwidth_Bps_median": container_bytes / elapsed.median,
            }
        )
    return rows


def direct_transfer_fits(rows: list[dict[str, Any]]) -> dict[tuple[str, str], Any]:
    fits = {}
    groups = group_by(rows, lambda row: (row["precision"], row["direction"]))
    for key, samples in groups.items():
        ordered = sorted(samples, key=lambda row: row["bytes"])
        if len(ordered) < 2:
            continue
        fits[key] = fit_transfer(
            [row["bytes"] for row in ordered],
            [row["worker_blocking_seconds_median"] for row in ordered],
        )
    return fits


def host_container_transfer_fits(
    rows: list[dict[str, Any]],
    *,
    elapsed_key: str,
) -> dict[tuple[str, str], Any]:
    """Fit against Python host-buffer bytes rather than useful payload bytes."""

    fits = {}
    groups = group_by(rows, lambda row: (row["precision"], row["direction"]))
    for key, samples in groups.items():
        ordered = sorted(samples, key=lambda row: row["host_container_bytes"])
        if len(ordered) < 2:
            continue
        fits[key] = fit_transfer(
            [row["host_container_bytes"] for row in ordered],
            [row[elapsed_key] for row in ordered],
        )
    return fits


def _accuracy_fields(validation: dict[str, Any] | None) -> dict[str, Any]:
    payload = validation or {}
    reference = payload.get("against_float64_reference") or {}
    return {
        "validation_method": payload.get("method"),
        "validation_max_abs_error": reference.get("max_abs_error"),
        "validation_mean_abs_error": reference.get("mean_abs_error"),
        "validation_rms_abs_error": reference.get("rms_abs_error"),
        "validation_relative_frobenius_error": reference.get("relative_frobenius_error"),
        "validation_relative_max_error": reference.get("relative_max_error"),
        "validation_normwise_residual": payload.get("normwise_residual"),
        "validation_scaled_residual": payload.get("scaled_residual"),
        "validation_k_times_unit_roundoff": payload.get("k_times_unit_roundoff"),
        "validation_rounded_max_abs_error": (
            (payload.get("against_correctly_rounded_reference") or {}).get("max_abs_error")
        ),
        "validation_rounded_relative_frobenius_error": (
            (payload.get("against_correctly_rounded_reference") or {}).get(
                "relative_frobenius_error"
            )
        ),
    }


def normalize_direct_gemm_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in records:
        timing = row["timing"]
        normalized.append(
            {
                **row,
                "timing": {
                    "host_h2d_s": timing["worker_h2d_s"],
                    "host_kernel_dispatch_s": timing["worker_kernel_dispatch_s"],
                    "host_post_dispatch_wait_and_d2h_s": timing[
                        "worker_post_dispatch_wait_and_d2h_s"
                    ],
                    "host_d2h_s": timing["worker_d2h_s"],
                    "component_sum_s": timing["worker_component_sum_s"],
                    "operational_e2e_s": timing["worker_operational_e2e_s"],
                    "device_kernel_s": timing["device_kernel_s"],
                    "device_kernel_cycles": timing["device_kernel_cycles"],
                },
            }
        )
    return normalized


def gemm_allocation_summaries(
    records: list[dict[str, Any]],
    *,
    host_scope: str,
) -> list[dict[str, Any]]:
    """Reduce repeated samples to one row per independent WSE allocation."""

    rows: list[dict[str, Any]] = []
    groups = group_by(
        records,
        lambda row: (
            row["run_id"],
            row["precision"],
            row["grid"]["width"],
            row["grid"]["height"],
            row["shape"]["n"],
            row["shape"]["k"],
            row["shape"]["m"],
        ),
    )
    for (
        run_id,
        precision,
        grid_width,
        grid_height,
        n,
        k,
        m,
    ), samples in sorted(groups.items()):
        shape = GemmShape(n, k, m)
        h2d_s = summarize(row["timing"]["host_h2d_s"] for row in samples)
        d2h_s = summarize(row["timing"]["host_d2h_s"] for row in samples)
        e2e_s = summarize(row["timing"]["operational_e2e_s"] for row in samples)
        residuals = [
            row.get("validation", {}).get("relative_residual")
            for row in samples
            if row.get("validation", {}).get("relative_residual") is not None
        ]
        kernel_s = summarize(row["timing"]["device_kernel_s"] for row in samples)
        max_cycles = summarize(row["timing"]["device_kernel_cycles"]["maximum"] for row in samples)
        median_pe_cycles = summarize(
            row["timing"]["device_kernel_cycles"]["median"] for row in samples
        )
        min_cycles = summarize(row["timing"]["device_kernel_cycles"]["minimum"] for row in samples)
        mean_cycle_values = [
            row["timing"]["device_kernel_cycles"]["mean"]
            for row in samples
            if "mean" in row["timing"]["device_kernel_cycles"]
        ]
        mean_pe_cycles = summarize(mean_cycle_values) if mean_cycle_values else None
        seconds_per_cycle = kernel_s.median / max_cycles.median
        rows.append(
            {
                "run_id": run_id,
                "host_scope": host_scope,
                "precision": precision,
                "grid_width": grid_width,
                "grid_height": grid_height,
                "active_pes": grid_width * grid_height,
                "n": n,
                "k": k,
                "m": m,
                "samples": len(samples),
                "h2d_seconds_median": h2d_s.median,
                "d2h_seconds_median": d2h_s.median,
                "e2e_seconds_median": e2e_s.median,
                "kernel_device_seconds_median": kernel_s.median,
                "kernel_device_cycles_median": max_cycles.median,
                "kernel_min_pe_seconds_median": min_cycles.median * seconds_per_cycle,
                "kernel_median_pe_cycles_median": median_pe_cycles.median,
                "kernel_median_pe_seconds_median": (median_pe_cycles.median * seconds_per_cycle),
                "kernel_min_pe_cycles_median": min_cycles.median,
                "kernel_mean_pe_cycles_median": (mean_pe_cycles.median if mean_pe_cycles else None),
                "kernel_mean_pe_seconds_median": (
                    mean_pe_cycles.median * seconds_per_cycle if mean_pe_cycles else None
                ),
                "kernel_pe_makespan_imbalance": (max_cycles.median / median_pe_cycles.median),
                "kernel_pe_max_to_mean_ratio": (
                    max_cycles.median / mean_pe_cycles.median if mean_pe_cycles else None
                ),
                "device_kernel_flops": shape.flops / kernel_s.median,
                "operational_e2e_flops": shape.flops / e2e_s.median,
                "validation_relative_residual": (
                    summarize(residuals).median if residuals else None
                ),
                **_accuracy_fields(samples[0].get("validation")),
            }
        )
    return rows


def device_allocation_summaries(
    allocations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize device performance across independent allocation medians."""

    rows: list[dict[str, Any]] = []
    groups = group_by(
        allocations,
        lambda row: (
            row["precision"],
            row["grid_width"],
            row["grid_height"],
        ),
    )
    for (
        precision,
        grid_width,
        grid_height,
    ), precision_allocations in sorted(groups.items()):
        first = precision_allocations[0]
        shape = GemmShape(first["n"], first["k"], first["m"])
        grid = Grid(grid_width, grid_height)
        seconds = summarize(row["kernel_device_seconds_median"] for row in precision_allocations)
        fastest_pe_seconds = summarize(
            row["kernel_min_pe_seconds_median"] for row in precision_allocations
        )
        median_pe_seconds = summarize(
            row["kernel_median_pe_seconds_median"] for row in precision_allocations
        )
        mean_pe_values = [
            row["kernel_mean_pe_seconds_median"]
            for row in precision_allocations
            if row["kernel_mean_pe_seconds_median"] is not None
        ]
        mean_pe_seconds = summarize(mean_pe_values) if mean_pe_values else None
        throughput = summarize(row["device_kernel_flops"] for row in precision_allocations)
        imbalance = summarize(row["kernel_pe_makespan_imbalance"] for row in precision_allocations)
        rows.append(
            {
                "precision": precision,
                "grid_width": grid_width,
                "grid_height": grid_height,
                "active_pes": grid.active_pes,
                "n": shape.n,
                "k": shape.k,
                "m": shape.m,
                "allocations": len(precision_allocations),
                "kernel_device_seconds_median": seconds.median,
                "kernel_device_seconds_min": min(
                    row["kernel_device_seconds_median"] for row in precision_allocations
                ),
                "kernel_device_seconds_max": max(
                    row["kernel_device_seconds_median"] for row in precision_allocations
                ),
                "kernel_fastest_pe_seconds_median": fastest_pe_seconds.median,
                "kernel_mean_pe_seconds_median": (
                    mean_pe_seconds.median if mean_pe_seconds else None
                ),
                "kernel_mean_pe_allocations": len(mean_pe_values),
                "kernel_median_pe_seconds_median": median_pe_seconds.median,
                "device_kernel_flops_median": throughput.median,
                "device_kernel_flops_p10": throughput.p10,
                "device_kernel_flops_p90": throughput.p90,
                "kernel_pe_makespan_imbalance_median": imbalance.median,
                "device_sram_oi_dsd_flops_per_byte": summa_implementation_sram_oi(
                    shape,
                    PRECISIONS[precision],
                    grid,
                    include_collective_receiver_writes=True,
                ),
            }
        )
    return rows


def gemm_summaries(
    records: list[dict[str, Any]],
    fits: dict[tuple[str, str], Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = group_by(
        records,
        lambda row: (
            row["precision"],
            row["shape"]["n"],
            row["shape"]["k"],
            row["shape"]["m"],
        ),
    )
    for (precision_name, n, k, m), samples in sorted(groups.items()):
        precision = PRECISIONS[precision_name]
        shape = GemmShape(n, k, m)
        validation = samples[0].get("validation") or {}
        h2d = summarize(row["timing"]["host_h2d_s"] for row in samples)
        kernel_dispatch = summarize(row["timing"]["host_kernel_dispatch_s"] for row in samples)
        wait_and_d2h = summarize(
            row["timing"]["host_post_dispatch_wait_and_d2h_s"] for row in samples
        )
        kernel_device = summarize(row["timing"]["device_kernel_s"] for row in samples)
        kernel_cycles = summarize(
            row["timing"]["device_kernel_cycles"]["maximum"] for row in samples
        )
        kernel_min_pe_cycles = summarize(
            row["timing"]["device_kernel_cycles"]["minimum"] for row in samples
        )
        kernel_median_pe_cycles = summarize(
            row["timing"]["device_kernel_cycles"]["median"] for row in samples
        )
        kernel_mean_pe_values = [
            row["timing"]["device_kernel_cycles"]["mean"]
            for row in samples
            if "mean" in row["timing"]["device_kernel_cycles"]
        ]
        kernel_mean_pe_cycles = summarize(kernel_mean_pe_values) if kernel_mean_pe_values else None
        seconds_per_cycle = kernel_device.median / kernel_cycles.median
        d2h = summarize(row["timing"]["host_d2h_s"] for row in samples)
        component_sum = summarize(row["timing"]["component_sum_s"] for row in samples)
        e2e = summarize(row["timing"]["operational_e2e_s"] for row in samples)
        device_throughput = shape.flops / kernel_device.median
        e2e_throughput = shape.flops / e2e.median

        predicted_e2e = None
        predicted_throughput = None
        if (precision_name, "h2d") in fits and (precision_name, "d2h") in fits:
            predicted_e2e = predict_cold_e2e_seconds(
                shape,
                precision,
                fits[(precision_name, "h2d")],
                fits[(precision_name, "d2h")],
                kernel_seconds=kernel_device.median,
                aggregate_h2d_calls=False,
            )
            predicted_throughput = shape.flops / predicted_e2e

        rows.append(
            {
                "precision": precision_name,
                "n": n,
                "k": k,
                "m": m,
                "mt": samples[0]["local_tile"]["mt"],
                "kt": samples[0]["local_tile"]["kt"],
                "nt": samples[0]["local_tile"]["nt"],
                "samples": len(samples),
                "allocations": len(
                    {row.get("run_id", "unidentified-allocation") for row in samples}
                ),
                "flops": shape.flops,
                "host_wire_bytes": shape.host_wire_bytes(precision),
                "host_container_bytes": samples[0].get(
                    "host_container_bytes",
                    4 * (n * k + k * m + n * m),
                ),
                "host_oi_flops_per_byte": shape.host_operational_intensity(precision),
                "host_container_oi_flops_per_byte": shape.flops
                / samples[0].get(
                    "host_container_bytes",
                    4 * (n * k + k * m + n * m),
                ),
                "device_sram_oi_dsd_flops_per_byte": summa_implementation_sram_oi(
                    shape,
                    precision,
                    include_collective_receiver_writes=True,
                ),
                "h2d_seconds_median": h2d.median,
                "h2d_seconds_p10": h2d.p10,
                "h2d_seconds_p90": h2d.p90,
                "kernel_dispatch_seconds_median": kernel_dispatch.median,
                "post_dispatch_wait_and_d2h_seconds_median": wait_and_d2h.median,
                "kernel_device_seconds_median": kernel_device.median,
                "kernel_device_seconds_p10": kernel_device.p10,
                "kernel_device_seconds_p90": kernel_device.p90,
                "kernel_device_cycles_median": kernel_cycles.median,
                "kernel_device_cycles_p10": kernel_cycles.p10,
                "kernel_device_cycles_p90": kernel_cycles.p90,
                "kernel_min_pe_cycles_median": kernel_min_pe_cycles.median,
                "kernel_min_pe_seconds_median": (kernel_min_pe_cycles.median * seconds_per_cycle),
                "kernel_mean_pe_cycles_median": (
                    kernel_mean_pe_cycles.median if kernel_mean_pe_cycles else None
                ),
                "kernel_mean_pe_seconds_median": (
                    kernel_mean_pe_cycles.median * seconds_per_cycle
                    if kernel_mean_pe_cycles
                    else None
                ),
                "kernel_median_pe_cycles_median": kernel_median_pe_cycles.median,
                "kernel_median_pe_seconds_median": (
                    kernel_median_pe_cycles.median * seconds_per_cycle
                ),
                "kernel_pe_makespan_imbalance": (
                    kernel_cycles.median / kernel_median_pe_cycles.median
                ),
                "d2h_seconds_median": d2h.median,
                "d2h_seconds_p10": d2h.p10,
                "d2h_seconds_p90": d2h.p90,
                "component_sum_seconds_median": component_sum.median,
                "e2e_seconds_median": e2e.median,
                "e2e_to_component_sum_ratio": e2e.median / component_sum.median,
                "e2e_seconds_p10": e2e.p10,
                "e2e_seconds_p90": e2e.p90,
                "device_kernel_flops": device_throughput,
                "operational_e2e_flops": e2e_throughput,
                "predicted_e2e_seconds": predicted_e2e,
                "predicted_e2e_flops": predicted_throughput,
                "validation_relative_residual": validation.get("relative_residual"),
                **_accuracy_fields(validation),
            }
        )
    return rows


def fit_gemm_cycle_models(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit the four-term SUMMA cycle model and annotate processed rows."""

    models = {}
    groups = group_by(rows, lambda row: (row["precision"],))
    for (precision_name,), samples in groups.items():
        if len(samples) < 4:
            continue
        precision = PRECISIONS[precision_name]
        features = []
        cycles = []
        for row in samples:
            mt, kt, nt = row["mt"], row["kt"], row["nt"]
            p = DEFAULT_GRID.width
            features.append(
                [
                    p * mt * kt * nt,
                    p * kt * nt,
                    p
                    * (
                        packed_collective_words(mt * kt, precision)
                        + packed_collective_words(kt * nt, precision)
                    ),
                    p,
                ]
            )
            cycles.append(row["kernel_device_cycles_median"])
        model = fit_summa_coefficients(features, cycles)
        models[precision_name] = model
        for row in samples:
            shape = GemmShape(row["n"], row["k"], row["m"])
            predicted = model.predict_cycles(shape, precision, DEFAULT_GRID)
            row["cycle_model_predicted_cycles"] = predicted
            row["cycle_model_relative_error"] = (
                predicted - row["kernel_device_cycles_median"]
            ) / row["kernel_device_cycles_median"]
    return models


def save_figure(fig: plt.Figure, figures: Path, name: str) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / f"{name}.svg", bbox_inches="tight")
    fig.savefig(figures / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_bandwidth(
    rows: list[dict[str, Any]],
    direct_rows: list[dict[str, Any]],
    figures: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
    for axis, precision in zip(axes, ("fp16", "fp32"), strict=True):
        for direction, marker in (("h2d", "o"), ("d2h", "s")):
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["precision"] == precision and row["direction"] == direction
                ),
                key=lambda row: row["bytes"],
            )
            axis.plot(
                [row["bytes"] / 1e6 for row in selected],
                [row["device_bandwidth_Bps_median"] / 1e9 for row in selected],
                marker=marker,
                label=f"{direction.upper()} WSE TSC bracket",
            )
            axis.plot(
                [row["bytes"] / 1e6 for row in selected],
                [row["host_blocking_bandwidth_Bps_median"] / 1e9 for row in selected],
                marker=marker,
                linestyle="--",
                label=f"{direction.upper()} user-node API",
            )
            direct = sorted(
                (
                    row
                    for row in direct_rows
                    if row["precision"] == precision and row["direction"] == direction
                ),
                key=lambda row: row["bytes"],
            )
            if direct:
                axis.plot(
                    [row["bytes"] / 1e6 for row in direct],
                    [row["worker_blocking_bandwidth_Bps_median"] / 1e9 for row in direct],
                    marker=marker,
                    linestyle=":",
                    label=f"{direction.upper()} appliance worker",
                )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(f"{precision.upper()} transfer payload")
        axis.set_xlabel("Payload (MB)")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("Effective bandwidth (GB/s)")
    axes[1].legend(fontsize=8)
    fig.suptitle("CS-3 host/WSE transfer bandwidth — 64x64 PEs, 16 I/O channels")
    fig.text(
        0.5,
        -0.02,
        "Source: ALCF WSE-3 measurements; median of repeated single transfers",
        ha="center",
        fontsize=8,
    )
    save_figure(fig, figures, "bandwidth")


def _plot_naive_roofs(
    axis: Any,
    oi: np.ndarray,
    bandwidth_by_precision: dict[str, float],
) -> None:
    for precision_name, color in (("fp16", "#4C78A8"), ("fp32", "#F58518")):
        if precision_name not in bandwidth_by_precision:
            continue
        peak = dense_compute_ceiling_flops(PRECISIONS[precision_name], DEFAULT_GRID)
        roof = np.minimum(peak, bandwidth_by_precision[precision_name] * oi)
        axis.plot(
            oi,
            roof / 1e12,
            color=color,
            label=f"{precision_name.upper()} roofline",
        )


def _scatter_square_series(
    axis: Any,
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
) -> None:
    for precision_name, marker in (("fp16", "o"), ("fp32", "s")):
        selected = sorted(
            (
                row
                for row in rows
                if row["precision"] == precision_name and row["n"] == row["k"] == row["m"]
            ),
            key=lambda row: row[x_key],
        )
        if not selected:
            continue
        axis.scatter(
            [row[x_key] for row in selected],
            [row[y_key] / 1e12 for row in selected],
            marker=marker,
            s=42,
            label=f"{precision_name.upper()} measured",
        )


def plot_device_roofline(rows: list[dict[str, Any]], figures: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.4, 5.2))
    oi = np.logspace(-2, 1, 400)
    sram_bw = scaled_sram_bandwidth_Bps(DEFAULT_GRID)
    _plot_naive_roofs(axis, oi, {"fp16": sram_bw, "fp32": sram_bw})
    _scatter_square_series(
        axis,
        rows,
        x_key="device_sram_oi_dsd_flops_per_byte",
        y_key="device_kernel_flops",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("SUMMA SRAM operational intensity (FLOP/byte)")
    axis.set_ylabel("Device kernel throughput (TFLOP/s)")
    axis.set_title("Device kernel roofline: SUMMA on 64x64 PEs")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)
    fig.text(
        0.5,
        0.01,
        "Naive roofline: 21 PB/s vendor SRAM scaled to 4,096 PEs; "
        "Q is implementation SRAM traffic; kernel time is the worst PE",
        ha="center",
        fontsize=8,
    )
    save_figure(fig, figures, "device-kernel-roofline")


def plot_e2e(
    rows: list[dict[str, Any]],
    figures: Path,
    transfer_fits_by_precision: dict[tuple[str, str], Any] | None = None,
) -> None:
    fig, axis = plt.subplots(figsize=(7.4, 5.2))
    oi = np.logspace(1, 3.4, 400)
    bandwidths = {}
    if transfer_fits_by_precision:
        for precision_name in ("fp16", "fp32"):
            if (precision_name, "h2d") not in transfer_fits_by_precision:
                continue
            bandwidths[precision_name] = effective_host_io_bandwidth_Bps(
                transfer_fits_by_precision[(precision_name, "h2d")],
                transfer_fits_by_precision[(precision_name, "d2h")],
            )
    _plot_naive_roofs(axis, oi, bandwidths)
    _scatter_square_series(
        axis,
        rows,
        x_key="host_oi_flops_per_byte",
        y_key="operational_e2e_flops",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Cold host-wire operational intensity (FLOP/byte)")
    axis.set_ylabel("Operational E2E throughput (TFLOP/s)")
    axis.set_title("E2E roofline: H2D(A,B) + SUMMA + D2H(C)")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)
    fig.text(
        0.5,
        0.01,
        r"Naive roofline $P=\min(\pi,\beta I)$ with worker $\beta$ from H2D/D2H fits; "
        "points are measured H2D + kernel + D2H",
        ha="center",
        fontsize=8,
    )
    save_figure(fig, figures, "e2e-roofline")


def plot_breakdown(
    rows: list[dict[str, Any]],
    figures: Path,
    *,
    scope: str,
    name: str,
) -> None:
    square = sorted(
        (row for row in rows if row["n"] == row["k"] == row["m"]),
        key=lambda row: (row["precision"], row["n"]),
    )
    labels = [f"{row['precision']}\n{row['n']}" for row in square]
    x = np.arange(len(square))
    h2d = np.asarray([row["h2d_seconds_median"] * 1e3 for row in square])
    kernel = np.asarray([row["kernel_device_seconds_median"] * 1e3 for row in square])
    d2h = np.asarray([row["d2h_seconds_median"] * 1e3 for row in square])
    measured_e2e = np.asarray([row["e2e_seconds_median"] * 1e3 for row in square])
    fig, axis = plt.subplots(figsize=(10.5, 4.8))
    axis.bar(x, h2d, label="H2D A+B")
    axis.bar(x, kernel, bottom=h2d, label="Device kernel")
    axis.bar(x, d2h, bottom=h2d + kernel, label="Isolated D2H C")
    axis.scatter(x, measured_e2e, marker="_", s=130, linewidths=2, label="Measured E2E")
    axis.set_xticks(x, labels)
    axis.set_xlabel("Precision and square dimension n=k=m")
    axis.set_ylabel("Median time (ms)")
    axis.set_title(f"Cold-input GEMM component time — {scope}")
    axis.legend()
    axis.grid(True, axis="y", alpha=0.25)
    fig.text(
        0.5,
        -0.08,
        "D2H is a repeated ready-result read; E2E ends at the first synchronizing C read",
        ha="center",
        fontsize=8,
    )
    save_figure(fig, figures, name)


def main() -> None:
    args = parse_args()
    if not args.bandwidth.exists() or not args.gemm.exists():
        raise FileNotFoundError("both bandwidth and GEMM raw JSONL files are required")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selected_shapes = {(shape["n"], shape["k"], shape["m"]) for shape in config["gemm"]["shapes"]}

    def is_selected_gemm(row: dict[str, Any]) -> bool:
        shape = row.get("shape", {})
        grid = row.get("grid", {})
        return (
            (shape.get("n"), shape.get("k"), shape.get("m")) in selected_shapes
            and grid.get("width") == config["grid"]["width"]
            and grid.get("height") == config["grid"]["height"]
        )

    bandwidth_records = [
        row
        for row in load_jsonl(args.bandwidth)
        if row.get("kind") == "bandwidth_sample" and not row.get("simulator", False)
    ]
    gemm_records = [
        row
        for row in load_jsonl(args.gemm)
        if row.get("kind") == "gemm_sample"
        and row.get("schema_version", 1) >= 2
        and not row.get("simulator", False)
        and is_selected_gemm(row)
    ]
    if not bandwidth_records or not gemm_records:
        raise ValueError("real-WSE bandwidth and GEMM sample records are required")
    direct_records = (
        [
            row
            for row in load_jsonl(args.bandwidth_direct)
            if row.get("kind") == "bandwidth_direct_sample"
        ]
        if args.bandwidth_direct.exists()
        else []
    )
    direct_gemm_records = (
        [
            row
            for row in load_jsonl(args.gemm_direct)
            if row.get("kind") == "gemm_direct_sample" and row.get("schema_version", 1) >= 2
            if is_selected_gemm(row)
        ]
        if args.gemm_direct.exists()
        else []
    )
    bandwidth_rows = bandwidth_summaries(bandwidth_records)
    direct_bandwidth_rows = direct_bandwidth_summaries(direct_records)
    fits = transfer_fits(bandwidth_rows)
    direct_fits = direct_transfer_fits(direct_bandwidth_rows)
    container_fits = host_container_transfer_fits(
        bandwidth_rows,
        elapsed_key="host_blocking_seconds_median",
    )
    direct_container_fits = host_container_transfer_fits(
        direct_bandwidth_rows,
        elapsed_key="worker_blocking_seconds_median",
    )
    normalized_direct_gemm_records = normalize_direct_gemm_records(direct_gemm_records)
    gemm_rows = gemm_summaries(gemm_records, fits)
    direct_gemm_rows = gemm_summaries(
        normalized_direct_gemm_records,
        direct_fits,
    )
    allocation_rows = gemm_allocation_summaries(
        gemm_records,
        host_scope="user_node",
    ) + gemm_allocation_summaries(
        normalized_direct_gemm_records,
        host_scope="appliance_worker",
    )
    device_allocation_rows = device_allocation_summaries(allocation_rows)
    for row in gemm_rows:
        row["host_scope"] = "user_node"
    for row in direct_gemm_rows:
        row["host_scope"] = "appliance_worker"
    cycle_models = fit_gemm_cycle_models(direct_gemm_rows or gemm_rows)

    args.processed.mkdir(parents=True, exist_ok=True)
    write_csv(args.processed / "bandwidth.csv", bandwidth_rows)
    if direct_bandwidth_rows:
        write_csv(args.processed / "bandwidth-direct.csv", direct_bandwidth_rows)
    write_csv(args.processed / "gemm.csv", gemm_rows)
    if direct_gemm_rows:
        write_csv(args.processed / "gemm-direct.csv", direct_gemm_rows)
    write_csv(args.processed / "gemm-allocations.csv", allocation_rows)
    write_csv(
        args.processed / "device-allocation-summary.csv",
        device_allocation_rows,
    )
    summary = {
        "grid": asdict(DEFAULT_GRID),
        "active_pes": DEFAULT_GRID.active_pes,
        "channels": 16,
        "nominal_clock_mhz": NOMINAL_CLOCK_MHZ,
        "gemm_record_schema": 2,
        "primary_gemm_host_scope": ("appliance_worker" if direct_gemm_rows else "user_node"),
        "selected_gemm_shapes": [{"n": n, "k": k, "m": m} for n, k, m in sorted(selected_shapes)],
        "gemm_measurements": {
            "appliance_worker": direct_gemm_rows,
            "user_node": gemm_rows,
        },
        "device_allocation_measurements": device_allocation_rows,
        "scaled_sram_bandwidth_Bps": scaled_sram_bandwidth_Bps(DEFAULT_GRID),
        "instruction_compute_ceiling_flops": {
            name: dense_compute_ceiling_flops(precision, DEFAULT_GRID)
            for name, precision in PRECISIONS.items()
        },
        "transfer_fits": {
            f"{precision}_{direction}": asdict(fit) for (precision, direction), fit in fits.items()
        },
        "direct_worker_transfer_fits": {
            f"{precision}_{direction}": asdict(fit)
            for (precision, direction), fit in direct_fits.items()
        },
        "user_node_host_container_fits": {
            f"{precision}_{direction}": asdict(fit)
            for (precision, direction), fit in container_fits.items()
        },
        "direct_worker_host_container_fits": {
            f"{precision}_{direction}": asdict(fit)
            for (precision, direction), fit in direct_container_fits.items()
        },
        "summa_cycle_models": {
            precision: asdict(model) for precision, model in cycle_models.items()
        },
    }
    (args.processed / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    plot_bandwidth(bandwidth_rows, direct_bandwidth_rows, args.figures)
    if not direct_gemm_rows:
        raise ValueError("worker/WSE GEMM samples are required for the roofline figures")
    plot_device_roofline(direct_gemm_rows, args.figures)
    plot_e2e(
        direct_gemm_rows,
        args.figures,
        transfer_fits_by_precision=direct_fits,
    )
    plot_breakdown(
        direct_gemm_rows,
        args.figures,
        scope="appliance worker / WSE",
        name="latency-breakdown-direct",
    )
    print(
        f"wrote {len(bandwidth_rows)} bandwidth groups and "
        f"{len(direct_gemm_rows)} worker GEMM groups"
    )


if __name__ == "__main__":
    main()
