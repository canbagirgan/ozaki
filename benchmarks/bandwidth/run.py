#!/usr/bin/env python3
"""Measure 16-channel H2D and D2H transfer rates on a 64x64 PE grid."""

from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType, MemcpyOrder
from cerebras.sdk.client import SdkRuntime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ozaki_roofline.records import append_jsonl, utc_now_iso  # noqa: E402
from ozaki_roofline.timing import synchronized_makespan_cycles  # noqa: E402

P = 64
CHANNELS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=("fp16", "fp32"), required=True)
    parser.add_argument("--direction", choices=("h2d", "d2h", "both"), default="both")
    parser.add_argument("--simulator", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--clock-mhz", type=float, default=1400.0)
    parser.add_argument(
        "--lengths",
        help="comma-separated elements per PE; defaults span 16 KiB to 64 MiB globally",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results" / "raw" / "bandwidth.jsonl",
    )
    return parser.parse_args()


def load_manifest(precision: str) -> tuple[Path, dict]:
    name = f"bandwidth-p64-{precision}-ch16"
    path = ROOT / "artifacts" / "bandwidth" / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"compile manifest not found: {path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def default_lengths(precision: str) -> list[int]:
    base = [1, 4, 16, 64, 256, 1024, 4096]
    return [*base, 8192] if precision == "fp16" else base


def measure_once(
    runner: SdkRuntime,
    *,
    direction: str,
    symbol_a: int,
    symbol_time: int,
    symbol_ref: int,
    host_input: np.ndarray,
    host_output: np.ndarray,
    elements_per_pe: int,
    memcpy_dtype: MemcpyDataType,
) -> tuple[int, float, float]:
    # First reproduce the SDK bandwidth-test sequence for device TSC timing.
    runner.call("f_sync", [], nonblock=False)
    runner.call("f_tic", [], nonblock=True)

    marker_start = time.perf_counter_ns()
    if direction == "h2d":
        transfer_task = runner.memcpy_h2d(
            symbol_a,
            host_input,
            0,
            0,
            P,
            P,
            elements_per_pe,
            streaming=False,
            data_type=memcpy_dtype,
            order=MemcpyOrder.COL_MAJOR,
            nonblock=True,
        )
    else:
        transfer_task = runner.memcpy_d2h(
            host_output,
            symbol_a,
            0,
            0,
            P,
            P,
            elements_per_pe,
            streaming=False,
            data_type=memcpy_dtype,
            order=MemcpyOrder.COL_MAJOR,
            nonblock=True,
        )
    runner.call("f_toc", [], nonblock=False)
    if transfer_task is None:
        raise RuntimeError("nonblocking transfer did not return an SDK task")
    transfer_task.wait()
    marker_end = time.perf_counter_ns()

    runner.call("f_memcpy_timestamps", [], nonblock=False)
    time_raw = np.zeros(P * P * 3, dtype=np.float32)
    runner.memcpy_d2h(
        time_raw,
        symbol_time,
        0,
        0,
        P,
        P,
        3,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False,
    )
    runner.call("f_reference_timestamps", [], nonblock=False)
    ref_raw = np.zeros(P * P * 2, dtype=np.float32)
    runner.memcpy_d2h(
        ref_raw,
        symbol_ref,
        0,
        0,
        P,
        P,
        2,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False,
    )

    cycles = synchronized_makespan_cycles(
        time_raw.reshape(P, P, 3),
        ref_raw.reshape(P, P, 2),
    )

    # Then time one independent blocking API call. This is the transfer cost
    # used by the E2E affine fit; it excludes the f_toc marker RPC above.
    host_start = time.perf_counter_ns()
    if direction == "h2d":
        runner.memcpy_h2d(
            symbol_a,
            host_input,
            0,
            0,
            P,
            P,
            elements_per_pe,
            streaming=False,
            data_type=memcpy_dtype,
            order=MemcpyOrder.COL_MAJOR,
            nonblock=False,
        )
    else:
        runner.memcpy_d2h(
            host_output,
            symbol_a,
            0,
            0,
            P,
            P,
            elements_per_pe,
            streaming=False,
            data_type=memcpy_dtype,
            order=MemcpyOrder.COL_MAJOR,
            nonblock=False,
        )
    host_end = time.perf_counter_ns()
    return (
        cycles,
        (marker_end - marker_start) * 1e-9,
        (host_end - host_start) * 1e-9,
    )


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.repetitions <= 0:
        raise ValueError("warmup must be nonnegative and repetitions positive")
    if args.clock_mhz <= 0:
        raise ValueError("clock frequency must be positive")

    manifest_path, manifest = load_manifest(args.precision)
    bytes_per_element = 2 if args.precision == "fp16" else 4
    memcpy_dtype = (
        MemcpyDataType.MEMCPY_16BIT if args.precision == "fp16" else MemcpyDataType.MEMCPY_32BIT
    )
    lengths = (
        [int(value) for value in args.lengths.split(",")]
        if args.lengths
        else default_lengths(args.precision)
    )
    capacity = int(manifest["capacity_elements_per_pe"])
    if not lengths or min(lengths) <= 0 or max(lengths) > capacity:
        raise ValueError(f"lengths must be in [1, {capacity}] elements per PE")
    directions = ("h2d", "d2h") if args.direction == "both" else (args.direction,)

    run_id = f"bandwidth-p64-{args.precision}-ch16-{int(time.time())}"
    runtime_startup_start = time.perf_counter_ns()
    runtime = SdkRuntime(
        manifest["artifact_path"],
        simulator=args.simulator,
        disable_version_check=True,
    )
    with runtime as runner:
        runtime_startup_s = (time.perf_counter_ns() - runtime_startup_start) * 1e-9
        symbol_a = runner.get_id("A")
        symbol_time = runner.get_id("time_memcpy")
        symbol_ref = runner.get_id("time_ref")

        for elements_per_pe in lengths:
            buffer_preparation_start = time.perf_counter_ns()
            total_elements = P * P * elements_per_pe
            host_input = (np.arange(total_elements, dtype=np.uint32) * 2654435761) & 0xFFFFFFFF
            if args.precision == "fp16":
                host_input &= 0xFFFF
            host_output = np.zeros(total_elements, dtype=np.uint32)
            buffer_preparation_s = (time.perf_counter_ns() - buffer_preparation_start) * 1e-9

            for direction in directions:
                # Ensure D2H has deterministic source data. This prefill is not timed.
                if direction == "d2h":
                    runner.memcpy_h2d(
                        symbol_a,
                        host_input,
                        0,
                        0,
                        P,
                        P,
                        elements_per_pe,
                        streaming=False,
                        data_type=memcpy_dtype,
                        order=MemcpyOrder.COL_MAJOR,
                        nonblock=False,
                    )

                for iteration in range(args.warmup + args.repetitions):
                    cycles, host_marker_sequence_s, host_blocking_s = measure_once(
                        runner,
                        direction=direction,
                        symbol_a=symbol_a,
                        symbol_time=symbol_time,
                        symbol_ref=symbol_ref,
                        host_input=host_input,
                        host_output=host_output,
                        elements_per_pe=elements_per_pe,
                        memcpy_dtype=memcpy_dtype,
                    )
                    if iteration < args.warmup:
                        continue

                    byte_count = total_elements * bytes_per_element
                    device_seconds = cycles / (args.clock_mhz * 1e6)
                    record = {
                        "schema_version": 1,
                        "kind": "bandwidth_sample",
                        "run_id": run_id,
                        "started_at": utc_now_iso(),
                        "sample_index": iteration - args.warmup,
                        "precision": args.precision,
                        "direction": direction,
                        "grid": {"width": P, "height": P, "active_pes": P * P},
                        "channels": CHANNELS,
                        "elements_per_pe": elements_per_pe,
                        "bytes": byte_count,
                        "host_container_bytes": total_elements * 4,
                        "device_cycles": cycles,
                        "clock_mhz_assumption": args.clock_mhz,
                        "device_seconds": device_seconds,
                        "host_marker_sequence_s": host_marker_sequence_s,
                        "host_blocking_s": host_blocking_s,
                        "device_bandwidth_Bps": byte_count / device_seconds,
                        "host_blocking_bandwidth_Bps": byte_count / host_blocking_s,
                        "excluded_timing": {
                            "host_buffer_preparation_s": buffer_preparation_s,
                            "runtime_startup_s": runtime_startup_s,
                        },
                        "simulator": args.simulator,
                        "memcpy_order": "COL_MAJOR",
                        "artifact_manifest": str(manifest_path),
                        "artifact_path": manifest["artifact_path"],
                        "client": {
                            "hostname": socket.gethostname(),
                            "python": platform.python_version(),
                        },
                    }
                    append_jsonl(args.results, record)
                    print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)

                if direction == "d2h":
                    expected = host_input if args.precision == "fp32" else host_input & 0xFFFF
                    actual = host_output if args.precision == "fp32" else host_output & 0xFFFF
                    if not np.array_equal(actual, expected):
                        mismatch = int(np.flatnonzero(actual != expected)[0])
                        raise AssertionError(
                            f"D2H validation failed at element {mismatch}: "
                            f"{actual[mismatch]} != {expected[mismatch]}"
                        )
        runtime_shutdown_start = time.perf_counter_ns()

    append_jsonl(
        args.results,
        {
            "schema_version": 1,
            "kind": "bandwidth_run_summary",
            "run_id": run_id,
            "finished_at": utc_now_iso(),
            "precision": args.precision,
            "directions": list(directions),
            "simulator": args.simulator,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "runtime_startup_s": runtime_startup_s,
            "runtime_shutdown_s": (time.perf_counter_ns() - runtime_shutdown_start) * 1e-9,
        },
    )


if __name__ == "__main__":
    main()
