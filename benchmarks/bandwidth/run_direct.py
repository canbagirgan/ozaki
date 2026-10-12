#!/usr/bin/env cs_python
"""Run blocking memcpy measurements directly on an appliance worker."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime

import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import (
    MemcpyDataType,
    MemcpyOrder,
    SdkRuntime,
)

P = 64
RECORD_PREFIX = "OZAKI_RECORD "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--cmaddr", required=True)
    parser.add_argument("--precision", choices=("fp16", "fp32"), required=True)
    parser.add_argument("--direction", choices=("h2d", "d2h", "both"), default="both")
    parser.add_argument("--lengths", required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    return parser.parse_args()


def emit(record: dict) -> None:
    print(RECORD_PREFIX + json.dumps(record, sort_keys=True, allow_nan=False), flush=True)


def main() -> None:
    args = parse_args()
    lengths = [int(value) for value in args.lengths.split(",")]
    capacity = 8192 if args.precision == "fp16" else 4096
    if (
        not lengths
        or min(lengths) <= 0
        or max(lengths) > capacity
        or args.warmup < 0
        or args.repetitions <= 0
    ):
        raise ValueError("invalid length or repetition configuration")

    bytes_per_element = 2 if args.precision == "fp16" else 4
    memcpy_dtype = (
        MemcpyDataType.MEMCPY_16BIT if args.precision == "fp16" else MemcpyDataType.MEMCPY_32BIT
    )
    directions = ("h2d", "d2h") if args.direction == "both" else (args.direction,)
    run_id = f"bandwidth-direct-p64-{args.precision}-{time.time_ns()}"

    runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
    startup_start = time.perf_counter_ns()
    runner.load()
    runner.run()
    startup_s = (time.perf_counter_ns() - startup_start) * 1e-9
    try:
        symbol_a = runner.get_id("A")
        for elements_per_pe in lengths:
            preparation_start = time.perf_counter_ns()
            total_elements = P * P * elements_per_pe
            host_input = (np.arange(total_elements, dtype=np.uint32) * 2654435761) & 0xFFFFFFFF
            if args.precision == "fp16":
                host_input &= 0xFFFF
            host_output = np.zeros(total_elements, dtype=np.uint32)
            preparation_s = (time.perf_counter_ns() - preparation_start) * 1e-9

            for direction in directions:
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
                    start = time.perf_counter_ns()
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
                    elapsed_s = (time.perf_counter_ns() - start) * 1e-9
                    if iteration < args.warmup:
                        continue

                    byte_count = total_elements * bytes_per_element
                    emit(
                        {
                            "schema_version": 1,
                            "kind": "bandwidth_direct_sample",
                            "run_id": run_id,
                            "started_at": datetime.now(UTC).isoformat(),
                            "sample_index": iteration - args.warmup,
                            "precision": args.precision,
                            "direction": direction,
                            "grid": {"width": P, "height": P, "active_pes": P * P},
                            "channels": 16,
                            "elements_per_pe": elements_per_pe,
                            "bytes": byte_count,
                            "host_container_bytes": total_elements * 4,
                            "worker_blocking_s": elapsed_s,
                            "worker_blocking_bandwidth_Bps": byte_count / elapsed_s,
                            "host_scope": "appliance_worker",
                            "excluded_timing": {
                                "worker_buffer_preparation_s": preparation_s,
                                "direct_runtime_startup_s": startup_s,
                            },
                        }
                    )

                if direction == "d2h":
                    expected = host_input if args.precision == "fp32" else host_input & 0xFFFF
                    actual = host_output if args.precision == "fp32" else host_output & 0xFFFF
                    if not np.array_equal(actual, expected):
                        mismatch = int(np.flatnonzero(actual != expected)[0])
                        raise AssertionError(f"D2H mismatch at element {mismatch}")
    finally:
        shutdown_start = time.perf_counter_ns()
        runner.stop()
        emit(
            {
                "schema_version": 1,
                "kind": "bandwidth_direct_run_summary",
                "run_id": run_id,
                "finished_at": datetime.now(UTC).isoformat(),
                "precision": args.precision,
                "runtime_startup_s": startup_s,
                "runtime_shutdown_s": (time.perf_counter_ns() - shutdown_start) * 1e-9,
            }
        )


if __name__ == "__main__":
    main()
