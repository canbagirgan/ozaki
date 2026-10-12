#!/usr/bin/env cs_python
"""Run cold-input SUMMA directly from an appliance worker."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import UTC, datetime

import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import (
    MemcpyDataType,
    MemcpyOrder,
    SdkRuntime,
)

RECORD_PREFIX = "OZAKI_RECORD "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--cmaddr", required=True)
    parser.add_argument("--precision", choices=("fp16", "fp32"), required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument(
        "--grid-size",
        type=int,
        choices=(64, 128, 256, 512, 640, 754),
        default=64,
        help="Square PE grid. 850 does not fit CS-3; 754 is the max even square.",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--clock-mhz", type=float, default=1400.0)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def emit(record: dict) -> None:
    print(RECORD_PREFIX + json.dumps(record, sort_keys=True, allow_nan=False), flush=True)


try:
    from ozaki_roofline.validation import compare_device_gemm
except ImportError:
    from validation import compare_device_gemm


def tile_matrix(
    matrix: np.ndarray,
    p: int,
    local_rows: int,
    local_cols: int,
) -> np.ndarray:
    return (
        matrix.reshape(p, local_rows, p, local_cols)
        .transpose(0, 2, 3, 1)
        .reshape(p, p, local_rows * local_cols)
        .copy()
    )


def untile_matrix(
    values: np.ndarray,
    p: int,
    local_rows: int,
    local_cols: int,
) -> np.ndarray:
    return (
        values.reshape(p, p, local_cols, local_rows)
        .transpose(0, 3, 1, 2)
        .reshape(p * local_rows, p * local_cols)
        .copy()
    )


def encode_input(tiled: np.ndarray, precision: str) -> np.ndarray:
    if precision == "fp32":
        return np.ascontiguousarray(tiled, dtype=np.float32).ravel()
    fp16 = np.ascontiguousarray(tiled, dtype=np.float16).ravel()
    return np.asarray(fp16.view(np.uint16), dtype=np.uint32)


def decode_output(raw: np.ndarray, precision: str) -> np.ndarray:
    if precision == "fp32":
        return raw.view(np.float32)
    return (raw & np.uint32(0xFFFF)).astype(np.uint16).view(np.float16)


def decode_cycle_summary(timer_values: np.ndarray) -> dict:
    words = np.ascontiguousarray(timer_values).view(np.uint32)
    start = (
        (words[..., 0] & 0xFFFF).astype(np.int64)
        | ((words[..., 0] >> 16).astype(np.int64) << 16)
        | ((words[..., 1] & 0xFFFF).astype(np.int64) << 32)
    )
    end = (
        (words[..., 1] >> 16).astype(np.int64)
        | ((words[..., 2] & 0xFFFF).astype(np.int64) << 16)
        | ((words[..., 2] >> 16).astype(np.int64) << 32)
    )
    durations = (end - start) % (1 << 48)
    return {
        "minimum": int(durations.min()),
        "mean": float(durations.mean()),
        "median": float(np.median(durations)),
        "maximum": int(durations.max()),
        "spread": int(durations.max() - durations.min()),
    }


def validate(a: np.ndarray, b: np.ndarray, c: np.ndarray, precision: str) -> dict:
    return compare_device_gemm(a, b, c, precision=precision)


def main() -> None:
    args = parse_args()
    p = args.grid_size
    if (
        min(args.n, args.k, args.m) <= 0
        or args.n % p
        or args.k % p
        or args.m % p
        or args.warmup < 0
        or args.repetitions <= 0
        or args.clock_mhz <= 0
    ):
        raise ValueError("invalid shape, repetition count, or clock")

    mt, kt, nt = args.n // p, args.k // p, args.m // p
    dtype = np.float16 if args.precision == "fp16" else np.float32
    bytes_per_element = 2 if args.precision == "fp16" else 4
    memcpy_dtype = (
        MemcpyDataType.MEMCPY_16BIT if args.precision == "fp16" else MemcpyDataType.MEMCPY_32BIT
    )
    rng = np.random.default_rng(args.seed)
    generation_start = time.perf_counter_ns()
    scale = 1.0 / math.sqrt(args.k)
    a = rng.normal(0.0, scale, size=(args.n, args.k)).astype(dtype)
    b = rng.normal(0.0, scale, size=(args.k, args.m)).astype(dtype)
    generation_s = (time.perf_counter_ns() - generation_start) * 1e-9
    packing_start = time.perf_counter_ns()
    a_host = encode_input(tile_matrix(a, p, mt, kt), args.precision)
    b_host = encode_input(tile_matrix(b, p, kt, nt), args.precision)
    packing_s = (time.perf_counter_ns() - packing_start) * 1e-9
    c_raw = np.zeros(args.n * args.m, dtype=np.uint32)
    timer_raw = np.zeros(p * p * 3, dtype=np.float32)

    flops = 2 * args.n * args.k * args.m
    h2d_bytes = bytes_per_element * (args.n * args.k + args.k * args.m)
    d2h_bytes = bytes_per_element * args.n * args.m
    run_id = f"summa-direct-p{p}-{args.precision}-{args.n}-{args.k}-{args.m}-{time.time_ns()}"

    runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
    startup_start = time.perf_counter_ns()
    runner.load()
    runner.run()
    startup_s = (time.perf_counter_ns() - startup_start) * 1e-9
    validation = None
    try:
        sym_a = runner.get_id("A")
        sym_b = runner.get_id("B")
        sym_c = runner.get_id("C")
        sym_timer = runner.get_id("timer")
        for iteration in range(args.warmup + args.repetitions):
            reset_start = time.perf_counter_ns()
            runner.launch("reset", nonblock=False)
            reset_s = (time.perf_counter_ns() - reset_start) * 1e-9

            e2e_start = time.perf_counter_ns()
            h2d_start = e2e_start
            runner.memcpy_h2d(
                sym_a,
                a_host,
                0,
                0,
                p,
                p,
                mt * kt,
                streaming=False,
                data_type=memcpy_dtype,
                order=MemcpyOrder.ROW_MAJOR,
                nonblock=True,
            )
            runner.memcpy_h2d(
                sym_b,
                b_host,
                0,
                0,
                p,
                p,
                kt * nt,
                streaming=False,
                data_type=memcpy_dtype,
                order=MemcpyOrder.ROW_MAJOR,
                nonblock=False,
            )
            h2d_end = time.perf_counter_ns()

            runner.launch("main", nonblock=False)
            kernel_end = time.perf_counter_ns()
            runner.memcpy_d2h(
                c_raw,
                sym_c,
                0,
                0,
                p,
                p,
                mt * nt,
                streaming=False,
                data_type=memcpy_dtype,
                order=MemcpyOrder.ROW_MAJOR,
                nonblock=False,
            )
            primary_d2h_end = time.perf_counter_ns()

            # The exported `main` function launches asynchronous PE tasks and
            # returns after dispatch. The first C read is the synchronization
            # point, so its wall time includes the outstanding kernel. Repeat
            # the now-ready C read to measure D2H without kernel wait time.
            clean_d2h_start = time.perf_counter_ns()
            runner.memcpy_d2h(
                c_raw,
                sym_c,
                0,
                0,
                p,
                p,
                mt * nt,
                streaming=False,
                data_type=memcpy_dtype,
                order=MemcpyOrder.ROW_MAJOR,
                nonblock=False,
            )
            clean_d2h_end = time.perf_counter_ns()

            timer_start = time.perf_counter_ns()
            runner.memcpy_d2h(
                timer_raw,
                sym_timer,
                0,
                0,
                p,
                p,
                3,
                streaming=False,
                data_type=MemcpyDataType.MEMCPY_32BIT,
                order=MemcpyOrder.ROW_MAJOR,
                nonblock=False,
            )
            cycle_summary = decode_cycle_summary(timer_raw.reshape(p, p, 3))
            timer_s = (time.perf_counter_ns() - timer_start) * 1e-9

            if iteration < args.warmup:
                continue
            post_start = time.perf_counter_ns()
            c = untile_matrix(decode_output(c_raw, args.precision), p, mt, nt)
            validation_s = 0.0
            if validation is None:
                validation_start = time.perf_counter_ns()
                validation = validate(a, b, c, args.precision)
                validation_s = (time.perf_counter_ns() - validation_start) * 1e-9
                emit(
                    {
                        "schema_version": 3,
                        "kind": "gemm_accuracy",
                        "run_id": run_id,
                        "seed": args.seed,
                        "precision": args.precision,
                        "shape": validation["shape"],
                        "validation": validation,
                    }
                )
            post_s = (time.perf_counter_ns() - post_start) * 1e-9

            h2d_s = (h2d_end - h2d_start) * 1e-9
            kernel_dispatch_s = (kernel_end - h2d_end) * 1e-9
            wait_and_d2h_s = (primary_d2h_end - kernel_end) * 1e-9
            d2h_s = (clean_d2h_end - clean_d2h_start) * 1e-9
            e2e_s = (primary_d2h_end - e2e_start) * 1e-9
            device_s = cycle_summary["maximum"] / (args.clock_mhz * 1e6)
            component_sum_s = h2d_s + device_s + d2h_s
            emit(
                {
                    "schema_version": 3,
                    "kind": "gemm_direct_sample",
                    "run_id": run_id,
                    "started_at": datetime.now(UTC).isoformat(),
                    "sample_index": iteration - args.warmup,
                    "precision": args.precision,
                    "shape": {"n": args.n, "k": args.k, "m": args.m},
                    "local_tile": {"mt": mt, "kt": kt, "nt": nt},
                    "grid": {"width": p, "height": p, "active_pes": p * p},
                    "channels": 16,
                    "host_scope": "appliance_worker",
                    "clock_mhz_assumption": args.clock_mhz,
                    "flops": flops,
                    "h2d_bytes": h2d_bytes,
                    "d2h_bytes": d2h_bytes,
                    "host_wire_bytes": h2d_bytes + d2h_bytes,
                    "host_container_bytes": a_host.nbytes + b_host.nbytes + c_raw.nbytes,
                    "host_oi_flops_per_byte": flops / (h2d_bytes + d2h_bytes),
                    "timing": {
                        "worker_h2d_s": h2d_s,
                        "worker_kernel_dispatch_s": kernel_dispatch_s,
                        "worker_post_dispatch_wait_and_d2h_s": wait_and_d2h_s,
                        "worker_d2h_s": d2h_s,
                        "worker_component_sum_s": component_sum_s,
                        "worker_operational_e2e_s": e2e_s,
                        "device_kernel_s": device_s,
                        "device_kernel_cycles": cycle_summary,
                    },
                    "throughput": {
                        "device_kernel_flops": flops / device_s,
                        "worker_operational_e2e_flops": flops / e2e_s,
                    },
                    "excluded_timing": {
                        "worker_data_generation_s": generation_s,
                        "worker_tiling_and_packing_s": packing_s,
                        "direct_runtime_startup_s": startup_s,
                        "device_reset_worker_call_s": reset_s,
                        "timer_readback_and_decode_s": timer_s,
                        "worker_untiling_and_validation_s": post_s,
                        "worker_validation_s": validation_s,
                    },
                    "validation": validation,
                }
            )
    finally:
        shutdown_start = time.perf_counter_ns()
        runner.stop()
        emit(
            {
                "schema_version": 2,
                "kind": "gemm_direct_run_summary",
                "run_id": run_id,
                "finished_at": datetime.now(UTC).isoformat(),
                "precision": args.precision,
                "shape": {"n": args.n, "k": args.k, "m": args.m},
                "runtime_startup_s": startup_s,
                "runtime_shutdown_s": (time.perf_counter_ns() - shutdown_start) * 1e-9,
            }
        )


if __name__ == "__main__":
    main()
