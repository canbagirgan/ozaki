#!/usr/bin/env cs_python
"""Run cold-input MeshGEMM from an appliance worker. SUMMA is untouched."""

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
from host_common import assignId, pe_kernel_cycles, tile_W, tile_X, untile_res

RECORD_PREFIX = "OZAKI_RECORD "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--cmaddr", required=True)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
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
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--clock-mhz", type=float, default=1400.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--input-kind",
        choices=("scaled", "full-mantissa", "ozimmu"),
        default="scaled",
        help=(
            "scaled = N(0, 1/sqrt(K)). full-mantissa = random 23-bit fp32 "
            "mantissa, exponent 2^0..2^7 (fp32 only). "
            "ozimmu = Uniform(-0.5, 0.5) * exp(phi * N(0, 1))."
        ),
    )
    parser.add_argument(
        "--phi",
        type=float,
        default=1.0,
        help="ozIMMU exponent-width phi (ignored unless --input-kind ozimmu).",
    )
    parser.add_argument(
        "--dump-pe-map",
        action="store_true",
        help="Emit the PxP TSC duration grid (use only for small grids).",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip host float64 reference GEMM (infeasible for large N).",
    )
    return parser.parse_args()


def emit(record: dict) -> None:
    print(RECORD_PREFIX + json.dumps(record, sort_keys=True, allow_nan=False), flush=True)


def input_record_fields(args: argparse.Namespace) -> dict:
    fields = {"input_kind": args.input_kind}
    if args.input_kind == "ozimmu":
        fields["phi"] = float(args.phi)
    return fields


try:
    from ozaki_roofline.split import full_mantissa_fp32, ozimmu_matrix
    from ozaki_roofline.validation import compare_device_gemm
except ImportError:
    from split import full_mantissa_fp32, ozimmu_matrix
    from validation import compare_device_gemm


def shuffle_b(b: np.ndarray, p: int) -> np.ndarray:
    """Place B tiles in the two-hop MeshGEMM order (WaferLLM host_common)."""

    k, m = b.shape
    kt, nt = k // p, m // p
    ind = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(p):
            if i == 0:
                ind[0, j] = j
            elif i == 1:
                _, ind[1, j] = assignId(ind[0, j], p)
            elif (i - 1) % 2 == 0:
                _, ind[i, j] = assignId(ind[i - 2, j], p)
            else:
                ind[i, j], _ = assignId(ind[i - 2, j], p)
    shuffled = np.empty_like(b)
    for i in range(p):
        for j in range(p):
            t = int(ind[i, j])
            shuffled[i * kt : (i + 1) * kt, j * nt : (j + 1) * nt] = b[
                t * kt : (t + 1) * kt, j * nt : (j + 1) * nt
            ]
    return shuffled


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
    durations = pe_kernel_cycles(timer_values)
    return {
        "minimum": int(durations.min()),
        "mean": float(durations.mean()),
        "median": float(np.median(durations)),
        "maximum": int(durations.max()),
        "spread": int(durations.max() - durations.min()),
    }


def main() -> None:
    args = parse_args()
    if args.dump_pe_map and args.grid_size >= 640:
        raise SystemExit(
            f"--dump-pe-map is forbidden on P={args.grid_size} (>=640)"
        )
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
    if args.input_kind == "full-mantissa" and args.precision != "fp32":
        raise ValueError("full-mantissa inputs are float32 bit patterns; use --precision fp32")
    if args.phi < 0.0:
        raise ValueError("phi must be >= 0")
    rng = np.random.default_rng(args.seed)
    generation_start = time.perf_counter_ns()
    if args.input_kind == "full-mantissa":
        a = full_mantissa_fp32(rng, (args.n, args.k))
        b = full_mantissa_fp32(rng, (args.k, args.m))
    elif args.input_kind == "ozimmu":
        a = ozimmu_matrix(rng, (args.n, args.k), phi=args.phi, dtype=np.float32)
        b = ozimmu_matrix(rng, (args.k, args.m), phi=args.phi, dtype=np.float32)
        if args.precision == "fp16":
            a = a.astype(np.float16)
            b = b.astype(np.float16)
    else:
        scale = 1.0 / math.sqrt(args.k)
        a = rng.normal(0.0, scale, size=(args.n, args.k)).astype(dtype)
        b = rng.normal(0.0, scale, size=(args.k, args.m)).astype(dtype)
    generation_s = (time.perf_counter_ns() - generation_start) * 1e-9
    packing_start = time.perf_counter_ns()
    a_host = encode_input(tile_X(a, p, mt, kt), args.precision)
    b_host = encode_input(tile_W(shuffle_b(b, p), p, kt, nt), args.precision)
    packing_s = (time.perf_counter_ns() - packing_start) * 1e-9
    c_raw = np.zeros(args.n * args.m, dtype=np.uint32)
    timer_raw = np.zeros(p * p * 3, dtype=np.float32)

    flops = 2 * args.n * args.k * args.m
    h2d_bytes = bytes_per_element * (args.n * args.k + args.k * args.m)
    d2h_bytes = bytes_per_element * args.n * args.m
    kind_tag = (
        f"{args.input_kind}-phi{args.phi:g}"
        if args.input_kind == "ozimmu"
        else args.input_kind
    )
    run_id = (
        f"meshgemm-direct-{kind_tag}-p{p}-{args.precision}-"
        f"{args.n}-{args.k}-{args.m}-{time.time_ns()}"
    )

    runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
    startup_start = time.perf_counter_ns()
    runner.load()
    runner.run()
    startup_s = (time.perf_counter_ns() - startup_start) * 1e-9
    validation = None
    try:
        sym_x = runner.get_id("X")
        sym_w = runner.get_id("W")
        sym_c = runner.get_id("res")
        sym_timer = runner.get_id("time_memcpy")
        for iteration in range(args.warmup + args.repetitions):
            reset_start = time.perf_counter_ns()
            runner.launch("init_task", nonblock=False)
            reset_s = (time.perf_counter_ns() - reset_start) * 1e-9

            e2e_start = time.perf_counter_ns()
            h2d_start = e2e_start
            runner.memcpy_h2d(
                sym_x,
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
                sym_w,
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

            runner.launch("meshgemm_host", np.int16(0), np.int16(1), nonblock=False)
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
            c = untile_res(decode_output(c_raw, args.precision), p, mt, nt)
            validation_s = 0.0
            if validation is None and not args.skip_validation:
                validation_start = time.perf_counter_ns()
                validation = compare_device_gemm(a, b, c, precision=args.precision)
                validation_s = (time.perf_counter_ns() - validation_start) * 1e-9
                emit(
                    {
                        "schema_version": 3,
                        "kind": "gemm_accuracy",
                        "algorithm": "meshgemm",
                        "run_id": run_id,
                        "seed": args.seed,
                        **input_record_fields(args),
                        "precision": args.precision,
                        "shape": validation["shape"],
                        "validation": validation,
                    }
                )
            elif validation is None:
                validation = {
                    "skipped": True,
                    "reason": (
                        "host float64 NumPy reference GEMM is infeasible "
                        f"for N={args.n} K={args.k} M={args.m}"
                    ),
                }
                emit(
                    {
                        "schema_version": 3,
                        "kind": "gemm_accuracy",
                        "algorithm": "meshgemm",
                        "run_id": run_id,
                        "seed": args.seed,
                        **input_record_fields(args),
                        "precision": args.precision,
                        "shape": {"n": args.n, "k": args.k, "m": args.m},
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
                    "algorithm": "meshgemm",
                    "run_id": run_id,
                    "started_at": datetime.now(UTC).isoformat(),
                    "sample_index": iteration - args.warmup,
                    **input_record_fields(args),
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
            if args.dump_pe_map:
                durations = pe_kernel_cycles(timer_raw.reshape(p, p, 3))
                emit(
                    {
                        "schema_version": 1,
                        "kind": "pe_duration_map",
                        "algorithm": "meshgemm",
                        "run_id": run_id,
                        "sample_index": iteration - args.warmup,
                        "precision": args.precision,
                        "shape": {"n": args.n, "k": args.k, "m": args.m},
                        "grid": {"width": p, "height": p},
                        "clock_mhz_assumption": args.clock_mhz,
                        "cycles": durations.astype(np.int64).ravel().tolist(),
                    }
                )
    finally:
        shutdown_start = time.perf_counter_ns()
        runner.stop()
        emit(
            {
                "schema_version": 2,
                "kind": "gemm_direct_run_summary",
                "algorithm": "meshgemm",
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
