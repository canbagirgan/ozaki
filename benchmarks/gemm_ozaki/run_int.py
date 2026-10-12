#!/usr/bin/env cs_python
"""Integer Ozaki: sequential i16 MeshGEMM products, reconstruct float64 on host.

One CS-3 job. Each triangle product reuses the same A/B/C SRAM buffers.
No @fmach / @fmachs / @fmacs on the device.
"""

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

RECORD_PREFIX = "OZAKI_INT_RECORD "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--cmaddr", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument(
        "--grid-size",
        type=int,
        choices=(64, 128, 256, 512, 640, 754),
        default=64,
    )
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--clock-mhz", type=float, default=1400.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--alpha", type=int, default=7)
    parser.add_argument("--n-slices", type=int, default=8)
    parser.add_argument("--dump-pe-map", action="store_true")
    parser.add_argument(
        "--single-product",
        action="store_true",
        help="Run A0@B0 only (kernel timing). Full triangle is the accuracy path.",
    )
    return parser.parse_args()


def emit(record: dict) -> None:
    print(RECORD_PREFIX + json.dumps(record, sort_keys=True, allow_nan=False), flush=True)


try:
    from ozaki_roofline.validation import compare_device_gemm
    from ozaki_roofline.integer_split import (
        integer_product_int64,
        k_reduction_exact,
        overflow_report,
        product_pairs,
        reconstruct_float64,
        reconstruct_gemm_fp64,
        slice_diagnostics_integer,
        split_shared_exponent,
    )
except ImportError:
    from validation import compare_device_gemm
    from integer_split import (
        integer_product_int64,
        k_reduction_exact,
        overflow_report,
        product_pairs,
        reconstruct_float64,
        reconstruct_gemm_fp64,
        slice_diagnostics_integer,
        split_shared_exponent,
    )


def shuffle_b(b: np.ndarray, p: int) -> np.ndarray:
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


def encode_i16(tiled: np.ndarray) -> np.ndarray:
    i16 = np.ascontiguousarray(tiled, dtype=np.int16).ravel()
    return np.asarray(i16.view(np.uint16), dtype=np.uint32)


def decode_i32(raw: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(raw).view(np.int32)


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
        raise SystemExit(f"--dump-pe-map is forbidden on P={args.grid_size} (>=640)")
    p = args.grid_size
    if (
        min(args.n, args.k, args.m) <= 0
        or args.n % p
        or args.k % p
        or args.m % p
        or args.warmup < 0
        or args.repetitions <= 0
        or args.clock_mhz <= 0
        or args.alpha < 1
        or args.n_slices < 1
    ):
        raise ValueError("invalid shape, slice params, or clock")

    mt, kt, nt = args.n // p, args.k // p, args.m // p
    fused_bytes = 4 * mt * kt + 4 * kt * nt + 4 * mt * nt
    pairs = product_pairs(args.n_slices, triangle=True)
    if args.single_product:
        pairs = [(0, 0)]
    k_exact = k_reduction_exact(args.alpha, args.k)

    rng = np.random.default_rng(args.seed)
    generation_start = time.perf_counter_ns()
    scale = 1.0 / math.sqrt(args.k)
    a = rng.normal(0.0, scale, size=(args.n, args.k)).astype(np.float64)
    b = rng.normal(0.0, scale, size=(args.k, args.m)).astype(np.float64)
    a_slices = split_shared_exponent(a, alpha=args.alpha, n_slices=args.n_slices, axis=1)
    b_slices = split_shared_exponent(b, alpha=args.alpha, n_slices=args.n_slices, axis=0)
    generation_s = (time.perf_counter_ns() - generation_start) * 1e-9

    packing_start = time.perf_counter_ns()
    a_host = [
        encode_i16(tile_X(np.asarray(word), p, mt, kt)) for word in a_slices.slices
    ]
    b_host = [
        encode_i16(tile_W(shuffle_b(np.asarray(word), p), p, kt, nt))
        for word in b_slices.slices
    ]
    packing_s = (time.perf_counter_ns() - packing_start) * 1e-9

    c_raw = np.zeros(args.n * args.m, dtype=np.uint32)
    timer_raw = np.zeros(p * p * 3, dtype=np.float32)
    run_id = f"ozaki-int-p{p}-{args.n}-{args.k}-{args.m}-{time.time_ns()}"
    clock_hz = args.clock_mhz * 1e6

    runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
    startup_start = time.perf_counter_ns()
    runner.load()
    runner.run()
    startup_s = (time.perf_counter_ns() - startup_start) * 1e-9

    sym_x = runner.get_id("X")
    sym_w = runner.get_id("W")
    sym_c = runner.get_id("res")
    sym_timer = runner.get_id("time_memcpy")

    rebuilt = np.zeros((args.n, args.m), dtype=np.float64)
    host_tiles: dict[tuple[int, int], np.ndarray] = {}
    device_vs_host_int = []
    cycle_summaries = []
    e2e_s_sum = 0.0
    kernel_dispatch_s_sum = 0.0
    last_cycle = None
    last_pe_map = None

    e_a = a_slices.exponents.reshape(args.n, 1).astype(np.int32)
    e_b = b_slices.exponents.reshape(1, args.m).astype(np.int32)

    accuracy_start = time.perf_counter_ns()
    for pair_index, (si, sj) in enumerate(pairs):
        runner.launch("init_task", nonblock=False)
        e2e_start = time.perf_counter_ns()
        runner.memcpy_h2d(
            sym_x,
            a_host[si],
            0,
            0,
            p,
            p,
            mt * kt,
            streaming=False,
            data_type=MemcpyDataType.MEMCPY_16BIT,
            order=MemcpyOrder.ROW_MAJOR,
            nonblock=True,
        )
        runner.memcpy_h2d(
            sym_w,
            b_host[sj],
            0,
            0,
            p,
            p,
            kt * nt,
            streaming=False,
            data_type=MemcpyDataType.MEMCPY_16BIT,
            order=MemcpyOrder.ROW_MAJOR,
            nonblock=False,
        )
        h2d_end = time.perf_counter_ns()
        runner.launch(
            "meshgemm_host",
            np.int16(args.warmup),
            np.int16(args.repetitions),
            nonblock=False,
        )
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
            data_type=MemcpyDataType.MEMCPY_32BIT,
            order=MemcpyOrder.ROW_MAJOR,
            nonblock=False,
        )
        e2e_s_sum += (time.perf_counter_ns() - e2e_start) * 1e-9
        kernel_dispatch_s_sum += (kernel_end - h2d_end) * 1e-9

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
        last_cycle = decode_cycle_summary(timer_raw.reshape(p, p, 3))
        cycle_summaries.append(last_cycle)
        if args.dump_pe_map:
            last_pe_map = pe_kernel_cycles(timer_raw.reshape(p, p, 3)).tolist()

        c_int = untile_res(decode_i32(c_raw), p, mt, nt).astype(np.int64, copy=False)
        host_int = integer_product_int64(a_slices.slices[si], b_slices.slices[sj])
        host_tiles[(si, sj)] = host_int
        device_vs_host_int.append(int(np.max(np.abs(c_int - host_int))))

        weight = (
            e_a
            + e_b
            - 2 * a_slices.int_bits
            + a_slices.shifts[si]
            + a_slices.shifts[sj]
        )
        rebuilt = rebuilt + np.ldexp(c_int.astype(np.float64), weight)
        print(
            f"OZAKI_INT_PROGRESS pair={pair_index + 1}/{len(pairs)} "
            f"i={si} j={sj} kernel_ms={last_cycle['mean'] / clock_hz * 1e3:.3f}",
            flush=True,
        )

    accuracy_wall_s = (time.perf_counter_ns() - accuracy_start) * 1e-9
    overflow = overflow_report(host_tiles)
    if args.single_product:
        oracle = np.ldexp(
            host_tiles[(0, 0)].astype(np.float64),
            e_a
            + e_b
            - 2 * a_slices.int_bits
            + a_slices.shifts[0]
            + a_slices.shifts[0],
        )
    else:
        oracle = reconstruct_gemm_fp64(a_slices, b_slices, triangle=True)

    validation = compare_device_gemm(a, b, rebuilt, precision="fp32")
    oracle_vs_f64 = compare_device_gemm(a, b, oracle, precision="fp32")
    tiny = float(np.finfo(np.float64).tiny)
    oracle_diff = rebuilt - oracle
    oracle_rel = float(np.linalg.norm(oracle_diff, ord="fro")) / max(
        float(np.linalg.norm(oracle, ord="fro")), tiny
    )

    mean_cycles = float(np.mean([c["mean"] for c in cycle_summaries]))
    kernel_s_one = mean_cycles / clock_hz
    kernel_s_all = kernel_s_one * len(pairs)

    record = {
        "schema_version": 1,
        "kind": "gemm_accuracy",
        "algorithm": "ozaki_integer",
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "precision": "i16x_i32",
        "alpha": args.alpha,
        "n_slices": args.n_slices,
        "n_products": len(pairs),
        "triangle": (not args.single_product),
        "single_product": bool(args.single_product),
        "k_reduction_exact": bool(k_exact),
        "fused_buffer_bytes": fused_bytes,
        "grid": {"width": p, "height": p},
        "shape": {"n": args.n, "k": args.k, "m": args.m},
        "local_tile": {"mt": mt, "kt": kt, "nt": nt},
        "clock_mhz": args.clock_mhz,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "startup_s": startup_s,
        "generation_s": generation_s,
        "packing_s": packing_s,
        "accuracy_wall_s": accuracy_wall_s,
        "e2e_s_sum": e2e_s_sum,
        "kernel_dispatch_s_sum": kernel_dispatch_s_sum,
        "kernel_s_one_product_tsc": kernel_s_one,
        "kernel_s_all_products_tsc": kernel_s_all,
        "kernel_ms_one_product_tsc": kernel_s_one * 1e3,
        "kernel_ms_all_products_tsc": kernel_s_all * 1e3,
        "tsc_cycles": {
            "one_product_mean": mean_cycles,
            "first": cycle_summaries[0],
            "last": cycle_summaries[-1],
        },
        "overflow": overflow,
        "device_vs_host_integer_max_abs": int(max(device_vs_host_int)),
        "device_matches_host_integer": bool(max(device_vs_host_int) == 0),
        "split": {
            "a": slice_diagnostics_integer(a, a_slices),
            "b": slice_diagnostics_integer(b, b_slices),
            "a_reconstruct_frobenius": float(
                np.linalg.norm(a - reconstruct_float64(a_slices), ord="fro")
            ),
        },
        "against_float64_reference": validation["against_float64_reference"],
        "host_oracle_vs_float64": oracle_vs_f64["against_float64_reference"],
        "device_vs_host_oracle_relative_frobenius": oracle_rel,
        "accumulate": "sequential_i16_gemm_i32_host_f64",
    }
    if last_pe_map is not None:
        record["pe_map_cycles"] = last_pe_map
    emit(record)
    runner.stop()


if __name__ == "__main__":
    main()
