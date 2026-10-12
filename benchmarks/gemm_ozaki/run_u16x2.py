#!/usr/bin/env cs_python
"""Sequential u16x2 MeshGEMM: four unsigned products, host schoolbook.

Inputs are seed-7 Gaussian N(0, 1/sqrt(K)) as float32, split at the 16-bit
word middle. Device stores u16 and zero-extends before scalar *. Host
reconstructs the uint32 schoolbook in int64. This is not Ozaki covering.
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

RECORD_PREFIX = "OZAKI_U16X2_RECORD "
PAIRS = ((0, 0), (0, 1), (1, 0), (1, 1))
PAIR_NAMES = {(0, 0): "A0B0", (0, 1): "A0B1", (1, 0): "A1B0", (1, 1): "A1B1"}


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
    parser.add_argument("--dump-pe-map", action="store_true")
    parser.add_argument(
        "--single-product",
        action="store_true",
        help="Run A0@B0 only (kernel timing).",
    )
    return parser.parse_args()


def emit(record: dict) -> None:
    print(RECORD_PREFIX + json.dumps(record, sort_keys=True, allow_nan=False), flush=True)


def progress(pair: int, total: int, msg: str) -> None:
    print(f"PROGRESS pair={pair}/{total} {msg}", flush=True)


try:
    from ozaki_roofline.fp32_bit_halves import (
        integer_product_int64,
        overflow_report_u16,
        schoolbook_uint32_gemm,
        split_fp32_uint16_halves,
        truncated_fp32_residual_oracle,
        wrap_u32,
    )
except ImportError:
    from fp32_bit_halves import (
        integer_product_int64,
        overflow_report_u16,
        schoolbook_uint32_gemm,
        split_fp32_uint16_halves,
        truncated_fp32_residual_oracle,
        wrap_u32,
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


def encode_u16(tiled: np.ndarray) -> np.ndarray:
    u16 = np.ascontiguousarray(tiled, dtype=np.uint16).ravel()
    return np.asarray(u16.view(np.uint16), dtype=np.uint32)


def decode_u32(raw: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(raw).view(np.uint32)


def decode_cycle_summary(timer_values: np.ndarray) -> dict:
    durations = pe_kernel_cycles(timer_values)
    return {
        "minimum": int(durations.min()),
        "mean": float(durations.mean()),
        "median": float(np.median(durations)),
        "maximum": int(durations.max()),
        "spread": int(durations.max() - durations.min()),
    }


def relative_frobenius(actual: np.ndarray, reference: np.ndarray) -> float:
    tiny = float(np.finfo(np.float64).tiny)
    diff = np.asarray(actual, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    ref_n = float(np.linalg.norm(reference, ord="fro"))
    return float(np.linalg.norm(diff, ord="fro")) / max(ref_n, tiny)


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
    ):
        raise ValueError("invalid shape or clock")

    mt, kt, nt = args.n // p, args.k // p, args.m // p
    fused_bytes = 4 * mt * kt + 4 * kt * nt + 4 * mt * nt
    clock_hz = args.clock_mhz * 1e6
    pairs = ((0, 0),) if args.single_product else PAIRS

    progress(0, len(pairs), "generate fp32 seed-7 Gaussians")
    rng = np.random.default_rng(args.seed)
    generation_start = time.perf_counter_ns()
    scale = 1.0 / math.sqrt(args.k)
    a32 = rng.normal(0.0, scale, size=(args.n, args.k)).astype(np.float32)
    b32 = rng.normal(0.0, scale, size=(args.k, args.m)).astype(np.float32)
    a_h = split_fp32_uint16_halves(a32)
    b_h = split_fp32_uint16_halves(b32)
    a_slices = (a_h.high, a_h.low)
    b_slices = (b_h.high, b_h.low)
    generation_s = (time.perf_counter_ns() - generation_start) * 1e-9

    packing_start = time.perf_counter_ns()
    a_host = [encode_u16(tile_X(word, p, mt, kt)) for word in a_slices]
    b_host = [encode_u16(tile_W(shuffle_b(word, p), p, kt, nt)) for word in b_slices]
    packing_s = (time.perf_counter_ns() - packing_start) * 1e-9

    c_raw = np.zeros(args.n * args.m, dtype=np.uint32)
    timer_raw = np.zeros(p * p * 3, dtype=np.float32)
    run_id = f"ozaki-u16x2-p{p}-{args.n}-{args.k}-{args.m}-{time.time_ns()}"

    runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
    startup_start = time.perf_counter_ns()
    runner.load()
    runner.run()
    startup_s = (time.perf_counter_ns() - startup_start) * 1e-9

    sym_x = runner.get_id("X")
    sym_w = runner.get_id("W")
    sym_c = runner.get_id("res")
    sym_timer = runner.get_id("time_memcpy")

    device_tiles: dict[tuple[int, int], np.ndarray] = {}
    host_tiles: dict[tuple[int, int], np.ndarray] = {}
    device_vs_host: dict[str, int] = {}
    device_vs_host_wrap: dict[str, int] = {}
    cycle_summaries = []
    e2e_s_sum = 0.0
    kernel_dispatch_s_sum = 0.0
    last_pe_map = None

    accuracy_start = time.perf_counter_ns()
    for pair_index, (si, sj) in enumerate(pairs):
        label = PAIR_NAMES[(si, sj)]
        progress(pair_index + 1, len(pairs), f"{label} init+H2D")
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
        progress(pair_index + 1, len(pairs), f"{label} kernel")
        runner.launch(
            "meshgemm_host",
            np.int16(args.warmup),
            np.int16(args.repetitions),
            nonblock=False,
        )
        kernel_end = time.perf_counter_ns()
        progress(pair_index + 1, len(pairs), f"{label} D2H")
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

        c_dev = untile_res(decode_u32(c_raw), p, mt, nt).astype(np.uint32, copy=False)
        host_int = integer_product_int64(a_slices[si], b_slices[sj])
        device_tiles[(si, sj)] = c_dev.astype(np.int64)
        host_tiles[(si, sj)] = host_int
        device_vs_host[label] = int(np.max(np.abs(device_tiles[(si, sj)] - host_int)))
        device_vs_host_wrap[label] = int(
            np.max(np.abs(device_tiles[(si, sj)] - wrap_u32(host_int).astype(np.int64)))
        )
        print(
            f"PROGRESS pair={pair_index + 1}/{len(pairs)} {label} "
            f"kernel_ms={last_cycle['mean'] / clock_hz * 1e3:.3f} "
            f"vs_host_int64={device_vs_host[label]} "
            f"vs_host_u32wrap={device_vs_host_wrap[label]}",
            flush=True,
        )

    accuracy_wall_s = (time.perf_counter_ns() - accuracy_start) * 1e-9
    overflow = overflow_report_u16(host_tiles, k=args.k)
    a64 = a32.astype(np.float64)
    b64 = b32.astype(np.float64)
    c_ref = np.matmul(a64, b64)
    _, c_trunc, c_resid = truncated_fp32_residual_oracle(a32, b32)
    trunc_relf = relative_frobenius(c_trunc, c_ref)
    resid_relf = relative_frobenius(c_resid, c_ref)
    if set(PAIRS).issubset(device_tiles) and set(PAIRS).issubset(host_tiles):
        c_int_device = schoolbook_uint32_gemm(
            a_h.high, a_h.low, b_h.high, b_h.low, products=device_tiles
        )
        c_int_host = schoolbook_uint32_gemm(
            a_h.high, a_h.low, b_h.high, b_h.low, products=host_tiles
        )
        headline_relf = relative_frobenius(c_int_device.astype(np.float64), c_ref)
        host_int_relf = relative_frobenius(c_int_host.astype(np.float64), c_ref)
        schoolbook_max_abs = int(np.max(np.abs(c_int_device - c_int_host)))
    else:
        headline_relf = None
        host_int_relf = None
        schoolbook_max_abs = -1

    mean_cycles = float(np.mean([c["mean"] for c in cycle_summaries]))
    kernel_s_one = mean_cycles / clock_hz
    kernel_s_all = kernel_s_one * 4

    record = {
        "schema_version": 1,
        "kind": "gemm_accuracy",
        "algorithm": "fp32_u16_bit_halves",
        "honest_note": (
            "NOT Ozaki covering of fp32 values; schoolbook mul of IEEE "
            "bit-halves in u16 with 32-bit K-acc that overflows at P=64."
        ),
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "precision": "u16x2_i32",
        "n_products": len(pairs),
        "single_product": bool(args.single_product),
        "fused_one_execute": False,
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
        "device_vs_host_integer_max_abs": device_vs_host,
        "device_vs_host_u32wrap_max_abs": device_vs_host_wrap,
        "device_matches_host_integer": bool(max(device_vs_host.values()) == 0),
        "device_matches_host_u32wrap": bool(max(device_vs_host_wrap.values()) == 0),
        "headline_relative_frobenius_vs_f64": headline_relf,
        "host_int64_schoolbook_relative_frobenius_vs_f64": host_int_relf,
        "truncated_fp32_highhalf_relative_frobenius_vs_f64": trunc_relf,
        "residual_float_oracle_relative_frobenius_vs_f64": resid_relf,
        "device_vs_host_schoolbook_max_abs": schoolbook_max_abs,
        "isa": "zero_extend_u16_to_u32_scalar_mul",
    }
    if last_pe_map is not None:
        record["pe_map_cycles"] = last_pe_map
    emit(record)
    runner.stop()


if __name__ == "__main__":
    main()
