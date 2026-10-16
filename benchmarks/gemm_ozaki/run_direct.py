#!/usr/bin/env cs_python
"""Fused Ozaki: one MeshGEMM sweep, 1/3/4/6/9/10/15/16 @fmachs products per step.

Default: one FP32 C. --twosum: Knuth TwoSum into (res, res_lo), host float64 C.
Omit --cmaddr to run on the SDK simulator (simfabric). Wafer launches pass it.
n_products=6 is three-slice triangle i+j<3 (skip A1B2, A2B1, A2B2).
n_products=9 is three-slice (A0+A1+A2). n_products=10 is four-slice triangle
i+j<4 with TwoSum (always). n_products=16 is four-slice full (16 products).
n_products=15 is five-slice of float64 (A0..A4), Ozaki triangle, dual fp32 C.
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

RECORD_PREFIX = "OZAKI_RECORD "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--cmaddr",
        default=None,
        help="CS-3 CM address. Omit for the Cerebras SDK simulator (simfabric).",
    )
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
            "mantissa, exponent 2^0..2^7 so A1/A2 bits are present. "
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
        "--precision",
        choices=("fp16", "fp32"),
        default="fp16",
        help="fp16 = fused float16 slices. fp32 = classic 2×f32 Ozaki of float64.",
    )
    parser.add_argument(
        "--n-products",
        type=int,
        choices=(1, 3, 4, 6, 9, 10, 15, 16),
        default=4,
        help="1 = A0B0; 3 = skip A1B1; 4 = two-slice; 6 = three-slice triangle i+j<3; 9 = three-slice; 10 = four-slice triangle+TwoSum; 15 = five-slice; 16 = four-slice full",
    )
    parser.add_argument(
        "--twosum",
        action="store_true",
        help="Read res and res_lo; reconstruct C in float64 on the host.",
    )
    parser.add_argument(
        "--twosum-period",
        type=int,
        default=1,
        help="Rank-1 slice products per TwoSum bucket. FP32: any integer 1..32767; FP16 n_products=3: 1,5,10,32,128. Must match compile.",
    )
    parser.add_argument(
        "--twosum-a0b0-only",
        action="store_true",
        help="TwoSum was applied only to A0B0. Recorded in the run id.",
    )
    parser.add_argument(
        "--twosum-cross-only",
        action="store_true",
        help="TwoSum was applied only to A0B1 and A1B0. Recorded in the run id.",
    )
    parser.add_argument(
        "--fp64-merge",
        action="store_true",
        help="Result words are the high and low halves of an on-wafer IEEE fp64 sum.",
    )
    parser.add_argument(
        "--cross-f16",
        action="store_true",
        help="A0B1/A1B0 accumulated in f16; compare against the cross-f16 oracle.",
    )
    parser.add_argument(
        "--dump-pe-map",
        action="store_true",
        help="Emit the PxP TSC duration grid (use only for small grids).",
    )
    parser.add_argument(
        "--high-precision",
        action="store_true",
        help="Generate sources in the host wide type; headline Rel Frob vs wide GEMM.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip host float64 reference GEMM (infeasible for large N).",
    )
    parser.add_argument(
        "--validate-subsample",
        type=int,
        default=0,
        help=(
            "If >0 with --skip-validation, compare C[:s,:s] vs a high-prec "
            "GEMM of A[:s,:] @ B[:,:s] (O(s^2 K), not N^3)."
        ),
    )
    return parser.parse_args()


def emit(record: dict) -> None:
    print(RECORD_PREFIX + json.dumps(record, sort_keys=True, allow_nan=False), flush=True)


PRODUCTS_A1 = (3, 4, 6, 9, 10, 15, 16)
PRODUCTS_A2 = (6, 9, 10, 15, 16)
PRODUCTS_A3 = (10, 15, 16)


def accumulate_tag(
    args: argparse.Namespace, *, fp32_ozaki: bool, five_slice: bool
) -> str:
    if fp32_ozaki:
        return (
            f"classic_ozaki_scalar_mul_vector_add_twosum_every{args.twosum_period}"
            if args.twosum else "classic_ozaki_4xfmacs_into_fp32_c"
        )
    if args.n_products == 10:
        return "fused_twosum_4slice_tri_fp32x2"
    if args.n_products == 16:
        return "fused_4slice_on_wafer_fp32"
    if five_slice:
        return "fused_5slice_triangle_dual_fp32"
    if args.n_products == 6:
        return "fused_3slice_tri_on_wafer_fp32"
    if args.n_products == 9:
        return (
            "fused_twosum_3slice_fp32x2"
            if args.twosum
            else "fused_3slice_on_wafer_fp32"
        )
    if args.n_products == 4:
        return "fused_twosum_fp32x2" if args.twosum else "fused_on_wafer_fp32"
    if args.n_products == 3:
        if args.cross_f16:
            return "fused_skip_a1b1_cross_f16"
        return (
            "fused_twosum_skip_a1b1_fp32x2"
            if args.twosum
            else "fused_skip_a1b1_on_wafer_fp32"
        )
    return "a0b0_twosum_fp32x2" if args.twosum else "a0b0_only_on_wafer_fp32"


def input_record_fields(args: argparse.Namespace) -> dict:
    fields = {"input_kind": args.input_kind}
    if args.input_kind == "ozimmu":
        fields["phi"] = float(args.phi)
    return fields


try:
    from ozaki_roofline.validation import compare_device_gemm
    from ozaki_roofline.split import (
        fifteen_product_dual_fp64,
        four_product_fp32,
        four_product_twosum_fp64,
        three_product_fp32,
        three_product_cross_f16,
        full_mantissa_fp32,
        nine_product_fp32,
        six_product_fp32,
        ozimmu_matrix,
        reconstruct_twosum_fp64,
        residual_split_fp16,
        residual_split_fp32,
        ten_product_twosum_fp64,
        sixteen_product_fp32,
        slice_diagnostics,
        fp16_covering_col_scales,
        fp16_covering_row_scales,
        apply_col_scales,
        apply_row_scales,
        unscale_gemm,
    )
    from ozaki_roofline.high_precision import (
        compare_device_vs_high,
        high_precision_info,
        ozimmu_matrix_high,
        scaled_gaussian_high,
        to_float64,
    )
except ImportError:
    from validation import compare_device_gemm
    from split import (
        fifteen_product_dual_fp64,
        four_product_fp32,
        four_product_twosum_fp64,
        three_product_fp32,
        three_product_cross_f16,
        full_mantissa_fp32,
        nine_product_fp32,
        six_product_fp32,
        ozimmu_matrix,
        reconstruct_twosum_fp64,
        residual_split_fp16,
        residual_split_fp32,
        ten_product_twosum_fp64,
        sixteen_product_fp32,
        slice_diagnostics,
        fp16_covering_col_scales,
        fp16_covering_row_scales,
        apply_col_scales,
        apply_row_scales,
        unscale_gemm,
    )
    from high_precision import (
        compare_device_vs_high,
        high_precision_info,
        ozimmu_matrix_high,
        scaled_gaussian_high,
        to_float64,
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


def encode_fp32(tiled: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(tiled, dtype=np.float32).ravel()


def encode_fp16(tiled: np.ndarray) -> np.ndarray:
    fp16 = np.ascontiguousarray(tiled, dtype=np.float16).ravel()
    return np.asarray(fp16.view(np.uint16), dtype=np.uint32)


def decode_fp32(raw: np.ndarray) -> np.ndarray:
    return raw.view(np.float32)


def assemble_fp64_halves(hi_f32: np.ndarray, lo_f32: np.ndarray) -> np.ndarray:
    """Join on-wafer IEEE fp64 halves. hi holds bits 63:32, lo holds bits 31:0."""

    hi_bits = np.ascontiguousarray(hi_f32, dtype=np.float32).view(np.uint32).astype(np.uint64)
    lo_bits = np.ascontiguousarray(lo_f32, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return (lo_bits | (hi_bits << np.uint64(32))).view(np.float64)


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
    if not 1 <= args.twosum_period <= 32767:
        raise ValueError("--twosum-period must be in [1, 32767]")
    if args.twosum_period != 1 and not args.twosum:
        raise ValueError("--twosum-period requires --twosum")
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
        or args.n_products not in (1, 3, 4, 6, 9, 10, 15, 16)
    ):
        raise ValueError("invalid shape, repetition count, or clock")

    mt, kt, nt = args.n // p, args.k // p, args.m // p
    fp32_ozaki = args.precision == "fp32"
    five_slice = args.n_products == 15
    four_slice_tri = args.n_products == 10
    if four_slice_tri:
        args.twosum = True
    dual_c = five_slice
    high_prec = bool((args.high_precision or four_slice_tri) and not (fp32_ozaki and args.twosum))
    if args.twosum_period != 1 and not fp32_ozaki and (
        args.n_products != 3 or args.twosum_period not in (5, 10, 32, 128)
    ):
        raise ValueError("FP16 periodic TwoSum requires n_products=3 and period 5, 10, 32, or 128")
    if fp32_ozaki:
        if args.fp64_merge:
            raise ValueError("--fp64-merge is only supported by the FP16 path")
        if args.n_products != 4:
            raise ValueError("classic 2×f32 Ozaki is n_products=4 only")
    if five_slice and args.twosum:
        raise ValueError("five-slice fp16x5 uses dual C without per-FMA TwoSum")
    if args.twosum_a0b0_only:
        if not args.twosum:
            raise ValueError("--twosum-a0b0-only requires --twosum")
        if args.n_products != 3:
            raise ValueError("--twosum-a0b0-only is n_products=3 only")
        if args.twosum_period != 1:
            raise ValueError("--twosum-a0b0-only applies TwoSum on every A0B0 product")
    if args.twosum_cross_only:
        if not args.twosum:
            raise ValueError("--twosum-cross-only requires --twosum")
        if args.n_products != 3:
            raise ValueError("--twosum-cross-only is n_products=3 only")
        if args.twosum_period != 1:
            raise ValueError("--twosum-cross-only applies TwoSum on every A0B1 and A1B0 product")
        if args.twosum_a0b0_only:
            raise ValueError("--twosum-cross-only cannot combine with --twosum-a0b0-only")
    if args.fp64_merge:
        if not args.twosum:
            raise ValueError("--fp64-merge requires --twosum")
    if args.cross_f16:
        if args.n_products != 3:
            raise ValueError("--cross-f16 is n_products=3 only")
        if args.twosum:
            raise ValueError("--cross-f16 cannot combine with --twosum")
        if fp32_ozaki:
            raise ValueError("--cross-f16 is the fp16 fused path only")
    if args.input_kind == "full-mantissa" and (fp32_ozaki or five_slice or high_prec):
        raise ValueError(
            "full-mantissa is the fp32 4-slice covering; use scaled or ozimmu for wide paths"
        )
    if args.phi < 0.0:
        raise ValueError("phi must be >= 0")
    # float128 A+B+C at N=16384 is ~12 GB; keep wide generation for validated sizes.
    wide_elems = args.n * args.k + args.k * args.m + args.n * args.m
    high_prec_generate = high_prec and (not args.skip_validation) and wide_elems * 16 < 2 * 1024**3
    rng = np.random.default_rng(args.seed)
    generation_start = time.perf_counter_ns()
    hp_info = high_precision_info() if high_prec else None
    src_dtype = np.float64 if (fp32_ozaki or five_slice or high_prec) else np.float32
    if high_prec_generate:
        if args.input_kind == "ozimmu":
            a = ozimmu_matrix_high(
                rng, (args.n, args.k), phi=args.phi, info=hp_info
            )
            b = ozimmu_matrix_high(
                rng, (args.k, args.m), phi=args.phi, info=hp_info
            )
        else:
            scale = 1.0 / math.sqrt(args.k)
            a = scaled_gaussian_high(
                rng, (args.n, args.k), scale=scale, info=hp_info
            )
            b = scaled_gaussian_high(
                rng, (args.k, args.m), scale=scale, info=hp_info
            )
    elif args.input_kind == "ozimmu":
        a = ozimmu_matrix(rng, (args.n, args.k), phi=args.phi, dtype=src_dtype)
        b = ozimmu_matrix(rng, (args.k, args.m), phi=args.phi, dtype=src_dtype)
    elif args.input_kind == "full-mantissa":
        a = full_mantissa_fp32(rng, (args.n, args.k))
        b = full_mantissa_fp32(rng, (args.k, args.m))
    else:
        scale = 1.0 / math.sqrt(args.k)
        a = rng.normal(0.0, scale, size=(args.n, args.k)).astype(src_dtype)
        b = rng.normal(0.0, scale, size=(args.k, args.m)).astype(src_dtype)
    a_src, b_src = a, b
    covering_row_scales = None
    covering_col_scales = None
    if four_slice_tri:
        covering_row_scales = fp16_covering_row_scales(a)
        covering_col_scales = fp16_covering_col_scales(b)
        a = apply_row_scales(a, covering_row_scales)
        b = apply_col_scales(b, covering_col_scales)
    if fp32_ozaki:
        n_slices = 2
        a_slices = residual_split_fp32(a)
        b_slices = residual_split_fp32(b)
    elif five_slice:
        n_slices = 5
        a_slices = residual_split_fp16(a, n_slices=5)
        b_slices = residual_split_fp16(b, n_slices=5)
    else:
        n_slices = {1: 1, 3: 2, 4: 2, 6: 3, 9: 3, 10: 4, 16: 4}[args.n_products]
        a_slices = residual_split_fp16(a, n_slices=max(n_slices, 2))
        b_slices = residual_split_fp16(b, n_slices=max(n_slices, 2))
    generation_s = (time.perf_counter_ns() - generation_start) * 1e-9
    packing_start = time.perf_counter_ns()
    encode = encode_fp32 if fp32_ozaki else encode_fp16
    a3_src = None
    b3_src = None
    a4_src = None
    b4_src = None
    if n_slices == 5:
        a0_src, a1_src, a2_src, a3_src, a4_src = a_slices
        b0_src, b1_src, b2_src, b3_src, b4_src = b_slices
    elif n_slices == 4:
        a0_src, a1_src, a2_src, a3_src = (
            a_slices.high,
            a_slices.mid,
            a_slices.extra,
            a_slices.tail,
        )
        b0_src, b1_src, b2_src, b3_src = (
            b_slices.high,
            b_slices.mid,
            b_slices.extra,
            b_slices.tail,
        )
    elif n_slices == 3:
        a0_src, a1_src, a2_src = a_slices.high, a_slices.mid, a_slices.extra
        b0_src, b1_src, b2_src = b_slices.high, b_slices.mid, b_slices.extra
    else:
        a0_src, a1_src, a2_src = a_slices.high, a_slices.low, None
        b0_src, b1_src, b2_src = b_slices.high, b_slices.low, None
    a0_host = encode(tile_X(a0_src, p, mt, kt))
    b0_host = encode(tile_W(shuffle_b(b0_src, p), p, kt, nt))
    a1_host = None
    b1_host = None
    a2_host = None
    b2_host = None
    a3_host = None
    b3_host = None
    a4_host = None
    b4_host = None
    if args.n_products in PRODUCTS_A1:
        a1_host = encode(tile_X(a1_src, p, mt, kt))
        b1_host = encode(tile_W(shuffle_b(b1_src, p), p, kt, nt))
    if args.n_products in PRODUCTS_A2:
        a2_host = encode(tile_X(a2_src, p, mt, kt))
        b2_host = encode(tile_W(shuffle_b(b2_src, p), p, kt, nt))
    if args.n_products in PRODUCTS_A3:
        a3_host = encode(tile_X(a3_src, p, mt, kt))
        b3_host = encode(tile_W(shuffle_b(b3_src, p), p, kt, nt))
    if args.n_products == 15:
        a4_host = encode(tile_X(a4_src, p, mt, kt))
        b4_host = encode(tile_W(shuffle_b(b4_src, p), p, kt, nt))
    packing_s = (time.perf_counter_ns() - packing_start) * 1e-9
    c_raw = np.zeros(args.n * args.m, dtype=np.uint32)
    read_lo = bool(args.twosum or dual_c)
    c_lo_raw = np.zeros(args.n * args.m, dtype=np.uint32) if read_lo else None
    timer_raw = np.zeros(p * p * 3, dtype=np.float32)
    input_memcpy = (
        MemcpyDataType.MEMCPY_32BIT if fp32_ozaki else MemcpyDataType.MEMCPY_16BIT
    )
    if four_slice_tri:
        # Single-buffer A2/A3/B2/B3. Same count as compile.py. T=32: 33536.
        fused_bytes = 12 * mt * kt + 12 * kt * nt + 8 * mt * nt + 24 * mt
    else:
        fused_bytes = (
            20 * mt * kt + 20 * kt * nt + 8 * mt * nt
            if five_slice
            else 16 * mt * kt + 16 * kt * nt + 4 * mt * nt
            if fp32_ozaki or args.n_products == 16
            else 12 * mt * kt + 12 * kt * nt + 4 * mt * nt
            if args.n_products in (6, 9)
            else 8 * mt * kt + 8 * kt * nt + 4 * mt * nt
        )
        if args.twosum:
            fused_bytes += 4 * mt * nt + 24 * mt
            if fp32_ozaki:
                fused_bytes += 4 * mt * nt
    if args.cross_f16:
        fused_bytes += 2 * mt * nt

    useful_flops = 2 * args.n * args.k * args.m
    ozaki_flops = args.n_products * useful_flops
    bytes_per_input = 4 if fp32_ozaki else 2
    h2d_bytes = bytes_per_input * (args.n * args.k + args.k * args.m) * n_slices
    d2h_bytes = 4 * args.n * args.m
    if args.twosum or dual_c:
        d2h_bytes *= 2
    if fp32_ozaki:
        mode = "fp32-fused"
    elif args.n_products == 1:
        mode = "a0b0"
    elif args.n_products == 15:
        mode = "fused-fp16x5"
    elif args.n_products == 10:
        mode = "fused-4slice-tri-twosum"
    elif args.n_products == 16:
        mode = "fused-4slice"
    elif args.n_products == 6:
        mode = "fused-3slice-tri"
    elif args.n_products == 9:
        mode = "fused-3slice"
    elif args.n_products == 3:
        mode = "fused-skip-a1b1-cross-f16" if args.cross_f16 else "fused-skip-a1b1"
    else:
        mode = "fused"
    if args.twosum and args.twosum_period != 1:
        mode = f"{mode}-twosum-every{args.twosum_period}"
    elif args.twosum and args.twosum_a0b0_only:
        mode = f"{mode}-twosum-a0b0"
    elif args.twosum and args.twosum_cross_only:
        mode = f"{mode}-twosum-cross"
    elif args.twosum:
        mode = f"{mode}-twosum"
    if args.fp64_merge:
        mode = f"{mode}-fp64"
    kind_tag = (
        f"{args.input_kind}-phi{args.phi:g}"
        if args.input_kind == "ozimmu"
        else args.input_kind
    )
    run_id = (
        f"ozaki-{mode}-{kind_tag}-p{p}-{args.n}-{args.k}-{args.m}-{time.time_ns()}"
    )
    algorithm = "ozaki_fp32_fused" if fp32_ozaki else "ozaki_fused"
    record_precision = (
        "fp64_2xfp32"
        if fp32_ozaki
        else "fp16x5_f64"
        if five_slice
        else "fp16x4_tri_twosum"
        if four_slice_tri
        else "fp16xfp32"
    )

    # cmaddr is None on the SDK simulator. The appliance simfabric worker
    # and a local cs_python run both treat a missing address as simfabric.
    runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
    startup_start = time.perf_counter_ns()
    runner.load()
    runner.run()
    startup_s = (time.perf_counter_ns() - startup_start) * 1e-9
    validation = None
    try:
        sym_a0 = runner.get_id("A0")
        sym_b0 = runner.get_id("B0")
        sym_a1 = runner.get_id("A1") if args.n_products in PRODUCTS_A1 else None
        sym_b1 = runner.get_id("B1") if args.n_products in PRODUCTS_A1 else None
        sym_a2 = runner.get_id("A2") if args.n_products in PRODUCTS_A2 else None
        sym_b2 = runner.get_id("B2") if args.n_products in PRODUCTS_A2 else None
        sym_a3 = runner.get_id("A3") if args.n_products in PRODUCTS_A3 else None
        sym_b3 = runner.get_id("B3") if args.n_products in PRODUCTS_A3 else None
        sym_a4 = runner.get_id("A4") if five_slice else None
        sym_b4 = runner.get_id("B4") if five_slice else None
        sym_c = runner.get_id("res")
        sym_c_lo = runner.get_id("res_lo") if read_lo else None
        sym_timer = runner.get_id("time_memcpy")
        for iteration in range(args.warmup + args.repetitions):
            reset_start = time.perf_counter_ns()
            runner.launch("init_task", nonblock=False)
            last_reset_s = (time.perf_counter_ns() - reset_start) * 1e-9

            e2e_start = time.perf_counter_ns()
            runner.memcpy_h2d(
                sym_a0,
                a0_host,
                0,
                0,
                p,
                p,
                mt * kt,
                streaming=False,
                data_type=input_memcpy,
                order=MemcpyOrder.ROW_MAJOR,
                nonblock=True,
            )
            if args.n_products in PRODUCTS_A1:
                runner.memcpy_h2d(
                    sym_a1,
                    a1_host,
                    0,
                    0,
                    p,
                    p,
                    mt * kt,
                    streaming=False,
                    data_type=input_memcpy,
                    order=MemcpyOrder.ROW_MAJOR,
                    nonblock=True,
                )
            if args.n_products in PRODUCTS_A2:
                runner.memcpy_h2d(
                    sym_a2,
                    a2_host,
                    0,
                    0,
                    p,
                    p,
                    mt * kt,
                    streaming=False,
                    data_type=input_memcpy,
                    order=MemcpyOrder.ROW_MAJOR,
                    nonblock=True,
                )
            if args.n_products in PRODUCTS_A3:
                runner.memcpy_h2d(
                    sym_a3,
                    a3_host,
                    0,
                    0,
                    p,
                    p,
                    mt * kt,
                    streaming=False,
                    data_type=input_memcpy,
                    order=MemcpyOrder.ROW_MAJOR,
                    nonblock=True,
                )
            if five_slice:
                runner.memcpy_h2d(
                    sym_a4,
                    a4_host,
                    0,
                    0,
                    p,
                    p,
                    mt * kt,
                    streaming=False,
                    data_type=input_memcpy,
                    order=MemcpyOrder.ROW_MAJOR,
                    nonblock=True,
                )
            runner.memcpy_h2d(
                sym_b0,
                b0_host,
                0,
                0,
                p,
                p,
                kt * nt,
                streaming=False,
                data_type=input_memcpy,
                order=MemcpyOrder.ROW_MAJOR,
                nonblock=args.n_products in PRODUCTS_A1,
            )
            if args.n_products in PRODUCTS_A1:
                runner.memcpy_h2d(
                    sym_b1,
                    b1_host,
                    0,
                    0,
                    p,
                    p,
                    kt * nt,
                    streaming=False,
                    data_type=input_memcpy,
                    order=MemcpyOrder.ROW_MAJOR,
                    nonblock=args.n_products in PRODUCTS_A2,
                )
            if args.n_products in PRODUCTS_A2:
                runner.memcpy_h2d(
                    sym_b2,
                    b2_host,
                    0,
                    0,
                    p,
                    p,
                    kt * nt,
                    streaming=False,
                    data_type=input_memcpy,
                    order=MemcpyOrder.ROW_MAJOR,
                    nonblock=args.n_products in PRODUCTS_A3,
                )
            if args.n_products in PRODUCTS_A3:
                runner.memcpy_h2d(
                    sym_b3,
                    b3_host,
                    0,
                    0,
                    p,
                    p,
                    kt * nt,
                    streaming=False,
                    data_type=input_memcpy,
                    order=MemcpyOrder.ROW_MAJOR,
                    nonblock=five_slice,
                )
            if five_slice:
                runner.memcpy_h2d(
                    sym_b4,
                    b4_host,
                    0,
                    0,
                    p,
                    p,
                    kt * nt,
                    streaming=False,
                    data_type=input_memcpy,
                    order=MemcpyOrder.ROW_MAJOR,
                    nonblock=False,
                )
            h2d_end = time.perf_counter_ns()
            h2d_s = (h2d_end - e2e_start) * 1e-9

            runner.launch("meshgemm_host", np.int16(0), np.int16(1), nonblock=False)
            kernel_end = time.perf_counter_ns()
            kernel_dispatch_s = (kernel_end - h2d_end) * 1e-9

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
                nonblock=read_lo,
            )
            if read_lo:
                runner.memcpy_d2h(
                    c_lo_raw,
                    sym_c_lo,
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
            primary_d2h_end = time.perf_counter_ns()
            e2e_s = (primary_d2h_end - e2e_start) * 1e-9
            wait_and_d2h_s = (primary_d2h_end - kernel_end) * 1e-9

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
                data_type=MemcpyDataType.MEMCPY_32BIT,
                order=MemcpyOrder.ROW_MAJOR,
                nonblock=read_lo,
            )
            if read_lo:
                runner.memcpy_d2h(
                    c_lo_raw,
                    sym_c_lo,
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
            d2h_s = (time.perf_counter_ns() - clean_d2h_start) * 1e-9

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
            last_timer_s = (time.perf_counter_ns() - timer_start) * 1e-9
            cycle_summary = decode_cycle_summary(timer_raw.reshape(p, p, 3))

            if iteration < args.warmup:
                continue

            post_start = time.perf_counter_ns()
            c_hi = untile_res(decode_fp32(c_raw), p, mt, nt).astype(np.float32, copy=False)
            c_lo = None
            if args.twosum:
                c_lo = untile_res(decode_fp32(c_lo_raw), p, mt, nt).astype(
                    np.float32, copy=False
                )
                if args.fp64_merge:
                    c = assemble_fp64_halves(c_hi, c_lo)
                else:
                    c = reconstruct_twosum_fp64(c_hi, c_lo)
                if covering_row_scales is not None:
                    c = unscale_gemm(c, covering_row_scales, covering_col_scales)
            elif dual_c:
                c_lo = untile_res(decode_fp32(c_lo_raw), p, mt, nt).astype(
                    np.float32, copy=False
                )
                c = np.asarray(c_hi, dtype=np.float64) + np.asarray(
                    c_lo, dtype=np.float64
                )
            else:
                c = c_hi
            validation_s = 0.0
            if validation is None and args.skip_validation:
                validation = {
                    "skipped": True,
                    "reason": (
                        "host float64 NumPy reference GEMM is infeasible "
                        f"for N={args.n} K={args.k} M={args.m}"
                    ),
                    "accumulate": accumulate_tag(
                        args, fp32_ozaki=fp32_ozaki, five_slice=five_slice
                    ),
                    "n_products": args.n_products,
                    "n_slices": n_slices,
                    "dual_c": bool(dual_c),
                    "twosum": bool(args.twosum),
                    "cross_f16": bool(args.cross_f16),
                    "shape": {"n": args.n, "k": args.k, "m": args.m},
                }
                validation.update(input_record_fields(args))
                if hp_info is not None:
                    validation["high_precision"] = {
                        "kind": hp_info["kind"],
                        "dtype_name": hp_info["dtype_name"],
                        "nmant": hp_info["nmant"],
                        "note": hp_info["note"],
                        "generated_in_high_type": bool(high_prec_generate),
                    }
                if fused_bytes is not None:
                    validation["fused_buffer_bytes"] = fused_bytes
                sub_n = int(args.validate_subsample)
                if (
                    sub_n > 0
                    and four_slice_tri
                    and hp_info is not None
                    and c is not None
                    and min(args.n, args.m) >= sub_n
                ):
                    sub_start = time.perf_counter_ns()
                    sub = compare_device_vs_high(
                        a_src[:sub_n, :],
                        b_src[:, :sub_n],
                        c[:sub_n, :sub_n],
                        info=hp_info,
                    )
                    validation["subsample_high_precision"] = {
                        "n": sub_n,
                        "k": args.k,
                        "m": sub_n,
                        "note": (
                            "Leading principal block vs high-prec GEMM of "
                            "the same rows/cols. Not a full-matrix headline."
                        ),
                        "against_high_precision_reference": sub[
                            "against_high_precision_reference"
                        ],
                        "against_float64_reference": sub[
                            "against_float64_reference"
                        ],
                        "seconds": (time.perf_counter_ns() - sub_start) * 1e-9,
                    }
                emit(
                    {
                        "schema_version": 3,
                        "kind": "gemm_accuracy",
                        "algorithm": algorithm,
                        "run_id": run_id,
                        "seed": args.seed,
                        **input_record_fields(args),
                        "n_products": args.n_products,
                        "n_slices": n_slices,
                        "dual_c": bool(dual_c),
                        "twosum": bool(args.twosum),
                        "twosum_period": int(args.twosum_period),
                        "twosum_a0b0_only": bool(args.twosum_a0b0_only),
                        "twosum_cross_only": bool(args.twosum_cross_only),
                        "fp64_merge": bool(args.fp64_merge),
                        "cross_f16": bool(args.cross_f16),
                        "precision": record_precision,
                        "fused_buffer_bytes": fused_bytes,
                        "shape": validation["shape"],
                        "validation": validation,
                    }
                )
            elif validation is None:
                validation_start = time.perf_counter_ns()
                if four_slice_tri or high_prec:
                    a64 = to_float64(a_src)
                    b64 = to_float64(b_src)
                    # float128 matmul is a software GEMM (~6 h at N=16384) and
                    # the wafer job was cancelled while it ran. Above 4096^3
                    # the full-matrix number is the float64 reference.
                    high_gemm_too_big = args.n * args.k * args.m > 4096**3
                    if high_gemm_too_big:
                        validation = compare_device_gemm(
                            a64, b64, c, precision="fp32"
                        )
                        validation["high_precision_gemm_skipped"] = (
                            "float128 matmul is software O(N^3); skipped above "
                            "4096^3. against_float64_reference is the full "
                            "float64 GEMM."
                        )
                        if hp_info is not None:
                            validation["high_precision"] = {
                                "kind": hp_info["kind"],
                                "dtype_name": hp_info["dtype_name"],
                                "nmant": hp_info["nmant"],
                                "note": hp_info["note"],
                                "generated_in_high_type": bool(high_prec_generate),
                            }
                    else:
                        validation = compare_device_vs_high(
                            a_src, b_src, c, info=hp_info
                        )
                        f64_cmp = compare_device_gemm(a64, b64, c, precision="fp32")
                        validation["against_float64_reference"] = f64_cmp[
                            "against_float64_reference"
                        ]
                    validation["shape"] = {"n": args.n, "k": args.k, "m": args.m}
                    validation["two_word_result"] = True
                    if covering_row_scales is not None:
                        validation["fp16_covering_scale"] = {
                            "row_scale_median": float(np.median(covering_row_scales)),
                            "col_scale_median": float(np.median(covering_col_scales)),
                            "row_scale_min": float(np.min(covering_row_scales)),
                            "col_scale_min": float(np.min(covering_col_scales)),
                        }
                    accumulate = accumulate_tag(
                        args, fp32_ozaki=fp32_ozaki, five_slice=five_slice
                    )
                    oracle_budget = args.n * args.k * args.m * args.n_products
                    if four_slice_tri and oracle_budget < 2_000_000:
                        host_oracle = ten_product_twosum_fp64(
                            a_slices.high,
                            a_slices.mid,
                            a_slices.extra,
                            a_slices.tail,
                            b_slices.high,
                            b_slices.mid,
                            b_slices.extra,
                            b_slices.tail,
                        )
                        if covering_row_scales is not None:
                            host_oracle = unscale_gemm(
                                host_oracle, covering_row_scales, covering_col_scales
                            )
                        oracle_high = compare_device_vs_high(
                            a_src, b_src, host_oracle, info=hp_info
                        )
                        oracle = compare_device_gemm(
                            a64, b64, host_oracle, precision="fp32"
                        )
                        validation["host_oracle_vs_high_precision"] = oracle_high[
                            "against_high_precision_reference"
                        ]
                        validation["host_oracle_vs_float64"] = oracle[
                            "against_float64_reference"
                        ]
                    else:
                        validation["host_oracle_skipped"] = (
                            "per-k TwoSum oracle is O(N K M n_products); "
                            f"budget={oracle_budget}"
                        )
                    validation["split"] = {
                        "a": slice_diagnostics(a, a_slices),
                        "b": slice_diagnostics(b, b_slices),
                    }
                    if hp_info is not None and not high_gemm_too_big:
                        validation["high_precision"]["generated_in_high_type"] = bool(
                            high_prec_generate
                        )
                    validation["accumulate"] = accumulate
                    validation["n_products"] = args.n_products
                    validation.update(input_record_fields(args))
                    validation["twosum"] = bool(args.twosum)
                    validation["cross_f16"] = bool(args.cross_f16)
                    validation["dual_c"] = bool(dual_c)
                    validation["n_slices"] = n_slices
                    if fused_bytes is not None:
                        validation["fused_buffer_bytes"] = fused_bytes
                else:
                    validation = compare_device_gemm(a, b, c, precision="fp32")
                    if fp32_ozaki:
                        if args.twosum:
                            host_oracle = four_product_twosum_fp64(
                                a_slices.high, a_slices.low, b_slices.high, b_slices.low,
                                period=args.twosum_period,
                            )
                        else:
                            host_oracle = four_product_fp32(
                                a_slices.high, a_slices.low, b_slices.high, b_slices.low
                            )
                        accumulate = accumulate_tag(args, fp32_ozaki=True, five_slice=False)
                        if args.twosum:
                            validation["primary_reference"] = "against_float64_reference"
                        if not args.twosum:
                            c_trunc = np.matmul(
                                np.asarray(a_slices.high, dtype=np.float32),
                                np.asarray(b_slices.high, dtype=np.float32),
                            ).astype(np.float32)
                            trunc = compare_device_gemm(a, b, c_trunc, precision="fp32")
                            validation["truncated_sgemm_vs_float64"] = trunc[
                                "against_float64_reference"
                            ]
                            ozaki_rel = validation["against_float64_reference"][
                                "relative_frobenius_error"
                            ]
                            trunc_rel = trunc["against_float64_reference"][
                                "relative_frobenius_error"
                            ]
                            validation["ozaki_beats_truncated_sgemm"] = bool(
                                ozaki_rel < trunc_rel
                            )
                    elif args.n_products == 16:
                        host_oracle = sixteen_product_fp32(
                            a_slices.high,
                            a_slices.mid,
                            a_slices.extra,
                            a_slices.tail,
                            b_slices.high,
                            b_slices.mid,
                            b_slices.extra,
                            b_slices.tail,
                        )
                        accumulate = "fused_4slice_on_wafer_fp32"
                    elif five_slice:
                        host_oracle, _oracle_hi, _oracle_lo = fifteen_product_dual_fp64(
                            a_slices.w0,
                            a_slices.w1,
                            a_slices.w2,
                            a_slices.w3,
                            a_slices.w4,
                            b_slices.w0,
                            b_slices.w1,
                            b_slices.w2,
                            b_slices.w3,
                            b_slices.w4,
                        )
                        accumulate = "fused_5slice_triangle_dual_fp32"
                        c_trunc = np.matmul(
                            np.asarray(a, dtype=np.float32),
                            np.asarray(b, dtype=np.float32),
                        ).astype(np.float32)
                        trunc = compare_device_gemm(a, b, c_trunc, precision="fp32")
                        hi_only = compare_device_gemm(a, b, c_hi, precision="fp32")
                        validation["truncated_sgemm_vs_float64"] = trunc[
                            "against_float64_reference"
                        ]
                        validation["c_hi_only_vs_float64"] = hi_only[
                            "against_float64_reference"
                        ]
                        ozaki_rel = validation["against_float64_reference"][
                            "relative_frobenius_error"
                        ]
                        trunc_rel = trunc["against_float64_reference"][
                            "relative_frobenius_error"
                        ]
                        validation["ozaki_beats_truncated_sgemm"] = bool(
                            ozaki_rel < trunc_rel
                        )
                        validation["dual_c"] = True
                        validation["n_slices"] = 5
                    elif args.n_products == 6:
                        host_oracle = six_product_fp32(
                            a_slices.high,
                            a_slices.mid,
                            a_slices.extra,
                            b_slices.high,
                            b_slices.mid,
                            b_slices.extra,
                        )
                        accumulate = "fused_3slice_tri_on_wafer_fp32"
                    elif args.n_products == 9:
                        host_oracle = nine_product_fp32(
                            a_slices.high,
                            a_slices.mid,
                            a_slices.extra,
                            b_slices.high,
                            b_slices.mid,
                            b_slices.extra,
                        )
                        accumulate = (
                            "fused_twosum_3slice_fp32x2"
                            if args.twosum
                            else "fused_3slice_on_wafer_fp32"
                        )
                    elif args.n_products == 4:
                        host_oracle = four_product_fp32(
                            a_slices.high, a_slices.low, b_slices.high, b_slices.low
                        )
                        accumulate = (
                            "fused_twosum_fp32x2"
                            if args.twosum
                            else "fused_on_wafer_fp32"
                        )
                    elif args.n_products == 3:
                        if args.cross_f16:
                            host_oracle = three_product_cross_f16(
                                a_slices.high, a_slices.low, b_slices.high, b_slices.low
                            )
                            accumulate = "fused_skip_a1b1_cross_f16"
                        else:
                            host_oracle = three_product_fp32(
                                a_slices.high, a_slices.low, b_slices.high, b_slices.low
                            )
                            accumulate = (
                                "fused_twosum_skip_a1b1_fp32x2"
                                if args.twosum
                                else "fused_skip_a1b1_on_wafer_fp32"
                            )
                    else:
                        host_oracle = np.matmul(
                            np.asarray(a_slices.high, dtype=np.float32),
                            np.asarray(b_slices.high, dtype=np.float32),
                        )
                        accumulate = (
                            "a0b0_twosum_fp32x2"
                            if args.twosum
                            else "a0b0_only_on_wafer_fp32"
                        )
                    oracle = compare_device_gemm(a, b, host_oracle, precision="fp32")
                    validation["split"] = {
                        "a": slice_diagnostics(a, a_slices),
                        "b": slice_diagnostics(b, b_slices),
                    }
                    validation["host_oracle_vs_float64"] = oracle[
                        "against_float64_reference"
                    ]
                    validation["accumulate"] = accumulate
                    validation["n_products"] = args.n_products
                    validation.update(input_record_fields(args))
                    validation["twosum"] = bool(args.twosum)
                    validation["cross_f16"] = bool(args.cross_f16)
                    validation["dual_c"] = bool(dual_c)
                    validation["n_slices"] = n_slices
                    if fused_bytes is not None:
                        validation["fused_buffer_bytes"] = fused_bytes
                validation_s = (time.perf_counter_ns() - validation_start) * 1e-9
                emit(
                    {
                        "schema_version": 3,
                        "kind": "gemm_accuracy",
                        "algorithm": algorithm,
                        "run_id": run_id,
                        "seed": args.seed,
                        **input_record_fields(args),
                        "n_products": args.n_products,
                        "n_slices": n_slices,
                        "dual_c": bool(dual_c),
                        "twosum": bool(args.twosum),
                        "twosum_period": int(args.twosum_period),
                        "twosum_a0b0_only": bool(args.twosum_a0b0_only),
                        "twosum_cross_only": bool(args.twosum_cross_only),
                        "fp64_merge": bool(args.fp64_merge),
                        "cross_f16": bool(args.cross_f16),
                        "precision": record_precision,
                        "fused_buffer_bytes": fused_bytes,
                        "shape": validation["shape"],
                        "validation": validation,
                    }
                )
            post_s = (time.perf_counter_ns() - post_start) * 1e-9

            device_s = cycle_summary["maximum"] / (args.clock_mhz * 1e6)
            device_median_s = cycle_summary["median"] / (args.clock_mhz * 1e6)
            component_sum_s = h2d_s + device_s + d2h_s
            emit(
                {
                    "schema_version": 3,
                    "kind": "gemm_direct_sample",
                    "algorithm": algorithm,
                    "run_id": run_id,
                    "started_at": datetime.now(UTC).isoformat(),
                    "sample_index": iteration - args.warmup,
                    "precision": record_precision,
                    "shape": {"n": args.n, "k": args.k, "m": args.m},
                    "local_tile": {"mt": mt, "kt": kt, "nt": nt},
                    "fused_buffer_bytes": fused_bytes,
                    "grid": {"width": p, "height": p, "active_pes": p * p},
                    "channels": 16,
                    "host_scope": "appliance_worker",
                    "clock_mhz_assumption": args.clock_mhz,
                    "n_products": args.n_products,
                    **input_record_fields(args),
                    "twosum": bool(args.twosum),
                    "cross_f16": bool(args.cross_f16),
                    "accumulate": validation["accumulate"],
                    "flops": useful_flops,
                    "ozaki_flops": ozaki_flops,
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
                        "device_kernel_median_s": device_median_s,
                        "device_kernel_cycles": cycle_summary,
                    },
                    "throughput": {
                        "device_kernel_useful_flops": useful_flops / device_s,
                        "device_kernel_ozaki_flops": ozaki_flops / device_s,
                        "worker_operational_e2e_useful_flops": useful_flops / e2e_s,
                    },
                    "excluded_timing": {
                        "worker_data_generation_s": generation_s,
                        "worker_tiling_and_packing_s": packing_s,
                        "direct_runtime_startup_s": startup_s,
                        "device_reset_worker_call_s": last_reset_s,
                        "timer_readback_and_decode_s": last_timer_s,
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
                        "algorithm": algorithm,
                        "run_id": run_id,
                        "sample_index": iteration - args.warmup,
                        "n_products": args.n_products,
                        "twosum": bool(args.twosum),
                        "twosum_period": int(args.twosum_period),
                        "twosum_a0b0_only": bool(args.twosum_a0b0_only),
                        "twosum_cross_only": bool(args.twosum_cross_only),
                        "fp64_merge": bool(args.fp64_merge),
                        "cross_f16": bool(args.cross_f16),
                        "precision": record_precision,
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
                "algorithm": algorithm,
                "run_id": run_id,
                "finished_at": datetime.now(UTC).isoformat(),
                "n_products": args.n_products,
                "twosum": bool(args.twosum),
                "cross_f16": bool(args.cross_f16),
                "precision": record_precision,
                "shape": {"n": args.n, "k": args.k, "m": args.m},
                "runtime_startup_s": startup_s,
                "runtime_shutdown_s": (time.perf_counter_ns() - shutdown_start) * 1e-9,
            }
        )


if __name__ == "__main__":
    main()
