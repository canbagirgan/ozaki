#!/usr/bin/env python3
"""Compile mixed FP16-input / FP32-accumulate MeshGEMM for Ozaki products."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import socket
import sys
import time
from pathlib import Path

from cerebras.sdk.client import SdkCompiler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ozaki_roofline.model import (  # noqa: E402
    CS3_FABRIC_DIMS,
    CS3_FABRIC_OFFSETS,
    GRID_SIZE_HELP,
    PRECISIONS,
    GemmShape,
    Grid,
    parse_grid_size,
    validate_summa_configuration,
)
from ozaki_roofline.records import utc_now_iso  # noqa: E402

FABRIC_DIMS = CS3_FABRIC_DIMS
FABRIC_OFFSETS = CS3_FABRIC_OFFSETS
# Start barrier is opt-in (--start-barrier) and must never be used for E2E.
CHANNELS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--precision",
        choices=("fp16", "fp32"),
        default="fp16",
        help="fp16 = fused float16 slices (n_products 1|3|4|6|9|10|15|16). fp32 = classic 2×f32 Ozaki.",
    )
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--grid-size", type=parse_grid_size, default=64, help=GRID_SIZE_HELP)
    parser.add_argument(
        "--n-products",
        type=int,
        choices=(1, 3, 4, 6, 9, 10, 15, 16),
        default=4,
        help="1 = A0B0; 3 = skip A1B1; 4 = two-slice; 6 = three-slice triangle i+j<3; 9 = three-slice; 10 = four-slice triangle+TwoSum; 15 = five-slice weight-split; 16 = four-slice full. fp32 is 4 only.",
    )
    parser.add_argument(
        "--twosum",
        action="store_true",
        help="Two-word C: Knuth TwoSum of each @fmachs product into (res, res_lo).",
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
        help="n_products=3, period 1: TwoSum only A0B0. A0B1 and A1B0 are plain @fmachs into C_hi.",
    )
    parser.add_argument(
        "--twosum-cross-only",
        action="store_true",
        help="n_products=3, period 1: TwoSum only A0B1 and A1B0. A0B0 is a plain @fmachs into C_hi.",
    )
    parser.add_argument(
        "--fp64-merge",
        action="store_true",
        help="On the wafer, write IEEE fp64(C_hi)+fp64(C_lo) into the two result words.",
    )
    parser.add_argument(
        "--cross-f16",
        action="store_true",
        help="n_products=3 only: A0B1/A1B0 via @fmach into f16, one @faddhs into C.",
    )
    parser.add_argument(
        "--data-budget-kib",
        type=int,
        default=40,
        help="Per-PE matrix-buffer budget in KiB (WSE-3 SRAM is 48 KiB).",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--start-barrier",
        action="store_true",
        help=(
            "Compile a fabric start barrier before the timed kernel. "
            "On-wafer kernel timing only. Never use for E2E."
        ),
    )
    return parser.parse_args()


def config_name(
    shape: GemmShape,
    grid_size: int,
    n_products: int = 4,
    start_barrier: bool = False,
    twosum: bool = False,
    precision: str = "fp16",
    cross_f16: bool = False,
    twosum_period: int = 1,
    twosum_a0b0_only: bool = False,
    twosum_cross_only: bool = False,
    fp64_merge: bool = False,
) -> str:
    if precision == "fp32":
        name = f"ozaki-fp32-fused-p{grid_size}-n{shape.n}-k{shape.k}-m{shape.m}"
        if twosum:
            name += f"-scalar-mul-vector-add-twosum-every{twosum_period}"
        if start_barrier:
            name += "-start-barrier"
        return name
    if n_products == 1:
        name = f"ozaki-fused-a0b0-p{grid_size}-n{shape.n}-k{shape.k}-m{shape.m}"
    elif n_products == 3:
        name = f"ozaki-fused-skip-a1b1-p{grid_size}-n{shape.n}-k{shape.k}-m{shape.m}"
    elif n_products == 6:
        name = f"ozaki-fused-3slice-tri-p{grid_size}-n{shape.n}-k{shape.k}-m{shape.m}"
    elif n_products == 9:
        name = f"ozaki-fused-3slice-p{grid_size}-n{shape.n}-k{shape.k}-m{shape.m}"
    elif n_products == 10:
        # Distinct from ozaki-fused-4slice (16-product). TwoSum is implied.
        name = f"ozaki-fused-twosum-4slice-tri-p{grid_size}-n{shape.n}-k{shape.k}-m{shape.m}"
    elif n_products == 15:
        name = f"ozaki-fused-fp16x5-p{grid_size}-n{shape.n}-k{shape.k}-m{shape.m}"
    elif n_products == 16:
        name = f"ozaki-fused-4slice-p{grid_size}-n{shape.n}-k{shape.k}-m{shape.m}"
    else:
        name = f"ozaki-fused-p{grid_size}-n{shape.n}-k{shape.k}-m{shape.m}"
    if twosum and n_products != 10:
        name = name.replace("ozaki-fused-", "ozaki-fused-twosum-", 1)
    if twosum_a0b0_only:
        name = name.replace("twosum-skip-a1b1", "twosum-a0b0-skip-a1b1", 1)
    if twosum_cross_only:
        name = name.replace("twosum-skip-a1b1", "twosum-cross-skip-a1b1", 1)
    if fp64_merge:
        name = name.replace("twosum-", "twosum-fp64-", 1)
    if twosum_period != 1:
        name = name.replace("twosum-", f"twosum-every{twosum_period}-", 1)
    if cross_f16:
        name = name.replace("skip-a1b1", "skip-a1b1-cross-f16", 1)
    if start_barrier:
        name += "-start-barrier"
    return name


def fused_buffer_bytes(
    mt: int,
    kt: int,
    nt: int,
    twosum: bool,
    n_products: int = 4,
    precision: str = "fp16",
    cross_f16: bool = False,
    twosum_period: int = 1,
) -> int:
    if precision == "fp32":
        # Double-buffer A0,A1,B0,B1 + one f32 C = 16+16+4 = 36 T^2 (square).
        return (
            16 * mt * kt + 16 * kt * nt + 4 * mt * nt
            + (8 * mt * nt + 24 * mt if twosum else 0)
        )
    if n_products == 10:
        # Always TwoSum. A0/A1/B0/B1 stay double-buffered; A2/A3/B2/B3
        # are single-buffered (recv aliases the A1/B1 send tile, then copy).
        # 12 T^2 A + 12 T^2 B + 8 T^2 C + 24 T scratch.
        # T=32: 33536. T=28: 25696. Full dbl-buf was 41728 and failed to link.
        return 12 * mt * kt + 12 * kt * nt + 8 * mt * nt + 24 * mt
    if n_products == 15:
        # Double-buffer A0..A4 and B0..B4 f16 + two f32 C (hi, lo) = 40 T^2 + 8 T^2.
        # T=28: 37632 < 40960. T=32: 49152 does not fit. T=40 does not fit.
        bytes_ = 20 * mt * kt + 20 * kt * nt + 8 * mt * nt
    elif n_products == 16:
        # Double-buffer A0..A3 and B0..B3 f16 + one f32 C = 32 T^2 + 4 T^2.
        # T=32: 36864 < 40960. T=40: 57600 does not fit.
        bytes_ = 16 * mt * kt + 16 * kt * nt + 4 * mt * nt
    elif n_products in (6, 9):
        bytes_ = 12 * mt * kt + 12 * kt * nt + 4 * mt * nt
    else:
        bytes_ = 8 * mt * kt + 8 * kt * nt + 4 * mt * nt
    if twosum:
        # Extra fp32 C_lo tile plus six scratch columns (p,s,z,t,u,zero).
        bytes_ += 4 * mt * nt + 24 * mt
    if twosum_period != 1:
        # fp32 bucket, one full C tile.
        bytes_ += 4 * mt * nt
    if cross_f16:
        bytes_ += 2 * mt * nt
    return bytes_


def main() -> None:
    args = parse_args()
    if not 1 <= args.twosum_period <= 32767:
        raise ValueError("--twosum-period must be in [1, 32767]")
    if args.twosum_period != 1 and not args.twosum:
        raise ValueError("--twosum-period requires --twosum")
    if args.precision == "fp32":
        if args.fp64_merge:
            raise ValueError("--fp64-merge is only supported by the FP16 path")
        if args.n_products != 4:
            raise ValueError("classic 2×f32 Ozaki is n_products=4 only")
    if args.twosum_period != 1:
        if not args.twosum:
            raise ValueError("--twosum-period requires --twosum")
        if args.precision != "fp32" and (
            args.n_products != 3 or args.twosum_period not in (5, 10, 32, 128)
        ):
            raise ValueError("FP16 periodic TwoSum requires n_products=3 and period 5, 10, 32, or 128")
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
    if args.n_products == 10:
        args.twosum = True
    if args.n_products == 15 and args.twosum:
        raise ValueError("five-slice fp16x5 uses dual C without per-FMA TwoSum")
    if args.cross_f16:
        if args.n_products != 3:
            raise ValueError("--cross-f16 is n_products=3 only")
        if args.twosum:
            raise ValueError("--cross-f16 cannot combine with --twosum")
        if args.precision == "fp32":
            raise ValueError("--cross-f16 is the fp16 fused path only")
    n_slices = (
        2
        if args.precision == "fp32"
        else {1: 1, 3: 2, 4: 2, 6: 3, 9: 3, 10: 4, 15: 5, 16: 4}[args.n_products]
    )
    shape = GemmShape(args.n, args.k, args.m)
    grid = Grid(args.grid_size, args.grid_size)
    # FP16 packing uses the fp16 five-buffer check; fused tiles are larger
    # (A0/A1/B0/B1 double-buffered f16 plus fp32 C = 20 T^2 for square tiles).
    # TwoSum adds a second fp32 C and scratch columns (~24 T^2 + 24 T).
    # Three-slice TwoSum is 32 T^2 + 24 T and does not fit T=40.
    # Five-slice fp16x5 (n_products=15) is 48 T^2; T=32 does not fit. Use T=28, N=1792.
    # FP32 fused: double-buffer A0,A1,B0,B1 + one C = 36 T^2; T=40 does not fit.
    mt, kt, nt = validate_summa_configuration(
        shape,
        PRECISIONS[args.precision],
        grid,
        data_budget_bytes=args.data_budget_kib * 1024,
    )
    fused_bytes = fused_buffer_bytes(
        mt,
        kt,
        nt,
        args.twosum,
        args.n_products,
        args.precision,
        args.cross_f16,
        args.twosum_period,
    )
    budget = args.data_budget_kib * 1024
    if fused_bytes > budget:
        raise ValueError(
            f"fused Ozaki buffers need {fused_bytes} bytes/PE, above the "
            f"{budget}-byte data budget"
        )

    name = config_name(
        shape,
        args.grid_size,
        args.n_products,
        args.start_barrier,
        args.twosum,
        args.precision,
        args.cross_f16,
        args.twosum_period,
        args.twosum_a0b0_only,
        args.twosum_cross_only,
        args.fp64_merge,
    )
    artifact_root = ROOT / "artifacts" / "gemm" / name
    manifest_path = ROOT / "artifacts" / "gemm" / f"{name}.json"
    if manifest_path.exists() and not args.force:
        print(manifest_path)
        return
    artifact_root.mkdir(parents=True, exist_ok=True)

    source_dir = ROOT / "benchmarks" / "gemm_mesh" / "src"
    if args.precision == "fp32":
        layout = "layout_fused_fp32.csl"
        params = (
            f"P:{args.grid_size},Mt:{mt},Kt:{kt},Nt:{nt},"
            f"start_barrier:{int(args.start_barrier)},"
            f"twosum:{int(args.twosum)},twosum_period:{args.twosum_period}"
        )
    else:
        layout = "layout_fused.csl"
        params = (
            f"P:{args.grid_size},Mt:{mt},Kt:{kt},Nt:{nt},"
            f"n_products:{args.n_products},n_slices:{n_slices},start_barrier:{int(args.start_barrier)},"
            f"twosum:{int(args.twosum)},twosum_period:{int(args.twosum_period)},"
            f"twosum_a0b0_only:{int(args.twosum_a0b0_only)},"
            f"twosum_cross_only:{int(args.twosum_cross_only)},"
            f"fp64_merge:{int(args.fp64_merge)},"
            f"cross_f16:{int(args.cross_f16)}"
        )
    options = " ".join(
        [
            "--arch=wse3",
            f"--fabric-dims={FABRIC_DIMS}",
            f"--fabric-offsets={FABRIC_OFFSETS}",
            f"--params={params}",
            "--memcpy",
            f"--channels={CHANNELS}",
            "--fp16-format=f16",
            "-o=out",
        ]
    )

    compile_start = time.perf_counter()
    with SdkCompiler(disable_version_check=True) as compiler:
        artifact_path = compiler.compile(
            str(source_dir),
            layout,
            options,
            str(artifact_root),
        )
    compile_client_wall_s = time.perf_counter() - compile_start

    if args.precision == "fp32":
        algorithm = "ozaki_fp32_fused"
        source = (
            "WaferLLM MeshGEMM 2×f32: scalar multiply, vector bucket add and TwoSum"
            if args.twosum else "WaferLLM MeshGEMM classic Ozaki: four @fmacs of 2×f32 slices"
        )
        precision_name = "fp64_2xfp32"
    elif args.n_products == 1:
        algorithm = "ozaki_fused"
        source = "WaferLLM MeshGEMM A0B0-only @fmachs"
        precision_name = "fp16xfp32"
    elif args.n_products == 3 and args.cross_f16:
        algorithm = "ozaki_fused"
        source = (
            "WaferLLM MeshGEMM fused Ozaki: A0B0 @fmachs, A0B1/A1B0 @fmach, skip A1B1"
        )
        precision_name = "fp16xfp32"
    elif args.n_products == 3:
        algorithm = "ozaki_fused"
        source = "WaferLLM MeshGEMM fused Ozaki: three @fmachs, skip A1B1, packed A0|A1 hop"
        precision_name = "fp16xfp32"
    elif args.n_products == 10:
        algorithm = "ozaki_fused"
        source = (
            "WaferLLM MeshGEMM fused Ozaki: 4 fp16 slices, triangle i+j<4 "
            "(10 @fmachs), Knuth TwoSum (C_hi, C_lo)"
        )
        precision_name = "fp16x4_tri_twosum"
    elif args.n_products == 16:
        algorithm = "ozaki_fused"
        source = "WaferLLM MeshGEMM fused Ozaki: sixteen @fmachs per four-hop step"
        precision_name = "fp16xfp32"
    elif args.n_products == 15:
        algorithm = "ozaki_fused"
        source = (
            "WaferLLM MeshGEMM fused Ozaki: 5 fp16 slices of float64, "
            "15 @fmachs triangle, dual fp32 C_hi/C_lo (no TwoSum)"
        )
        precision_name = "fp16x5_f64"
    elif args.n_products == 6:
        algorithm = "ozaki_fused"
        source = (
            "WaferLLM MeshGEMM fused Ozaki: six @fmachs triangle i+j<3, "
            "packed A0|A1|A2 hop"
        )
        precision_name = "fp16xfp32"
    elif args.n_products == 9 and args.twosum:
        algorithm = "ozaki_fused"
        source = "WaferLLM MeshGEMM fused Ozaki: nine @fmachs, Knuth TwoSum (C_hi, C_lo)"
        precision_name = "fp16xfp32"
    elif args.n_products == 9:
        algorithm = "ozaki_fused"
        source = "WaferLLM MeshGEMM fused Ozaki: nine @fmachs, packed A0|A1|A2 hop"
        precision_name = "fp16xfp32"
    elif args.twosum:
        algorithm = "ozaki_fused"
        source = "WaferLLM MeshGEMM fused Ozaki: four @fmachs, Knuth TwoSum (C_hi, C_lo)"
        precision_name = "fp16xfp32"
    else:
        algorithm = "ozaki_fused"
        source = "WaferLLM MeshGEMM fused Ozaki: four @fmachs, packed A0|A1 hop"
        precision_name = "fp16xfp32"
    manifest = {
        "schema_version": 1,
        "kind": "gemm_compile",
        "algorithm": algorithm,
        "n_products": args.n_products,
        "n_slices": n_slices,
        "dual_c": args.n_products == 15,
        "twosum": bool(args.twosum or args.n_products == 10),
        "twosum_period": int(args.twosum_period),
        "twosum_a0b0_only": bool(args.twosum_a0b0_only),
        "twosum_cross_only": bool(args.twosum_cross_only),
        "fp64_merge": bool(args.fp64_merge),
        "cross_f16": bool(args.cross_f16),
        "source": source,
        "created_at": utc_now_iso(),
        "name": name,
        "precision": precision_name,
        "grid": {"width": grid.width, "height": grid.height},
        "channels": CHANNELS,
        "shape": {"n": shape.n, "k": shape.k, "m": shape.m},
        "local_tile": {"mt": mt, "kt": kt, "nt": nt},
        "fused_buffer_bytes": fused_bytes,
        "fabric_dims": FABRIC_DIMS,
        "fabric_offsets": FABRIC_OFFSETS,
        "layout": layout,
        "start_barrier": bool(args.start_barrier),
        "compiler_options": options,
        "compile_client_wall_s": compile_client_wall_s,
        "artifact_path": artifact_path,
        "client": {
            "hostname": socket.gethostname(),
            "python": platform.python_version(),
            "cerebras_sdk": importlib.metadata.version("cerebras-sdk"),
            "cerebras_appliance": importlib.metadata.version("cerebras-appliance"),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
