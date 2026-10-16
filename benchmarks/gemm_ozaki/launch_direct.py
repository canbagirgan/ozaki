#!/usr/bin/env python3
"""Launch fused Ozaki GEMM (one kernel, 1/3/4/6/9/10/15/16 products per two-hop).

Wafer (default) uses SdkLauncher and a CS-3. ``--simulator`` is the Cerebras
SDK simulator (simfabric): same compile artifact, no CS-3 allocation, and no
``%CMADDR%``. E2E simulator runs stay barrier-free.
"""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import socket
import sys
import time
from pathlib import Path

from cerebras.sdk.client import SdkLauncher

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ozaki_roofline.model import (  # noqa: E402
    GRID_SIZE_HELP,
    NOMINAL_CLOCK_MHZ,
    GemmShape,
    Grid,
    PRECISIONS,
    parse_grid_size,
    reject_large_pe_map,
    validate_summa_configuration,
)
from ozaki_roofline.records import append_jsonl  # noqa: E402

RECORD_PREFIX = "OZAKI_RECORD "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaled-residual-fscale", action="store_true", help="Use scalar fscale_f32 only at the final compensated cross merge.")
    parser.add_argument(
        "--scaled-residual",
        action="store_true",
        help="3 products: store fp16(2**16 * fp32 residual), merge cross sum at 2**-16.",
    )
    parser.add_argument(
        "--precision",
        choices=("fp16", "fp32"),
        default="fp16",
        help="fp16 = fused float16 slices. fp32 = classic 2×f32 Ozaki.",
    )
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--grid-size", type=parse_grid_size, default=64, help=GRID_SIZE_HELP)
    parser.add_argument(
        "--data-budget-kib",
        type=int,
        default=40,
        help="Per-PE matrix-buffer budget in KiB (WSE-3 SRAM is 48 KiB).",
    )
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
        help="Launch the two-word TwoSum artifact (res + res_lo).",
    )
    parser.add_argument(
        "--twosum-period",
        type=int,
        choices=(1, 5, 10, 32, 128),
        default=1,
        help="Must match the compile. 5 or 10 = that many products in a bucket, then one TwoSum.",
    )
    parser.add_argument(
        "--twosum-a0b0-only",
        action="store_true",
        help="Launch the artifact that TwoSums only A0B0.",
    )
    parser.add_argument(
        "--twosum-cross-only",
        action="store_true",
        help="Launch the artifact that TwoSums only A0B1 and A1B0.",
    )
    parser.add_argument(
        "--fp64-merge",
        action="store_true",
        help="Launch the artifact that stores IEEE fp64(C_hi)+fp64(C_lo) on the wafer.",
    )
    parser.add_argument(
        "--cross-f16",
        action="store_true",
        help="Launch the n_products=3 artifact with A0B1/A1B0 in f16.",
    )
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
        "--simulator",
        action="store_true",
        help=(
            "Run on the Cerebras SDK simulator (simfabric). Does not allocate "
            "a CS-3. Uses the same compile artifact as a wafer launch. "
            "E2E: omit --start-barrier and --dump-pe-map."
        ),
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--clock-mhz", type=float, default=NOMINAL_CLOCK_MHZ)
    parser.add_argument(
        "--start-barrier",
        action="store_true",
        help=(
            "Launch the start-barrier artifact used for on-wafer kernel "
            "timing. Never use for E2E."
        ),
    )
    parser.add_argument(
        "--dump-pe-map",
        action="store_true",
        help=(
            "Ask the worker to emit the PxP TSC duration grid. Does not "
            "enable the start barrier; pass --start-barrier for on-wafer maps."
        ),
    )
    parser.add_argument(
        "--high-precision",
        action="store_true",
        help="Generate sources in the host wide type and headline-check vs wide GEMM.",
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
        help="Forwarded to the worker: high-prec check of C[:s,:s] when skipping full GEMM.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results" / "raw" / "gemm-ozaki-direct.jsonl",
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
    scaled_residual: bool = False,
    scaled_residual_fscale: bool = False,
) -> str:
    if precision == "fp32":
        name = f"ozaki-fp32-fused-p{grid_size}-n{shape.n}-k{shape.k}-m{shape.m}"
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
    if scaled_residual:
        name += "-scaled-residual" + ("-final-merge" if twosum else "")
        if scaled_residual_fscale:
            name += "-fscale"
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
        return 16 * mt * kt + 16 * kt * nt + 4 * mt * nt
    if n_products == 10:
        # Matches compile.py and the kernel: A2/A3/B2/B3 are single-buffered.
        # 12 T^2 A + 12 T^2 B + 8 T^2 C + 24 T scratch. T=32: 33536.
        # Full double-buffer was 41728 and does not fit the 40960-byte cap.
        return 12 * mt * kt + 12 * kt * nt + 8 * mt * nt + 24 * mt
    if n_products == 15:
        bytes_ = 20 * mt * kt + 20 * kt * nt + 8 * mt * nt
    elif n_products == 16:
        bytes_ = 16 * mt * kt + 16 * kt * nt + 4 * mt * nt
    elif n_products in (6, 9):
        bytes_ = 12 * mt * kt + 12 * kt * nt + 4 * mt * nt
    else:
        bytes_ = 8 * mt * kt + 8 * kt * nt + 4 * mt * nt
    if twosum:
        bytes_ += 4 * mt * nt + 24 * mt
    if twosum_period != 1:
        bytes_ += 4 * mt * nt
    if cross_f16:
        bytes_ += 2 * mt * nt
    return bytes_


def load_manifest(
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
    scaled_residual: bool = False,
    scaled_residual_fscale: bool = False,
) -> dict:
    path = (
        ROOT
        / "artifacts"
        / "gemm"
        / f"{config_name(shape, grid_size, n_products, start_barrier, twosum, precision, cross_f16, twosum_period, twosum_a0b0_only, twosum_cross_only, fp64_merge, scaled_residual, scaled_residual_fscale)}.json"
    )
    if not path.exists():
        raise FileNotFoundError(f"compile manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    if args.scaled_residual_fscale and not (args.scaled_residual and args.twosum):
        raise ValueError("--scaled-residual-fscale requires --scaled-residual --twosum")
    if args.scaled_residual and (
        args.precision != "fp16" or args.n_products != 3
        or args.cross_f16
        or (args.twosum and (args.twosum_period != 10
            or args.twosum_a0b0_only or args.twosum_cross_only or args.fp64_merge))
    ):
        raise ValueError("--scaled-residual requires fp16, n_products=3, no cross-f16; TwoSum supports period 10 only")
    if args.simulator and args.start_barrier:
        raise ValueError(
            "--simulator is a barrier-free E2E simfabric run; omit --start-barrier"
        )
    default_results = ROOT / "results" / "raw" / "gemm-ozaki-direct.jsonl"
    if args.simulator and args.results == default_results:
        args.results = ROOT / "results" / "raw" / "gemm-ozaki-simulator.jsonl"
    if args.precision == "fp32":
        if args.twosum:
            raise ValueError("classic 2×f32 Ozaki has no TwoSum on this path")
        if args.n_products != 4:
            raise ValueError("classic 2×f32 Ozaki is n_products=4 only")
    if args.twosum_period != 1:
        if not args.twosum:
            raise ValueError("--twosum-period requires --twosum")
        if args.n_products != 3:
            raise ValueError("--twosum-period is n_products=3 only")
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
    reject_large_pe_map(args.grid_size, args.dump_pe_map)
    shape = GemmShape(args.n, args.k, args.m)
    grid = Grid(args.grid_size, args.grid_size)
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
    if args.scaled_residual:
        fused_bytes += (12 if args.twosum else 4) * mt * nt
    budget = args.data_budget_kib * 1024
    if fused_bytes > budget:
        raise ValueError(
            f"fused Ozaki buffers need {fused_bytes} bytes/PE, above the "
            f"{budget}-byte data budget"
        )
    manifest = load_manifest(
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
        args.scaled_residual,
        args.scaled_residual_fscale,
    )
    if int(manifest.get("n_products", 4)) != int(args.n_products):
        raise RuntimeError(
            "compile artifact n_products="
            f"{manifest.get('n_products')} does not match launch "
            f"--n-products={args.n_products}"
        )
    if int(manifest.get("twosum_period", 1)) != int(args.twosum_period):
        raise RuntimeError(
            "compile artifact twosum_period="
            f"{manifest.get('twosum_period', 1)} does not match launch "
            f"--twosum-period={args.twosum_period}"
        )
    if bool(manifest.get("fp64_merge", False)) != bool(args.fp64_merge):
        raise RuntimeError(
            "compile artifact fp64_merge="
            f"{manifest.get('fp64_merge', False)} does not match launch "
            f"--fp64-merge={args.fp64_merge}"
        )
    if bool(manifest.get("twosum_cross_only", False)) != bool(args.twosum_cross_only):
        raise RuntimeError(
            "compile artifact twosum_cross_only="
            f"{manifest.get('twosum_cross_only', False)} does not match launch "
            f"--twosum-cross-only={args.twosum_cross_only}"
        )
    if bool(manifest.get("twosum_a0b0_only", False)) != bool(args.twosum_a0b0_only):
        raise RuntimeError(
            "compile artifact twosum_a0b0_only="
            f"{manifest.get('twosum_a0b0_only', False)} does not match launch "
            f"--twosum-a0b0-only={args.twosum_a0b0_only}"
        )
    if bool(manifest.get("twosum")) != bool(args.twosum):
        raise RuntimeError(
            "compile artifact twosum="
            f"{manifest.get('twosum')} does not match launch --twosum="
            f"{args.twosum}"
        )
    if bool(manifest.get("scaled_residual_fscale", False)) != args.scaled_residual_fscale:
        raise ValueError("compile artifact scaling implementation does not match launch")
    if bool(manifest.get("scaled_residual", False)) != args.scaled_residual:
        raise ValueError("compile artifact scaled_residual does not match launch")
    if bool(manifest.get("cross_f16")) != bool(args.cross_f16):
        raise RuntimeError(
            "compile artifact cross_f16="
            f"{manifest.get('cross_f16')} does not match launch --cross-f16="
            f"{args.cross_f16}"
        )
    if bool(manifest.get("start_barrier")) != bool(args.start_barrier):
        raise RuntimeError(
            "compile artifact start_barrier="
            f"{manifest.get('start_barrier')} does not match launch "
            f"--start-barrier={args.start_barrier}; E2E must use a "
            "no-barrier compile"
        )
    here = Path(__file__).resolve().parent
    mesh = ROOT / "benchmarks" / "gemm_mesh"
    worker_cmd = [
        "cs_python",
        "run_direct.py",
        "--name",
        "out",
    ]
    # SdkLauncher rejects %CMADDR% when simulator=True. Omitting --cmaddr
    # makes the worker SdkRuntime attach to simfabric.
    if not args.simulator:
        worker_cmd.extend(["--cmaddr", "%CMADDR%"])
    worker_cmd.extend(
        [
            "--n",
            str(shape.n),
            "--k",
            str(shape.k),
            "--m",
            str(shape.m),
            "--grid-size",
            str(args.grid_size),
            "--warmup",
            str(args.warmup),
            "--repetitions",
            str(args.repetitions),
            "--clock-mhz",
            str(args.clock_mhz),
            "--n-products",
            str(args.n_products),
            "--input-kind",
            args.input_kind,
            "--phi",
            str(args.phi),
        ]
    )
    command = shlex.join(worker_cmd)
    if args.precision == "fp32":
        command = shlex.join(shlex.split(command) + ["--precision", "fp32"])
    if args.twosum:
        command = shlex.join(shlex.split(command) + ["--twosum"])
    if args.twosum_period != 1:
        command = shlex.join(
            shlex.split(command) + ["--twosum-period", str(args.twosum_period)]
        )
    if args.twosum_a0b0_only:
        command = shlex.join(shlex.split(command) + ["--twosum-a0b0-only"])
    if args.twosum_cross_only:
        command = shlex.join(shlex.split(command) + ["--twosum-cross-only"])
    if args.fp64_merge:
        command = shlex.join(shlex.split(command) + ["--fp64-merge"])
    if args.high_precision or args.n_products == 10:
        command = shlex.join(shlex.split(command) + ["--high-precision"])
    if args.scaled_residual_fscale:
        command = shlex.join(shlex.split(command) + ["--scaled-residual-fscale"])
    if args.scaled_residual:
        command = shlex.join(shlex.split(command) + ["--scaled-residual"])
    if args.cross_f16:
        command = shlex.join(shlex.split(command) + ["--cross-f16"])
    if args.dump_pe_map:
        command = shlex.join(shlex.split(command) + ["--dump-pe-map"])
    if args.skip_validation:
        command = shlex.join(shlex.split(command) + ["--skip-validation"])
    if args.validate_subsample:
        command = shlex.join(
            shlex.split(command)
            + ["--validate-subsample", str(args.validate_subsample)]
        )

    launcher_start = time.perf_counter_ns()
    with SdkLauncher(
        manifest["artifact_path"],
        simulator=args.simulator,
        disable_version_check=True,
    ) as launcher:
        launcher_startup_s = (time.perf_counter_ns() - launcher_start) * 1e-9
        launcher.stage(str(here / "run_direct.py"))
        launcher.stage(str(mesh / "host_common.py"))
        launcher.stage(str(ROOT / "src" / "ozaki_roofline" / "validation.py"))
        launcher.stage(str(ROOT / "src" / "ozaki_roofline" / "split.py"))
        launcher.stage(str(ROOT / "src" / "ozaki_roofline" / "high_precision.py"))
        run_start = time.perf_counter_ns()
        response = launcher.run(command)
        launcher_run_s = (time.perf_counter_ns() - run_start) * 1e-9

    parsed = 0
    for line in response.splitlines():
        if not line.startswith(RECORD_PREFIX):
            continue
        record = json.loads(line.removeprefix(RECORD_PREFIX))
        record["start_barrier"] = bool(args.start_barrier)
        record["simulator"] = bool(args.simulator)
        record["launcher"] = {
            "submission_hostname": socket.gethostname(),
            "submission_python": platform.python_version(),
            "launcher_startup_s": launcher_startup_s,
            "launcher_run_s": launcher_run_s,
            "execution_mode": (
                "sdk_launcher_simulator"
                if args.simulator
                else "sdk_launcher_worker_direct"
            ),
        }
        append_jsonl(args.results, record)
        print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
        parsed += 1
    if parsed == 0:
        raise RuntimeError("Ozaki worker emitted no OZAKI_RECORD lines:\n" + response)


if __name__ == "__main__":
    main()
