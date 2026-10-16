#!/usr/bin/env python3
"""Compile and launch one CS-3 GEMM measurement.

``--grid-size`` is the square PE count P. ``--tile`` is the local tile T.
For a square problem ``N = K = M = T * P``. Pass ``--n`` or ``--tile``;
if both are set they must agree.

``--timing kernel`` compiles and launches the fabric start barrier used for
on-wafer TSC maps. ``--timing e2e`` leaves that barrier off.

Examples (from the repo root, on an ALCF CS-3 user node)::

    python scripts/measure.py --list-methods

    python scripts/measure.py --dry-run \\
        --method 4slice-tri --grid-size 512 --tile 32 --timing kernel

    python scripts/measure.py \\
        --method fp32 --grid-size 512 --n 16384 --timing kernel
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ozaki_roofline.campaign import METHODS, measurement_commands  # noqa: E402
from ozaki_roofline.model import NOMINAL_CLOCK_MHZ  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list-methods", action="store_true")
    parser.add_argument("--method", choices=tuple(METHODS))
    parser.add_argument(
        "--twosum-period", type=int, default=None,
        help="FP32 Ozaki TwoSum: override products per bucket (1..32767).",
    )
    parser.add_argument("--grid-size", type=int)
    parser.add_argument("--n", type=int, help="Rows of A and C. Square default: K = M = N.")
    parser.add_argument("--k", type=int, help="Shared dimension. Default: N.")
    parser.add_argument("--m", type=int, help="Columns of B and C. Default: N.")
    parser.add_argument("--tile", type=int, help="Local tile T. Square N = T * grid size.")
    parser.add_argument("--timing", choices=("kernel", "e2e"), help="kernel enables --start-barrier.")
    parser.add_argument(
        "--input-kind",
        choices=("scaled", "full-mantissa", "ozimmu"),
        default="ozimmu",
    )
    parser.add_argument("--phi", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7, help="Host-skip only. Wafer workers use seed 7.")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--clock-mhz", type=float, default=NOMINAL_CLOCK_MHZ)
    parser.add_argument("--data-budget-kib", type=int, default=40)
    parser.add_argument("--alpha", type=int, default=7, help="int method only.")
    parser.add_argument("--n-slices", type=int, default=8, help="int method only.")
    parser.add_argument("--single-product", action="store_true", help="int and u16x2 only.")
    parser.add_argument("--simulator", action="store_true")
    parser.add_argument("--dump-pe-map", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--high-precision", action="store_true")
    parser.add_argument("--validate-subsample", type=int, default=0)
    parser.add_argument("--stage", choices=("compile", "launch", "both"), default="both")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_methods:
        for name, method in METHODS.items():
            print(f"{name}\t{method.summary}")
        return
    if args.method is None:
        raise SystemExit("pass --method and --timing, or --list-methods")
    if args.timing is None:
        if args.method == "host-skip":
            args.timing = "e2e"
        else:
            raise SystemExit("pass --timing kernel or --timing e2e")
    commands = measurement_commands(
        ROOT,
        args.method,
        grid_size=args.grid_size,
        n=args.n,
        k=args.k,
        m=args.m,
        tile=args.tile,
        timing=args.timing,
        input_kind=args.input_kind,
        phi=args.phi,
        warmup=args.warmup,
        repetitions=args.repetitions,
        clock_mhz=args.clock_mhz,
        data_budget_kib=args.data_budget_kib,
        simulator=args.simulator,
        dump_pe_map=args.dump_pe_map,
        force=args.force,
        skip_validation=args.skip_validation,
        high_precision=args.high_precision,
        validate_subsample=args.validate_subsample,
        alpha=args.alpha,
        n_slices=args.n_slices,
        single_product=args.single_product,
        seed=args.seed,
        stage=args.stage,
        twosum_period=args.twosum_period,
    )
    for command in commands:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
