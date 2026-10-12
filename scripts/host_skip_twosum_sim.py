#!/usr/bin/env python3
"""Host-only accuracy check for 2-slice skip Ozaki and Knuth TwoSum.

Runs entirely on the host. No Cerebras SDK, no wafer, no compile.
Models the n_products=3 kernel: A0B0+A0B1+A1B0, fp16 slices, fp32
``@fmachs`` products, optional Knuth TwoSum into (C_hi, C_lo).

Prints relative Frobenius versus float64(A)@float64(B) for the FP32
baseline, skip without TwoSum, every-product TwoSum, and blocked
periods 5, 10, 32, and 128. ``--phi`` only changes ozIMMU generation:
A_ij = Uniform(-0.5, 0.5) * exp(phi * N(0, 1)).

Example (from the repo root):

    PYTHONPATH=src python3 scripts/host_skip_twosum_sim.py --n 64 --phi 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ozaki_roofline.host_skip_sim import (  # noqa: E402
    TWOSUM_PERIODS,
    ozimmu_problem,
    variant_rel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Host numeric simulator for 2-slice skip Ozaki "
            "(n_products=3) with Knuth TwoSum. No wafer and no CSL compile."
        ),
        epilog=(
            "Example: PYTHONPATH=src python3 scripts/host_skip_twosum_sim.py "
            "--n 64 --phi 1"
        ),
    )
    parser.add_argument("--n", type=int, default=64, help="Rows of A and of C.")
    parser.add_argument("--k", type=int, default=None, help="Shared dimension. Default: N.")
    parser.add_argument("--m", type=int, default=None, help="Columns of B and of C. Default: N.")
    parser.add_argument(
        "--phi",
        type=float,
        default=1.0,
        help="ozIMMU exponent width. Paper set includes 0.1, 1, 2, 4.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Exit 1 unless every-product TwoSum beats fp32 and skip, "
            "and period 128 is between those two or not more than 4× the fp32 error."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k = args.n if args.k is None else args.k
    m = args.n if args.m is None else args.m
    if min(args.n, k, m) <= 0 or args.phi < 0.0:
        raise SystemExit("n, k, m must be positive and phi must be >= 0")
    a, b = ozimmu_problem(args.n, k, m, phi=args.phi, seed=args.seed)
    rel = variant_rel(a, b, periods=TWOSUM_PERIODS)
    print(
        f"host skip-TwoSum sim  N={args.n} K={k} M={m}  "
        f"phi={args.phi}  seed={args.seed}  input=ozimmu",
        flush=True,
    )
    order = [
        "fp32",
        "skip",
        "twosum-every",
        "twosum-every-5",
        "twosum-every-10",
        "twosum-every-32",
        "twosum-every-128",
        "twosum-a0b0-only",
        "twosum-cross-only",
    ]
    for name in order:
        print(f"{name:22s}  Rel {rel[name]:.6e}", flush=True)
    every = rel["twosum-every"]
    fp32 = rel["fp32"]
    skip = rel["skip"]
    blocked = rel["twosum-every-128"]
    lo, hi = (every, fp32) if every <= fp32 else (fp32, every)
    between = lo <= blocked <= hi
    mild = blocked <= fp32 * 4.0
    print(
        f"check  twosum-every < fp32: {every < fp32}  "
        f"twosum-every < skip: {every < skip}  "
        f"period-128 between every and fp32: {between}  "
        f"period-128 <= 4*fp32: {mild}",
        flush=True,
    )
    if args.check and not (every < fp32 and every < skip and (between or mild)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
