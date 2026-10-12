#!/usr/bin/env python3
"""Host oracles for the FP64-class Ozaki campaign. Small N only."""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ozaki_roofline.high_precision import (  # noqa: E402
    compare_device_vs_high,
    high_precision_info,
    ozimmu_matrix_high,
    scaled_gaussian_high,
)
from ozaki_roofline.split import (  # noqa: E402
    residual_split_fp16,
    ten_product_dual_fp64,
    ten_product_twosum_fp64,
    fifteen_product_twosum_fp64,
    nine_product_twosum_fp64,
    fp16_covering_row_scales,
    fp16_covering_col_scales,
    apply_row_scales,
    apply_col_scales,
    unscale_gemm,
)


def _words4(matrix):
    s = residual_split_fp16(matrix, n_slices=4)
    return s.high, s.mid, s.extra, s.tail


def _words5(matrix):
    s = residual_split_fp16(matrix, n_slices=5)
    return s.w0, s.w1, s.w2, s.w3, s.w4


def _words3(matrix):
    s = residual_split_fp16(matrix, n_slices=3)
    return s.high, s.mid, s.extra


def main() -> None:
    info = high_precision_info()
    rng = np.random.default_rng(7)
    cases = []
    for kind, n, k, m in (
        ("ozimmu_phi1", 8, 32, 8),
        ("ozimmu_phi1", 12, 64, 10),
        ("scaled_gaussian", 8, 32, 8),
        ("scaled_gaussian", 12, 64, 10),
    ):
        if kind == "ozimmu_phi1":
            a = ozimmu_matrix_high(rng, (n, k), phi=1.0, info=info)
            b = ozimmu_matrix_high(rng, (k, m), phi=1.0, info=info)
        else:
            scale = 1.0 / math.sqrt(k)
            a = scaled_gaussian_high(rng, (n, k), scale=scale, info=info)
            b = scaled_gaussian_high(rng, (k, m), scale=scale, info=info)
        a4, b4 = _words4(a), _words4(b)
        sa, sb = fp16_covering_row_scales(a), fp16_covering_col_scales(b)
        a_s = apply_row_scales(a, sa)
        b_s = apply_col_scales(b, sb)
        a4s, b4s = _words4(a_s), _words4(b_s)
        a5, b5 = _words5(a), _words5(b)
        a3, b3 = _words3(a), _words3(b)
        schemes = {
            "ten_twosum": ten_product_twosum_fp64(*a4, *b4),
            "ten_twosum_covering_scale": unscale_gemm(
                ten_product_twosum_fp64(*a4s, *b4s), sa, sb
            ),
            "ten_dual": ten_product_dual_fp64(*a4, *b4)[0],
            "fifteen_twosum": fifteen_product_twosum_fp64(*a5, *b5),
            "nine_twosum": nine_product_twosum_fp64(*a3, *b3),
        }
        for name, c in schemes.items():
            report = compare_device_vs_high(a, b, c, info=info)
            rec = {
                "kind": "host_ozaki_fp64_oracle",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "scheme": name,
                "input_kind": kind,
                "shape": {"n": n, "k": k, "m": m},
                "high_precision": {
                    "kind": info["kind"],
                    "nmant": info["nmant"],
                    "dtype_name": info["dtype_name"],
                },
                "rel_frob_vs_high": report["against_high_precision_reference"][
                    "relative_frobenius_error"
                ],
                "rel_frob_vs_f64": report["against_float64_reference"][
                    "relative_frobenius_error"
                ],
                "hits_1e12": bool(
                    report["against_high_precision_reference"]["relative_frobenius_error"]
                    <= 1e-12
                ),
            }
            cases.append(rec)
            print(json.dumps(rec, sort_keys=True), flush=True)
    out = ROOT / "results" / "raw" / "ozaki-fp64-host-oracle.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        for rec in cases:
            handle.write(json.dumps(rec, sort_keys=True) + "\n")
    print(f"wrote {len(cases)} records to {out}", flush=True)


if __name__ == "__main__":
    main()
