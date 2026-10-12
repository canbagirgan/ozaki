"""Host numeric simulator for the 2-slice skip Ozaki kernels.

No Cerebras SDK and no wafer. This is the accuracy model of
``benchmarks/gemm_mesh/src/meshgemm_fused.csl`` for ``n_products=3``
(A0B0 + A0B1 + A1B0, A1B1 skipped):

* each product is an f16×f16 value added in f32, matching ``@fmachs``
  (the product is exact in f32);
* Knuth TwoSum is ``twosum_add``: TwoSum(C_hi, p), then C_lo = fl(C_lo+e)
  with the second-order error discarded;
* a blocked period counts products across K, accumulates that many with
  plain f32 adds into a bucket, then one ``twosum_add``, and flushes a
  non-empty tail at the end;
* ``a0b0`` / ``cross`` TwoSum only the named products. The others are
  plain f32 adds into C_hi, not into the bucket.

The accuracy oracle is ``float64(C_hi) + float64(C_lo)`` against
``float64(A) @ float64(B)``. ``--fp64-merge`` on the wafer stores the
IEEE bit halves of that sum; ``assemble_fp64_halves`` joins them. This
module does not emulate the CSL bit-twiddling adder.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from ozaki_roofline.split import (
    ozimmu_matrix,
    reconstruct_twosum_fp64,
    residual_split_fp16,
    twosum_add,
)
from ozaki_roofline.validation import compare_device_gemm

# Periods the fused kernel accepts. Period 1 is every product (bucket
# length 1), which matches gemv_twosum_step.
TWOSUM_PERIODS = (1, 5, 10, 32, 128)
PAIR_ORDER = ("a0b0", "a0b1", "a1b0")


def assemble_fp64_halves(hi_f32: np.ndarray, lo_f32: np.ndarray) -> np.ndarray:
    """Join IEEE fp64 halves. hi holds bits 63:32, lo holds bits 31:0.

    Same bit join as ``assemble_fp64_halves`` in
    ``benchmarks/gemm_ozaki/run_direct.py``. The two arrays are raw
    words, not the numeric TwoSum values.
    """

    hi_bits = np.ascontiguousarray(hi_f32, dtype=np.float32).view(np.uint32).astype(
        np.uint64
    )
    lo_bits = np.ascontiguousarray(lo_f32, dtype=np.float32).view(np.uint32).astype(
        np.uint64
    )
    return (lo_bits | (hi_bits << np.uint64(32))).view(np.float64)


def pack_fp64_halves(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a float64 into the two f32 words ``assemble_fp64_halves`` joins."""

    bits = np.ascontiguousarray(value, dtype=np.float64).view(np.uint64)
    lo_bits = (bits & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    hi_bits = (bits >> np.uint64(32)).astype(np.uint32)
    return hi_bits.view(np.float32), lo_bits.view(np.float32)


def relative_frobenius(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Relative Frobenius of ``c`` versus ``float64(A) @ float64(B)``."""

    report = compare_device_gemm(a, b, c, precision="fp32")
    return float(report["against_float64_reference"]["relative_frobenius_error"])


def native_fp32_gemm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """NumPy float32 matmul, the host stand-in for the FP32 baseline."""

    return np.matmul(
        np.asarray(a, dtype=np.float32),
        np.asarray(b, dtype=np.float32),
    )


def _as_f32_slices(
    a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a_slices = residual_split_fp16(np.asarray(a, dtype=np.float32), n_slices=2)
    b_slices = residual_split_fp16(np.asarray(b, dtype=np.float32), n_slices=2)
    return (
        np.asarray(a_slices.high, dtype=np.float32),
        np.asarray(a_slices.low, dtype=np.float32),
        np.asarray(b_slices.high, dtype=np.float32),
        np.asarray(b_slices.low, dtype=np.float32),
    )


def _f16_product(left: np.ndarray, right: np.ndarray, k: int) -> np.ndarray:
    """One ``@fmachs`` outer product at this K: exact f16×f16, stored in f32."""

    return (left[:, k : k + 1] * right[k : k + 1, :]).astype(np.float32, copy=False)


def _product_mode(name: str, selective: Optional[str]) -> str:
    if selective is None:
        return "bucket"
    if selective == "a0b0":
        return "twosum" if name == "a0b0" else "hi"
    if selective == "cross":
        return "twosum" if name != "a0b0" else "hi"
    raise ValueError("selective must be None, 'a0b0', or 'cross'")


def skip_accumulate(
    a: np.ndarray,
    b: np.ndarray,
    *,
    period: int = 1,
    selective: Optional[str] = None,
    twosum: bool = True,
) -> np.ndarray:
    """2-slice skip GEMM (3 products) with the wafer's accumulator.

    ``twosum=False`` is plain f32 ``@fmachs`` into one C (no C_lo).
    ``period`` counts products across K. Selective modes require period 1
    and apply TwoSum only to the named products.
    """

    if twosum and period not in TWOSUM_PERIODS:
        raise ValueError(f"twosum period must be one of {TWOSUM_PERIODS}")
    if selective is not None and not twosum:
        raise ValueError("selective TwoSum requires twosum")
    if selective is not None and period != 1:
        raise ValueError("selective TwoSum is period 1 only")
    if selective not in (None, "a0b0", "cross"):
        raise ValueError("selective must be None, 'a0b0', or 'cross'")

    a0, a1, b0, b1 = _as_f32_slices(a, b)
    n, k_dim = a0.shape
    m = b0.shape[1]
    if a1.shape != (n, k_dim) or b1.shape != (k_dim, m) or b0.shape[0] != k_dim:
        raise ValueError("slice shapes disagree")

    hi = np.zeros((n, m), dtype=np.float32)
    lo = np.zeros((n, m), dtype=np.float32)
    bucket = np.zeros((n, m), dtype=np.float32)
    count = 0
    pairs = {"a0b0": (a0, b0), "a0b1": (a0, b1), "a1b0": (a1, b0)}

    def flush() -> None:
        nonlocal hi, lo, bucket, count
        hi, lo = twosum_add(hi, lo, bucket)
        bucket = np.zeros((n, m), dtype=np.float32)
        count = 0

    for k in range(k_dim):
        for name in PAIR_ORDER:
            left, right = pairs[name]
            product = _f16_product(left, right, k)
            if not twosum:
                hi = (hi + product).astype(np.float32, copy=False)
                continue
            mode = _product_mode(name, selective)
            if mode == "hi":
                hi = (hi + product).astype(np.float32, copy=False)
                continue
            if mode == "twosum":
                hi, lo = twosum_add(hi, lo, product)
                continue
            bucket = (bucket + product).astype(np.float32, copy=False)
            count += 1
            if count == period:
                flush()

    if twosum and selective is None and count > 0:
        flush()
    if not twosum:
        return hi
    return reconstruct_twosum_fp64(hi, lo)


def ozimmu_problem(
    n: int,
    k: int,
    m: int,
    *,
    phi: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Float32 ozIMMU A (n, k) and B (k, m). ``phi`` only scales generation."""

    rng = np.random.default_rng(seed)
    a = ozimmu_matrix(rng, (n, k), phi=phi, dtype=np.float32)
    b = ozimmu_matrix(rng, (k, m), phi=phi, dtype=np.float32)
    return a, b


def variant_rel(
    a: np.ndarray,
    b: np.ndarray,
    *,
    periods: Iterable[int] = TWOSUM_PERIODS,
) -> dict[str, float]:
    """Relative Frobenius of each host variant versus float64(A)@float64(B)."""

    rel = {
        "fp32": relative_frobenius(a, b, native_fp32_gemm(a, b)),
        "skip": relative_frobenius(a, b, skip_accumulate(a, b, twosum=False)),
        "twosum-every": relative_frobenius(
            a, b, skip_accumulate(a, b, period=1, twosum=True)
        ),
        "twosum-a0b0-only": relative_frobenius(
            a, b, skip_accumulate(a, b, period=1, selective="a0b0")
        ),
        "twosum-cross-only": relative_frobenius(
            a, b, skip_accumulate(a, b, period=1, selective="cross")
        ),
    }
    for period in periods:
        if period == 1:
            continue
        rel[f"twosum-every-{period}"] = relative_frobenius(
            a, b, skip_accumulate(a, b, period=period, twosum=True)
        )
    return rel


def fp64_merge_matches_float64_add(hi: np.ndarray, lo: np.ndarray) -> bool:
    """True when bit-assembly of the float64 sum recovers that sum.

    Finite normals (and other finite values) are exact: packing
    ``float64(hi)+float64(lo)`` and calling ``assemble_fp64_halves``
    is the host ``--fp64-merge`` reconstruct path. Non-finite entries
    are skipped; an all-non-finite pair returns False.
    """

    oracle = reconstruct_twosum_fp64(hi, lo)
    finite = np.isfinite(oracle) & np.isfinite(hi) & np.isfinite(lo)
    if not np.any(finite):
        return False
    packed_hi, packed_lo = pack_fp64_halves(oracle)
    assembled = assemble_fp64_halves(packed_hi, packed_lo)
    return bool(np.array_equal(assembled[finite], oracle[finite]))
