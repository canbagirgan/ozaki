"""IEEE float32 word split into unsigned 16-bit halves (not Ozaki covering).

Each 32-bit word is cut in the middle:

    bits = float32.view(uint32)
    high = (bits >> 16).astype(uint16)   # A0 / B0
    low  = (bits & 0xFFFF).astype(uint16)  # A1 / B1

The wafer path multiplies those uint16 tiles and accumulates in 32 bits.
Schoolbook reconstruction of the uint32 bit-pattern is

    u = high<<16 | low
    C_int = (A0@B0)<<32 + (A0@B1 + A1@B0)<<16 + A1@B1

That is integer arithmetic on IEEE payloads, not a residual covering of
the float32 values. A truncated-fp32 (bf16-style) float oracle is provided
only as a comment: ``A0_f = bitcast(A0<<16)``, ``A1_f = A - A0_f``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

U32_LIMIT = 1 << 32
I32_ABS_LIMIT = 1 << 31


class Fp32Halves(NamedTuple):
    """Unsigned 16-bit halves of a float32 matrix."""

    high: np.ndarray
    low: np.ndarray


def split_fp32_uint16_halves(matrix: np.ndarray) -> Fp32Halves:
    """Split float32 (or float32-bit-identical) values into u16 high/low halves."""

    if matrix.ndim != 2:
        raise ValueError("expected a 2-D matrix")
    bits = np.asarray(matrix, dtype=np.float32).view(np.uint32)
    high = np.asarray((bits >> np.uint32(16)) & np.uint32(0xFFFF), dtype=np.uint16)
    low = np.asarray(bits & np.uint32(0xFFFF), dtype=np.uint16)
    return Fp32Halves(high=high, low=low)


def reconstruct_uint32(halves: Fp32Halves) -> np.ndarray:
    """Rebuild the original uint32 payloads: ``high<<16 | low``."""

    high = np.asarray(halves.high, dtype=np.uint32)
    low = np.asarray(halves.low, dtype=np.uint32)
    return (high << np.uint32(16)) | low


def truncated_fp32(high: np.ndarray) -> np.ndarray:
    """bf16-style truncated float32: bitcast(high << 16), low bits zero."""

    bits = np.asarray(high, dtype=np.uint32) << np.uint32(16)
    return bits.view(np.float32)


def integer_product_int64(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Exact uint16 GEMM in int64 (host oracle, no 32-bit wrap).

    When every product fits in the float64 mantissa (``(2^16-1)^2 * K < 2^53``),
    use DGEMM and round back; that is exact and much faster than int64 GEMM.
    """

    left16 = np.asarray(left)
    right16 = np.asarray(right)
    k = int(left16.shape[1])
    worst = ((1 << 16) - 1) ** 2 * k
    if worst < (1 << 53):
        return np.rint(
            np.matmul(left16.astype(np.float64), right16.astype(np.float64))
        ).astype(np.int64)
    return np.matmul(left16.astype(np.int64), right16.astype(np.int64))


def schoolbook_uint32_gemm(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    products: dict[tuple[int, int], np.ndarray] | None = None,
) -> np.ndarray:
    """Integer schoolbook of ``(A0<<16|A1) @ (B0<<16|B1)`` in int64.

    ``products[(i, j)]`` is ``Ai @ Bj``. Values may be wrapping device u32
    zero-extended to int64, or exact host int64 tiles.
    """

    if products is None:
        tiles = {
            (0, 0): integer_product_int64(a0, b0),
            (0, 1): integer_product_int64(a0, b1),
            (1, 0): integer_product_int64(a1, b0),
            (1, 1): integer_product_int64(a1, b1),
        }
    else:
        tiles = products
    p00 = np.asarray(tiles[(0, 0)], dtype=np.int64)
    p01 = np.asarray(tiles[(0, 1)], dtype=np.int64)
    p10 = np.asarray(tiles[(1, 0)], dtype=np.int64)
    p11 = np.asarray(tiles[(1, 1)], dtype=np.int64)
    return (p00 << 32) + ((p01 + p10) << 16) + p11


def wrap_u32(values: np.ndarray) -> np.ndarray:
    """Low 32 bits of an int64 tile, as uint32."""

    return np.asarray(values, dtype=np.int64).astype(np.uint64).astype(np.uint32)


def overflow_report_u16(
    products: dict[tuple[int, int], np.ndarray],
    *,
    k: int,
) -> dict[str, float]:
    """How badly uint16×uint16 K-sums overflow i32 / u32."""

    max_abs = 0
    n_overflow_i32 = 0
    n_overflow_u32 = 0
    n_elem = 0
    for tile in products.values():
        abs_tile = np.abs(np.asarray(tile, dtype=np.int64))
        if abs_tile.size:
            max_abs = max(max_abs, int(abs_tile.max()))
        n_overflow_i32 += int(np.count_nonzero(abs_tile >= I32_ABS_LIMIT))
        n_overflow_u32 += int(np.count_nonzero(abs_tile >= U32_LIMIT))
        n_elem += int(abs_tile.size)
    worst = ((1 << 16) - 1) ** 2 * int(k)
    return {
        "max_abs_product": float(max_abs),
        "n_overflow_i32": float(n_overflow_i32),
        "n_overflow_u32": float(n_overflow_u32),
        "overflow_fraction_i32": float(n_overflow_i32) / max(n_elem, 1),
        "overflow_fraction_u32": float(n_overflow_u32) / max(n_elem, 1),
        "fits_i32": float(n_overflow_i32 == 0),
        "fits_u32": float(n_overflow_u32 == 0),
        "worst_case_u16_product": float(worst),
        "k_reduction_bound_bits": float(32.0 + np.log2(max(k, 1))),
        "overflow_expected_at_p64": float(k >= 64),
    }


def truncated_fp32_residual_oracle(
    a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Host float oracle: truncated high-halves plus true residuals.

    Not the wafer integer path. ``A0_f = bitcast(A0<<16)``, ``A1_f = A - A0_f``.
    """

    a32 = np.asarray(a, dtype=np.float32)
    b32 = np.asarray(b, dtype=np.float32)
    a_h, _ = split_fp32_uint16_halves(a32)
    b_h, _ = split_fp32_uint16_halves(b32)
    a0_f = truncated_fp32(a_h)
    b0_f = truncated_fp32(b_h)
    a64 = a32.astype(np.float64)
    b64 = b32.astype(np.float64)
    a1_f = a64 - a0_f.astype(np.float64)
    b1_f = b64 - b0_f.astype(np.float64)
    c_trunc = np.matmul(a0_f.astype(np.float64), b0_f.astype(np.float64))
    c_resid = (
        c_trunc
        + np.matmul(a0_f.astype(np.float64), b1_f)
        + np.matmul(a1_f, b0_f.astype(np.float64))
        + np.matmul(a1_f, b1_f)
    )
    return a0_f, c_trunc, c_resid
