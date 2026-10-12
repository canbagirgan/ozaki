"""ozIMMU-style shared-exponent integer slices of float64.

WSE-3 has no vector integer ``@mul`` / ``@mac`` and i8 is not a memory-DSD
type, so slices are packed in i16 containers. The device kernel multiplies
those i16 values with the scalar ``*`` operator and accumulates in i32.

A row of A (column of B) shares one exponent from ``np.frexp``. The scaled
mantissa is a signed integer of at most 52 magnitude bits so ``rint`` stays
exact in float64, then split high-to-low into ``alpha``-bit slices.
Reconstruction is a power-of-two scale of the integer GEMM products; no
TwoSum is required when the K-reduction is exact in signed i32
(``2*alpha + log2(K) < 31``).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

# float64 can rint integers exactly only up to 2^53. Use 52 magnitude bits
# plus sign so |mant| < 2^52 after scaling by the shared frexp exponent.
INT_BITS = 52


class IntegerSlices(NamedTuple):
    """Shared-exponent integer slices of one float64 matrix."""

    slices: tuple[np.ndarray, ...]
    exponents: np.ndarray
    shifts: tuple[int, ...]
    alpha: int
    int_bits: int
    axis: int


def product_pairs(n_slices: int, *, triangle: bool = True) -> list[tuple[int, int]]:
    """Ozaki pairs. Triangle keeps ``i + j < n_slices`` (``s(s+1)/2`` products)."""

    if n_slices < 1:
        raise ValueError("n_slices must be positive")
    pairs: list[tuple[int, int]] = []
    for i in range(n_slices):
        for j in range(n_slices):
            if triangle and i + j >= n_slices:
                continue
            pairs.append((i, j))
    return pairs


def k_reduction_exact(alpha: int, k: int, *, signed_bits: int = 31) -> bool:
    """True if every slice product reduces along K without i32 overflow.

    Worst-case unsigned ``alpha``-bit values: ``((2^alpha)-1)^2 * K < 2^signed_bits``.
    """

    if alpha < 1 or k < 1 or signed_bits < 1:
        raise ValueError("alpha, k, and signed_bits must be positive")
    max_abs = (1 << alpha) - 1
    return int(max_abs) * int(max_abs) * int(k) < (1 << signed_bits)


def slice_shifts(alpha: int, n_slices: int, int_bits: int = INT_BITS) -> tuple[int, ...]:
    """Place-value shifts for high-to-low ``alpha``-bit chunks of ``int_bits``."""

    if alpha < 1 or n_slices < 1 or int_bits < 1:
        raise ValueError("alpha, n_slices, and int_bits must be positive")
    shifts: list[int] = []
    bit_remain = int_bits
    for _ in range(n_slices):
        shift = max(bit_remain - alpha, 0)
        shifts.append(shift)
        bit_remain = shift
    return tuple(shifts)


def split_shared_exponent(
    matrix: np.ndarray,
    *,
    alpha: int = 7,
    n_slices: int = 8,
    axis: int = 1,
    int_bits: int = INT_BITS,
) -> IntegerSlices:
    """Split a 2-D float64 matrix into signed i16 slices with a shared exponent.

    ``axis=1`` is per-row (A). ``axis=0`` is per-column (B). Each group is
    scaled with the ``frexp`` exponent so the integer mantissa uses
    ``int_bits`` magnitude bits (exact in float64), then split high-to-low.
    """

    if matrix.ndim != 2:
        raise ValueError("expected a 2-D matrix")
    if alpha < 1 or alpha > 7:
        raise ValueError("alpha must be in 1..7 so products fit i16 containers")
    if n_slices < 1 or n_slices > 16:
        raise ValueError("n_slices must be in 1..16")
    if axis not in (0, 1):
        raise ValueError("axis must be 0 (columns) or 1 (rows)")
    if int_bits < alpha or int_bits > 52:
        raise ValueError("int_bits must be in alpha..52")

    source = np.asarray(matrix, dtype=np.float64)
    absmax = np.max(np.abs(source), axis=axis, keepdims=True)
    _, exp = np.frexp(np.where(absmax == 0.0, 1.0, absmax))
    exp = np.where(absmax == 0.0, 0, exp).astype(np.int32)
    # absmax = m * 2^exp with m in [0.5, 1) => scaled max is in [2^{51}, 2^{52}).
    scale_exp = int_bits - exp
    scaled = np.ldexp(source, scale_exp)
    limit = 1 << int_bits
    mant = np.clip(np.rint(scaled), -(limit - 1), limit - 1).astype(np.int64)

    shifts = slice_shifts(alpha, n_slices, int_bits)
    remaining = mant
    words: list[np.ndarray] = []
    for shift in shifts:
        sl = remaining >> np.int64(shift)
        remaining = remaining - (sl << np.int64(shift))
        words.append(np.asarray(sl, dtype=np.int16))
    return IntegerSlices(
        slices=tuple(words),
        exponents=exp,
        shifts=shifts,
        alpha=alpha,
        int_bits=int_bits,
        axis=axis,
    )


def slice_weight_exp(slices: IntegerSlices, index: int) -> np.ndarray:
    """Power-of-two exponent that turns one integer slice back into float64."""

    return slices.exponents.astype(np.int32) - slices.int_bits + slices.shifts[index]


def reconstruct_float64(slices: IntegerSlices) -> np.ndarray:
    """Sum the integer slices in float64 with the stored shared exponents."""

    rebuilt = np.zeros(slices.slices[0].shape, dtype=np.float64)
    for index, word in enumerate(slices.slices):
        rebuilt = rebuilt + np.ldexp(
            np.asarray(word, dtype=np.float64), slice_weight_exp(slices, index)
        )
    return rebuilt


def integer_product_int64(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Exact integer GEMM in int64 (host oracle for one slice pair)."""

    return np.matmul(
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
    )


def reconstruct_gemm_fp64(
    a_slices: IntegerSlices,
    b_slices: IntegerSlices,
    products: np.ndarray | None = None,
    *,
    triangle: bool = True,
) -> np.ndarray:
    """Rebuild float64 C from integer slice products.

    If ``products`` is None, compute them with int64 matmul. ``products[p, q]``
    is the integer ``A_p @ B_q``. Exponents are per-row of A and per-column of B.
    """

    n_a = len(a_slices.slices)
    n_b = len(b_slices.slices)
    if n_a != n_b:
        raise ValueError("A and B must use the same number of slices")
    if a_slices.alpha != b_slices.alpha or a_slices.int_bits != b_slices.int_bits:
        raise ValueError("A and B must use the same alpha and int_bits")
    if a_slices.shifts != b_slices.shifts:
        raise ValueError("A and B must use the same slice shifts")
    n, k = a_slices.slices[0].shape
    k_b, m = b_slices.slices[0].shape
    if k != k_b:
        raise ValueError("inner dimensions of A and B slices do not match")

    pairs = product_pairs(n_a, triangle=triangle)
    if products is None:
        tiles = {
            (i, j): integer_product_int64(a_slices.slices[i], b_slices.slices[j])
            for i, j in pairs
        }
    else:
        if products.shape[:2] != (n_a, n_b):
            raise ValueError("products leading axes must be (n_slices, n_slices)")
        tiles = {(i, j): products[i, j] for i, j in pairs}

    e_a = a_slices.exponents.reshape(n, 1).astype(np.int32)
    e_b = b_slices.exponents.reshape(1, m).astype(np.int32)
    rebuilt = np.zeros((n, m), dtype=np.float64)
    for i, j in pairs:
        weight = (
            e_a
            + e_b
            - 2 * a_slices.int_bits
            + a_slices.shifts[i]
            + a_slices.shifts[j]
        )
        rebuilt = rebuilt + np.ldexp(np.asarray(tiles[(i, j)], dtype=np.float64), weight)
    return rebuilt


def slice_diagnostics_integer(
    source: np.ndarray, slices: IntegerSlices
) -> dict[str, float]:
    """How close the integer slices are to the float64 source."""

    source64 = np.asarray(source, dtype=np.float64)
    rebuilt = reconstruct_float64(slices)
    residual = source64 - rebuilt
    source_fro = float(np.linalg.norm(source64, ord="fro"))
    tiny = float(np.finfo(np.float64).tiny)
    max_abs_slice = max(float(np.max(np.abs(word))) for word in slices.slices)
    return {
        "source_frobenius": source_fro,
        "reconstruction_frobenius_error": float(np.linalg.norm(residual, ord="fro")),
        "relative_reconstruction_error": float(np.linalg.norm(residual, ord="fro"))
        / max(source_fro, tiny),
        "max_abs_error": float(np.max(np.abs(residual))),
        "n_slices": float(len(slices.slices)),
        "alpha": float(slices.alpha),
        "int_bits": float(slices.int_bits),
        "max_abs_slice": max_abs_slice,
        "k_reduction_bound_alpha_bits": float(
            2 * slices.alpha + np.log2(max(source64.shape[1], 1))
        ),
    }


def overflow_report(
    products: dict[tuple[int, int], np.ndarray],
    *,
    signed_bits: int = 31,
) -> dict[str, float]:
    """Host-side check that integer products fit in signed i32."""

    limit = 1 << signed_bits
    max_abs = 0
    n_overflow = 0
    n_elem = 0
    for tile in products.values():
        abs_tile = np.abs(np.asarray(tile, dtype=np.int64))
        max_abs = max(max_abs, int(abs_tile.max()) if abs_tile.size else 0)
        n_overflow += int(np.count_nonzero(abs_tile >= limit))
        n_elem += int(abs_tile.size)
    return {
        "max_abs_product": float(max_abs),
        "n_overflow_i32": float(n_overflow),
        "overflow_fraction": float(n_overflow) / max(n_elem, 1),
        "fits_i32": float(n_overflow == 0),
    }
