"""Ozaki residual split: float32 into float16 slices, or float64 into float32.

Matches the initial document for two float16 slices of a float32 matrix:

    A0 = fp16(A)
    A1 = fp16(A - fp32(A0))

and the same for B. Reconstruction is the four-product sum

    C = A0 B0 + A0 B1 + A1 B0 + A1 B1

with each product accumulated in float32. A1 B1 is kept.

Classic Ozaki for extra-accurate DGEMM uses two float32 slices of float64:

    A0 = fp32(A)
    A1 = fp32(A - fp64(A0))

Two float32 words cover 48 mantissa bits vs float64's 53; this is not a
full covering of binary64. Do not add a third float32 slice here.

Three float16 slices continue the residual from float32 bits only (no invented
precision). The third word is leftover after two float16 roundings:

    A2 = fp16(A - fp32(A0) - fp32(A1))

Nine products cover 3×11 = 33 > 24 mantissa bits. Knuth TwoSum of those
products into (C_hi, C_lo) is reconstructed on the host as float64(hi)+float64(lo).

Four float16 slices of a **float32** source continue one more residual
step (A3 is leftover after three fp16 roundings of fp32). Sixteen
products into one fp32 C is the full 4×4 wafer plan. On 1/sqrt(K)
Gaussians A2 and A3 typically underflow to zero.

Four float16 slices of a **wide** source (float64, or the host high
type: IEEE binary128 / x87 long double / mpmath quad) keep the residual
in that wide type (same recipe as five-slice, stop at A3). 4×11 = 44
significand bits: enough for a ~1e-12 target (~40 bits), short of
binary64's 53. The wafer plan that can hit 1e-12 is the Ozaki triangle
``i+j < 4`` (10 ``@fmachs``, not 11 and not 16) with Knuth TwoSum of
each ``@fmachs`` into ``(C_hi, C_lo)``. Host ``float64(hi)+float64(lo)``.
Weight-split dual C without per-FMA TwoSum stays f32-class (~1e-6).
Eleven products is not a complete weight class (10, then 13).

Five float16 slices of a **true float64** source cover 55 > 53 significand
bits:

    A0 = fp16(A)
    A1 = fp16(A - float64(A0))
    ...
    A4 = fp16(residual after A0..A3)

The fast wafer plan is the Ozaki triangle (``i+j < 5`` → 15 ``@fmachs``,
not 25), with high-weight products (``i+j <= 1``) into ``C_hi`` and the
rest into ``C_lo``. Host reconstructs ``float64(C_hi) + float64(C_lo)``.
No per-FMA Knuth TwoSum.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

# A float32 has 24 significand bits. Two fp16 words cover 22; the extra ~2
# bits land in A2 on about one quarter of random mantissas. A3 is identically
# zero for any finite float32 (3*11=33>24). Exponent band [2^0, 2^8) keeps
# A0 finite in fp16 and GEMM products inside float32 at K=2048.
FULL_MANTISSA_EXP_LO = 0
FULL_MANTISSA_EXP_HI = 7

# Ootomo et al., ozIMMU Eq. (6): Uniform(-0.5, 0.5) * exp(phi * N(0, 1)).
# Paper sweep is phi in {0.1, 1, 2, 4}. Default 1 is the first non-trivial
# width; phi=4 can overflow fp16.
OZIMMU_PHI_DEFAULT = 1.0

# Naive residual fp16 underflows A3 for O(1) values (leftover < 2^{-24}).
# Scale each A row / B column into the fp16 finite range before splitting.
FP16_FINITE_MAX = 65504.0
FP16_COVERING_FRAC = 0.98


def _max_abs_float64(matrix: np.ndarray, axis: int) -> np.ndarray:
    arr = np.asarray(matrix)
    if arr.dtype == object:
        try:
            from ozaki_roofline.high_precision import to_float64
        except ImportError:
            from high_precision import to_float64

        abs64 = np.abs(to_float64(arr))
    else:
        abs64 = np.abs(np.asarray(arr, dtype=np.float64))
    return np.max(abs64, axis=axis)


def fp16_covering_row_scales(
    matrix: np.ndarray, *, frac: float = FP16_COVERING_FRAC
) -> np.ndarray:
    """Per-row scale so max|row| maps to ``frac * 65504`` (fp16 finite max)."""

    mx = _max_abs_float64(matrix, axis=1)
    return np.where(mx == 0.0, 1.0, (FP16_FINITE_MAX * frac) / mx)


def fp16_covering_col_scales(
    matrix: np.ndarray, *, frac: float = FP16_COVERING_FRAC
) -> np.ndarray:
    """Per-column scale so max|col| maps to ``frac * 65504``."""

    mx = _max_abs_float64(matrix, axis=0)
    return np.where(mx == 0.0, 1.0, (FP16_FINITE_MAX * frac) / mx)


def apply_row_scales(matrix: np.ndarray, scales: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix)
    sc = np.asarray(scales)
    if arr.dtype == object:
        out = np.empty_like(arr)
        for i in range(arr.shape[0]):
            factor = sc[i]
            for j in range(arr.shape[1]):
                out[i, j] = arr[i, j] * factor
        return out
    return arr * sc.astype(arr.dtype, copy=False)[:, None]


def apply_col_scales(matrix: np.ndarray, scales: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix)
    sc = np.asarray(scales)
    if arr.dtype == object:
        out = np.empty_like(arr)
        for j in range(arr.shape[1]):
            factor = sc[j]
            for i in range(arr.shape[0]):
                out[i, j] = arr[i, j] * factor
        return out
    return arr * sc.astype(arr.dtype, copy=False)[None, :]


def unscale_gemm(
    c: np.ndarray, row_scales: np.ndarray, col_scales: np.ndarray
) -> np.ndarray:
    """Undo per-row A and per-column B covering scales after GEMM."""

    return np.asarray(c, dtype=np.float64) / (
        np.asarray(row_scales, dtype=np.float64)[:, None]
        * np.asarray(col_scales, dtype=np.float64)[None, :]
    )



def _wide_float_dtype(dtype: np.dtype) -> np.dtype | None:
    """float64 or a NumPy type wider than binary64 (float128/longdouble)."""

    if dtype == np.float64:
        return np.dtype(np.float64)
    if dtype == np.float32 or dtype == np.float16:
        return None
    try:
        nmant = int(np.finfo(dtype).nmant)
    except (TypeError, ValueError, OverflowError):
        return None
    if nmant > 52:
        return np.dtype(dtype)
    return None


def ozimmu_matrix(
    rng: np.random.Generator,
    shape: tuple[int, int],
    *,
    phi: float = OZIMMU_PHI_DEFAULT,
    dtype: type = np.float32,
) -> np.ndarray:
    """ozIMMU random matrix: signed uniform mantissa, lognormal scale.

    ``phi`` stretches the exponent range. Larger ``phi`` needs more Ozaki
    slices to hold the same accuracy.
    """

    if len(shape) != 2:
        raise ValueError("expected a 2-D shape")
    if phi < 0.0:
        raise ValueError("phi must be >= 0")
    mantissa = rng.uniform(-0.5, 0.5, size=shape)
    scale = np.exp(phi * rng.standard_normal(size=shape))
    return (mantissa * scale).astype(dtype, copy=False)


def full_mantissa_fp32(
    rng: np.random.Generator,
    shape: tuple[int, int],
    *,
    exp_lo: int = FULL_MANTISSA_EXP_LO,
    exp_hi: int = FULL_MANTISSA_EXP_HI,
) -> np.ndarray:
    """Float32 with an independent 23-bit mantissa and a safe fp16 exponent.

    Every significand bit is random, so the bits that 2-slice Ozaki drops
    are actually present. ``residual_split_fp16(..., n_slices=4)`` then
    keeps A0 and A1 live, A2 on ~1/4 of entries, and A3 empty.
    """

    if len(shape) != 2:
        raise ValueError("expected a 2-D shape")
    if exp_lo > exp_hi:
        raise ValueError("exp_lo must be <= exp_hi")
    sign = rng.integers(0, 2, size=shape, dtype=np.uint32)
    mantissa = rng.integers(0, 1 << 23, size=shape, dtype=np.uint32)
    exponent = rng.integers(exp_lo + 127, exp_hi + 128, size=shape, dtype=np.uint32)
    bits = (sign << np.uint32(31)) | (exponent << np.uint32(23)) | mantissa
    return bits.view(np.float32)


class TwoSlice(NamedTuple):
    """High and residual slices of one matrix (float16 of f32, or float32 of f64)."""

    high: np.ndarray
    low: np.ndarray


class ThreeSlice(NamedTuple):
    """Three residual float16 slices of one float32 matrix (A0, A1, A2)."""

    high: np.ndarray
    mid: np.ndarray
    extra: np.ndarray


class FourSlice(NamedTuple):
    """Four residual float16 slices (A0..A3) of float32 or float64."""

    high: np.ndarray
    mid: np.ndarray
    extra: np.ndarray
    tail: np.ndarray


class FiveSlice(NamedTuple):
    """Five residual float16 slices of one float64 matrix (A0..A4)."""

    w0: np.ndarray
    w1: np.ndarray
    w2: np.ndarray
    w3: np.ndarray
    w4: np.ndarray


def residual_split_fp32(matrix: np.ndarray) -> TwoSlice:
    """Split a 2-D array stored as float64 into two float32 slices.

    ``A0 = float32(A)`` and ``A1 = float32(A - float64(A0))``. The source is
    converted to float64 first so extra bits are not discarded before the
    residual. Reconstruction ``float64(A0) + float64(A1)`` is closer to ``A``
    than ``A0`` alone, but two float32 words cannot recover all 53 bits.
    """

    if matrix.ndim != 2:
        raise ValueError("expected a 2-D matrix")
    source = np.asarray(matrix, dtype=np.float64)
    high = source.astype(np.float32)
    residual = source - np.asarray(high, dtype=np.float64)
    return TwoSlice(high=high, low=residual.astype(np.float32))


def residual_split_fp16(
    matrix: np.ndarray, n_slices: int = 2
) -> TwoSlice | ThreeSlice | FourSlice | FiveSlice:
    """Split a 2-D array into two..five residual float16 slices.

    ``n_slices`` 2 or 3 converts to float32 first. ``n_slices=4`` of a
    float32 source does the same (A3 cannot invent bits past float32).
    ``n_slices=4`` of a **float64** or **float128/longdouble** source
    keeps the residual in that wide type. ``n_slices=5`` keeps float64
    residual. Object (mpmath) arrays use ``residual_split_fp16_high``.
    """

    if matrix.ndim != 2:
        raise ValueError("expected a 2-D matrix")
    if n_slices not in (2, 3, 4, 5):
        raise ValueError("n_slices must be 2, 3, 4, or 5")
    source_arr = np.asarray(matrix)
    if source_arr.dtype == object:
        try:
            from ozaki_roofline.high_precision import residual_split_fp16_high
        except ImportError:
            from high_precision import residual_split_fp16_high

        if n_slices not in (4, 5):
            raise ValueError("object/mpmath split is n_slices=4 or 5")
        return residual_split_fp16_high(source_arr, n_slices=n_slices)
    wide = _wide_float_dtype(source_arr.dtype)
    if n_slices == 5:
        source = np.asarray(matrix, dtype=np.float64)
        words: list[np.ndarray] = []
        residual = source
        for _ in range(n_slices):
            word = residual.astype(np.float16)
            words.append(word)
            residual = residual - np.asarray(word, dtype=np.float64)
        return FiveSlice(*words)
    if n_slices == 4 and wide is not None:
        source = np.asarray(matrix, dtype=wide)
        words = []
        residual = source
        for _ in range(4):
            word = residual.astype(np.float16)
            words.append(word)
            residual = residual - np.asarray(word, dtype=wide)
        return FourSlice(
            high=words[0], mid=words[1], extra=words[2], tail=words[3]
        )
    source = np.asarray(matrix, dtype=np.float32)
    high = source.astype(np.float16)
    residual = source - np.asarray(high, dtype=np.float32)
    mid = residual.astype(np.float16)
    if n_slices == 2:
        return TwoSlice(high=high, low=mid)
    leftover = residual - np.asarray(mid, dtype=np.float32)
    extra = leftover.astype(np.float16)
    if n_slices == 3:
        return ThreeSlice(high=high, mid=mid, extra=extra)
    rest = leftover - np.asarray(extra, dtype=np.float32)
    tail = rest.astype(np.float16)
    return FourSlice(high=high, mid=mid, extra=extra, tail=tail)


def reconstruct_fp32(slices: TwoSlice | ThreeSlice | FourSlice) -> np.ndarray:
    """Sum the float16 slices in float32. Not exact A when a residual underflows."""

    rebuilt = np.asarray(slices.high, dtype=np.float32)
    if isinstance(slices, FourSlice):
        rebuilt = rebuilt + np.asarray(slices.mid, dtype=np.float32)
        rebuilt = rebuilt + np.asarray(slices.extra, dtype=np.float32)
        rebuilt = rebuilt + np.asarray(slices.tail, dtype=np.float32)
        return rebuilt
    if isinstance(slices, ThreeSlice):
        rebuilt = rebuilt + np.asarray(slices.mid, dtype=np.float32)
        rebuilt = rebuilt + np.asarray(slices.extra, dtype=np.float32)
        return rebuilt
    return rebuilt + np.asarray(slices.low, dtype=np.float32)


def reconstruct_high(
    slices: TwoSlice | FourSlice | FiveSlice, dtype: np.dtype
) -> np.ndarray:
    """Sum fp16 (or fp32) slices in ``dtype`` (float64 or float128/longdouble)."""

    if isinstance(slices, FiveSlice):
        rebuilt = np.asarray(slices.w0, dtype=dtype)
        rebuilt = rebuilt + np.asarray(slices.w1, dtype=dtype)
        rebuilt = rebuilt + np.asarray(slices.w2, dtype=dtype)
        rebuilt = rebuilt + np.asarray(slices.w3, dtype=dtype)
        return rebuilt + np.asarray(slices.w4, dtype=dtype)
    if isinstance(slices, FourSlice):
        rebuilt = np.asarray(slices.high, dtype=dtype)
        rebuilt = rebuilt + np.asarray(slices.mid, dtype=dtype)
        rebuilt = rebuilt + np.asarray(slices.extra, dtype=dtype)
        return rebuilt + np.asarray(slices.tail, dtype=dtype)
    return np.asarray(slices.high, dtype=dtype) + np.asarray(
        slices.low, dtype=dtype
    )


def reconstruct_fp64(slices: TwoSlice | FourSlice | FiveSlice) -> np.ndarray:
    """Sum slices in float64.

    Two float32 words (classic Ozaki), four or five float16 words of a
    wide source. Closer to the source than the high word alone.
    """

    if isinstance(slices, FiveSlice):
        rebuilt = np.asarray(slices.w0, dtype=np.float64)
        rebuilt = rebuilt + np.asarray(slices.w1, dtype=np.float64)
        rebuilt = rebuilt + np.asarray(slices.w2, dtype=np.float64)
        rebuilt = rebuilt + np.asarray(slices.w3, dtype=np.float64)
        return rebuilt + np.asarray(slices.w4, dtype=np.float64)
    if isinstance(slices, FourSlice):
        rebuilt = np.asarray(slices.high, dtype=np.float64)
        rebuilt = rebuilt + np.asarray(slices.mid, dtype=np.float64)
        rebuilt = rebuilt + np.asarray(slices.extra, dtype=np.float64)
        return rebuilt + np.asarray(slices.tail, dtype=np.float64)
    return np.asarray(slices.high, dtype=np.float64) + np.asarray(
        slices.low, dtype=np.float64
    )


def knuth_twosum(
    a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Error-free transformation of ``a + b`` in float32: ``s + e = a + b``."""

    a32 = np.asarray(a, dtype=np.float32)
    b32 = np.asarray(b, dtype=np.float32)
    s = a32 + b32
    z = s - a32
    e = (a32 - (s - z)) + (b32 - z)
    return s, e


def twosum_add(
    hi: np.ndarray, lo: np.ndarray, p: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Add a float32 ``p`` into the two-word accumulator ``(hi, lo)``."""

    s, e1 = knuth_twosum(hi, p)
    v, _e2 = knuth_twosum(lo, e1)
    return s, v


def reconstruct_twosum_fp64(hi: np.ndarray, lo: np.ndarray) -> np.ndarray:
    """Host D2H reconstruction: float32 words added in float64 (exact)."""

    return np.asarray(hi, dtype=np.float64) + np.asarray(lo, dtype=np.float64)


def four_product_fp32(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
) -> np.ndarray:
    """Host oracle for the wafer plan: four float32 GEMMs, float64 add, float32 C.

    Each ``numpy.matmul`` of float32 slices is a stand-in for ``@fmachs``
    (fp16 inputs, fp32 accumulate) with a different reduction order. The four
    products are summed in float64 so the add does not eat the extra bits.
    """

    def product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.matmul(
            np.asarray(left, dtype=np.float32),
            np.asarray(right, dtype=np.float32),
        ).astype(np.float64)

    reconstructed = (
        product(a0, b0) + product(a0, b1) + product(a1, b0) + product(a1, b1)
    )
    return reconstructed.astype(np.float32)


def three_product_fp32(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
) -> np.ndarray:
    """Same as ``four_product_fp32`` without the last A1B1 GEMM."""

    def product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.matmul(
            np.asarray(left, dtype=np.float32),
            np.asarray(right, dtype=np.float32),
        ).astype(np.float64)

    reconstructed = product(a0, b0) + product(a0, b1) + product(a1, b0)
    return reconstructed.astype(np.float32)


def three_product_cross_f16(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
) -> np.ndarray:
    """Host stand-in for A0B0 ``@fmachs`` plus A0B1/A1B0 ``@fmach``.

    Device reduction is per-k ``@fmach`` into one f16 tile, then one
    ``@faddhs`` into C. This oracle does two f16 GEMMs, adds in f16,
    and promotes once. Order differs from the wafer.
    """

    c = np.matmul(
        np.asarray(a0, dtype=np.float32),
        np.asarray(b0, dtype=np.float32),
    )
    cross = np.matmul(
        np.asarray(a0, dtype=np.float16),
        np.asarray(b1, dtype=np.float16),
    ).astype(np.float16)
    cross = (
        cross.astype(np.float32)
        + np.matmul(
            np.asarray(a1, dtype=np.float16),
            np.asarray(b0, dtype=np.float16),
        ).astype(np.float32)
    ).astype(np.float16)
    return (c + cross.astype(np.float32)).astype(np.float32)


def nine_product_fp32(
    a0: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    """Host oracle: nine float32 GEMMs, float64 add, float32 C."""

    def product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.matmul(
            np.asarray(left, dtype=np.float32),
            np.asarray(right, dtype=np.float32),
        ).astype(np.float64)

    reconstructed = (
        product(a0, b0)
        + product(a0, b1)
        + product(a0, b2)
        + product(a1, b0)
        + product(a1, b1)
        + product(a1, b2)
        + product(a2, b0)
        + product(a2, b1)
        + product(a2, b2)
    )
    return reconstructed.astype(np.float32)


def six_product_fp32(
    a0: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    """Host oracle: Ozaki triangle ``i+j<3`` (six products), float64 add, float32 C.

    Same as ``nine_product_fp32`` minus A1B2, A2B1, A2B2.
    """

    def product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.matmul(
            np.asarray(left, dtype=np.float32),
            np.asarray(right, dtype=np.float32),
        ).astype(np.float64)

    reconstructed = (
        product(a0, b0)
        + product(a0, b1)
        + product(a0, b2)
        + product(a1, b0)
        + product(a1, b1)
        + product(a2, b0)
    )
    return reconstructed.astype(np.float32)


def sixteen_product_fp32(
    a0: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    a3: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    b3: np.ndarray,
) -> np.ndarray:
    """Host oracle: sixteen float32 GEMMs, float64 add, float32 C.

    High-to-low product order matches the wafer kernel: A0B0 first, A3B3 last.
    """

    def product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.matmul(
            np.asarray(left, dtype=np.float32),
            np.asarray(right, dtype=np.float32),
        ).astype(np.float64)

    reconstructed = np.float64(0.0)
    for left in (a0, a1, a2, a3):
        for right in (b0, b1, b2, b3):
            reconstructed = reconstructed + product(left, right)
    return reconstructed.astype(np.float32)


def _fp32_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.matmul(
        np.asarray(left, dtype=np.float32),
        np.asarray(right, dtype=np.float32),
    )


def _triangle_product_dual_fp64(
    a_words: tuple[np.ndarray, ...],
    b_words: tuple[np.ndarray, ...],
    *,
    weight_limit: int,
    hi_max_weight: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ozaki triangle ``i+j < weight_limit`` into two fp32 tiles.

    Products with ``i+j <= hi_max_weight`` accumulate in ``C_hi``; the rest
    go to ``C_lo``. Returns ``(float64(hi)+float64(lo), C_hi, C_lo)``.
    """

    n = np.asarray(a_words[0]).shape[0]
    m = np.asarray(b_words[0]).shape[1]
    c_hi = np.zeros((n, m), dtype=np.float32)
    c_lo = np.zeros((n, m), dtype=np.float32)
    for i, left in enumerate(a_words):
        for j, right in enumerate(b_words):
            if i + j >= weight_limit:
                continue
            product = _fp32_product(left, right)
            if i + j <= hi_max_weight:
                c_hi = (c_hi + product).astype(np.float32)
            else:
                c_lo = (c_lo + product).astype(np.float32)
    rebuilt = np.asarray(c_hi, dtype=np.float64) + np.asarray(c_lo, dtype=np.float64)
    return rebuilt, c_hi, c_lo


def ten_product_dual_fp64(
    a0: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    a3: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    b3: np.ndarray,
    *,
    hi_max_weight: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ozaki triangle ``i+j < 4`` into two fp32 tiles, reconstructed in f64.

    Weight-split dual C **without** per-FMA TwoSum. Stays f32-class
    (~1e-6); do not use this as the 1e-12 accuracy story. See
    ``ten_product_twosum_fp64``.
    """

    return _triangle_product_dual_fp64(
        (a0, a1, a2, a3),
        (b0, b1, b2, b3),
        weight_limit=4,
        hi_max_weight=hi_max_weight,
    )


def fifteen_product_dual_fp64(
    a0: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    a3: np.ndarray,
    a4: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    b3: np.ndarray,
    b4: np.ndarray,
    *,
    hi_max_weight: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ozaki triangle ``i+j < 5`` into two fp32 tiles, reconstructed in f64.

    Products with ``i+j <= hi_max_weight`` accumulate in ``C_hi``; the rest
    go to ``C_lo``. Returns ``(float64(hi)+float64(lo), C_hi, C_lo)``.
    Matches the fast wafer plan: 15 ``@fmachs``, no per-FMA TwoSum.
    """

    return _triangle_product_dual_fp64(
        (a0, a1, a2, a3, a4),
        (b0, b1, b2, b3, b4),
        weight_limit=5,
        hi_max_weight=hi_max_weight,
    )


def fifteen_product_fp32(
    a0: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    a3: np.ndarray,
    a4: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    b3: np.ndarray,
    b4: np.ndarray,
) -> np.ndarray:
    """Same 15 products into one fp32 C (shows why a single word is not f64-class)."""

    a_words = (a0, a1, a2, a3, a4)
    b_words = (b0, b1, b2, b3, b4)
    reconstructed = np.float64(0.0)
    for i, left in enumerate(a_words):
        for j, right in enumerate(b_words):
            if i + j >= 5:
                continue
            reconstructed = reconstructed + _fp32_product(left, right).astype(np.float64)
    return reconstructed.astype(np.float32)


def four_product_twosum_fp64(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
) -> np.ndarray:
    """Device-like oracle: four rank-1 fp32 products, Knuth TwoSum into (hi, lo).

    Inner products are float32 multiplies (stand-in for ``@fmachs(p, 0, A, b)``).
    The returned array is ``float64(hi) + float64(lo)``.
    """

    a0_32 = np.asarray(a0, dtype=np.float32)
    a1_32 = np.asarray(a1, dtype=np.float32)
    b0_32 = np.asarray(b0, dtype=np.float32)
    b1_32 = np.asarray(b1, dtype=np.float32)
    n, k = a0_32.shape
    m = b0_32.shape[1]
    hi = np.zeros((n, m), dtype=np.float32)
    lo = np.zeros((n, m), dtype=np.float32)
    pairs = ((a0_32, b0_32), (a0_32, b1_32), (a1_32, b0_32), (a1_32, b1_32))
    for kk in range(k):
        for left, right in pairs:
            product = left[:, kk : kk + 1] * right[kk : kk + 1, :]
            hi, lo = twosum_add(hi, lo, product)
    return reconstruct_twosum_fp64(hi, lo)


def nine_product_twosum_fp64(
    a0: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    """Device-like oracle: nine rank-1 fp32 products, Knuth TwoSum into (hi, lo).

    High-to-low product order matches the wafer kernel:
    A0B0, A0B1, A0B2, A1B0, A1B1, A1B2, A2B0, A2B1, A2B2.
    Small tests only; do not run at N=2048 on the worker.
    """

    words = [
        np.asarray(word, dtype=np.float32)
        for word in (a0, a1, a2, b0, b1, b2)
    ]
    a0_32, a1_32, a2_32, b0_32, b1_32, b2_32 = words
    n, k = a0_32.shape
    m = b0_32.shape[1]
    hi = np.zeros((n, m), dtype=np.float32)
    lo = np.zeros((n, m), dtype=np.float32)
    pairs = (
        (a0_32, b0_32),
        (a0_32, b1_32),
        (a0_32, b2_32),
        (a1_32, b0_32),
        (a1_32, b1_32),
        (a1_32, b2_32),
        (a2_32, b0_32),
        (a2_32, b1_32),
        (a2_32, b2_32),
    )
    for kk in range(k):
        for left, right in pairs:
            product = left[:, kk : kk + 1] * right[kk : kk + 1, :]
            hi, lo = twosum_add(hi, lo, product)
    return reconstruct_twosum_fp64(hi, lo)


def ten_product_twosum_fp64(
    a0: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    a3: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    b3: np.ndarray,
) -> np.ndarray:
    """Device-like oracle: triangle ``i+j < 4``, Knuth TwoSum into (hi, lo).

    Ten products, high-to-low order matching the wafer kernel:
    A0B0, A0B1, A0B2, A0B3, A1B0, A1B1, A1B2, A2B0, A2B1, A3B0.
    Not eleven: 11 is not a complete Ozaki weight class.
    Small tests only.
    """

    words = [
        np.asarray(word, dtype=np.float32)
        for word in (a0, a1, a2, a3, b0, b1, b2, b3)
    ]
    a0_32, a1_32, a2_32, a3_32, b0_32, b1_32, b2_32, b3_32 = words
    n, k = a0_32.shape
    m = b0_32.shape[1]
    hi = np.zeros((n, m), dtype=np.float32)
    lo = np.zeros((n, m), dtype=np.float32)
    pairs = (
        (a0_32, b0_32),
        (a0_32, b1_32),
        (a0_32, b2_32),
        (a0_32, b3_32),
        (a1_32, b0_32),
        (a1_32, b1_32),
        (a1_32, b2_32),
        (a2_32, b0_32),
        (a2_32, b1_32),
        (a3_32, b0_32),
    )
    for kk in range(k):
        for left, right in pairs:
            product = left[:, kk : kk + 1] * right[kk : kk + 1, :]
            hi, lo = twosum_add(hi, lo, product)
    return reconstruct_twosum_fp64(hi, lo)


def fifteen_product_twosum_fp64(
    a0: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    a3: np.ndarray,
    a4: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    b3: np.ndarray,
    b4: np.ndarray,
) -> np.ndarray:
    """Device-like oracle: triangle ``i+j < 5``, Knuth TwoSum into (hi, lo).

    Fifteen products. Small tests only. Not the weight-split dual-C path.
    """

    words = [
        np.asarray(word, dtype=np.float32)
        for word in (a0, a1, a2, a3, a4, b0, b1, b2, b3, b4)
    ]
    a0_32, a1_32, a2_32, a3_32, a4_32, b0_32, b1_32, b2_32, b3_32, b4_32 = words
    n, k = a0_32.shape
    m = b0_32.shape[1]
    hi = np.zeros((n, m), dtype=np.float32)
    lo = np.zeros((n, m), dtype=np.float32)
    pairs = (
        (a0_32, b0_32),
        (a0_32, b1_32),
        (a0_32, b2_32),
        (a0_32, b3_32),
        (a0_32, b4_32),
        (a1_32, b0_32),
        (a1_32, b1_32),
        (a1_32, b2_32),
        (a1_32, b3_32),
        (a2_32, b0_32),
        (a2_32, b1_32),
        (a2_32, b2_32),
        (a3_32, b0_32),
        (a3_32, b1_32),
        (a4_32, b0_32),
    )
    for kk in range(k):
        for left, right in pairs:
            product = left[:, kk : kk + 1] * right[kk : kk + 1, :]
            hi, lo = twosum_add(hi, lo, product)
    return reconstruct_twosum_fp64(hi, lo)


def slice_diagnostics(
    source: np.ndarray, slices: TwoSlice | ThreeSlice | FourSlice | FiveSlice
) -> dict[str, float]:
    """How much of ``source`` the slices still hold.

    Wide sources (float64 or float128/longdouble) reconstruct in that
    type. Float32 sources keep the float16-slice reconstruction in float32.
    """

    tiny = float(np.finfo(np.float64).tiny)
    if isinstance(slices, FiveSlice):
        source64 = np.asarray(source, dtype=np.float64)
        rebuilt = reconstruct_fp64(slices)
        residual = source64 - rebuilt
        source_fro = float(np.linalg.norm(source64, ord="fro"))
        fp16_sub = np.float32(2.0**-14)

        def _zero_frac(word: np.ndarray) -> float:
            return float(np.mean(word == 0))

        def _sub_frac(word: np.ndarray) -> float:
            return float(
                np.mean(
                    (word != 0)
                    & (np.abs(word.astype(np.float32)) < fp16_sub)
                )
            )

        return {
            "source_frobenius": source_fro,
            "reconstruction_frobenius_error": float(
                np.linalg.norm(residual, ord="fro")
            ),
            "relative_reconstruction_error": float(np.linalg.norm(residual, ord="fro"))
            / max(source_fro, tiny),
            "high_plus_low_max_abs_error": float(np.max(np.abs(residual))),
            "n_slices": 5.0,
            "w0_zero_fraction": _zero_frac(slices.w0),
            "w1_zero_fraction": _zero_frac(slices.w1),
            "w2_zero_fraction": _zero_frac(slices.w2),
            "w3_zero_fraction": _zero_frac(slices.w3),
            "w4_zero_fraction": _zero_frac(slices.w4),
            "w2_subnormal_fraction": _sub_frac(slices.w2),
            "w3_subnormal_fraction": _sub_frac(slices.w3),
            "w4_subnormal_fraction": _sub_frac(slices.w4),
        }
    source_arr = np.asarray(source)
    wide = _wide_float_dtype(source_arr.dtype)
    if wide is not None and isinstance(slices, FourSlice):
        source_w = np.asarray(source, dtype=wide)
        rebuilt = reconstruct_high(slices, wide)
        residual = source_w - rebuilt
        source_fro = float(np.linalg.norm(source_w, ord="fro"))
        fp16_sub = np.float32(2.0**-14)

        def _zero_frac4(word: np.ndarray) -> float:
            return float(np.mean(word == 0))

        def _sub_frac4(word: np.ndarray) -> float:
            return float(
                np.mean(
                    (word != 0)
                    & (np.abs(word.astype(np.float32)) < fp16_sub)
                )
            )

        return {
            "source_frobenius": source_fro,
            "reconstruction_frobenius_error": float(
                np.linalg.norm(residual, ord="fro")
            ),
            "relative_reconstruction_error": float(np.linalg.norm(residual, ord="fro"))
            / max(source_fro, tiny),
            "high_plus_low_max_abs_error": float(np.max(np.abs(residual))),
            "n_slices": 4.0,
            "high_zero_fraction": _zero_frac4(slices.high),
            "mid_zero_fraction": _zero_frac4(slices.mid),
            "extra_zero_fraction": _zero_frac4(slices.extra),
            "tail_zero_fraction": _zero_frac4(slices.tail),
            "extra_subnormal_fraction": _sub_frac4(slices.extra),
            "tail_subnormal_fraction": _sub_frac4(slices.tail),
        }
    if np.asarray(source).dtype == np.float64 and isinstance(slices, TwoSlice):
        source64 = np.asarray(source, dtype=np.float64)
        rebuilt = reconstruct_fp64(slices)
        residual = source64 - rebuilt
        source_fro = float(np.linalg.norm(source64, ord="fro"))
        metrics: dict[str, float] = {
            "source_frobenius": source_fro,
            "reconstruction_frobenius_error": float(
                np.linalg.norm(residual, ord="fro")
            ),
            "relative_reconstruction_error": float(np.linalg.norm(residual, ord="fro"))
            / max(source_fro, tiny),
            "high_plus_low_max_abs_error": float(np.max(np.abs(residual))),
            "n_slices": 2.0,
            "low_slice_zero_fraction": float(np.mean(slices.low == 0)),
            "low_slice_subnormal_fraction": float(
                np.mean(
                    (slices.low != 0)
                    & (
                        np.abs(slices.low.astype(np.float32))
                        < np.float32(2.0**-126)
                    )
                )
            ),
        }
        return metrics

    source32 = np.asarray(source, dtype=np.float32)
    rebuilt = reconstruct_fp32(slices)
    residual = source32 - rebuilt
    source64 = source32.astype(np.float64)
    source_fro = float(np.linalg.norm(source64, ord="fro"))
    metrics: dict[str, float] = {
        "source_frobenius": source_fro,
        "reconstruction_frobenius_error": float(
            np.linalg.norm(residual.astype(np.float64), ord="fro")
        ),
        "relative_reconstruction_error": float(
            np.linalg.norm(residual.astype(np.float64), ord="fro")
        )
        / max(source_fro, tiny),
        "high_plus_low_max_abs_error": float(np.max(np.abs(residual))),
        "n_slices": (
            4.0
            if isinstance(slices, FourSlice)
            else 3.0
            if isinstance(slices, ThreeSlice)
            else 2.0
        ),
    }
    if isinstance(slices, (ThreeSlice, FourSlice)):
        metrics["mid_slice_zero_fraction"] = float(np.mean(slices.mid == 0))
        metrics["extra_slice_zero_fraction"] = float(np.mean(slices.extra == 0))
        metrics["extra_slice_subnormal_fraction"] = float(
            np.mean(
                (slices.extra != 0)
                & (
                    np.abs(slices.extra.astype(np.float32))
                    < np.float32(2.0**-14)
                )
            )
        )
        if isinstance(slices, FourSlice):
            metrics["tail_slice_zero_fraction"] = float(np.mean(slices.tail == 0))
            metrics["tail_slice_subnormal_fraction"] = float(
                np.mean(
                    (slices.tail != 0)
                    & (
                        np.abs(slices.tail.astype(np.float32))
                        < np.float32(2.0**-14)
                    )
                )
            )
    else:
        metrics["low_slice_zero_fraction"] = float(np.mean(slices.low == 0))
        metrics["low_slice_subnormal_fraction"] = float(
            np.mean(
                (slices.low != 0)
                & (np.abs(slices.low.astype(np.float32)) < np.float32(2.0**-14))
            )
        )
    return metrics
