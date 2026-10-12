"""Wider-than-binary64 host arithmetic for the 1e-12 Ozaki headline.

The wafer has no fp64 FPU. A 1e-12 relative Frobenius number is only
meaningful against a reference that is itself finer than ~1e-12. NumPy
``float64(A) @ float64(B)`` already has ~K·2^{-52} noise (~1e-13 at
K~2e3), so it cannot be the headline.

Resolution order
----------------
1. NumPy dtype with ``nmant >= 112`` (true IEEE binary128, 113-bit
   significand). Rare on x86_64.
2. NumPy ``float128`` / ``longdouble``. On this ALCF Linux x86_64 that
   is **80-bit x87 long double** (64-bit significand, 16-byte storage),
   **not** IEEE binary128. K·2^{-64} at K~2e3 is ~1e-16, still below
   1e-12. Documented in ``high_precision_info()``. Prefer this over
   mpmath: the appliance worker ships mpmath, which is slow and used
   to steal the path from x87 (then crashed on ``mpf(float16)``).
3. Optional ``mpmath`` at dps=34 (~113 bits) if no NumPy wide type.

Inputs are generated in the high type (extra mantissa bits filled from
independent float64 uniforms). Do not generate float64 and promote.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_BINARY128_NMANT = 112
_MPMATH_DPS = 34


def _numpy_wide_dtype() -> np.dtype | None:
    for name in ("float128", "longdouble"):
        if not hasattr(np, name):
            continue
        dtype = np.dtype(getattr(np, name))
        try:
            nmant = int(np.finfo(dtype).nmant)
        except (ValueError, OverflowError):
            continue
        if nmant > 52:
            return dtype
    return None


def _mpmath_available() -> bool:
    try:
        import mpmath  # noqa: F401
    except ImportError:
        return False
    return True


def high_precision_info() -> dict[str, Any]:
    """Describe the host type used for 1e-12 references.

    ``kind`` is one of ``ieee_binary128``, ``mpmath_quad``,
    ``x87_longdouble``, or ``float64`` (last-resort; not enough for 1e-12).
    """

    wide = _numpy_wide_dtype()
    if wide is not None:
        nmant = int(np.finfo(wide).nmant)
        if nmant >= _BINARY128_NMANT:
            return {
                "kind": "ieee_binary128",
                "dtype": wide,
                "dtype_name": wide.name,
                "nmant": nmant,
                "itemsize": int(wide.itemsize),
                "note": (
                    "NumPy dtype with nmant>=112: true IEEE binary128 "
                    "(113-bit significand)."
                ),
            }
    if wide is not None:
        nmant = int(np.finfo(wide).nmant)
        return {
            "kind": "x87_longdouble",
            "dtype": wide,
            "dtype_name": wide.name,
            "nmant": nmant,
            "itemsize": int(wide.itemsize),
            "note": (
                "NumPy float128/longdouble on this platform is 80-bit x87 "
                "extended precision (64-bit significand), not IEEE "
                "binary128 (113-bit). Enough for a 1e-12 headline "
                "(K*2^{-64} << 1e-12); not a full quad covering. "
                "Preferred over mpmath when both exist."
            ),
        }
    if _mpmath_available():
        import mpmath

        return {
            "kind": "mpmath_quad",
            "dtype": np.dtype(object),
            "dtype_name": "mpmath.mpf",
            "nmant": 113,
            "itemsize": None,
            "mpmath_dps": _MPMATH_DPS,
            "note": (
                f"mpmath mpf at dps={_MPMATH_DPS} (~113-bit significand). "
                "Optional; used only when NumPy has no wider-than-f64 type."
            ),
            "mpmath_version": getattr(mpmath, "__version__", ""),
        }
    return {
        "kind": "float64",
        "dtype": np.dtype(np.float64),
        "dtype_name": "float64",
        "nmant": 52,
        "itemsize": 8,
        "note": (
            "No wider-than-binary64 type. Cannot honestly claim 1e-12 "
            "against this reference."
        ),
    }


def high_precision_dtype() -> np.dtype:
    return high_precision_info()["dtype"]


def _as_high_scalar(value: float, info: dict[str, Any] | None = None) -> Any:
    info = info or high_precision_info()
    if info["kind"] == "mpmath_quad":
        import mpmath

        with mpmath.workdps(_MPMATH_DPS):
            return mpmath.mpf(value)
    return info["dtype"].type(value)


def _combine_unit_interval(
    u0: np.ndarray, u1: np.ndarray, u2: np.ndarray, info: dict[str, Any]
) -> np.ndarray:
    """Independent float64 uniforms → one high-precision sample in (0, 1)."""

    if info["kind"] == "mpmath_quad":
        import mpmath

        out = np.empty(u0.shape, dtype=object)
        scale1 = mpmath.mpf(2) ** -53
        scale2 = mpmath.mpf(2) ** -106
        with mpmath.workdps(_MPMATH_DPS):
            for idx in np.ndindex(u0.shape):
                out[idx] = (
                    mpmath.mpf(float(u0[idx]))
                    + mpmath.mpf(float(u1[idx])) * scale1
                    + mpmath.mpf(float(u2[idx])) * scale2
                )
        return out
    dtype = info["dtype"]
    two = dtype.type(2.0)
    return (
        u0.astype(dtype)
        + u1.astype(dtype) * (two ** dtype.type(-53))
        + u2.astype(dtype) * (two ** dtype.type(-106))
    )


def random_unit_high(
    rng: np.random.Generator, shape: tuple[int, ...], *, info: dict[str, Any] | None = None
) -> np.ndarray:
    """Open unit-interval samples with mantissa bits past binary64."""

    info = info or high_precision_info()
    # Exclude 0 so later log() in Box-Muller is defined.
    u0 = rng.random(size=shape)
    u0 = np.maximum(u0, np.finfo(np.float64).tiny)
    u1 = rng.random(size=shape)
    u2 = rng.random(size=shape)
    return _combine_unit_interval(u0, u1, u2, info)


def standard_normal_high(
    rng: np.random.Generator, shape: tuple[int, int], *, info: dict[str, Any] | None = None
) -> np.ndarray:
    """Box-Muller Gaussians in the high type. Not float64-then-promote."""

    info = info or high_precision_info()
    u1 = random_unit_high(rng, shape, info=info)
    u2 = random_unit_high(rng, shape, info=info)
    if info["kind"] == "mpmath_quad":
        import mpmath

        out = np.empty(shape, dtype=object)
        with mpmath.workdps(_MPMATH_DPS):
            two = mpmath.mpf(2)
            pi = mpmath.pi
            for idx in np.ndindex(shape):
                r = mpmath.sqrt(-two * mpmath.log(u1[idx]))
                out[idx] = r * mpmath.cos(two * pi * u2[idx])
        return out
    dtype = info["dtype"]
    two = dtype.type(2.0)
    pi = dtype.type(np.pi)
    radius = np.sqrt(-two * np.log(u1))
    return radius * np.cos(two * pi * u2)


def ozimmu_matrix_high(
    rng: np.random.Generator,
    shape: tuple[int, int],
    *,
    phi: float = 1.0,
    info: dict[str, Any] | None = None,
) -> np.ndarray:
    """ozIMMU matrix with mantissa and scale drawn in the high type."""

    if len(shape) != 2:
        raise ValueError("expected a 2-D shape")
    if phi < 0.0:
        raise ValueError("phi must be >= 0")
    info = info or high_precision_info()
    unit = random_unit_high(rng, shape, info=info)
    gauss = standard_normal_high(rng, shape, info=info)
    if info["kind"] == "mpmath_quad":
        import mpmath

        out = np.empty(shape, dtype=object)
        half = mpmath.mpf("0.5")
        phi_m = mpmath.mpf(phi)
        with mpmath.workdps(_MPMATH_DPS):
            for idx in np.ndindex(shape):
                mantissa = unit[idx] - half
                out[idx] = mantissa * mpmath.exp(phi_m * gauss[idx])
        return out
    dtype = info["dtype"]
    half = dtype.type(0.5)
    mantissa = unit - half
    scale = np.exp(dtype.type(phi) * gauss)
    return mantissa * scale


def scaled_gaussian_high(
    rng: np.random.Generator,
    shape: tuple[int, int],
    *,
    scale: float,
    info: dict[str, Any] | None = None,
) -> np.ndarray:
    """N(0, scale^2) in the high type (scale applied after the Gaussian)."""

    info = info or high_precision_info()
    gauss = standard_normal_high(rng, shape, info=info)
    if info["kind"] == "mpmath_quad":
        import mpmath

        s = mpmath.mpf(scale)
        out = np.empty(shape, dtype=object)
        with mpmath.workdps(_MPMATH_DPS):
            for idx in np.ndindex(shape):
                out[idx] = gauss[idx] * s
        return out
    return gauss * info["dtype"].type(scale)


def to_float64(matrix: np.ndarray) -> np.ndarray:
    """Truncate a high-precision matrix to binary64 (loses extra bits)."""

    if matrix.dtype == object:
        return np.vectorize(float, otypes=[np.float64])(matrix)
    return np.asarray(matrix, dtype=np.float64)


def promote_to_high(matrix: np.ndarray, *, info: dict[str, Any] | None = None) -> np.ndarray:
    info = info or high_precision_info()
    if info["kind"] == "mpmath_quad":
        import mpmath

        out = np.empty(matrix.shape, dtype=object)
        with mpmath.workdps(_MPMATH_DPS):
            for idx in np.ndindex(matrix.shape):
                raw = matrix[idx]
                out[idx] = mpmath.mpf(float(raw) if not isinstance(raw, str) else raw)
        return out
    return np.asarray(matrix, dtype=info["dtype"])


def gemm_high(
    a: np.ndarray, b: np.ndarray, *, info: dict[str, Any] | None = None
) -> np.ndarray:
    """C = A @ B in the high type. Not a float64 GEMM promoted afterwards."""

    info = info or high_precision_info()
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("A and B must be 2-D")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"inner dimensions disagree: A is {a.shape}, B is {b.shape}")
    if info["kind"] == "mpmath_quad":
        import mpmath

        n, k = a.shape
        m = b.shape[1]
        out = np.empty((n, m), dtype=object)
        with mpmath.workdps(_MPMATH_DPS):
            zero = mpmath.mpf(0)
            for i in range(n):
                for j in range(m):
                    acc = zero
                    for kk in range(k):
                        acc = acc + a[i, kk] * b[kk, j]
                    out[i, j] = acc
        return out
    a_h = np.asarray(a, dtype=info["dtype"])
    b_h = np.asarray(b, dtype=info["dtype"])
    return np.matmul(a_h, b_h)


def _high_frobenius(matrix: np.ndarray, info: dict[str, Any]) -> Any:
    if info["kind"] == "mpmath_quad":
        import mpmath

        acc = mpmath.mpf(0)
        with mpmath.workdps(_MPMATH_DPS):
            for idx in np.ndindex(matrix.shape):
                acc = acc + matrix[idx] * matrix[idx]
            return mpmath.sqrt(acc)
    return np.linalg.norm(np.asarray(matrix, dtype=info["dtype"]), ord="fro")


def _high_to_float(value: Any) -> float:
    return float(value)


def residual_split_fp16_high(
    matrix: np.ndarray,
    n_slices: int = 4,
    *,
    info: dict[str, Any] | None = None,
) -> Any:
    """Residual fp16 split keeping the residual in the high type.

    Each word is ``fp16(residual)``; the subtraction is **not** done in
    float32 or float64. Import of ``FourSlice`` is local to avoid a cycle.
    """

    try:
        from ozaki_roofline.split import FiveSlice, FourSlice
    except ImportError:
        from split import FiveSlice, FourSlice

    if matrix.ndim != 2:
        raise ValueError("expected a 2-D matrix")
    if n_slices not in (4, 5):
        raise ValueError("high-precision split is n_slices=4 or 5")
    info = info or high_precision_info()
    residual = promote_to_high(matrix, info=info)
    words: list[np.ndarray] = []
    for _ in range(n_slices):
        if info["kind"] == "mpmath_quad":
            word = np.empty(residual.shape, dtype=np.float16)
            for idx in np.ndindex(residual.shape):
                word[idx] = np.float16(float(residual[idx]))
            promoted = promote_to_high(word, info=info)
            next_residual = np.empty(residual.shape, dtype=object)
            for idx in np.ndindex(residual.shape):
                next_residual[idx] = residual[idx] - promoted[idx]
            residual = next_residual
        else:
            word = residual.astype(np.float16)
            residual = residual - np.asarray(word, dtype=info["dtype"])
        words.append(word)
    if n_slices == 5:
        return FiveSlice(*words)
    return FourSlice(high=words[0], mid=words[1], extra=words[2], tail=words[3])


def relative_frobenius_high(
    actual: np.ndarray,
    reference: np.ndarray,
    *,
    info: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Rel Frob of ``actual`` vs ``reference`` in the high type, reported as float."""

    info = info or high_precision_info()
    actual_h = promote_to_high(actual, info=info)
    reference_h = promote_to_high(reference, info=info)
    if info["kind"] == "mpmath_quad":
        diff = np.empty(actual_h.shape, dtype=object)
        for idx in np.ndindex(actual_h.shape):
            diff[idx] = actual_h[idx] - reference_h[idx]
    else:
        diff = actual_h - reference_h
    fro = _high_frobenius(diff, info)
    ref_fro = _high_frobenius(reference_h, info)
    tiny = 2.0 ** -1024
    ref_fro_f = max(_high_to_float(ref_fro), tiny)
    return {
        "frobenius_error": _high_to_float(fro),
        "reference_frobenius_norm": _high_to_float(ref_fro),
        "relative_frobenius_error": _high_to_float(fro) / ref_fro_f,
    }


def compare_device_vs_high(
    a: np.ndarray,
    b: np.ndarray,
    c_device: np.ndarray,
    *,
    info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Headline 1e-12 metric: device C (f64 hi+lo) vs high-precision GEMM of A, B.

    Also reports vs ``float64(A) @ float64(B)`` so the noisy f64 reference
    is visible and is not mistaken for the headline.
    """

    info = info or high_precision_info()
    c_ref = gemm_high(a, b, info=info)
    against_high = relative_frobenius_high(c_device, c_ref, info=info)
    a64 = to_float64(a)
    b64 = to_float64(b)
    c64_ref = np.matmul(a64, b64)
    c64 = np.asarray(c_device, dtype=np.float64)
    diff64 = c64 - c64_ref
    fro64 = float(np.linalg.norm(diff64, ord="fro"))
    ref64 = float(np.linalg.norm(c64_ref, ord="fro"))
    tiny = float(np.finfo(np.float64).tiny)
    return {
        "method": "high_precision_reference",
        "high_precision": {
            "kind": info["kind"],
            "dtype_name": info["dtype_name"],
            "nmant": info["nmant"],
            "note": info["note"],
        },
        "against_high_precision_reference": {
            **against_high,
            "note": (
                "Headline for 1e-12: high-precision GEMM of the same "
                "sources used for the Ozaki split. Device C is "
                "float64(C_hi)+float64(C_lo) promoted into the high type."
            ),
        },
        "against_float64_reference": {
            "frobenius_error": fro64,
            "reference_frobenius_norm": ref64,
            "relative_frobenius_error": fro64 / max(ref64, tiny),
            "note": (
                "float64(A)@float64(B). Noisy at ~K*2^{-52}; not the "
                "1e-12 headline."
            ),
        },
        "two_word_result": bool(np.asarray(c_device).dtype == np.float64),
    }
