"""Compare a device GEMM result against a host NumPy reference.

The mathematical reference is the float64 product of the same A and B values
that were sent to the device. Every finite IEEE binary16 and binary32 value
is exactly representable in binary64, so this promotion does not change the
inputs. NumPy's float64 matmul is then an accurate proxy for the exact
product: for K=2560 the DGEMM rounding is on the order of K * 2^{-53} ~ 3e-13,
which is far below binary32 and binary16 unit roundoff.

Device FP16 uses @fmach (binary16 accumulation). NumPy's `@` on float16 arrays
is not used as a reference because it typically accumulates in a wider type
and would not match the wafer kernel.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Round-to-nearest unit roundoff u = 2^{1-p}/2 = 2^{-p}.
# binary16: 11 significant bits -> u = 2^{-11}
# binary32: 24 significant bits -> u = 2^{-24}
UNIT_ROUNDOFF = {
    "fp16": 2.0**-11,
    "fp32": 2.0**-24,
}


def _as_float(value: np.generic | float) -> float:
    return float(np.asarray(value, dtype=np.float64))


def _require_finite(name: str, array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        bad = int(np.size(array) - np.isfinite(array).sum())
        raise AssertionError(f"{name} contains {bad} NaN or infinity values")


def _error_metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Normwise and entrywise errors. Both arrays must be float64 and finite."""

    difference = actual - reference
    abs_difference = np.abs(difference)
    frobenius_error = _as_float(np.linalg.norm(difference, ord="fro"))
    reference_frobenius = _as_float(np.linalg.norm(reference, ord="fro"))
    max_abs_reference = _as_float(np.max(np.abs(reference))) if reference.size else 0.0
    max_abs_error = _as_float(np.max(abs_difference)) if abs_difference.size else 0.0
    tiny = float(np.finfo(np.float64).tiny)
    if abs_difference.size:
        p50, p99, p999 = np.percentile(abs_difference, [50.0, 99.0, 99.9])
        mean_abs = _as_float(np.mean(abs_difference))
        rms_abs = _as_float(np.sqrt(np.mean(difference * difference)))
    else:
        p50 = p99 = p999 = mean_abs = rms_abs = 0.0
    return {
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs,
        "rms_abs_error": rms_abs,
        "abs_error_p50": _as_float(p50),
        "abs_error_p99": _as_float(p99),
        "abs_error_p99_9": _as_float(p999),
        "frobenius_error": frobenius_error,
        "reference_frobenius_norm": reference_frobenius,
        "relative_frobenius_error": frobenius_error / max(reference_frobenius, tiny),
        "max_abs_reference": max_abs_reference,
        "relative_max_error": max_abs_error / max(max_abs_reference, tiny),
    }


def compare_device_gemm(
    a: np.ndarray,
    b: np.ndarray,
    c_device: np.ndarray,
    *,
    precision: str,
) -> dict[str, Any]:
    """Return a JSON-serializable comparison of device C against NumPy.

    Parameters
    ----------
    a, b:
        Host matrices in the same global layout as the untilled device result.
        Usually the arrays packed to the wafer (device dtype). Float64 A and B
        are also accepted: then ``C_ref = A @ B`` uses those true float64
        values and does not round them to the device dtype first.
    c_device:
        Untiled device result. Same dtype as the device tiles, or float64 when
        the host reconstructed a two-word (TwoSum) accumulator.
    precision:
        ``fp16`` or ``fp32``.
    """

    if precision not in UNIT_ROUNDOFF:
        raise ValueError(f"unsupported precision: {precision}")
    expected_dtype = np.float16 if precision == "fp16" else np.float32
    if a.ndim != 2 or b.ndim != 2 or c_device.ndim != 2:
        raise ValueError("A, B, and C must be 2-D")
    n, k = a.shape
    k_b, m = b.shape
    if k != k_b:
        raise ValueError(f"inner dimensions disagree: A is {a.shape}, B is {b.shape}")
    if c_device.shape != (n, m):
        raise ValueError(f"C shape {c_device.shape} does not match A@B = {(n, m)}")
    wide_result = c_device.dtype == np.float64
    source_is_float64 = a.dtype == np.float64 and b.dtype == np.float64
    if source_is_float64:
        if a.dtype != np.float64 or b.dtype != np.float64:
            raise ValueError("float64 A and B must share dtype")
    elif a.dtype != expected_dtype or b.dtype != expected_dtype:
        raise ValueError(
            f"expected dtype {expected_dtype} for {precision}, "
            f"got A={a.dtype}, B={b.dtype}, C={c_device.dtype}"
        )
    if not wide_result and c_device.dtype != expected_dtype:
        raise ValueError(
            f"expected dtype {expected_dtype} for {precision}, "
            f"got A={a.dtype}, B={b.dtype}, C={c_device.dtype}"
        )

    _require_finite("device C", c_device)
    _require_finite("host A", a)
    _require_finite("host B", b)

    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    c_device64 = np.asarray(c_device, dtype=np.float64)
    c_ref = np.matmul(a64, b64)
    _require_finite("float64 NumPy reference C", c_ref)

    against_float64 = _error_metrics(c_device64, c_ref)
    a_fro = _as_float(np.linalg.norm(a64, ord="fro"))
    b_fro = _as_float(np.linalg.norm(b64, ord="fro"))
    tiny = float(np.finfo(np.float64).tiny)
    product_scale = max(a_fro * b_fro, tiny)
    unit_roundoff = UNIT_ROUNDOFF[precision]
    k_u = float(k) * unit_roundoff
    normwise_residual = against_float64["frobenius_error"] / product_scale
    scaled_residual = against_float64["frobenius_error"] / max(
        float(k) * unit_roundoff * product_scale,
        tiny,
    )

    c_rounded = np.asarray(c_ref, dtype=expected_dtype)
    against_rounded = _error_metrics(c_device64, np.asarray(c_rounded, dtype=np.float64))
    if wide_result:
        n_match_rounded = int(
            np.count_nonzero(c_device64 == np.asarray(c_rounded, dtype=np.float64))
        )
    else:
        n_match_rounded = int(np.count_nonzero(c_device == c_rounded))

    a32 = np.asarray(a, dtype=np.float32)
    b32 = np.asarray(b, dtype=np.float32)
    c_host32 = np.matmul(a32, b32)
    _require_finite("float32 NumPy C", c_host32)
    against_host32 = _error_metrics(c_device64, np.asarray(c_host32, dtype=np.float64))

    return {
        "method": "float64_numpy_reference",
        "precision": precision,
        "shape": {"n": int(n), "k": int(k), "m": int(m)},
        "n_elements": int(n * m),
        "device_dtype": "float64" if wide_result else np.dtype(expected_dtype).name,
        "source_dtype": "float64" if source_is_float64 else np.dtype(expected_dtype).name,
        "two_word_result": bool(wide_result),
        "reference": {
            "dtype": "float64",
            "operator": "numpy.matmul",
            "input_promotion": (
                "true float64 A and B; C_ref = numpy.matmul(A, B) without "
                "rounding the source to the device dtype"
                if source_is_float64
                else "exact bit-pattern promotion of A and B to float64"
            ),
        },
        "unit_roundoff": unit_roundoff,
        "k_times_unit_roundoff": k_u,
        "classical_gemm_bound_applies": bool(k_u < 1.0),
        "a_frobenius_norm": a_fro,
        "b_frobenius_norm": b_fro,
        "checksums": {
            "a_sum": _as_float(np.sum(a64)),
            "b_sum": _as_float(np.sum(b64)),
            "c_device_sum": _as_float(np.sum(c_device64)),
            "c_ref_sum": _as_float(np.sum(c_ref)),
        },
        "normwise_residual": normwise_residual,
        "scaled_residual": scaled_residual,
        "against_float64_reference": against_float64,
        "against_correctly_rounded_reference": {
            **against_rounded,
            "n_entries_matching": n_match_rounded,
            "fraction_matching": n_match_rounded / max(int(n * m), 1),
            "note": (
                "Exact float64 product rounded to the device dtype. Isolates "
                "accumulation and reduction-order error from unavoidable output rounding."
            ),
        },
        "against_numpy_float32_gemm": {
            **against_host32,
            "note": (
                "NumPy float32 matmul of the same A and B values. For fp32 this "
                "is a same-precision host GEMM with a different summation order. "
                "For fp16 this is a wider-accumulation host product, not @fmach."
            ),
        },
    }
