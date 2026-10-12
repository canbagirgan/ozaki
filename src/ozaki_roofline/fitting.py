"""Small dependency-light fitting helpers for benchmark results."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import median

import numpy as np

from .model import SummaCoefficients, TransferFit


@dataclass(frozen=True)
class DistributionSummary:
    count: int
    minimum: float
    p10: float
    median: float
    p90: float
    maximum: float


def percentile(values: Sequence[float], probability: float) -> float:
    """Linear-interpolated percentile with no external statistics package."""

    if not values:
        raise ValueError("at least one value is required")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize(values: Iterable[float]) -> DistributionSummary:
    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("at least one value is required")
    return DistributionSummary(
        count=len(materialized),
        minimum=min(materialized),
        p10=percentile(materialized, 0.10),
        median=float(median(materialized)),
        p90=percentile(materialized, 0.90),
        maximum=max(materialized),
    )


def fit_transfer(byte_counts: Sequence[int], elapsed_seconds: Sequence[float]) -> TransferFit:
    """Fit t = alpha + bytes / beta by ordinary least squares.

    A negative fitted intercept is clamped to zero and the slope is then
    re-fitted through the origin. This keeps the physical model valid while
    making the adjustment explicit and deterministic.
    """

    if len(byte_counts) != len(elapsed_seconds) or len(byte_counts) < 2:
        raise ValueError("matching byte/time sequences with at least two points are required")
    x = np.asarray(byte_counts, dtype=np.float64)
    y = np.asarray(elapsed_seconds, dtype=np.float64)
    if np.any(x < 0) or np.any(y <= 0):
        raise ValueError("byte counts must be nonnegative and times positive")

    design = np.column_stack([np.ones_like(x), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    if intercept < 0:
        intercept = 0.0
        slope = float(np.dot(x, y) / np.dot(x, x))
    if slope <= 0:
        raise ValueError("measurements do not produce a positive transfer slope")
    return TransferFit(float(intercept), float(1.0 / slope))


def fit_summa_coefficients(
    features: Sequence[Sequence[float]], cycles: Sequence[float]
) -> SummaCoefficients:
    """Fit a nonnegative four-term per-step SUMMA cycle model.

    Each feature row must already contain:
    [P*Mt*Kt*Nt, P*Kt*Nt, P*(A_words+B_words), P].

    With only four coefficients, enumerating all nonempty active sets provides
    a small dependency-free nonnegative least-squares solution.
    """

    if len(features) != len(cycles) or len(features) < 4:
        raise ValueError("at least four matching feature/cycle rows are required")
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(cycles, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 4:
        raise ValueError("SUMMA feature rows must contain exactly four terms")
    if np.any(x < 0) or np.any(y <= 0):
        raise ValueError("SUMMA features must be nonnegative and cycles positive")

    best_coefficients = None
    best_residual = float("inf")
    for mask in range(1, 1 << x.shape[1]):
        active = [column for column in range(x.shape[1]) if mask & (1 << column)]
        active_x = x[:, active]
        scales = np.linalg.norm(active_x, axis=0)
        if np.any(scales == 0):
            continue
        scaled_coefficients = np.linalg.lstsq(active_x / scales, y, rcond=None)[0]
        active_coefficients = scaled_coefficients / scales
        if np.any(active_coefficients < -1e-10):
            continue
        coefficients = np.zeros(x.shape[1], dtype=np.float64)
        coefficients[active] = np.maximum(active_coefficients, 0)
        residual = float(np.dot(x @ coefficients - y, x @ coefficients - y))
        if residual < best_residual:
            best_coefficients = coefficients
            best_residual = residual

    if best_coefficients is None:
        raise ValueError("could not fit nonnegative SUMMA coefficients")
    return SummaCoefficients(*(float(value) for value in best_coefficients))
