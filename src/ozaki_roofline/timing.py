"""Decoders for WSE timestamp buffers used by the benchmark kernels."""

from __future__ import annotations

import numpy as np


def unpack_synchronized_timestamps(
    time_values: np.ndarray,
    reference_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode packed 48-bit start/end clocks from the SDK bandwidth test.

    The returned arrays have the synchronized reference and deterministic
    Manhattan propagation correction removed.
    """

    if time_values.shape[-1] != 3 or reference_values.shape[-1] != 2:
        raise ValueError("time and reference buffers must end in dimensions 3 and 2")
    if time_values.shape[:-1] != reference_values.shape[:-1]:
        raise ValueError("time and reference PE grids must match")

    time_words = np.ascontiguousarray(time_values).view(np.uint32)
    ref_words = np.ascontiguousarray(reference_values).view(np.uint32)

    start = (
        (time_words[..., 0] & 0xFFFF).astype(np.int64)
        | ((time_words[..., 0] >> 16).astype(np.int64) << 16)
        | ((time_words[..., 1] & 0xFFFF).astype(np.int64) << 32)
    )
    end = (
        (time_words[..., 1] >> 16).astype(np.int64)
        | ((time_words[..., 2] & 0xFFFF).astype(np.int64) << 16)
        | ((time_words[..., 2] >> 16).astype(np.int64) << 32)
    )
    reference = (
        (ref_words[..., 0] & 0xFFFF).astype(np.int64)
        | ((ref_words[..., 0] >> 16).astype(np.int64) << 16)
        | ((ref_words[..., 1] & 0xFFFF).astype(np.int64) << 32)
    )

    py, px = np.indices(reference.shape)
    corrected_reference = reference - (px + py)
    return start - corrected_reference, end - corrected_reference


def synchronized_makespan_cycles(
    time_values: np.ndarray,
    reference_values: np.ndarray,
) -> int:
    starts, ends = unpack_synchronized_timestamps(time_values, reference_values)
    cycles = int(ends.max() - starts.min())
    if cycles <= 0:
        raise ValueError(f"invalid synchronized makespan: {cycles} cycles")
    return cycles
