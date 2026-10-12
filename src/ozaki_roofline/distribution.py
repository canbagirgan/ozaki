"""Host-side tensor layouts for the SDK SUMMA implementation."""

from __future__ import annotations

import numpy as np


def tile_matrix(matrix: np.ndarray, p: int, local_rows: int, local_cols: int) -> np.ndarray:
    """Convert global row-major data to PE-row/PE-col/local-column/local-row."""

    expected = (p * local_rows, p * local_cols)
    if matrix.shape != expected:
        raise ValueError(f"matrix shape {matrix.shape} does not match expected {expected}")
    return (
        matrix.reshape(p, local_rows, p, local_cols)
        .transpose(0, 2, 3, 1)
        .reshape(p, p, local_rows * local_cols)
        .copy()
    )


def untile_matrix(values: np.ndarray, p: int, local_rows: int, local_cols: int) -> np.ndarray:
    """Invert :func:`tile_matrix` for a flat or PE-tiled array."""

    expected_size = p * p * local_rows * local_cols
    if values.size != expected_size:
        raise ValueError(f"received {values.size} values, expected {expected_size}")
    return (
        values.reshape(p, p, local_cols, local_rows)
        .transpose(0, 3, 1, 2)
        .reshape(p * local_rows, p * local_cols)
        .copy()
    )


def encode_memcpy_input(tiled: np.ndarray, precision_name: str) -> np.ndarray:
    """Prepare one-u32-container-per-element input expected by SdkRuntime."""

    if precision_name == "fp32":
        return np.ascontiguousarray(tiled, dtype=np.float32).ravel()
    if precision_name != "fp16":
        raise ValueError(f"unsupported precision: {precision_name}")
    fp16 = np.ascontiguousarray(tiled, dtype=np.float16).ravel()
    return np.asarray(fp16.view(np.uint16), dtype=np.uint32)


def decode_memcpy_output(raw: np.ndarray, precision_name: str) -> np.ndarray:
    """Decode SdkRuntime output containers to their logical element type."""

    words = np.ascontiguousarray(raw, dtype=np.uint32)
    if precision_name == "fp32":
        return words.view(np.float32)
    if precision_name != "fp16":
        raise ValueError(f"unsupported precision: {precision_name}")
    return (words & np.uint32(0xFFFF)).astype(np.uint16).view(np.float16)
