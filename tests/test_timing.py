import unittest

import numpy as np

from ozaki_roofline.timing import (
    synchronized_makespan_cycles,
    unpack_synchronized_timestamps,
)


def as_f32_words(words: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(words, dtype=np.uint32).view(np.float32)


def pack_time(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    words = np.empty((*start.shape, 3), dtype=np.uint32)
    words[..., 0] = (start & 0xFFFF) | (((start >> 16) & 0xFFFF) << 16)
    words[..., 1] = ((start >> 32) & 0xFFFF) | ((end & 0xFFFF) << 16)
    words[..., 2] = ((end >> 16) & 0xFFFF) | (((end >> 32) & 0xFFFF) << 16)
    return as_f32_words(words)


def pack_reference(reference: np.ndarray) -> np.ndarray:
    words = np.empty((*reference.shape, 2), dtype=np.uint32)
    words[..., 0] = (reference & 0xFFFF) | (((reference >> 16) & 0xFFFF) << 16)
    words[..., 1] = (reference >> 32) & 0xFFFF
    return as_f32_words(words)


class TimestampTests(unittest.TestCase):
    def test_unpack_and_makespan(self) -> None:
        reference = np.full((2, 3), 1_000_000, dtype=np.int64)
        py, px = np.indices(reference.shape)
        corrected = reference - (px + py)
        desired_start = np.asarray([[10, 11, 12], [13, 14, 15]], dtype=np.int64)
        desired_end = np.asarray([[110, 121, 132], [143, 154, 165]], dtype=np.int64)
        raw_start = corrected + desired_start
        raw_end = corrected + desired_end

        starts, ends = unpack_synchronized_timestamps(
            pack_time(raw_start, raw_end),
            pack_reference(reference),
        )
        np.testing.assert_array_equal(starts, desired_start)
        np.testing.assert_array_equal(ends, desired_end)
        self.assertEqual(
            synchronized_makespan_cycles(
                pack_time(raw_start, raw_end),
                pack_reference(reference),
            ),
            155,
        )

    def test_shape_validation(self) -> None:
        with self.assertRaises(ValueError):
            unpack_synchronized_timestamps(
                np.zeros((2, 2, 2), dtype=np.float32),
                np.zeros((2, 2, 2), dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
