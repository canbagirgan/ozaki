import unittest

import numpy as np

from ozaki_roofline.distribution import (
    decode_memcpy_output,
    encode_memcpy_input,
    tile_matrix,
    untile_matrix,
)


class DistributionTests(unittest.TestCase):
    def test_tile_round_trip_rectangular(self) -> None:
        matrix = np.arange(6 * 8, dtype=np.float32).reshape(6, 8)
        tiled = tile_matrix(matrix, p=2, local_rows=3, local_cols=4)
        self.assertEqual(tiled.shape, (2, 2, 12))
        np.testing.assert_array_equal(untile_matrix(tiled, 2, 3, 4), matrix)

    def test_fp32_memcpy_round_trip(self) -> None:
        values = np.asarray([1.25, -2.5, 3.75], dtype=np.float32)
        encoded = encode_memcpy_input(values, "fp32")
        decoded = decode_memcpy_output(encoded.view(np.uint32), "fp32")
        np.testing.assert_array_equal(decoded, values)

    def test_fp16_memcpy_round_trip(self) -> None:
        values = np.asarray([1.25, -2.5, 3.75], dtype=np.float16)
        encoded = encode_memcpy_input(values, "fp16")
        self.assertEqual(encoded.dtype, np.uint32)
        decoded = decode_memcpy_output(encoded, "fp16")
        np.testing.assert_array_equal(decoded, values)

    def test_shape_validation(self) -> None:
        with self.assertRaises(ValueError):
            tile_matrix(np.zeros((4, 4)), p=2, local_rows=1, local_cols=2)
        with self.assertRaises(ValueError):
            untile_matrix(np.zeros(7), p=2, local_rows=2, local_cols=2)


if __name__ == "__main__":
    unittest.main()
