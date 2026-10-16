"""Scaled low slices preserve residuals that would underflow unscaled FP16."""
import unittest
import numpy as np
from ozaki_roofline.split import scaled_residual_split_fp16


class ScaledResidualTest(unittest.TestCase):
    def test_scale_before_cast_preserves_tiny_residual(self):
        a = np.array([[np.nextafter(np.float32(1), np.float32(2)) * 2**-10]], dtype=np.float32)
        high, low = scaled_residual_split_fp16(a)
        residual = a - high.astype(np.float32)
        self.assertEqual(residual.astype(np.float16).item(), 0)
        self.assertNotEqual(low.item(), 0)
        np.testing.assert_array_equal(high.astype(np.float32) + np.ldexp(low.astype(np.float32), -16), a)

    def test_cross_merge_restores_units(self):
        a = np.array([[1.0001, -0.5001], [0.12501, -2.0001]], dtype=np.float32)
        b = a.T.copy()
        a0, a1 = (x.astype(np.float32) for x in scaled_residual_split_fp16(a))
        b0, b1 = (x.astype(np.float32) for x in scaled_residual_split_fp16(b))
        c = a0 @ b0 + np.ldexp(a0 @ b1 + a1 @ b0, -16)
        np.testing.assert_allclose(c, a.astype(np.float64) @ b.astype(np.float64), rtol=2e-7, atol=1e-8)

    def test_scaled_residual_overflow_is_rejected(self):
        with self.assertRaises(ValueError):
            scaled_residual_split_fp16(np.array([[32769]], dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
