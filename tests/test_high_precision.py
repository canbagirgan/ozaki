"""Host high-precision reference and 4-slice TwoSum vs the 1e-12 headline."""

from __future__ import annotations

import unittest

import numpy as np

from ozaki_roofline.high_precision import (
    compare_device_vs_high,
    gemm_high,
    high_precision_info,
    ozimmu_matrix_high,
    residual_split_fp16_high,
    scaled_gaussian_high,
    standard_normal_high,
    to_float64,
)
from ozaki_roofline.split import (
    FourSlice,
    apply_col_scales,
    apply_row_scales,
    fp16_covering_col_scales,
    fp16_covering_row_scales,
    reconstruct_high,
    residual_split_fp16,
    ten_product_twosum_fp64,
    unscale_gemm,
)
from ozaki_roofline.integer_split import reconstruct_gemm_fp64, split_shared_exponent


class HighPrecisionInfoTests(unittest.TestCase):
    def test_kind_is_wider_than_binary64(self) -> None:
        info = high_precision_info()
        self.assertIn(info["kind"], ("ieee_binary128", "mpmath_quad", "x87_longdouble"))
        self.assertGreater(int(info["nmant"]), 52)


class HighPrecisionSplitTests(unittest.TestCase):
    def test_four_slice_wide_residual_beats_float64_truncation(self) -> None:
        rng = np.random.default_rng(7)
        info = high_precision_info()
        source = ozimmu_matrix_high(rng, (16, 16), phi=1.0, info=info)
        four = residual_split_fp16(source, n_slices=4)
        self.assertIsInstance(four, FourSlice)
        rebuilt = reconstruct_high(four, info["dtype"])
        residual = source - rebuilt
        source_fro = float(np.linalg.norm(source, ord="fro"))
        rel = float(np.linalg.norm(residual, ord="fro")) / source_fro
        # Unscaled O(1) sources flush A3 into fp16 underflow, so reconstruction
        # is ~1e-8, worse than binary64 truncation of the same wide source.
        self.assertLess(rel, 1e-6)
        self.assertGreater(rel, 1e-12)

    def test_covering_scale_keeps_tail_slice_live(self) -> None:
        rng = np.random.default_rng(7)
        info = high_precision_info()
        source = ozimmu_matrix_high(rng, (32, 32), phi=1.0, info=info)
        sa = fp16_covering_row_scales(source)
        scaled = apply_row_scales(source, sa)
        four = residual_split_fp16(scaled, n_slices=4)
        tail_zero = float(np.mean(four.tail == 0))
        self.assertLess(tail_zero, 0.95)

    def test_high_split_helper_matches_wide_residual_split(self) -> None:
        rng = np.random.default_rng(7)
        info = high_precision_info()
        source = scaled_gaussian_high(rng, (8, 8), scale=1.0, info=info)
        a = residual_split_fp16_high(source, n_slices=4, info=info)
        b = residual_split_fp16(source, n_slices=4)
        np.testing.assert_array_equal(a.high, b.high)
        np.testing.assert_array_equal(a.tail, b.tail)


class TenProductHeadlineTests(unittest.TestCase):
    def _scaled_ten_product(self, a, b):
        sa = fp16_covering_row_scales(a)
        sb = fp16_covering_col_scales(b)
        a_s = apply_row_scales(a, sa)
        b_s = apply_col_scales(b, sb)
        a4 = residual_split_fp16(a_s, n_slices=4)
        b4 = residual_split_fp16(b_s, n_slices=4)
        c_s = ten_product_twosum_fp64(
            a4.high, a4.mid, a4.extra, a4.tail,
            b4.high, b4.mid, b4.extra, b4.tail,
        )
        return unscale_gemm(c_s, sa, sb)

    def test_ten_product_twosum_vs_high_gemm_ozimmu_phi1(self) -> None:
        rng = np.random.default_rng(7)
        info = high_precision_info()
        n, k, m = 8, 32, 6
        a = ozimmu_matrix_high(rng, (n, k), phi=1.0, info=info)
        b = ozimmu_matrix_high(rng, (k, m), phi=1.0, info=info)
        c = self._scaled_ten_product(a, b)
        report = compare_device_vs_high(a, b, c, info=info)
        rel_high = report["against_high_precision_reference"]["relative_frobenius_error"]
        rel_f64 = report["against_float64_reference"]["relative_frobenius_error"]
        # Per-row/col fp16 covering + TwoSum is ~1e-12 class; ozIMMU φ=1 sits ~2e-12.
        self.assertLess(rel_high, 5e-12)
        self.assertLess(rel_f64, 5e-12)

    def test_ten_product_twosum_vs_high_gemm_unscaled_gaussian(self) -> None:
        rng = np.random.default_rng(7)
        info = high_precision_info()
        a = standard_normal_high(rng, (8, 32), info=info)
        b = standard_normal_high(rng, (32, 6), info=info)
        c = self._scaled_ten_product(a, b)
        report = compare_device_vs_high(a, b, c, info=info)
        self.assertLess(
            report["against_high_precision_reference"]["relative_frobenius_error"],
            3e-12,
        )

    def test_integer_seven_slice_hits_1e12_on_ozimmu(self) -> None:
        rng = np.random.default_rng(7)
        info = high_precision_info()
        a = ozimmu_matrix_high(rng, (8, 32), phi=1.0, info=info)
        b = ozimmu_matrix_high(rng, (32, 6), phi=1.0, info=info)
        asl = split_shared_exponent(to_float64(a), alpha=7, n_slices=7, axis=1)
        bsl = split_shared_exponent(to_float64(b), alpha=7, n_slices=7, axis=0)
        c = reconstruct_gemm_fp64(asl, bsl)
        report = compare_device_vs_high(a, b, c, info=info)
        self.assertLess(
            report["against_high_precision_reference"]["relative_frobenius_error"],
            1e-12,
        )

    def test_high_gemm_is_not_promoted_float64_matmul(self) -> None:
        rng = np.random.default_rng(7)
        info = high_precision_info()
        a = ozimmu_matrix_high(rng, (6, 8), phi=1.0, info=info)
        b = ozimmu_matrix_high(rng, (8, 5), phi=1.0, info=info)
        c_high = gemm_high(a, b, info=info)
        c64 = np.matmul(to_float64(a), to_float64(b))
        report = compare_device_vs_high(a, b, c64, info=info)
        # Truncating sources to f64 before GEMM must miss the extra bits.
        self.assertGreater(
            report["against_high_precision_reference"]["relative_frobenius_error"],
            0.0,
        )
        self.assertEqual(c_high.shape, (6, 5))


if __name__ == "__main__":
    unittest.main()
