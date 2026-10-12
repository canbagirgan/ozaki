import math
import unittest

import numpy as np

from ozaki_roofline.integer_split import (
    IntegerSlices,
    integer_product_int64,
    k_reduction_exact,
    overflow_report,
    product_pairs,
    reconstruct_float64,
    reconstruct_gemm_fp64,
    slice_diagnostics_integer,
    split_shared_exponent,
)


class ProductPairTests(unittest.TestCase):
    def test_triangle_count(self) -> None:
        self.assertEqual(len(product_pairs(8, triangle=True)), 36)
        self.assertEqual(len(product_pairs(8, triangle=False)), 64)
        self.assertEqual(product_pairs(2, triangle=True), [(0, 0), (0, 1), (1, 0)])


class ExactnessTests(unittest.TestCase):
    def test_alpha7_k3584_fits_i32(self) -> None:
        self.assertTrue(k_reduction_exact(7, 3584))
        self.assertTrue(k_reduction_exact(7, 1024))

    def test_too_wide_alpha_overflows(self) -> None:
        self.assertFalse(k_reduction_exact(15, 4096))


class IntegerSplitTests(unittest.TestCase):
    def test_rejects_non_2d(self) -> None:
        with self.assertRaises(ValueError):
            split_shared_exponent(np.ones(4, dtype=np.float64))

    def test_rejects_bad_alpha(self) -> None:
        source = np.ones((2, 2), dtype=np.float64)
        with self.assertRaises(ValueError):
            split_shared_exponent(source, alpha=8)

    def test_power_of_two_is_exact(self) -> None:
        source = np.asarray([[1.0, -2.0], [0.5, 4.0]], dtype=np.float64)
        slices = split_shared_exponent(source, alpha=7, n_slices=8, axis=1)
        rebuilt = reconstruct_float64(slices)
        np.testing.assert_allclose(rebuilt, source, rtol=0, atol=0)

    def test_row_split_reconstructs_gaussians(self) -> None:
        rng = np.random.default_rng(7)
        k = 32
        source = rng.normal(0.0, 1.0 / math.sqrt(k), size=(8, k)).astype(np.float64)
        slices = split_shared_exponent(source, alpha=7, n_slices=8, axis=1)
        diag = slice_diagnostics_integer(source, slices)
        self.assertLess(diag["relative_reconstruction_error"], 1e-15)
        self.assertLessEqual(diag["max_abs_slice"], 2**8)

    def test_column_split_matches_row_split_of_transpose(self) -> None:
        rng = np.random.default_rng(7)
        source = rng.normal(0.0, 0.1, size=(5, 7)).astype(np.float64)
        by_col = split_shared_exponent(source, alpha=7, n_slices=8, axis=0)
        by_row = split_shared_exponent(source.T, alpha=7, n_slices=8, axis=1)
        np.testing.assert_array_equal(by_col.exponents.ravel(), by_row.exponents.ravel())
        np.testing.assert_allclose(
            reconstruct_float64(by_col), source, rtol=1e-13, atol=1e-16
        )


class IntegerGemmOracleTests(unittest.TestCase):
    def test_oracle_near_float64_matmul(self) -> None:
        rng = np.random.default_rng(7)
        n, k, m = 12, 24, 10
        a = rng.normal(0.0, 1.0 / math.sqrt(k), size=(n, k)).astype(np.float64)
        b = rng.normal(0.0, 1.0 / math.sqrt(k), size=(k, m)).astype(np.float64)
        a_slices = split_shared_exponent(a, alpha=7, n_slices=8, axis=1)
        b_slices = split_shared_exponent(b, alpha=7, n_slices=8, axis=0)
        rebuilt = reconstruct_gemm_fp64(a_slices, b_slices, triangle=True)
        reference = np.matmul(a, b)
        rel = np.linalg.norm(rebuilt - reference, ord="fro") / np.linalg.norm(
            reference, ord="fro"
        )
        self.assertLess(float(rel), 1e-13)

    def test_triangle_beats_few_slices(self) -> None:
        rng = np.random.default_rng(7)
        n, k, m = 8, 16, 8
        a = rng.normal(0.0, 1.0 / math.sqrt(k), size=(n, k)).astype(np.float64)
        b = rng.normal(0.0, 1.0 / math.sqrt(k), size=(k, m)).astype(np.float64)
        reference = np.matmul(a, b)

        def rel_err(n_slices: int) -> float:
            a_s = split_shared_exponent(a, alpha=7, n_slices=n_slices, axis=1)
            b_s = split_shared_exponent(b, alpha=7, n_slices=n_slices, axis=0)
            rebuilt = reconstruct_gemm_fp64(a_s, b_s, triangle=True)
            return float(
                np.linalg.norm(rebuilt - reference, ord="fro")
                / np.linalg.norm(reference, ord="fro")
            )

        self.assertLess(rel_err(8), rel_err(2))

    def test_int32_overflow_report_clean_for_small_k(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.normal(0.0, 0.2, size=(4, 8)).astype(np.float64)
        b = rng.normal(0.0, 0.2, size=(8, 4)).astype(np.float64)
        a_slices = split_shared_exponent(a, alpha=7, n_slices=8, axis=1)
        b_slices = split_shared_exponent(b, alpha=7, n_slices=8, axis=0)
        products = {
            (i, j): integer_product_int64(a_slices.slices[i], b_slices.slices[j])
            for i, j in product_pairs(8, triangle=True)
        }
        report = overflow_report(products)
        self.assertEqual(report["n_overflow_i32"], 0.0)
        self.assertTrue(report["fits_i32"])
        self.assertTrue(k_reduction_exact(7, 8))

    def test_host_int32_matches_int64_when_exact(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.integers(-20, 21, size=(5, 6), dtype=np.int16)
        b = rng.integers(-20, 21, size=(6, 4), dtype=np.int16)
        c64 = integer_product_int64(a, b)
        c32 = np.matmul(a.astype(np.int32), b.astype(np.int32))
        np.testing.assert_array_equal(c32.astype(np.int64), c64)


if __name__ == "__main__":
    unittest.main()
