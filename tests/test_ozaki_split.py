import math
import unittest

import numpy as np

from ozaki_roofline.split import (
    FiveSlice,
    FourSlice,
    ThreeSlice,
    fifteen_product_dual_fp64,
    four_product_fp32,
    three_product_fp32,
    three_product_cross_f16,
    four_product_twosum_fp64,
    fifteen_product_twosum_fp64,
    full_mantissa_fp32,
    ozimmu_matrix,
    knuth_twosum,
    nine_product_fp32,
    nine_product_twosum_fp64,
    six_product_fp32,
    ten_product_dual_fp64,
    ten_product_twosum_fp64,
    reconstruct_fp32,
    reconstruct_fp64,
    reconstruct_twosum_fp64,
    residual_split_fp16,
    residual_split_fp32,
    sixteen_product_fp32,
    slice_diagnostics,
)
from ozaki_roofline.validation import compare_device_gemm


class ResidualSplitTests(unittest.TestCase):
    def test_already_float16_has_zero_low_slice(self) -> None:
        source = np.asarray([[0.5, 1.0], [-2.0, 0.25]], dtype=np.float16).astype(
            np.float32
        )
        slices = residual_split_fp16(source)
        self.assertTrue(np.array_equal(slices.high, source.astype(np.float16)))
        self.assertTrue(np.all(slices.low == 0))
        np.testing.assert_array_equal(reconstruct_fp32(slices), source)

    def test_exact_power_of_two_residual_is_kept(self) -> None:
        # 1 + 2^{-12}: high word is 1.0, residual is exactly representable in fp16.
        source = np.asarray([[1.0 + 2.0**-12]], dtype=np.float32)
        slices = residual_split_fp16(source)
        self.assertEqual(float(slices.high[0, 0]), 1.0)
        self.assertEqual(float(slices.low[0, 0]), 2.0**-12)
        np.testing.assert_array_equal(reconstruct_fp32(slices), source)

    def test_reconstruction_beats_high_word_alone(self) -> None:
        rng = np.random.default_rng(7)
        source = rng.normal(0.0, 1.0, size=(8, 8)).astype(np.float32)
        slices = residual_split_fp16(source)
        high_only = slices.high.astype(np.float32)
        rebuilt = reconstruct_fp32(slices)
        high_err = np.linalg.norm((source - high_only).astype(np.float64), ord="fro")
        two_err = np.linalg.norm((source - rebuilt).astype(np.float64), ord="fro")
        self.assertLess(two_err, high_err)
        diag = slice_diagnostics(source, slices)
        self.assertLess(diag["relative_reconstruction_error"], 1e-4)

    def test_rejects_non_2d(self) -> None:
        with self.assertRaises(ValueError):
            residual_split_fp16(np.ones(4, dtype=np.float32))

    def test_rejects_bad_n_slices(self) -> None:
        source = np.ones((2, 2), dtype=np.float32)
        with self.assertRaises(ValueError):
            residual_split_fp16(source, n_slices=1)
        with self.assertRaises(ValueError):
            residual_split_fp16(source, n_slices=6)

    def test_three_slice_reconstruction_beats_two_slice(self) -> None:
        rng = np.random.default_rng(7)
        source = rng.normal(0.0, 1.0, size=(16, 16)).astype(np.float32)
        two = residual_split_fp16(source, n_slices=2)
        three = residual_split_fp16(source, n_slices=3)
        self.assertIsInstance(three, ThreeSlice)
        two_err = np.linalg.norm(
            (source - reconstruct_fp32(two)).astype(np.float64), ord="fro"
        )
        three_err = np.linalg.norm(
            (source - reconstruct_fp32(three)).astype(np.float64), ord="fro"
        )
        self.assertLess(three_err, two_err)
        two_diag = slice_diagnostics(source, two)
        three_diag = slice_diagnostics(source, three)
        self.assertLess(
            three_diag["relative_reconstruction_error"],
            two_diag["relative_reconstruction_error"],
        )

    def test_third_slice_not_identically_zero_on_gaussians(self) -> None:
        rng = np.random.default_rng(7)
        source = rng.normal(0.0, 1.0, size=(32, 32)).astype(np.float32)
        three = residual_split_fp16(source, n_slices=3)
        self.assertFalse(np.all(three.extra == 0))
        self.assertGreater(float(np.mean(three.extra != 0)), 0.0)

    def test_four_slice_reconstruction_matches_float32_source(self) -> None:
        rng = np.random.default_rng(7)
        source = rng.normal(0.0, 1.0, size=(16, 16)).astype(np.float32)
        four = residual_split_fp16(source, n_slices=4)
        self.assertIsInstance(four, FourSlice)
        rebuilt = reconstruct_fp32(four)
        diag = slice_diagnostics(source, four)
        self.assertEqual(diag["n_slices"], 4.0)
        self.assertLess(diag["relative_reconstruction_error"], 1e-6)
        np.testing.assert_allclose(
            rebuilt.astype(np.float64), source.astype(np.float64), rtol=0.0, atol=1e-6
        )

    def test_scaled_gaussians_flush_a2_and_a3(self) -> None:
        rng = np.random.default_rng(7)
        k = 2048
        scale = 1.0 / math.sqrt(k)
        source = rng.normal(0.0, scale, size=(64, 64)).astype(np.float32)
        four = residual_split_fp16(source, n_slices=4)
        self.assertGreater(float(np.mean(four.extra == 0)), 0.99)
        self.assertGreater(float(np.mean(four.tail == 0)), 0.99)

    def test_ozimmu_phi1_keeps_a1_and_does_not_flush_a2(self) -> None:
        rng = np.random.default_rng(7)
        source = ozimmu_matrix(rng, (64, 64), phi=1.0)
        four = residual_split_fp16(source, n_slices=4)
        self.assertTrue(np.isfinite(source).all())
        self.assertLess(float(np.mean(four.mid == 0)), 0.05)
        self.assertLess(float(np.mean(four.extra == 0)), 0.99)

    def test_full_mantissa_occupies_the_fp32_covering(self) -> None:
        rng = np.random.default_rng(7)
        source = full_mantissa_fp32(rng, (64, 64))
        two = residual_split_fp16(source, n_slices=2)
        four = residual_split_fp16(source, n_slices=4)
        rebuilt = reconstruct_fp32(four)
        two_err = slice_diagnostics(source, two)["relative_reconstruction_error"]
        four_err = slice_diagnostics(source, four)["relative_reconstruction_error"]
        self.assertLess(four_err, 1e-7)
        self.assertLess(four_err, two_err)
        self.assertLess(float(np.mean(four.high == 0)), 0.01)
        self.assertLess(float(np.mean(four.mid == 0)), 0.01)
        extra_zero = float(np.mean(four.extra == 0))
        self.assertGreater(extra_zero, 0.5)
        self.assertLess(extra_zero, 0.9)
        self.assertGreater(float(np.mean(four.tail == 0)), 0.99)
        np.testing.assert_array_equal(rebuilt, source)
        self.assertTrue(np.isfinite(source).all())
        self.assertLess(float(np.max(np.abs(source))), 256.0)
        self.assertGreater(float(np.min(np.abs(source))), 0.5)


class FiveSliceFp64Tests(unittest.TestCase):
    def test_scaled_one_over_sqrt_k_flushes_a2_through_a4(self) -> None:
        # fp16 min subnormal is 2^{-24}. After A0/A1 the residual of
        # N(0, 1/sqrt(K)) sits below that, so A2..A4 are identically 0
        # even when the source is true float64.
        rng = np.random.default_rng(7)
        k = 1792
        scale = 1.0 / math.sqrt(k)
        source = rng.normal(0.0, scale, size=(64, 64)).astype(np.float64)
        five = residual_split_fp16(source, n_slices=5)
        self.assertIsInstance(five, FiveSlice)
        diag = slice_diagnostics(source, five)
        self.assertEqual(diag["n_slices"], 5.0)
        self.assertGreater(diag["w1_zero_fraction"], 0.0)
        self.assertLess(diag["w1_zero_fraction"], 0.05)
        self.assertEqual(diag["w2_zero_fraction"], 1.0)
        self.assertEqual(diag["w3_zero_fraction"], 1.0)
        self.assertEqual(diag["w4_zero_fraction"], 1.0)

    def test_unscaled_gaussians_keep_a2_live_a3_a4_flush(self) -> None:
        rng = np.random.default_rng(7)
        source = rng.normal(0.0, 1.0, size=(64, 64)).astype(np.float64)
        five = residual_split_fp16(source, n_slices=5)
        diag = slice_diagnostics(source, five)
        self.assertLess(diag["w2_zero_fraction"], 1.0)
        self.assertEqual(diag["w3_zero_fraction"], 1.0)
        self.assertEqual(diag["w4_zero_fraction"], 1.0)
        self.assertLess(diag["relative_reconstruction_error"], 1e-7)

    def test_five_slice_beats_float32_truncation_of_the_matrix(self) -> None:
        rng = np.random.default_rng(7)
        source = rng.normal(0.0, 1.0, size=(16, 16)).astype(np.float64)
        five = residual_split_fp16(source, n_slices=5)
        five_err = np.linalg.norm(source - reconstruct_fp64(five), ord="fro")
        trunc_err = np.linalg.norm(source - source.astype(np.float32).astype(np.float64), ord="fro")
        self.assertLess(five_err, trunc_err)

    def test_f32_source_promoted_to_f64_cannot_invent_a2(self) -> None:
        rng = np.random.default_rng(7)
        k = 2048
        scale = 1.0 / math.sqrt(k)
        source32 = rng.normal(0.0, scale, size=(32, 32)).astype(np.float32)
        five_from_f32 = residual_split_fp16(
            source32.astype(np.float64), n_slices=5
        )
        self.assertGreater(float(np.mean(five_from_f32.w2 == 0)), 0.99)


class FifteenProductTests(unittest.TestCase):
    def test_triangle_dual_is_f64_result_near_truncated_sgemm(self) -> None:
        # At 1/sqrt(K), A2..A4 flush, so 15 products collapse to the A0/A1
        # quadrilateral. Dual C cannot beat native SGEMM; it should still
        # be the same order and better than fp16-only GEMM.
        rng = np.random.default_rng(7)
        k = 64
        scale = 1.0 / math.sqrt(k)
        a = rng.normal(0.0, scale, size=(16, k)).astype(np.float64)
        b = rng.normal(0.0, scale, size=(k, 12)).astype(np.float64)
        a5 = residual_split_fp16(a, n_slices=5)
        b5 = residual_split_fp16(b, n_slices=5)
        self.assertGreater(float(np.mean(a5.w2 == 0)), 0.99)
        c_dual, c_hi, _c_lo = fifteen_product_dual_fp64(
            a5.w0, a5.w1, a5.w2, a5.w3, a5.w4,
            b5.w0, b5.w1, b5.w2, b5.w3, b5.w4,
        )
        c_trunc = np.matmul(a.astype(np.float32), b.astype(np.float32)).astype(
            np.float32
        )
        c_fp16 = np.matmul(
            a.astype(np.float16).astype(np.float64),
            b.astype(np.float16).astype(np.float64),
        )
        dual = compare_device_gemm(a, b, c_dual, precision="fp32")
        trunc = compare_device_gemm(a, b, c_trunc, precision="fp32")
        fp16 = compare_device_gemm(a, b, c_fp16, precision="fp32")
        dual_rel = dual["against_float64_reference"]["relative_frobenius_error"]
        trunc_rel = trunc["against_float64_reference"]["relative_frobenius_error"]
        fp16_rel = fp16["against_float64_reference"]["relative_frobenius_error"]
        self.assertTrue(dual["two_word_result"])
        self.assertLess(dual_rel, 50.0 * trunc_rel)
        self.assertLess(dual_rel, fp16_rel)
        hi_only = compare_device_gemm(a, b, c_hi, precision="fp32")
        self.assertLessEqual(
            dual_rel,
            hi_only["against_float64_reference"]["relative_frobenius_error"] * 1.05,
        )

    def test_fifteen_products_is_s_times_s_plus_one_over_two(self) -> None:
        self.assertEqual(5 * 6 // 2, 15)

    def test_sram_five_slice_dual_c_fits_t28_not_t32(self) -> None:
        def bytes_(t: int) -> int:
            return 20 * t * t + 20 * t * t + 8 * t * t

        self.assertEqual(bytes_(28), 37632)
        self.assertLess(bytes_(28), 40 * 1024)
        self.assertEqual(bytes_(32), 49152)
        self.assertGreater(bytes_(32), 40 * 1024)
        self.assertGreater(bytes_(40), 40 * 1024)


class FourProductTests(unittest.TestCase):
    def test_four_products_beat_high_word_gemm_vs_float64(self) -> None:
        rng = np.random.default_rng(7)
        k = 64
        scale = 1.0 / math.sqrt(k)
        a = rng.normal(0.0, scale, size=(32, k)).astype(np.float32)
        b = rng.normal(0.0, scale, size=(k, 24)).astype(np.float32)
        a_slices = residual_split_fp16(a)
        b_slices = residual_split_fp16(b)
        c_ozaki = four_product_fp32(
            a_slices.high, a_slices.low, b_slices.high, b_slices.low
        )
        c_high = np.matmul(
            a_slices.high.astype(np.float32), b_slices.high.astype(np.float32)
        ).astype(np.float32)
        ozaki = compare_device_gemm(a, b, c_ozaki, precision="fp32")
        high = compare_device_gemm(a, b, c_high, precision="fp32")
        self.assertLess(
            ozaki["against_float64_reference"]["relative_frobenius_error"],
            high["against_float64_reference"]["relative_frobenius_error"],
        )
        self.assertLess(
            ozaki["against_float64_reference"]["relative_frobenius_error"],
            5e-4,
        )

    def test_k2560_host_oracle_is_near_native_float32_not_float16(self) -> None:
        rng = np.random.default_rng(7)
        n = k = m = 64
        scale = 1.0 / math.sqrt(k)
        a = rng.normal(0.0, scale, size=(n, k)).astype(np.float32)
        b = rng.normal(0.0, scale, size=(k, m)).astype(np.float32)
        a_slices = residual_split_fp16(a)
        b_slices = residual_split_fp16(b)
        c_ozaki = four_product_fp32(
            a_slices.high, a_slices.low, b_slices.high, b_slices.low
        )
        c_native32 = np.matmul(a, b).astype(np.float32)
        c_fp16_acc = np.matmul(
            a.astype(np.float16).astype(np.float64),
            b.astype(np.float16).astype(np.float64),
        ).astype(np.float32)
        ozaki = compare_device_gemm(a, b, c_ozaki, precision="fp32")
        native = compare_device_gemm(a, b, c_native32, precision="fp32")
        truncated = compare_device_gemm(a, b, c_fp16_acc, precision="fp32")
        ozaki_rel = ozaki["against_float64_reference"]["relative_frobenius_error"]
        native_rel = native["against_float64_reference"]["relative_frobenius_error"]
        trunc_rel = truncated["against_float64_reference"]["relative_frobenius_error"]
        self.assertLess(ozaki_rel, trunc_rel)
        self.assertLess(ozaki_rel / native_rel, 50.0)
        self.assertGreaterEqual(ozaki_rel, native_rel * 0.5)

    def test_skip_a1b1_is_close_to_four_products_on_scaled(self) -> None:
        rng = np.random.default_rng(7)
        k = 2560
        scale = 1.0 / math.sqrt(k)
        a = rng.normal(0.0, scale, size=(64, k)).astype(np.float32)
        b = rng.normal(0.0, scale, size=(k, 64)).astype(np.float32)
        a_slices = residual_split_fp16(a)
        b_slices = residual_split_fp16(b)
        c4 = four_product_fp32(
            a_slices.high, a_slices.low, b_slices.high, b_slices.low
        )
        c3 = three_product_fp32(
            a_slices.high, a_slices.low, b_slices.high, b_slices.low
        )
        four = compare_device_gemm(a, b, c4, precision="fp32")
        three = compare_device_gemm(a, b, c3, precision="fp32")
        rel4 = four["against_float64_reference"]["relative_frobenius_error"]
        rel3 = three["against_float64_reference"]["relative_frobenius_error"]
        self.assertLess(abs(rel3 - rel4) / max(rel4, 1e-30), 0.05)

    def test_cross_f16_is_near_three_products_on_scaled(self) -> None:
        rng = np.random.default_rng(7)
        k = 2560
        scale = 1.0 / math.sqrt(k)
        a = rng.normal(0.0, scale, size=(64, k)).astype(np.float32)
        b = rng.normal(0.0, scale, size=(k, 64)).astype(np.float32)
        a_slices = residual_split_fp16(a)
        b_slices = residual_split_fp16(b)
        c3 = three_product_fp32(
            a_slices.high, a_slices.low, b_slices.high, b_slices.low
        )
        c_x = three_product_cross_f16(
            a_slices.high, a_slices.low, b_slices.high, b_slices.low
        )
        three = compare_device_gemm(a, b, c3, precision="fp32")
        cross = compare_device_gemm(a, b, c_x, precision="fp32")
        rel3 = three["against_float64_reference"]["relative_frobenius_error"]
        relx = cross["against_float64_reference"]["relative_frobenius_error"]
        self.assertLess(relx / max(rel3, 1e-30), 4.0)
        self.assertGreaterEqual(relx, rel3 * 0.5)


class SixteenProductTests(unittest.TestCase):
    def test_sixteen_products_match_four_on_scaled_gaussians(self) -> None:
        rng = np.random.default_rng(7)
        k = 256
        scale = 1.0 / math.sqrt(k)
        a = rng.normal(0.0, scale, size=(8, k)).astype(np.float32)
        b = rng.normal(0.0, scale, size=(k, 6)).astype(np.float32)
        a4 = residual_split_fp16(a, n_slices=4)
        b4 = residual_split_fp16(b, n_slices=4)
        a2 = residual_split_fp16(a, n_slices=2)
        b2 = residual_split_fp16(b, n_slices=2)
        c16 = sixteen_product_fp32(
            a4.high, a4.mid, a4.extra, a4.tail, b4.high, b4.mid, b4.extra, b4.tail
        )
        c4 = four_product_fp32(a2.high, a2.low, b2.high, b2.low)
        sixteen = compare_device_gemm(a, b, c16, precision="fp32")
        four = compare_device_gemm(a, b, c4, precision="fp32")
        rel16 = sixteen["against_float64_reference"]["relative_frobenius_error"]
        rel4 = four["against_float64_reference"]["relative_frobenius_error"]
        self.assertLess(abs(rel16 - rel4) / max(rel4, 1e-30), 0.05)
        self.assertGreater(float(np.mean(a4.extra == 0)), 0.9)
        self.assertGreater(float(np.mean(a4.tail == 0)), 0.9)


class TwoSumTests(unittest.TestCase):
    def test_knuth_twosum_is_error_free_in_float64(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.normal(size=(32, 32)).astype(np.float32)
        b = rng.normal(size=(32, 32)).astype(np.float32)
        s, e = knuth_twosum(a, b)
        rebuilt = s.astype(np.float64) + e.astype(np.float64)
        exact = a.astype(np.float64) + b.astype(np.float64)
        np.testing.assert_array_equal(rebuilt, exact)

    def test_twosum_four_product_beats_single_word_fp32_c(self) -> None:
        rng = np.random.default_rng(7)
        k = 64
        scale = 1.0 / math.sqrt(k)
        a = rng.normal(0.0, scale, size=(16, k)).astype(np.float32)
        b = rng.normal(0.0, scale, size=(k, 12)).astype(np.float32)
        a_slices = residual_split_fp16(a)
        b_slices = residual_split_fp16(b)
        c_word = four_product_fp32(
            a_slices.high, a_slices.low, b_slices.high, b_slices.low
        )
        c_two = four_product_twosum_fp64(
            a_slices.high, a_slices.low, b_slices.high, b_slices.low
        )
        word = compare_device_gemm(a, b, c_word, precision="fp32")
        two = compare_device_gemm(a, b, c_two, precision="fp32")
        self.assertTrue(two["two_word_result"])
        self.assertLess(
            two["against_float64_reference"]["relative_frobenius_error"],
            word["against_float64_reference"]["relative_frobenius_error"],
        )

    def test_reconstruct_twosum_fp64_is_exact_f32_add(self) -> None:
        hi = np.asarray([[1.0]], dtype=np.float32)
        lo = np.asarray([[2.0**-24]], dtype=np.float32)
        rebuilt = reconstruct_twosum_fp64(hi, lo)
        self.assertEqual(rebuilt[0, 0], 1.0 + 2.0**-24)

    def test_nine_product_twosum_beats_four_product_vs_float64(self) -> None:
        rng = np.random.default_rng(7)
        k = 32
        # Unscaled so the third fp16 word is live; 1/sqrt(k) flushes A2 to zero.
        a = rng.normal(0.0, 1.0, size=(8, k)).astype(np.float32)
        b = rng.normal(0.0, 1.0, size=(k, 6)).astype(np.float32)
        a2 = residual_split_fp16(a, n_slices=2)
        b2 = residual_split_fp16(b, n_slices=2)
        a3 = residual_split_fp16(a, n_slices=3)
        b3 = residual_split_fp16(b, n_slices=3)
        c_four = four_product_twosum_fp64(a2.high, a2.low, b2.high, b2.low)
        c_nine = nine_product_twosum_fp64(
            a3.high, a3.mid, a3.extra, b3.high, b3.mid, b3.extra
        )
        four = compare_device_gemm(a, b, c_four, precision="fp32")
        nine = compare_device_gemm(a, b, c_nine, precision="fp32")
        self.assertTrue(nine["two_word_result"])
        self.assertFalse(np.all(a3.extra == 0))
        self.assertLess(
            nine["against_float64_reference"]["relative_frobenius_error"],
            four["against_float64_reference"]["relative_frobenius_error"],
        )
        self.assertLess(
            nine["against_float64_reference"]["relative_frobenius_error"],
            5e-7,
        )

    def test_six_products_is_triangle_i_plus_j_lt_3(self) -> None:
        self.assertEqual(sum(1 for i in range(3) for j in range(3) if i + j < 3), 6)
        rng = np.random.default_rng(7)
        a = rng.normal(0.0, 1.0, size=(8, 16)).astype(np.float32)
        b = rng.normal(0.0, 1.0, size=(16, 6)).astype(np.float32)
        a3 = residual_split_fp16(a, n_slices=3)
        b3 = residual_split_fp16(b, n_slices=3)
        c6 = six_product_fp32(
            a3.high, a3.mid, a3.extra, b3.high, b3.mid, b3.extra
        )
        c9 = nine_product_fp32(
            a3.high, a3.mid, a3.extra, b3.high, b3.mid, b3.extra
        )

        def product(left, right):
            return np.matmul(
                np.asarray(left, dtype=np.float32),
                np.asarray(right, dtype=np.float32),
            ).astype(np.float64)

        skipped = (
            product(a3.mid, b3.extra)
            + product(a3.extra, b3.mid)
            + product(a3.extra, b3.extra)
        )
        rebuilt = (c6.astype(np.float64) + skipped).astype(np.float32)
        np.testing.assert_allclose(rebuilt, c9, rtol=1e-6, atol=1e-6)

    def test_ten_products_is_triangle_i_plus_j_lt_4(self) -> None:
        self.assertEqual(sum(1 for i in range(4) for j in range(4) if i + j < 4), 10)
        self.assertNotEqual(sum(1 for i in range(4) for j in range(4) if i + j <= 4), 11)
        self.assertEqual(sum(1 for i in range(4) for j in range(4) if i + j <= 4), 13)

    def test_ten_product_twosum_beats_weight_split_dual(self) -> None:
        rng = np.random.default_rng(7)
        k = 32
        a = rng.normal(0.0, 1.0, size=(8, k)).astype(np.float64)
        b = rng.normal(0.0, 1.0, size=(k, 6)).astype(np.float64)
        a4 = residual_split_fp16(a, n_slices=4)
        b4 = residual_split_fp16(b, n_slices=4)
        c_dual, _hi, _lo = ten_product_dual_fp64(
            a4.high, a4.mid, a4.extra, a4.tail,
            b4.high, b4.mid, b4.extra, b4.tail,
        )
        c_two = ten_product_twosum_fp64(
            a4.high, a4.mid, a4.extra, a4.tail,
            b4.high, b4.mid, b4.extra, b4.tail,
        )
        dual = compare_device_gemm(a, b, c_dual, precision="fp32")
        two = compare_device_gemm(a, b, c_two, precision="fp32")
        self.assertTrue(two["two_word_result"])
        self.assertLess(
            two["against_float64_reference"]["relative_frobenius_error"],
            dual["against_float64_reference"]["relative_frobenius_error"],
        )

    def test_sram_ten_product_twosum_t32_fits_40kib(self) -> None:
        def bytes_(t: int) -> int:
            # A0/A1 dbl-buf + A2/A3 single-buf (scheme F).
            return 12 * t * t + 12 * t * t + 8 * t * t + 24 * t

        self.assertEqual(bytes_(32), 33536)
        self.assertLess(bytes_(32), 40 * 1024)
        self.assertLess(bytes_(32), 48 * 1024)
        self.assertEqual(bytes_(28), 25760)
        self.assertLess(bytes_(28), 40 * 1024)
        self.assertAlmostEqual(bytes_(32) / (48 * 1024), 33536 / (48 * 1024), places=4)

    def test_fifteen_product_twosum_exists(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.normal(0.0, 1.0, size=(4, 8)).astype(np.float64)
        b = rng.normal(0.0, 1.0, size=(8, 3)).astype(np.float64)
        a5 = residual_split_fp16(a, n_slices=5)
        b5 = residual_split_fp16(b, n_slices=5)
        c = fifteen_product_twosum_fp64(
            a5.w0, a5.w1, a5.w2, a5.w3, a5.w4,
            b5.w0, b5.w1, b5.w2, b5.w3, b5.w4,
        )
        self.assertEqual(c.dtype, np.float64)
        self.assertEqual(c.shape, (4, 3))


class ResidualSplitFp32Tests(unittest.TestCase):
    def test_already_float32_has_tiny_low_slice(self) -> None:
        source = np.asarray([[0.5, 1.0], [-2.0, 0.25]], dtype=np.float32).astype(
            np.float64
        )
        slices = residual_split_fp32(source)
        np.testing.assert_array_equal(slices.high, source.astype(np.float32))
        self.assertTrue(np.all(slices.low == 0))
        np.testing.assert_array_equal(reconstruct_fp64(slices), source)

    def test_fp32_reconstruction_beats_high_word_vs_float64(self) -> None:
        rng = np.random.default_rng(7)
        source = rng.normal(0.0, 1.0, size=(16, 16)).astype(np.float64)
        slices = residual_split_fp32(source)
        high_only = slices.high.astype(np.float64)
        rebuilt = reconstruct_fp64(slices)
        high_err = np.linalg.norm(source - high_only, ord="fro")
        two_err = np.linalg.norm(source - rebuilt, ord="fro")
        self.assertLess(two_err, high_err)
        diag = slice_diagnostics(source, slices)
        self.assertLess(diag["relative_reconstruction_error"], 1e-7)
        self.assertEqual(diag["n_slices"], 2.0)

    def test_rejects_non_2d(self) -> None:
        with self.assertRaises(ValueError):
            residual_split_fp32(np.ones(4, dtype=np.float64))

    def test_four_fp32_products_beat_truncated_sgemm_vs_float64(self) -> None:
        rng = np.random.default_rng(7)
        k = 64
        scale = 1.0 / math.sqrt(k)
        a = rng.normal(0.0, scale, size=(32, k)).astype(np.float64)
        b = rng.normal(0.0, scale, size=(k, 24)).astype(np.float64)
        a_slices = residual_split_fp32(a)
        b_slices = residual_split_fp32(b)
        c_ozaki = four_product_fp32(
            a_slices.high, a_slices.low, b_slices.high, b_slices.low
        )
        c_trunc = np.matmul(a_slices.high, b_slices.high).astype(np.float32)
        ozaki = compare_device_gemm(a, b, c_ozaki, precision="fp32")
        trunc = compare_device_gemm(a, b, c_trunc, precision="fp32")
        self.assertEqual(ozaki["source_dtype"], "float64")
        self.assertLess(
            ozaki["against_float64_reference"]["relative_frobenius_error"],
            trunc["against_float64_reference"]["relative_frobenius_error"],
        )


if __name__ == "__main__":
    unittest.main()
