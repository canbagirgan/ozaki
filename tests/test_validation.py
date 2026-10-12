import json
import math
import unittest

import numpy as np

from ozaki_roofline.validation import UNIT_ROUNDOFF, compare_device_gemm


class ValidationTests(unittest.TestCase):
    def test_identical_float64_proxy_is_zero_for_float32_numpy(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(size=(8, 5)).astype(np.float32)
        b = rng.normal(size=(5, 6)).astype(np.float32)
        host32 = (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float32)
        report = compare_device_gemm(a, b, host32, precision="fp32")
        host = report["against_numpy_float32_gemm"]
        self.assertEqual(host["max_abs_error"], 0.0)
        self.assertEqual(host["relative_frobenius_error"], 0.0)

    def test_known_two_by_two_fp32(self) -> None:
        a = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        exact = np.asarray([[19.0, 22.0], [43.0, 50.0]], dtype=np.float32)
        report = compare_device_gemm(a, b, exact, precision="fp32")
        ref = report["against_float64_reference"]
        self.assertLess(ref["max_abs_error"], 1e-6)
        self.assertLess(ref["relative_frobenius_error"], 1e-6)

    def test_perturbed_fp32_reports_the_injected_error(self) -> None:
        a = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        b = np.asarray([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
        c = (a @ b).astype(np.float32)
        c[0, 0] += np.float32(0.25)
        report = compare_device_gemm(a, b, c, precision="fp32")
        self.assertAlmostEqual(
            report["against_float64_reference"]["max_abs_error"],
            0.25,
            places=6,
        )

    def test_fp16_inputs_are_promoted_exactly(self) -> None:
        a = np.asarray([[0.5, 0.25], [0.125, 1.0]], dtype=np.float16)
        b = np.asarray([[2.0, 0.0], [0.0, 4.0]], dtype=np.float16)
        exact16 = np.asarray([[1.0, 1.0], [0.25, 4.0]], dtype=np.float16)
        report = compare_device_gemm(a, b, exact16, precision="fp16")
        self.assertEqual(report["unit_roundoff"], UNIT_ROUNDOFF["fp16"])
        self.assertLess(report["against_float64_reference"]["max_abs_error"], 1e-8)

    def test_nonfinite_device_result_raises(self) -> None:
        a = np.ones((2, 2), dtype=np.float32)
        b = np.ones((2, 2), dtype=np.float32)
        c = np.ones((2, 2), dtype=np.float32)
        c[0, 1] = np.nan
        with self.assertRaises(AssertionError):
            compare_device_gemm(a, b, c, precision="fp32")

    def test_shape_mismatch_raises(self) -> None:
        a = np.ones((2, 3), dtype=np.float32)
        b = np.ones((3, 4), dtype=np.float32)
        c = np.ones((2, 2), dtype=np.float32)
        with self.assertRaises(ValueError):
            compare_device_gemm(a, b, c, precision="fp32")

    def test_dtype_mismatch_raises(self) -> None:
        a = np.ones((2, 2), dtype=np.float32)
        b = np.ones((2, 2), dtype=np.float32)
        c = np.ones((2, 2), dtype=np.float16)
        with self.assertRaises(ValueError):
            compare_device_gemm(a, b, c, precision="fp32")

    def test_report_is_json_serializable(self) -> None:
        rng = np.random.default_rng(1)
        a = rng.normal(size=(4, 3)).astype(np.float32)
        b = rng.normal(size=(3, 5)).astype(np.float32)
        c = (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)
        report = compare_device_gemm(a, b, c, precision="fp32")
        encoded = json.dumps(report, allow_nan=False)
        self.assertIn("against_float64_reference", encoded)
        self.assertEqual(report["device_dtype"], "float32")

    def test_scaled_residual_matches_injected_frobenius_error(self) -> None:
        a = np.eye(2, dtype=np.float32)
        b = np.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
        c = np.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
        c[0, 0] += np.float32(0.5)
        report = compare_device_gemm(a, b, c, precision="fp32")
        self.assertAlmostEqual(
            report["against_float64_reference"]["frobenius_error"],
            0.5,
            places=6,
        )
        a_fro = float(np.linalg.norm(a.astype(np.float64), ord="fro"))
        b_fro = float(np.linalg.norm(b.astype(np.float64), ord="fro"))
        expected_scaled = 0.5 / (2.0 * UNIT_ROUNDOFF["fp32"] * a_fro * b_fro)
        self.assertAlmostEqual(report["scaled_residual"], expected_scaled, places=6)

    def test_correctly_rounded_reference_is_exact_for_tiny_fp16(self) -> None:
        a = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16)
        b = np.asarray([[0.5, 1.5], [2.0, 0.25]], dtype=np.float16)
        exact16 = b.copy()
        report = compare_device_gemm(a, b, exact16, precision="fp16")
        rounded = report["against_correctly_rounded_reference"]
        self.assertEqual(rounded["n_entries_matching"], 4)
        self.assertEqual(rounded["max_abs_error"], 0.0)

    def test_float64_two_word_result_is_accepted(self) -> None:
        a = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        c64 = np.matmul(a.astype(np.float64), b.astype(np.float64))
        report = compare_device_gemm(a, b, c64, precision="fp32")
        self.assertTrue(report["two_word_result"])
        self.assertEqual(report["device_dtype"], "float64")
        self.assertLess(report["against_float64_reference"]["relative_frobenius_error"], 1e-15)

    def test_float64_source_uses_true_matmul_not_truncated_promotion(self) -> None:
        rng = np.random.default_rng(7)
        k = 32
        scale = 1.0 / math.sqrt(k)
        a64 = rng.normal(0.0, scale, size=(8, k)).astype(np.float64)
        b64 = rng.normal(0.0, scale, size=(k, 6)).astype(np.float64)
        c_ref = np.matmul(a64, b64)
        c_trunc = np.matmul(a64.astype(np.float32), b64.astype(np.float32)).astype(
            np.float32
        )
        true_ref = compare_device_gemm(a64, b64, c_trunc, precision="fp32")
        promoted = compare_device_gemm(
            a64.astype(np.float32),
            b64.astype(np.float32),
            c_trunc,
            precision="fp32",
        )
        self.assertEqual(true_ref["source_dtype"], "float64")
        self.assertEqual(promoted["source_dtype"], "float32")
        self.assertGreater(
            true_ref["against_float64_reference"]["relative_frobenius_error"],
            promoted["against_float64_reference"]["relative_frobenius_error"],
        )
        exact = compare_device_gemm(
            a64, b64, c_ref.astype(np.float32), precision="fp32"
        )
        self.assertLess(
            exact["against_float64_reference"]["relative_frobenius_error"],
            true_ref["against_float64_reference"]["relative_frobenius_error"],
        )


if __name__ == "__main__":
    unittest.main()
