import math
import unittest

import numpy as np

from ozaki_roofline.fp32_bit_halves import (
    integer_product_int64,
    overflow_report_u16,
    reconstruct_uint32,
    schoolbook_uint32_gemm,
    split_fp32_uint16_halves,
    truncated_fp32,
    truncated_fp32_residual_oracle,
    wrap_u32,
)


class SplitTests(unittest.TestCase):
    def test_roundtrip_bits(self) -> None:
        rng = np.random.default_rng(7)
        source = rng.normal(0.0, 0.1, size=(5, 7)).astype(np.float32)
        halves = split_fp32_uint16_halves(source)
        bits = reconstruct_uint32(halves)
        rebuilt = bits.view(np.float32)
        np.testing.assert_array_equal(rebuilt, source)
        np.testing.assert_array_equal(halves.high.dtype, np.uint16)
        np.testing.assert_array_equal(halves.low.dtype, np.uint16)

    def test_literal_shift_mask(self) -> None:
        source = np.asarray([[1.0, -2.5], [0.0, math.pi]], dtype=np.float32)
        bits = source.view(np.uint32)
        halves = split_fp32_uint16_halves(source)
        np.testing.assert_array_equal(halves.high, (bits >> 16).astype(np.uint16))
        np.testing.assert_array_equal(halves.low, (bits & 0xFFFF).astype(np.uint16))

    def test_low_half_bit15_is_unsigned(self) -> None:
        # Mantissa low half often has bit 15 set; signed i16 would be negative.
        bits = np.asarray([[0x3F80FFFF]], dtype=np.uint32)
        source = bits.view(np.float32)
        halves = split_fp32_uint16_halves(source)
        self.assertEqual(int(halves.low[0, 0]), 0xFFFF)
        self.assertGreater(int(halves.low[0, 0]), 32767)


class SchoolbookTests(unittest.TestCase):
    def test_matches_uint32_matmul(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.normal(0.0, 0.2, size=(6, 5)).astype(np.float32)
        b = rng.normal(0.0, 0.2, size=(5, 4)).astype(np.float32)
        ah = split_fp32_uint16_halves(a)
        bh = split_fp32_uint16_halves(b)
        got = schoolbook_uint32_gemm(ah.high, ah.low, bh.high, bh.low)
        a_u = reconstruct_uint32(ah).astype(np.int64)
        b_u = reconstruct_uint32(bh).astype(np.int64)
        np.testing.assert_array_equal(got, np.matmul(a_u, b_u))

    def test_signed_mul_differs_when_bit15_set(self) -> None:
        a = np.asarray([[0xFFFF]], dtype=np.uint16)
        b = np.asarray([[1]], dtype=np.uint16)
        unsigned = integer_product_int64(a, b)
        signed = np.matmul(a.astype(np.int16).astype(np.int64), b.astype(np.int64))
        self.assertEqual(int(unsigned[0, 0]), 65535)
        self.assertEqual(int(signed[0, 0]), -1)

    def test_overflow_at_k64(self) -> None:
        a = np.full((2, 64), 65535, dtype=np.uint16)
        b = np.full((64, 2), 65535, dtype=np.uint16)
        products = {(0, 0): integer_product_int64(a, b)}
        report = overflow_report_u16(products, k=64)
        self.assertEqual(report["fits_i32"], 0.0)
        self.assertEqual(report["fits_u32"], 0.0)
        self.assertEqual(report["overflow_expected_at_p64"], 1.0)

    def test_wrap_u32_matches_mod(self) -> None:
        values = np.asarray([[(1 << 32) + 3, 7]], dtype=np.int64)
        np.testing.assert_array_equal(wrap_u32(values), np.asarray([[3, 7]], dtype=np.uint32))


class TruncatedFloatOracleTests(unittest.TestCase):
    def test_high_half_is_truncated_fp32(self) -> None:
        source = np.asarray([[1.23456789]], dtype=np.float32)
        halves = split_fp32_uint16_halves(source)
        trunc = truncated_fp32(halves.high)
        bits = source.view(np.uint32) & np.uint32(0xFFFF0000)
        np.testing.assert_array_equal(trunc.view(np.uint32), bits)

    def test_residual_oracle_near_source_product(self) -> None:
        rng = np.random.default_rng(7)
        k = 8
        a = rng.normal(0.0, 1.0 / math.sqrt(k), size=(4, k)).astype(np.float32)
        b = rng.normal(0.0, 1.0 / math.sqrt(k), size=(k, 3)).astype(np.float32)
        _, _, c_resid = truncated_fp32_residual_oracle(a, b)
        reference = np.matmul(a.astype(np.float64), b.astype(np.float64))
        rel = np.linalg.norm(c_resid - reference, ord="fro") / np.linalg.norm(
            reference, ord="fro"
        )
        self.assertLess(float(rel), 1e-6)


if __name__ == "__main__":
    unittest.main()
