"""Host numeric checks for 2-slice skip Ozaki. No wafer and no CSL compile."""

from __future__ import annotations

import unittest

import numpy as np

from ozaki_roofline.host_skip_sim import (
    assemble_fp64_halves,
    fp64_merge_matches_float64_add,
    ozimmu_problem,
    pack_fp64_halves,
    skip_accumulate,
    variant_rel,
)
from ozaki_roofline.split import reconstruct_twosum_fp64


class HostSkipSimTests(unittest.TestCase):
    def test_period_longer_than_the_product_stream_matches_skip(self) -> None:
        # 3 products per K. K=4 is 12 products, so period 128 is one tail flush
        # of a bucket that saw the same f32 adds as plain skip.
        a, b = ozimmu_problem(8, 4, 6, phi=1.0, seed=7)
        skip = skip_accumulate(a, b, twosum=False)
        blocked = skip_accumulate(a, b, period=128, twosum=True)
        np.testing.assert_array_equal(blocked, skip.astype(np.float64))

    def test_selective_modes_differ_from_every_product(self) -> None:
        a, b = ozimmu_problem(8, 8, 8, phi=1.0, seed=7)
        every = skip_accumulate(a, b, period=1)
        a0b0 = skip_accumulate(a, b, period=1, selective="a0b0")
        cross = skip_accumulate(a, b, period=1, selective="cross")
        self.assertFalse(np.array_equal(every, a0b0))
        self.assertFalse(np.array_equal(every, cross))
        self.assertFalse(np.array_equal(a0b0, cross))

    def test_fp64_bit_assemble_matches_float64_add(self) -> None:
        rng = np.random.default_rng(7)
        hi = rng.normal(size=(32, 16)).astype(np.float32)
        lo = (rng.normal(size=(32, 16)) * np.float32(2.0**-20)).astype(np.float32)
        hi[np.abs(hi) < np.finfo(np.float32).tiny] = np.float32(1.0)
        lo[np.abs(lo) < np.finfo(np.float32).tiny] = np.float32(2.0**-20)
        self.assertTrue(fp64_merge_matches_float64_add(hi, lo))
        oracle = reconstruct_twosum_fp64(hi, lo)
        packed_hi, packed_lo = pack_fp64_halves(oracle)
        np.testing.assert_array_equal(
            assemble_fp64_halves(packed_hi, packed_lo), oracle
        )
        # A known finite normal pair whose sum is not a float32.
        hi_one = np.asarray([[np.float32(1.0)]])
        lo_eps = np.asarray([[np.float32(2.0**-24)]])
        self.assertTrue(fp64_merge_matches_float64_add(hi_one, lo_eps))
        self.assertEqual(
            float(reconstruct_twosum_fp64(hi_one, lo_eps)[0, 0]),
            1.0 + 2.0**-24,
        )

    def test_ozimmu_ordering_at_small_n(self) -> None:
        for phi in (0.1, 1.0, 2.0):
            a, b = ozimmu_problem(64, 64, 64, phi=phi, seed=7)
            rel = variant_rel(a, b)
            every = rel["twosum-every"]
            fp32 = rel["fp32"]
            skip = rel["skip"]
            blocked = rel["twosum-every-128"]
            self.assertLess(every, fp32, msg=f"phi={phi} every={every} fp32={fp32}")
            self.assertLess(every, skip, msg=f"phi={phi} every={every} skip={skip}")
            lo, hi = (every, fp32) if every <= fp32 else (fp32, every)
            between = lo <= blocked <= hi
            not_wild = blocked <= fp32 * 4.0
            self.assertTrue(
                between or not_wild,
                msg=(
                    f"phi={phi} every={every} period128={blocked} "
                    f"fp32={fp32} skip={skip}"
                ),
            )
            self.assertTrue(np.isfinite(list(rel.values())).all())


if __name__ == "__main__":
    unittest.main()
