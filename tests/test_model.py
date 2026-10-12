import unittest

from ozaki_roofline.fitting import fit_transfer, summarize
from ozaki_roofline.model import (
    CANONICAL_P754_N,
    CANONICAL_P754_TILE,
    MAX_EVEN_SQUARE_P,
    PRECISIONS,
    GemmShape,
    Grid,
    TransferFit,
    dense_compute_ceiling_flops,
    device_ridge_point_flops_per_byte,
    effective_host_io_bandwidth_Bps,
    packed_collective_words,
    parse_grid_size,
    reject_large_pe_map,
    scaled_sram_bandwidth_Bps,
    summa_data_footprint_bytes,
    validate_cs3_square_placement,
    validate_summa_configuration,
)


class GemmShapeTests(unittest.TestCase):
    def test_flops_bytes_and_host_oi(self) -> None:
        shape = GemmShape(n=256, k=512, m=128)
        fp16 = PRECISIONS["fp16"]
        self.assertEqual(shape.flops, 2 * 256 * 512 * 128)
        self.assertEqual(shape.h2d_bytes(fp16), 2 * (256 * 512 + 512 * 128))
        self.assertEqual(shape.d2h_bytes(fp16), 2 * 256 * 128)
        self.assertAlmostEqual(
            shape.host_operational_intensity(fp16),
            shape.flops / shape.host_wire_bytes(fp16),
        )

    def test_default_grid_tiles(self) -> None:
        self.assertEqual(GemmShape(1024, 2048, 512).local_tiles(), (16, 32, 8))
        self.assertEqual(
            GemmShape(2560, 2560, 2560).local_tiles(Grid(128, 128)),
            (20, 20, 20),
        )
        with self.assertRaises(ValueError):
            GemmShape(1000, 2048, 512).local_tiles()


class RoofTests(unittest.TestCase):
    def test_64_square_instruction_ceilings(self) -> None:
        self.assertAlmostEqual(
            dense_compute_ceiling_flops(PRECISIONS["fp16"]),
            91.7504e12,
        )
        self.assertAlmostEqual(
            dense_compute_ceiling_flops(PRECISIONS["fp32"]),
            5.7344e12,
        )

    def test_scaled_sram_bandwidth_and_ridges(self) -> None:
        self.assertAlmostEqual(scaled_sram_bandwidth_Bps(), 21e15 * 4096 / 900_000)
        self.assertAlmostEqual(
            device_ridge_point_flops_per_byte(PRECISIONS["fp16"]),
            0.96,
        )
        self.assertAlmostEqual(
            device_ridge_point_flops_per_byte(PRECISIONS["fp32"]),
            0.06,
        )

    def test_effective_host_io_bandwidth(self) -> None:
        h2d = TransferFit(0.0, 18e9)
        d2h = TransferFit(0.0, 3e9)
        self.assertAlmostEqual(
            effective_host_io_bandwidth_Bps(h2d, d2h),
            3.0 / (2.0 / 18e9 + 1.0 / 3e9),
        )


class SummaTests(unittest.TestCase):
    def test_five_buffer_footprint(self) -> None:
        self.assertEqual(summa_data_footprint_bytes(40, 40, 40, PRECISIONS["fp32"]), 32_000)
        self.assertEqual(summa_data_footprint_bytes(40, 40, 40, PRECISIONS["fp16"]), 16_000)

    def test_validate_packing_and_budget(self) -> None:
        self.assertEqual(
            validate_summa_configuration(GemmShape(64 * 40, 64 * 40, 64 * 40), PRECISIONS["fp16"]),
            (40, 40, 40),
        )
        with self.assertRaises(ValueError):
            validate_summa_configuration(GemmShape(64 * 3, 64 * 3, 64), PRECISIONS["fp16"])
        with self.assertRaises(ValueError):
            validate_summa_configuration(GemmShape(64 * 64, 64 * 64, 64 * 64), PRECISIONS["fp32"])

    def test_collective_word_packing(self) -> None:
        self.assertEqual(packed_collective_words(10, PRECISIONS["fp16"]), 5)
        self.assertEqual(packed_collective_words(10, PRECISIONS["fp32"]), 10)


class Cs3GridTests(unittest.TestCase):
    def test_850_square_does_not_fit(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            parse_grid_size(850)
        message = str(ctx.exception)
        self.assertIn("850", message)
        self.assertIn("754", message)
        self.assertIn("16588", message)
        with self.assertRaises(ValueError):
            validate_cs3_square_placement(850)
        with self.assertRaises(ValueError):
            validate_summa_configuration(
                GemmShape(850 * 20, 850 * 20, 850 * 20),
                PRECISIONS["fp16"],
                Grid(850, 850),
            )

    def test_754_whitelist_and_canonical_16k(self) -> None:
        self.assertEqual(MAX_EVEN_SQUARE_P, 754)
        self.assertEqual(parse_grid_size("754"), 754)
        self.assertEqual(CANONICAL_P754_TILE, 22)
        self.assertEqual(CANONICAL_P754_N, 16588)
        self.assertEqual(
            validate_summa_configuration(
                GemmShape(16588, 16588, 16588),
                PRECISIONS["fp16"],
                Grid(754, 754),
            ),
            (22, 22, 22),
        )
        self.assertEqual(
            validate_summa_configuration(
                GemmShape(15080, 15080, 15080),
                PRECISIONS["fp16"],
                Grid(754, 754),
            ),
            (20, 20, 20),
        )
        with self.assertRaises(ValueError):
            validate_summa_configuration(
                GemmShape(754 * 19, 754 * 19, 754 * 19),
                PRECISIONS["fp16"],
                Grid(754, 754),
            )

    def test_pe_map_forbidden_on_large_p(self) -> None:
        reject_large_pe_map(64, True)
        reject_large_pe_map(754, False)
        with self.assertRaises(ValueError):
            reject_large_pe_map(640, True)
        with self.assertRaises(ValueError):
            reject_large_pe_map(754, True)


class FittingTests(unittest.TestCase):
    def test_transfer_fit(self) -> None:
        latency = 200e-6
        bandwidth = 5e9
        sizes = [1_000_000, 2_000_000, 4_000_000]
        times = [latency + size / bandwidth for size in sizes]
        fit = fit_transfer(sizes, times)
        self.assertAlmostEqual(fit.latency_s, latency)
        self.assertAlmostEqual(fit.bandwidth_Bps, bandwidth, places=-2)

    def test_distribution_summary(self) -> None:
        result = summarize([1, 2, 3, 4, 5])
        self.assertEqual(result.count, 5)
        self.assertEqual(result.median, 3)
        self.assertEqual(result.p10, 1.4)
        self.assertEqual(result.p90, 4.6)


if __name__ == "__main__":
    unittest.main()
