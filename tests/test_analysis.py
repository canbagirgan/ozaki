import unittest

from scripts.analyze import (
    bandwidth_summaries,
    device_allocation_summaries,
    fit_gemm_cycle_models,
    gemm_allocation_summaries,
    gemm_summaries,
    host_container_transfer_fits,
    normalize_direct_gemm_records,
    transfer_fits,
)


class AnalysisTests(unittest.TestCase):
    def test_aggregation_and_directional_prediction(self) -> None:
        bandwidth_records = []
        for precision in ("fp16", "fp32"):
            for direction, bandwidth in (("h2d", 100e9), ("d2h", 20e9)):
                for byte_count in (1_000_000, 4_000_000):
                    elapsed = 1e-3 + byte_count / bandwidth
                    bandwidth_records.append(
                        {
                            "precision": precision,
                            "direction": direction,
                            "bytes": byte_count,
                            "channels": 16,
                            "elements_per_pe": 1,
                            "device_seconds": elapsed,
                            "host_blocking_s": elapsed,
                            "device_bandwidth_Bps": byte_count / elapsed,
                            "host_blocking_bandwidth_Bps": byte_count / elapsed,
                        }
                    )

        bandwidth_rows = bandwidth_summaries(bandwidth_records)
        fits = transfer_fits(bandwidth_rows)
        container_fits = host_container_transfer_fits(
            bandwidth_rows,
            elapsed_key="host_blocking_seconds_median",
        )
        self.assertEqual(len(fits), 4)
        self.assertEqual(len(container_fits), 4)
        self.assertAlmostEqual(fits[("fp32", "h2d")].latency_s, 1e-3)
        self.assertAlmostEqual(fits[("fp32", "h2d")].bandwidth_Bps, 100e9, delta=1e4)

        gemm_record = {
            "precision": "fp32",
            "grid": {"width": 64, "height": 64},
            "shape": {"n": 256, "k": 256, "m": 256},
            "local_tile": {"mt": 4, "kt": 4, "nt": 4},
            "timing": {
                "host_h2d_s": 0.002,
                "host_kernel_dispatch_s": 0.0001,
                "host_post_dispatch_wait_and_d2h_s": 0.0069,
                "host_d2h_s": 0.004,
                "component_sum_s": 0.0085,
                "operational_e2e_s": 0.009,
                "device_kernel_s": 0.0025,
                "device_kernel_cycles": {
                    "minimum": 2_000_000,
                    "mean": 2_075_000,
                    "median": 2_100_000,
                    "maximum": 2_187_500,
                },
            },
            "validation": {"relative_residual": 1e-6},
        }
        gemm_rows = gemm_summaries([gemm_record], fits)
        self.assertEqual(len(gemm_rows), 1)
        self.assertGreater(gemm_rows[0]["device_kernel_flops"], 0)
        self.assertGreater(gemm_rows[0]["predicted_e2e_seconds"], 0)
        self.assertEqual(gemm_rows[0]["validation_relative_residual"], 1e-6)

        gemm_record["run_id"] = "allocation-1"
        allocation_rows = gemm_allocation_summaries(
            [gemm_record],
            host_scope="user_node",
        )
        device_rows = device_allocation_summaries(allocation_rows)
        self.assertEqual(allocation_rows[0]["samples"], 1)
        self.assertAlmostEqual(
            allocation_rows[0]["kernel_mean_pe_seconds_median"],
            2_075_000 / 875_000_000,
        )
        self.assertEqual(device_rows[0]["allocations"], 1)
        self.assertAlmostEqual(
            device_rows[0]["kernel_mean_pe_seconds_median"],
            2_075_000 / 875_000_000,
        )
        self.assertGreater(device_rows[0]["device_kernel_flops_median"], 0)

    def test_cycle_model_fit(self) -> None:
        rows = []
        for tile in (4, 8, 16, 24):
            rows.append(
                {
                    "precision": "fp32",
                    "n": tile * 64,
                    "k": tile * 64,
                    "m": tile * 64,
                    "mt": tile,
                    "kt": tile,
                    "nt": tile,
                    "kernel_device_cycles_median": float(
                        64 * (2 * tile**3 + 10 * tile**2 + 3 * tile**2 + 100)
                    ),
                }
            )
        models = fit_gemm_cycle_models(rows)
        self.assertIn("fp32", models)
        self.assertIn("cycle_model_predicted_cycles", rows[0])

    def test_direct_gemm_timing_normalization(self) -> None:
        records = [
            {
                "timing": {
                    "worker_h2d_s": 0.001,
                    "worker_kernel_dispatch_s": 0.0001,
                    "worker_post_dispatch_wait_and_d2h_s": 0.0049,
                    "worker_d2h_s": 0.003,
                    "worker_component_sum_s": 0.0055,
                    "worker_operational_e2e_s": 0.006,
                    "device_kernel_s": 0.0015,
                    "device_kernel_cycles": {"maximum": 1_312_500},
                    "timer_readback_s": 0.004,
                }
            }
        ]
        normalized = normalize_direct_gemm_records(records)
        self.assertEqual(normalized[0]["timing"]["host_h2d_s"], 0.001)
        self.assertEqual(normalized[0]["timing"]["operational_e2e_s"], 0.006)
        self.assertNotIn("timer_readback_s", normalized[0]["timing"])


if __name__ == "__main__":
    unittest.main()
