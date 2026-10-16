"""Shape and command tests for the parameterized measurement entry point."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ozaki_roofline.campaign import measurement_commands, resolve_shape  # noqa: E402


def _cmds(**overrides: object) -> list[list[str]]:
    params = dict(
        grid_size=512,
        n=None,
        k=None,
        m=None,
        tile=32,
        timing="kernel",
        input_kind="ozimmu",
        phi=1.0,
        warmup=1,
        repetitions=3,
        clock_mhz=1400.0,
        data_budget_kib=40,
        simulator=False,
        dump_pe_map=False,
        force=False,
        skip_validation=False,
        high_precision=False,
        validate_subsample=0,
        alpha=7,
        n_slices=8,
        single_product=False,
        seed=7,
        stage="both",
    )
    params.update(overrides)
    return measurement_commands(ROOT, **params)  # type: ignore[arg-type]


class ResolveShapeTest(unittest.TestCase):
    def test_tile_sets_square_n(self) -> None:
        p, n, k, m, tile = resolve_shape(512, None, None, None, 32, grid_required=True)
        self.assertEqual((p, n, k, m, tile), (512, 16384, 16384, 16384, 32))

    def test_n_sets_tile(self) -> None:
        _p, n, _k, _m, tile = resolve_shape(512, 16384, None, None, None, grid_required=True)
        self.assertEqual(n, 16384)
        self.assertEqual(tile, 32)

    def test_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            resolve_shape(512, 8192, None, None, 32, grid_required=True)


class CommandTest(unittest.TestCase):
    def test_fp32_twosum_methods(self) -> None:
        for name, period in (("fp32-ozaki-twosum", 1), ("fp32-ozaki-twosum-every10", 10)):
            for command in _cmds(method_name=name, tile=28):
                self.assertEqual(command[command.index("--precision") + 1], "fp32")
                self.assertEqual(command[command.index("--n-products") + 1], "4")
                self.assertIn("--twosum", command)
                if period != 1:
                    self.assertEqual(command[command.index("--twosum-period") + 1], str(period))

    def test_fp32_twosum_period_override(self) -> None:
        for command in _cmds(method_name="fp32-ozaki-twosum", twosum_period=7):
            self.assertEqual(command[command.index("--twosum-period") + 1], "7")
        for period in (0, 32768):
            with self.assertRaises(ValueError):
                _cmds(method_name="fp32-ozaki-twosum", twosum_period=period)
        with self.assertRaises(ValueError):
            _cmds(method_name="fp32", twosum_period=7)

    def test_four_slice_triangle_kernel(self) -> None:
        compile_cmd, launch_cmd = _cmds(method_name="4slice-tri")
        self.assertTrue(compile_cmd[1].endswith("benchmarks/gemm_ozaki/compile.py"))
        self.assertIn("--n-products", compile_cmd)
        self.assertEqual(compile_cmd[compile_cmd.index("--n-products") + 1], "10")
        self.assertIn("--start-barrier", compile_cmd)
        self.assertIn("--start-barrier", launch_cmd)
        self.assertEqual(launch_cmd[launch_cmd.index("--n") + 1], "16384")
        self.assertEqual(launch_cmd[launch_cmd.index("--grid-size") + 1], "512")
        self.assertEqual(launch_cmd[launch_cmd.index("--input-kind") + 1], "ozimmu")

    def test_e2e_has_no_barrier(self) -> None:
        compile_cmd, launch_cmd = _cmds(method_name="skip-a1b1", timing="e2e", n=16384, tile=None)
        self.assertNotIn("--start-barrier", compile_cmd)
        self.assertNotIn("--start-barrier", launch_cmd)

    def test_period_flags(self) -> None:
        compile_cmd, _launch = _cmds(method_name="twosum-every10")
        self.assertIn("--twosum", compile_cmd)
        self.assertEqual(compile_cmd[compile_cmd.index("--twosum-period") + 1], "10")
        self.assertEqual(compile_cmd[compile_cmd.index("--n-products") + 1], "3")

    def test_fp32_mesh(self) -> None:
        compile_cmd, _launch = _cmds(method_name="fp32")
        self.assertTrue(compile_cmd[1].endswith("benchmarks/gemm_mesh/compile.py"))
        self.assertEqual(compile_cmd[compile_cmd.index("--precision") + 1], "fp32")

    def test_pe_map_rejected_on_large_grid(self) -> None:
        with self.assertRaises(ValueError):
            _cmds(method_name="fp32", grid_size=640, tile=4, dump_pe_map=True)

    def test_simulator_rejects_kernel_timing(self) -> None:
        with self.assertRaises(ValueError):
            _cmds(method_name="skip-a1b1", simulator=True)

    def test_host_skip_uses_n(self) -> None:
        (cmd,) = _cmds(method_name="host-skip", grid_size=None, n=64, tile=None, stage="launch")
        self.assertTrue(cmd[1].endswith("scripts/host_skip_twosum_sim.py"))
        self.assertNotIn("--start-barrier", cmd)
        self.assertEqual(cmd[cmd.index("--n") + 1], "64")


if __name__ == "__main__":
    unittest.main()
