#!/usr/bin/env python3
"""Run cold-input SUMMA samples on a real WSE-3 or the appliance simulator."""

from __future__ import annotations

import argparse
import json
import math
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType, MemcpyOrder
from cerebras.sdk.client import SdkRuntime, sdk_utils

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ozaki_roofline.distribution import (  # noqa: E402
    decode_memcpy_output,
    encode_memcpy_input,
    tile_matrix,
    untile_matrix,
)
from ozaki_roofline.model import (  # noqa: E402
    GRID_SIZE_HELP,
    PRECISIONS,
    GemmShape,
    Grid,
    parse_grid_size,
    validate_summa_configuration,
)
from ozaki_roofline.records import append_jsonl, utc_now_iso  # noqa: E402
from ozaki_roofline.validation import compare_device_gemm  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=sorted(PRECISIONS), required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--grid-size", type=parse_grid_size, default=64, help=GRID_SIZE_HELP)
    parser.add_argument("--simulator", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--clock-mhz", type=float, default=1400.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results" / "raw" / "gemm.jsonl",
    )
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def config_name(precision: str, shape: GemmShape, grid_size: int) -> str:
    return f"summa-p{grid_size}-{precision}-n{shape.n}-k{shape.k}-m{shape.m}"


def load_manifest(precision: str, shape: GemmShape, grid_size: int) -> tuple[Path, dict]:
    name = config_name(precision, shape, grid_size)
    path = ROOT / "artifacts" / "gemm" / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"compile manifest not found: {path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def decode_cycles(timer_values: np.ndarray) -> dict[str, float | int]:
    per_pe = np.asarray(
        [
            int(sdk_utils.calculate_cycles(timer_values[py, px, :]))
            for py in range(timer_values.shape[0])
            for px in range(timer_values.shape[1])
        ],
        dtype=np.int64,
    )
    return {
        "minimum": int(per_pe.min()),
        "mean": float(per_pe.mean()),
        "median": float(np.median(per_pe)),
        "maximum": int(per_pe.max()),
        "spread": int(per_pe.max() - per_pe.min()),
    }


def validation_metrics(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    precision: str,
) -> dict:
    return compare_device_gemm(a, b, c, precision=precision)


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.repetitions <= 0:
        raise ValueError("warmup must be nonnegative and repetitions positive")
    if args.clock_mhz <= 0:
        raise ValueError("clock frequency must be positive")

    precision = PRECISIONS[args.precision]
    shape = GemmShape(args.n, args.k, args.m)
    grid = Grid(args.grid_size, args.grid_size)
    mt, kt, nt = validate_summa_configuration(shape, precision, grid)
    manifest_path, manifest = load_manifest(args.precision, shape, args.grid_size)

    p = grid.width
    generation_start = time.perf_counter_ns()
    rng = np.random.default_rng(args.seed)
    scale = 1.0 / math.sqrt(shape.k)
    dtype = np.float16 if args.precision == "fp16" else np.float32
    a = rng.normal(0.0, scale, size=(shape.n, shape.k)).astype(dtype)
    b = rng.normal(0.0, scale, size=(shape.k, shape.m)).astype(dtype)
    generation_end = time.perf_counter_ns()
    packing_start = generation_end
    a_host = encode_memcpy_input(tile_matrix(a, p, mt, kt), args.precision)
    b_host = encode_memcpy_input(tile_matrix(b, p, kt, nt), args.precision)
    packing_end = time.perf_counter_ns()

    memcpy_dtype = (
        MemcpyDataType.MEMCPY_16BIT if args.precision == "fp16" else MemcpyDataType.MEMCPY_32BIT
    )
    memcpy_order = MemcpyOrder.ROW_MAJOR
    c_raw = np.zeros(shape.n * shape.m, dtype=np.uint32)
    timer_raw = np.zeros(p * p * 3, dtype=np.float32)

    run_id = f"{config_name(args.precision, shape, args.grid_size)}-{time.time_ns()}"
    common = {
        "schema_version": 3,
        "kind": "gemm_sample",
        "run_id": run_id,
        "started_at": utc_now_iso(),
        "precision": args.precision,
        "shape": {"n": shape.n, "k": shape.k, "m": shape.m},
        "local_tile": {"mt": mt, "kt": kt, "nt": nt},
        "grid": {"width": p, "height": p, "active_pes": p * p},
        "channels": manifest["channels"],
        "artifact_manifest": str(manifest_path),
        "artifact_path": manifest["artifact_path"],
        "simulator": args.simulator,
        "clock_mhz_assumption": args.clock_mhz,
        "flops": shape.flops,
        "h2d_bytes": shape.h2d_bytes(precision),
        "d2h_bytes": shape.d2h_bytes(precision),
        "host_wire_bytes": shape.host_wire_bytes(precision),
        "host_container_bytes": a_host.nbytes + b_host.nbytes + c_raw.nbytes,
        "host_oi_flops_per_byte": shape.host_operational_intensity(precision),
        "client": {
            "hostname": socket.gethostname(),
            "python": platform.python_version(),
        },
    }

    validation: dict | None = None
    runtime_startup_start = time.perf_counter_ns()
    runtime = SdkRuntime(
        manifest["artifact_path"],
        simulator=args.simulator,
        disable_version_check=True,
    )
    with runtime as runner:
        runtime_startup_s = (time.perf_counter_ns() - runtime_startup_start) * 1e-9
        sym_a = runner.get_id("A")
        sym_b = runner.get_id("B")
        sym_c = runner.get_id("C")
        sym_timer = runner.get_id("timer")

        total_iterations = args.warmup + args.repetitions
        for iteration in range(total_iterations):
            reset_start = time.perf_counter_ns()
            runner.launch("reset", nonblock=False)
            reset_end = time.perf_counter_ns()

            e2e_start = time.perf_counter_ns()
            h2d_start = e2e_start
            runner.memcpy_h2d(
                sym_a,
                a_host,
                0,
                0,
                p,
                p,
                mt * kt,
                streaming=False,
                data_type=memcpy_dtype,
                order=memcpy_order,
                nonblock=True,
            )
            runner.memcpy_h2d(
                sym_b,
                b_host,
                0,
                0,
                p,
                p,
                kt * nt,
                streaming=False,
                data_type=memcpy_dtype,
                order=memcpy_order,
                nonblock=False,
            )
            h2d_end = time.perf_counter_ns()

            kernel_dispatch_start = h2d_end
            runner.launch("main", nonblock=False)
            kernel_dispatch_end = time.perf_counter_ns()

            primary_d2h_start = kernel_dispatch_end
            runner.memcpy_d2h(
                c_raw,
                sym_c,
                0,
                0,
                p,
                p,
                mt * nt,
                streaming=False,
                data_type=memcpy_dtype,
                order=memcpy_order,
                nonblock=False,
            )
            primary_d2h_end = time.perf_counter_ns()

            # The first C read is also the synchronization point for the
            # asynchronously launched PE tasks. Repeat the now-ready C read so
            # D2H transfer time is not conflated with outstanding kernel work.
            clean_d2h_start = time.perf_counter_ns()
            runner.memcpy_d2h(
                c_raw,
                sym_c,
                0,
                0,
                p,
                p,
                mt * nt,
                streaming=False,
                data_type=memcpy_dtype,
                order=memcpy_order,
                nonblock=False,
            )
            clean_d2h_end = time.perf_counter_ns()

            timer_readback_start = time.perf_counter_ns()
            runner.memcpy_d2h(
                timer_raw,
                sym_timer,
                0,
                0,
                p,
                p,
                3,
                streaming=False,
                data_type=MemcpyDataType.MEMCPY_32BIT,
                order=memcpy_order,
                nonblock=False,
            )
            cycle_summary = decode_cycles(timer_raw.reshape(p, p, 3))
            timer_readback_end = time.perf_counter_ns()

            measured = iteration >= args.warmup
            sample_index = iteration - args.warmup
            if measured:
                postprocess_start = time.perf_counter_ns()
                c = untile_matrix(
                    decode_memcpy_output(c_raw, args.precision),
                    p,
                    mt,
                    nt,
                )
                validation_s = 0.0
                if validation is None and not args.skip_validation:
                    validation_start = time.perf_counter_ns()
                    validation = validation_metrics(a, b, c, args.precision)
                    validation_s = (time.perf_counter_ns() - validation_start) * 1e-9
                postprocess_end = time.perf_counter_ns()

                host_h2d_s = (h2d_end - h2d_start) * 1e-9
                host_kernel_dispatch_s = (kernel_dispatch_end - kernel_dispatch_start) * 1e-9
                host_wait_and_d2h_s = (primary_d2h_end - primary_d2h_start) * 1e-9
                host_d2h_s = (clean_d2h_end - clean_d2h_start) * 1e-9
                operational_e2e_s = (primary_d2h_end - e2e_start) * 1e-9
                device_kernel_s = cycle_summary["maximum"] / (args.clock_mhz * 1e6)
                component_sum_s = host_h2d_s + device_kernel_s + host_d2h_s
                record = {
                    **common,
                    "sample_index": sample_index,
                    "timing": {
                        "host_h2d_s": host_h2d_s,
                        "host_kernel_dispatch_s": host_kernel_dispatch_s,
                        "host_post_dispatch_wait_and_d2h_s": host_wait_and_d2h_s,
                        "host_d2h_s": host_d2h_s,
                        "component_sum_s": component_sum_s,
                        "operational_e2e_s": operational_e2e_s,
                        "device_kernel_s": device_kernel_s,
                        "device_kernel_cycles": cycle_summary,
                    },
                    "excluded_timing": {
                        "host_data_generation_s": (generation_end - generation_start) * 1e-9,
                        "host_tiling_and_packing_s": (packing_end - packing_start) * 1e-9,
                        "runtime_startup_s": runtime_startup_s,
                        "device_reset_host_call_s": (reset_end - reset_start) * 1e-9,
                        "timer_readback_and_decode_s": (timer_readback_end - timer_readback_start)
                        * 1e-9,
                        "host_untiling_and_validation_s": (postprocess_end - postprocess_start)
                        * 1e-9,
                        "host_validation_s": validation_s,
                    },
                    "throughput": {
                        "device_kernel_flops": shape.flops / device_kernel_s,
                        "operational_e2e_flops": shape.flops / operational_e2e_s,
                    },
                    "validation": validation,
                }
                append_jsonl(args.results, record)
                print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
        runtime_shutdown_start = time.perf_counter_ns()

    run_summary = {
        "schema_version": 2,
        "kind": "gemm_run_summary",
        "run_id": run_id,
        "finished_at": utc_now_iso(),
        "precision": args.precision,
        "shape": {"n": shape.n, "k": shape.k, "m": shape.m},
        "simulator": args.simulator,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "runtime_startup_s": runtime_startup_s,
        "runtime_shutdown_s": (time.perf_counter_ns() - runtime_shutdown_start) * 1e-9,
    }
    append_jsonl(args.results, run_summary)


if __name__ == "__main__":
    main()
