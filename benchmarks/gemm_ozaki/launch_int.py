#!/usr/bin/env python3
"""Launch integer-slice Ozaki MeshGEMM on the worker (one CS-3 job)."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import socket
import sys
import time
from pathlib import Path

from cerebras.sdk.client import SdkLauncher

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ozaki_roofline.model import (  # noqa: E402
    GRID_SIZE_HELP,
    NOMINAL_CLOCK_MHZ,
    PRECISIONS,
    GemmShape,
    Grid,
    parse_grid_size,
    reject_large_pe_map,
    validate_summa_configuration,
)
from ozaki_roofline.records import append_jsonl  # noqa: E402

RECORD_PREFIX = "OZAKI_INT_RECORD "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--grid-size", type=parse_grid_size, default=64, help=GRID_SIZE_HELP)
    parser.add_argument("--data-budget-kib", type=int, default=40)
    parser.add_argument("--alpha", type=int, default=7)
    parser.add_argument("--n-slices", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--clock-mhz", type=float, default=NOMINAL_CLOCK_MHZ)
    parser.add_argument(
        "--start-barrier",
        action="store_true",
        help=(
            "Launch the start-barrier artifact used for on-wafer kernel "
            "timing. Never use for E2E."
        ),
    )
    parser.add_argument(
        "--dump-pe-map",
        action="store_true",
        help="Ask the worker to emit the PxP TSC duration grid.",
    )
    parser.add_argument(
        "--single-product",
        action="store_true",
        help="Time one integer GEMM (A0@B0) only. Used for the start-barrier run.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results" / "raw" / "gemm-ozaki-int-direct.jsonl",
    )
    return parser.parse_args()


def config_name(shape: GemmShape, grid_size: int, start_barrier: bool = False) -> str:
    name = f"ozaki-int-p{grid_size}-n{shape.n}-k{shape.k}-m{shape.m}"
    if start_barrier:
        name += "-start-barrier"
    return name


def integer_buffer_bytes(mt: int, kt: int, nt: int) -> int:
    return 4 * mt * kt + 4 * kt * nt + 4 * mt * nt


def load_manifest(shape: GemmShape, grid_size: int, start_barrier: bool) -> dict:
    path = ROOT / "artifacts" / "gemm" / f"{config_name(shape, grid_size, start_barrier)}.json"
    if not path.exists():
        raise FileNotFoundError(f"compile manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    reject_large_pe_map(args.grid_size, args.dump_pe_map)
    shape = GemmShape(args.n, args.k, args.m)
    grid = Grid(args.grid_size, args.grid_size)
    mt, kt, nt = validate_summa_configuration(
        shape,
        PRECISIONS["fp16"],
        grid,
        data_budget_bytes=args.data_budget_kib * 1024,
    )
    fused_bytes = integer_buffer_bytes(mt, kt, nt)
    budget = args.data_budget_kib * 1024
    if fused_bytes > budget:
        raise ValueError(
            f"integer MeshGEMM buffers need {fused_bytes} bytes/PE, above the "
            f"{budget}-byte data budget"
        )
    manifest = load_manifest(shape, args.grid_size, args.start_barrier)
    if bool(manifest.get("start_barrier")) != bool(args.start_barrier):
        raise RuntimeError(
            "compile artifact start_barrier="
            f"{manifest.get('start_barrier')} does not match launch "
            f"--start-barrier={args.start_barrier}; E2E must use a "
            "no-barrier compile"
        )
    here = Path(__file__).resolve().parent
    mesh = ROOT / "benchmarks" / "gemm_mesh"
    command_parts = [
        "cs_python",
        "run_int.py",
        "--name",
        "out",
        "--cmaddr",
        "%CMADDR%",
        "--n",
        str(shape.n),
        "--k",
        str(shape.k),
        "--m",
        str(shape.m),
        "--grid-size",
        str(args.grid_size),
        "--warmup",
        str(args.warmup),
        "--repetitions",
        str(args.repetitions),
        "--clock-mhz",
        str(args.clock_mhz),
        "--alpha",
        str(args.alpha),
        "--n-slices",
        str(args.n_slices),
    ]
    if args.dump_pe_map:
        command_parts.append("--dump-pe-map")
    if args.single_product:
        command_parts.append("--single-product")
    command = shlex.join(command_parts)

    launcher_start = time.perf_counter_ns()
    with SdkLauncher(
        manifest["artifact_path"],
        simulator=False,
        disable_version_check=True,
    ) as launcher:
        launcher_startup_s = (time.perf_counter_ns() - launcher_start) * 1e-9
        launcher.stage(str(here / "run_int.py"))
        launcher.stage(str(mesh / "host_common.py"))
        launcher.stage(str(ROOT / "src" / "ozaki_roofline" / "validation.py"))
        launcher.stage(str(ROOT / "src" / "ozaki_roofline" / "integer_split.py"))
        run_start = time.perf_counter_ns()
        response = launcher.run(command)
        launcher_run_s = (time.perf_counter_ns() - run_start) * 1e-9

    parsed = 0
    for line in response.splitlines():
        if not line.startswith(RECORD_PREFIX):
            continue
        record = json.loads(line.removeprefix(RECORD_PREFIX))
        record["start_barrier"] = bool(args.start_barrier)
        record["launcher"] = {
            "submission_hostname": socket.gethostname(),
            "submission_python": platform.python_version(),
            "launcher_startup_s": launcher_startup_s,
            "launcher_run_s": launcher_run_s,
            "execution_mode": "sdk_launcher_worker_direct",
        }
        append_jsonl(args.results, record)
        print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
        parsed += 1
    if parsed == 0:
        raise RuntimeError(
            "integer Ozaki worker emitted no OZAKI_INT_RECORD lines:\n" + response
        )


if __name__ == "__main__":
    main()
