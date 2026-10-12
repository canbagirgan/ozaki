#!/usr/bin/env python3
"""Launch MeshGEMM on the appliance worker, separately from SUMMA."""

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

RECORD_PREFIX = "OZAKI_RECORD "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=sorted(PRECISIONS), default="fp16")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--grid-size", type=parse_grid_size, default=64, help=GRID_SIZE_HELP)
    parser.add_argument(
        "--data-budget-kib",
        type=int,
        default=40,
        help="Per-PE matrix-buffer budget in KiB (WSE-3 SRAM is 48 KiB).",
    )
    parser.add_argument(
        "--input-kind",
        choices=("scaled", "full-mantissa", "ozimmu"),
        default="scaled",
        help=(
            "scaled = N(0, 1/sqrt(K)). full-mantissa = random 23-bit fp32 "
            "mantissa, exponent 2^0..2^7 (fp32 only). "
            "ozimmu = Uniform(-0.5, 0.5) * exp(phi * N(0, 1))."
        ),
    )
    parser.add_argument(
        "--phi",
        type=float,
        default=1.0,
        help="ozIMMU exponent-width phi (ignored unless --input-kind ozimmu).",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
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
        help=(
            "Ask the worker to emit the PxP TSC duration grid. Does not "
            "enable the start barrier; pass --start-barrier for on-wafer maps."
        ),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results" / "raw" / "gemm-mesh-direct.jsonl",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip host float64 reference GEMM (infeasible for large N).",
    )
    return parser.parse_args()


def config_name(
    precision: str, shape: GemmShape, grid_size: int, start_barrier: bool = False
) -> str:
    name = f"meshgemm-p{grid_size}-{precision}-n{shape.n}-k{shape.k}-m{shape.m}"
    if start_barrier:
        name += "-start-barrier"
    return name


def load_manifest(
    precision: str, shape: GemmShape, grid_size: int, start_barrier: bool = False
) -> dict:
    path = (
        ROOT
        / "artifacts"
        / "gemm"
        / f"{config_name(precision, shape, grid_size, start_barrier)}.json"
    )
    if not path.exists():
        raise FileNotFoundError(f"compile manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    reject_large_pe_map(args.grid_size, args.dump_pe_map)
    shape = GemmShape(args.n, args.k, args.m)
    grid = Grid(args.grid_size, args.grid_size)
    validate_summa_configuration(
        shape,
        PRECISIONS[args.precision],
        grid,
        data_budget_bytes=args.data_budget_kib * 1024,
    )
    manifest = load_manifest(
        args.precision, shape, args.grid_size, args.start_barrier
    )
    if bool(manifest.get("start_barrier")) != bool(args.start_barrier):
        raise RuntimeError(
            "compile artifact start_barrier="
            f"{manifest.get('start_barrier')} does not match launch "
            f"--start-barrier={args.start_barrier}; E2E must use a "
            "no-barrier compile"
        )
    here = Path(__file__).resolve().parent
    command = shlex.join(
        [
            "cs_python",
            "run_direct.py",
            "--name",
            "out",
            "--cmaddr",
            "%CMADDR%",
            "--precision",
            args.precision,
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
            "--input-kind",
            args.input_kind,
            "--phi",
            str(args.phi),
        ]
    )
    if args.dump_pe_map:
        command = shlex.join(shlex.split(command) + ["--dump-pe-map"])
    if args.skip_validation:
        command = shlex.join(shlex.split(command) + ["--skip-validation"])

    launcher_start = time.perf_counter_ns()
    with SdkLauncher(
        manifest["artifact_path"],
        simulator=False,
        disable_version_check=True,
    ) as launcher:
        launcher_startup_s = (time.perf_counter_ns() - launcher_start) * 1e-9
        launcher.stage(str(here / "run_direct.py"))
        launcher.stage(str(here / "host_common.py"))
        launcher.stage(str(ROOT / "src" / "ozaki_roofline" / "validation.py"))
        launcher.stage(str(ROOT / "src" / "ozaki_roofline" / "split.py"))
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
        raise RuntimeError("MeshGEMM worker emitted no OZAKI_RECORD lines:\n" + response)


if __name__ == "__main__":
    main()
