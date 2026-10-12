#!/usr/bin/env python3
"""Launch direct worker/WSE SUMMA with SdkLauncher."""

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
    validate_summa_configuration,
)
from ozaki_roofline.records import append_jsonl  # noqa: E402

RECORD_PREFIX = "OZAKI_RECORD "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=sorted(PRECISIONS), required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--grid-size", type=parse_grid_size, default=64, help=GRID_SIZE_HELP)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--clock-mhz", type=float, default=NOMINAL_CLOCK_MHZ)
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results" / "raw" / "gemm-direct.jsonl",
    )
    return parser.parse_args()


def config_name(precision: str, shape: GemmShape, grid_size: int) -> str:
    return f"summa-p{grid_size}-{precision}-n{shape.n}-k{shape.k}-m{shape.m}"


def load_manifest(precision: str, shape: GemmShape, grid_size: int) -> dict:
    path = ROOT / "artifacts" / "gemm" / f"{config_name(precision, shape, grid_size)}.json"
    if not path.exists():
        raise FileNotFoundError(f"compile manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    shape = GemmShape(args.n, args.k, args.m)
    grid = Grid(args.grid_size, args.grid_size)
    validate_summa_configuration(shape, PRECISIONS[args.precision], grid)
    manifest = load_manifest(args.precision, shape, args.grid_size)
    direct_script = Path(__file__).resolve().parent / "run_direct.py"
    command = shlex.join(
        [
            "cs_python",
            direct_script.name,
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
        ]
    )

    launcher_start = time.perf_counter_ns()
    with SdkLauncher(
        manifest["artifact_path"],
        simulator=False,
        disable_version_check=True,
    ) as launcher:
        launcher_startup_s = (time.perf_counter_ns() - launcher_start) * 1e-9
        launcher.stage(str(direct_script))
        launcher.stage(str(ROOT / "src" / "ozaki_roofline" / "validation.py"))
        run_start = time.perf_counter_ns()
        response = launcher.run(command)
        launcher_run_s = (time.perf_counter_ns() - run_start) * 1e-9

    parsed = 0
    for line in response.splitlines():
        if not line.startswith(RECORD_PREFIX):
            continue
        record = json.loads(line.removeprefix(RECORD_PREFIX))
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
        raise RuntimeError(f"direct benchmark returned no records:\n{response}")


if __name__ == "__main__":
    main()
