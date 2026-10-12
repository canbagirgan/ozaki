#!/usr/bin/env python3
"""Launch the direct worker/WSE bandwidth benchmark with SdkLauncher."""

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

from ozaki_roofline.records import append_jsonl  # noqa: E402

RECORD_PREFIX = "OZAKI_RECORD "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=("fp16", "fp32"), required=True)
    parser.add_argument("--direction", choices=("h2d", "d2h", "both"), default="both")
    parser.add_argument("--lengths", required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results" / "raw" / "bandwidth-direct.jsonl",
    )
    return parser.parse_args()


def load_manifest(precision: str) -> dict:
    path = ROOT / "artifacts" / "bandwidth" / f"bandwidth-p64-{precision}-ch16.json"
    if not path.exists():
        raise FileNotFoundError(f"compile manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.precision)
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
            "--direction",
            args.direction,
            "--lengths",
            args.lengths,
            "--warmup",
            str(args.warmup),
            "--repetitions",
            str(args.repetitions),
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
