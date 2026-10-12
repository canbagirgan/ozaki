#!/usr/bin/env python3
"""Reproducible orchestration for square-grid CS-3 experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "compile",
            "simulator-smoke",
            "wafer-smoke",
            "wafer-full",
            "wafer-direct-smoke",
            "wafer-direct-full",
        ),
    )
    parser.add_argument(
        "--precision",
        choices=("fp16", "fp32", "both"),
        default="both",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "p64.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-bandwidth", action="store_true")
    return parser.parse_args()


def selected_precisions(value: str) -> tuple[str, ...]:
    return ("fp16", "fp32") if value == "both" else (value,)


def run(command: Iterable[object], *, dry_run: bool) -> None:
    rendered = [str(part) for part in command]
    print("+", " ".join(rendered), flush=True)
    if not dry_run:
        subprocess.run(rendered, cwd=ROOT, check=True)


def gemm_command(
    python: str,
    action: str,
    precision: str,
    shape: dict,
    *,
    grid_size: int,
    results: Path | None = None,
    simulator: bool = False,
    warmup: int = 1,
    repetitions: int = 5,
    clock_mhz: float = 1400.0,
) -> list[object]:
    command: list[object] = [
        python,
        f"benchmarks/gemm_summa/{action}.py",
        "--precision",
        precision,
        "--n",
        shape["n"],
        "--k",
        shape["k"],
        "--m",
        shape["m"],
        "--grid-size",
        grid_size,
    ]
    if action == "run":
        command.extend(
            [
                "--warmup",
                warmup,
                "--repetitions",
                repetitions,
                "--clock-mhz",
                clock_mhz,
            ]
        )
        if simulator:
            command.append("--simulator")
            command.extend(
                [
                    "--results",
                    ROOT / "results" / "raw" / "gemm-simulator.jsonl",
                ]
            )
        elif results is not None:
            command.extend(["--results", results])
    return command


def direct_gemm_command(
    python: str,
    precision: str,
    shape: dict,
    *,
    grid_size: int,
    results: Path,
    warmup: int,
    repetitions: int,
    clock_mhz: float,
) -> list[object]:
    return [
        python,
        "benchmarks/gemm_summa/launch_direct.py",
        "--precision",
        precision,
        "--n",
        shape["n"],
        "--k",
        shape["k"],
        "--m",
        shape["m"],
        "--grid-size",
        grid_size,
        "--warmup",
        warmup,
        "--repetitions",
        repetitions,
        "--clock-mhz",
        clock_mhz,
        "--results",
        results,
    ]


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    grid = config["grid"]
    if (
        grid["width"] != grid["height"]
        or grid["width"] not in (64, 128)
        or config["channels"] != 16
    ):
        raise ValueError("the suite supports square 64 or 128 grids with 16 channels")
    grid_size = int(grid["width"])
    if grid_size != 64 and not args.skip_bandwidth:
        raise ValueError("128x128 runs currently require --skip-bandwidth")
    suffix = "" if grid_size == 64 else f"-p{grid_size}"
    gemm_results = ROOT / "results" / "raw" / f"gemm{suffix}.jsonl"
    direct_gemm_results = ROOT / "results" / "raw" / f"gemm-direct{suffix}.jsonl"

    python = sys.executable
    precisions = selected_precisions(args.precision)
    shapes = config["gemm"]["shapes"]
    clock_mhz = float(config["clock_mhz"])

    if args.stage == "compile":
        for precision in precisions:
            if not args.skip_bandwidth:
                run(
                    [python, "benchmarks/bandwidth/compile.py", "--precision", precision],
                    dry_run=args.dry_run,
                )
            for shape in shapes:
                run(
                    gemm_command(
                        python,
                        "compile",
                        precision,
                        shape,
                        grid_size=grid_size,
                    ),
                    dry_run=args.dry_run,
                )
        return

    if args.stage == "simulator-smoke":
        for precision in precisions:
            run(
                gemm_command(
                    python,
                    "run",
                    precision,
                    shapes[0],
                    grid_size=grid_size,
                    simulator=True,
                    warmup=0,
                    repetitions=1,
                    clock_mhz=clock_mhz,
                ),
                dry_run=args.dry_run,
            )
        return

    if args.stage == "wafer-smoke":
        for precision in precisions:
            if not args.skip_bandwidth:
                run(
                    [
                        python,
                        "benchmarks/bandwidth/run.py",
                        "--precision",
                        precision,
                        "--direction",
                        "both",
                        "--lengths",
                        "16",
                        "--warmup",
                        "0",
                        "--repetitions",
                        "1",
                        "--clock-mhz",
                        clock_mhz,
                    ],
                    dry_run=args.dry_run,
                )
            run(
                gemm_command(
                    python,
                    "run",
                    precision,
                    shapes[0],
                    grid_size=grid_size,
                    results=gemm_results,
                    warmup=0,
                    repetitions=1,
                    clock_mhz=clock_mhz,
                ),
                dry_run=args.dry_run,
            )
        return

    if args.stage == "wafer-direct-smoke":
        for precision in precisions:
            if not args.skip_bandwidth:
                run(
                    [
                        python,
                        "benchmarks/bandwidth/launch_direct.py",
                        "--precision",
                        precision,
                        "--direction",
                        "both",
                        "--lengths",
                        "16",
                        "--warmup",
                        "0",
                        "--repetitions",
                        "1",
                    ],
                    dry_run=args.dry_run,
                )
            run(
                direct_gemm_command(
                    python,
                    precision,
                    shapes[0],
                    grid_size=grid_size,
                    results=direct_gemm_results,
                    warmup=0,
                    repetitions=1,
                    clock_mhz=clock_mhz,
                ),
                dry_run=args.dry_run,
            )
        return

    if args.stage == "wafer-direct-full":
        for precision in precisions:
            lengths = config["bandwidth"][f"{precision}_elements_per_pe"]
            if not args.skip_bandwidth:
                run(
                    [
                        python,
                        "benchmarks/bandwidth/launch_direct.py",
                        "--precision",
                        precision,
                        "--direction",
                        "both",
                        "--lengths",
                        ",".join(str(value) for value in lengths),
                        "--warmup",
                        config["bandwidth"]["warmup"],
                        "--repetitions",
                        config["bandwidth"]["repetitions"],
                    ],
                    dry_run=args.dry_run,
                )
            for shape in shapes:
                run(
                    direct_gemm_command(
                        python,
                        precision,
                        shape,
                        grid_size=grid_size,
                        results=direct_gemm_results,
                        warmup=config["gemm"]["warmup"],
                        repetitions=config["gemm"]["repetitions"],
                        clock_mhz=clock_mhz,
                    ),
                    dry_run=args.dry_run,
                )
        return

    for precision in precisions:
        lengths = config["bandwidth"][f"{precision}_elements_per_pe"]
        if not args.skip_bandwidth:
            run(
                [
                    python,
                    "benchmarks/bandwidth/run.py",
                    "--precision",
                    precision,
                    "--direction",
                    "both",
                    "--lengths",
                    ",".join(str(value) for value in lengths),
                    "--warmup",
                    config["bandwidth"]["warmup"],
                    "--repetitions",
                    config["bandwidth"]["repetitions"],
                    "--clock-mhz",
                    clock_mhz,
                ],
                dry_run=args.dry_run,
            )
        for shape in shapes:
            run(
                gemm_command(
                    python,
                    "run",
                    precision,
                    shape,
                    grid_size=grid_size,
                    results=gemm_results,
                    warmup=config["gemm"]["warmup"],
                    repetitions=config["gemm"]["repetitions"],
                    clock_mhz=clock_mhz,
                ),
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()
