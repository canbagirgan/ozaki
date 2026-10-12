#!/usr/bin/env python3
"""Compile the WaferLLM MeshGEMM kernel as a separate artifact from SUMMA."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import socket
import sys
import time
from pathlib import Path

from cerebras.sdk.client import SdkCompiler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ozaki_roofline.model import (  # noqa: E402
    CS3_FABRIC_DIMS,
    CS3_FABRIC_OFFSETS,
    GRID_SIZE_HELP,
    PRECISIONS,
    GemmShape,
    Grid,
    parse_grid_size,
    validate_summa_configuration,
)
from ozaki_roofline.records import utc_now_iso  # noqa: E402

FABRIC_DIMS = CS3_FABRIC_DIMS
FABRIC_OFFSETS = CS3_FABRIC_OFFSETS
# Same I/O width as SUMMA. Kernel time is independent of this; E2E is not.
# Start barrier is opt-in (--start-barrier) and must never be used for E2E.
CHANNELS = 16


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
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--start-barrier",
        action="store_true",
        help=(
            "Compile a fabric start barrier before the timed kernel. "
            "On-wafer kernel timing only. Never use for E2E."
        ),
    )
    return parser.parse_args()


def config_name(
    precision: str, shape: GemmShape, grid_size: int, start_barrier: bool = False
) -> str:
    name = f"meshgemm-p{grid_size}-{precision}-n{shape.n}-k{shape.k}-m{shape.m}"
    if start_barrier:
        name += "-start-barrier"
    return name


def main() -> None:
    args = parse_args()
    precision = PRECISIONS[args.precision]
    shape = GemmShape(args.n, args.k, args.m)
    grid = Grid(args.grid_size, args.grid_size)
    mt, kt, nt = validate_summa_configuration(
        shape,
        precision,
        grid,
        data_budget_bytes=args.data_budget_kib * 1024,
    )

    name = config_name(args.precision, shape, args.grid_size, args.start_barrier)
    artifact_root = ROOT / "artifacts" / "gemm" / name
    manifest_path = ROOT / "artifacts" / "gemm" / f"{name}.json"
    if manifest_path.exists() and not args.force:
        print(manifest_path)
        return
    artifact_root.mkdir(parents=True, exist_ok=True)

    source_dir = Path(__file__).resolve().parent / "src"
    layout = "layout.csl" if args.precision == "fp16" else "layout_fp32.csl"
    options = " ".join(
        [
            "--arch=wse3",
            f"--fabric-dims={FABRIC_DIMS}",
            f"--fabric-offsets={FABRIC_OFFSETS}",
            f"--params=P:{args.grid_size},Mt:{mt},Kt:{kt},Nt:{nt},start_barrier:{int(args.start_barrier)}",
            "--memcpy",
            f"--channels={CHANNELS}",
            "--fp16-format=f16",
            "-o=out",
        ]
    )

    compile_start = time.perf_counter()
    with SdkCompiler(disable_version_check=True) as compiler:
        artifact_path = compiler.compile(
            str(source_dir),
            layout,
            options,
            str(artifact_root),
        )
    compile_client_wall_s = time.perf_counter() - compile_start

    manifest = {
        "schema_version": 1,
        "kind": "gemm_compile",
        "algorithm": "meshgemm",
        "source": (
            "WaferLLM MeshGEMM (Apache-2.0)"
            if args.precision == "fp16"
            else "WaferLLM MeshGEMM ported to FP32 @fmacs"
        ),
        "created_at": utc_now_iso(),
        "name": name,
        "precision": args.precision,
        "grid": {"width": grid.width, "height": grid.height},
        "channels": CHANNELS,
        "shape": {"n": shape.n, "k": shape.k, "m": shape.m},
        "local_tile": {"mt": mt, "kt": kt, "nt": nt},
        "fabric_dims": FABRIC_DIMS,
        "fabric_offsets": FABRIC_OFFSETS,
        "layout": layout,
        "start_barrier": bool(args.start_barrier),
        "compiler_options": options,
        "compile_client_wall_s": compile_client_wall_s,
        "artifact_path": artifact_path,
        "client": {
            "hostname": socket.gethostname(),
            "python": platform.python_version(),
            "cerebras_sdk": importlib.metadata.version("cerebras-sdk"),
            "cerebras_appliance": importlib.metadata.version("cerebras-appliance"),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
