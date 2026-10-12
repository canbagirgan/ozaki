#!/usr/bin/env python3
"""Compile fixed-64x64, 16-channel transfer benchmark artifacts."""

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

from ozaki_roofline.records import utc_now_iso  # noqa: E402

FABRIC_DIMS = "762,1172"
FABRIC_OFFSETS = "4,1"
CHANNELS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=("fp16", "fp32"), required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    name = f"bandwidth-p64-{args.precision}-ch16"
    artifact_root = ROOT / "artifacts" / "bandwidth" / name
    manifest_path = ROOT / "artifacts" / "bandwidth" / f"{name}.json"
    if manifest_path.exists() and not args.force:
        print(manifest_path)
        return
    artifact_root.mkdir(parents=True, exist_ok=True)

    layout = "layout_16.csl" if args.precision == "fp16" else "layout_32.csl"
    options = " ".join(
        [
            "--arch=wse3",
            f"--fabric-dims={FABRIC_DIMS}",
            f"--fabric-offsets={FABRIC_OFFSETS}",
            "--memcpy",
            f"--channels={CHANNELS}",
            "--fp16-format=f16",
            "-o=out",
        ]
    )
    compile_start = time.perf_counter()
    with SdkCompiler(disable_version_check=True) as compiler:
        artifact_path = compiler.compile(
            str(Path(__file__).resolve().parent / "src"),
            layout,
            options,
            str(artifact_root),
        )
    compile_client_wall_s = time.perf_counter() - compile_start

    manifest = {
        "schema_version": 1,
        "kind": "bandwidth_compile",
        "created_at": utc_now_iso(),
        "name": name,
        "precision": args.precision,
        "grid": {"width": 64, "height": 64},
        "channels": CHANNELS,
        "capacity_elements_per_pe": 8192 if args.precision == "fp16" else 4096,
        "fabric_dims": FABRIC_DIMS,
        "fabric_offsets": FABRIC_OFFSETS,
        "layout": layout,
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
