"""Build compile/launch commands for the CS-3 Ozaki measurements.

Square GEMMs use one local tile ``T = N / P``. Pass ``N`` or ``T``; if both
are set they must agree. ``K`` and ``M`` default to ``N``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from ozaki_roofline.model import parse_grid_size, reject_large_pe_map


@dataclass(frozen=True)
class Method:
    """One runnable kernel variant."""

    family: str
    summary: str
    precision: str = "fp16"
    n_products: int | None = None
    twosum: bool = False
    twosum_period: int = 1
    twosum_a0b0_only: bool = False
    twosum_cross_only: bool = False
    fp64_merge: bool = False
    cross_f16: bool = False


METHODS: dict[str, Method] = {
    "fp32": Method("mesh", "Native MeshGEMM FP32 (@fmacs).", precision="fp32"),
    "fp16": Method("mesh", "Native MeshGEMM FP16 (@fmach).", precision="fp16"),
    "a0b0": Method("ozaki", "One product, A0B0.", n_products=1),
    "skip-a1b1": Method(
        "ozaki",
        "2-slice triangle: A0B0+A0B1+A1B0, skip A1B1.",
        n_products=3,
    ),
    "twosum": Method(
        "ozaki",
        "skip-a1b1 plus Knuth TwoSum on every product.",
        n_products=3,
        twosum=True,
    ),
    "twosum-every5": Method(
        "ozaki",
        "skip-a1b1, TwoSum after every 5 products.",
        n_products=3,
        twosum=True,
        twosum_period=5,
    ),
    "twosum-every10": Method(
        "ozaki",
        "skip-a1b1, TwoSum after every 10 products.",
        n_products=3,
        twosum=True,
        twosum_period=10,
    ),
    "twosum-every32": Method(
        "ozaki",
        "skip-a1b1, TwoSum after every 32 products.",
        n_products=3,
        twosum=True,
        twosum_period=32,
    ),
    "twosum-every128": Method(
        "ozaki",
        "skip-a1b1, TwoSum after every 128 products.",
        n_products=3,
        twosum=True,
        twosum_period=128,
    ),
    "twosum-a0b0": Method(
        "ozaki",
        "skip-a1b1, TwoSum only on A0B0.",
        n_products=3,
        twosum=True,
        twosum_a0b0_only=True,
    ),
    "twosum-cross": Method(
        "ozaki",
        "skip-a1b1, TwoSum only on A0B1 and A1B0.",
        n_products=3,
        twosum=True,
        twosum_cross_only=True,
    ),
    "twosum-fp64": Method(
        "ozaki",
        "skip-a1b1 TwoSum, IEEE fp64 merge on the wafer.",
        n_products=3,
        twosum=True,
        fp64_merge=True,
    ),
    "cross-f16": Method(
        "ozaki",
        "skip-a1b1 with A0B1/A1B0 accumulated in fp16.",
        n_products=3,
        cross_f16=True,
    ),
    "two-slice": Method("ozaki", "2-slice Ozaki, four products.", n_products=4),
    "twosum-two-slice": Method(
        "ozaki",
        "2-slice Ozaki with Knuth TwoSum.",
        n_products=4,
        twosum=True,
    ),
    "3slice-tri": Method(
        "ozaki",
        "3-slice triangle, i+j<3 (6 products).",
        n_products=6,
    ),
    "3slice": Method("ozaki", "3-slice Ozaki, 9 products.", n_products=9),
    "4slice-tri": Method(
        "ozaki",
        "4-slice triangle i+j<4 (10 products) with Knuth TwoSum.",
        n_products=10,
    ),
    "fp16x5": Method(
        "ozaki",
        "5-slice weight-split, 15 products, dual fp32 C, no per-FMA TwoSum.",
        n_products=15,
    ),
    "4slice": Method("ozaki", "4-slice Ozaki, 16 products.", n_products=16),
    "fp32-ozaki": Method(
        "ozaki",
        "Classic 2×fp32 Ozaki of float64 inputs.",
        precision="fp32",
        n_products=4,
    ),
    "int": Method("int", "Integer-slice Ozaki (shared exponent, i16 products)."),
    "u16x2": Method("u16", "Packed uint16×uint16 Ozaki."),
    "host-skip": Method(
        "host",
        "Host numeric check of skip-a1b1 and TwoSum periods. No wafer.",
    ),
}


def resolve_shape(
    grid_size: int | None,
    n: int | None,
    k: int | None,
    m: int | None,
    tile: int | None,
    *,
    grid_required: bool,
) -> tuple[int | None, int, int, int, int | None]:
    """Return ``(P, N, K, M, T)`` with ``T = N/P`` when a grid is set."""

    if grid_required and grid_size is None:
        raise ValueError("pass --grid-size")
    p = parse_grid_size(grid_size) if grid_size is not None else None
    if n is None and tile is None:
        raise ValueError("pass --n or --tile")
    if p is None and n is None:
        raise ValueError("host-skip needs --n, or both --grid-size and --tile")
    if n is None:
        assert p is not None and tile is not None
        n = tile * p
    if p is not None and tile is not None and n != tile * p:
        raise ValueError(f"N={n} does not equal T*P={tile}*{p}={tile * p}")
    if p is not None and n % p != 0:
        raise ValueError(f"N={n} is not a multiple of grid size {p}")
    if k is None:
        k = n
    if m is None:
        m = n
    derived: int | None = None
    if p is not None:
        if k % p != 0 or m % p != 0:
            raise ValueError(f"K={k} and M={m} must be multiples of grid size {p}")
        derived = n // p
        if tile is not None and (k // p != tile or m // p != tile):
            raise ValueError(
                f"T={tile} does not match local tiles "
                f"N/P={n // p}, K/P={k // p}, M/P={m // p}"
            )
    elif tile is not None and (k != n or m != n):
        raise ValueError("without a grid, K and M must equal N")
    return p, n, k, m, derived


def measurement_commands(
    root: Path,
    method_name: str,
    *,
    grid_size: int | None,
    n: int | None,
    k: int | None,
    m: int | None,
    tile: int | None,
    timing: str,
    input_kind: str,
    phi: float,
    warmup: int,
    repetitions: int,
    clock_mhz: float,
    data_budget_kib: int,
    simulator: bool,
    dump_pe_map: bool,
    force: bool,
    skip_validation: bool,
    high_precision: bool,
    validate_subsample: int,
    alpha: int,
    n_slices: int,
    single_product: bool,
    seed: int,
    stage: str,
) -> list[list[str]]:
    """Argv lists for the requested stage. Does not run them."""

    if method_name not in METHODS:
        known = ", ".join(METHODS)
        raise ValueError(f"unknown method {method_name!r}; choose from {known}")
    if timing not in {"kernel", "e2e"}:
        raise ValueError("timing must be 'kernel' or 'e2e'")
    if stage not in {"compile", "launch", "both"}:
        raise ValueError("stage must be compile, launch, or both")
    method = METHODS[method_name]
    if simulator and timing == "kernel":
        raise ValueError(
            "--simulator is a barrier-free simfabric run; use --timing e2e"
        )
    if simulator and method.family != "ozaki":
        raise ValueError("--simulator is implemented for the fused Ozaki launcher")
    if method.family not in {"ozaki", "mesh"} and input_kind != "ozimmu":
        raise ValueError(
            f"--input-kind is for fused Ozaki and MeshGEMM, not {method_name}"
        )

    grid_required = method.family != "host"
    p, n_dim, k_dim, m_dim, _tile = resolve_shape(
        grid_size, n, k, m, tile, grid_required=grid_required
    )
    if dump_pe_map:
        if p is None:
            raise ValueError("--dump-pe-map needs a grid")
        reject_large_pe_map(p, True)

    py = sys.executable
    if method.family == "host":
        if stage == "compile":
            raise ValueError("host-skip has no compile step")
        return [
            [
                py,
                str(root / "scripts" / "host_skip_twosum_sim.py"),
                "--n",
                str(n_dim),
                "--k",
                str(k_dim),
                "--m",
                str(m_dim),
                "--phi",
                str(phi),
                "--seed",
                str(seed),
            ]
        ]

    assert p is not None
    shape = [
        "--n",
        str(n_dim),
        "--k",
        str(k_dim),
        "--m",
        str(m_dim),
        "--grid-size",
        str(p),
        "--data-budget-kib",
        str(data_budget_kib),
    ]
    barrier = ["--start-barrier"] if timing == "kernel" else []
    compile_cmd: list[str]
    launch_cmd: list[str]
    if method.family == "mesh":
        compile_cmd = [
            py,
            str(root / "benchmarks" / "gemm_mesh" / "compile.py"),
            "--precision",
            method.precision,
            *shape,
            *barrier,
        ]
        launch_cmd = [
            py,
            str(root / "benchmarks" / "gemm_mesh" / "launch_direct.py"),
            "--precision",
            method.precision,
            *shape,
            "--warmup",
            str(warmup),
            "--repetitions",
            str(repetitions),
            "--clock-mhz",
            str(clock_mhz),
            "--input-kind",
            input_kind,
            "--phi",
            str(phi),
            *barrier,
        ]
    elif method.family == "ozaki":
        variant = _ozaki_flags(method)
        compile_cmd = [
            py,
            str(root / "benchmarks" / "gemm_ozaki" / "compile.py"),
            "--precision",
            method.precision,
            *shape,
            *variant,
            *barrier,
        ]
        launch_cmd = [
            py,
            str(root / "benchmarks" / "gemm_ozaki" / "launch_direct.py"),
            "--precision",
            method.precision,
            *shape,
            *variant,
            "--warmup",
            str(warmup),
            "--repetitions",
            str(repetitions),
            "--clock-mhz",
            str(clock_mhz),
            "--input-kind",
            input_kind,
            "--phi",
            str(phi),
            *barrier,
        ]
        if high_precision:
            launch_cmd.append("--high-precision")
        if validate_subsample:
            launch_cmd.extend(["--validate-subsample", str(validate_subsample)])
        if simulator:
            launch_cmd.append("--simulator")
    elif method.family == "int":
        compile_cmd = [py, str(root / "benchmarks" / "gemm_ozaki" / "compile_int.py"), *shape, *barrier]
        launch_cmd = [
            py,
            str(root / "benchmarks" / "gemm_ozaki" / "launch_int.py"),
            *shape,
            "--alpha",
            str(alpha),
            "--n-slices",
            str(n_slices),
            "--warmup",
            str(warmup),
            "--repetitions",
            str(repetitions),
            "--clock-mhz",
            str(clock_mhz),
            *barrier,
        ]
        if single_product:
            launch_cmd.append("--single-product")
    else:
        compile_cmd = [
            py,
            str(root / "benchmarks" / "gemm_ozaki" / "compile_u16x2.py"),
            *shape,
            *barrier,
        ]
        launch_cmd = [
            py,
            str(root / "benchmarks" / "gemm_ozaki" / "launch_u16x2.py"),
            *shape,
            "--warmup",
            str(warmup),
            "--repetitions",
            str(repetitions),
            "--clock-mhz",
            str(clock_mhz),
            *barrier,
        ]
        if single_product:
            launch_cmd.append("--single-product")

    if force:
        compile_cmd.append("--force")
    if dump_pe_map:
        launch_cmd.append("--dump-pe-map")
    if skip_validation and method.family in {"mesh", "ozaki"}:
        launch_cmd.append("--skip-validation")
    if stage == "compile":
        return [compile_cmd]
    if stage == "launch":
        return [launch_cmd]
    return [compile_cmd, launch_cmd]


def _ozaki_flags(method: Method) -> list[str]:
    assert method.n_products is not None
    flags = ["--n-products", str(method.n_products)]
    if method.twosum:
        flags.append("--twosum")
    if method.twosum_period != 1:
        flags.extend(["--twosum-period", str(method.twosum_period)])
    if method.twosum_a0b0_only:
        flags.append("--twosum-a0b0-only")
    if method.twosum_cross_only:
        flags.append("--twosum-cross-only")
    if method.fp64_merge:
        flags.append("--fp64-merge")
    if method.cross_f16:
        flags.append("--cross-f16")
    return flags
