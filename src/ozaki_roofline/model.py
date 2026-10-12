"""Analytical definitions used by the CS-3 GEMM roofline study.

The module deliberately keeps three quantities separate:

* vendor hardware specifications,
* instruction-derived dense ceilings for the PE rectangle in use, and
* measured host-transfer and kernel rates.

The WSE-3 marketing value of 125 PFLOP/s is a sparse-AI number and is not
used as the dense GEMM ceiling.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class Precision:
    """A pure input/accumulator/output precision."""

    name: str
    bytes_per_element: int
    fmac_lanes_per_pe: int
    cycles_per_vector_fmac: float = 1.0

    def __post_init__(self) -> None:
        if self.bytes_per_element <= 0:
            raise ValueError("bytes_per_element must be positive")
        if self.fmac_lanes_per_pe <= 0:
            raise ValueError("fmac_lanes_per_pe must be positive")
        if self.cycles_per_vector_fmac <= 0:
            raise ValueError("cycles_per_vector_fmac must be positive")


PRECISIONS: Mapping[str, Precision] = {
    # WSE-3 @fmach is SIMD-8.
    "fp16": Precision("fp16", bytes_per_element=2, fmac_lanes_per_pe=8),
    # Pure FP32 @fmacs is scalar. SIMD-4 @fmachs is mixed FP16/FP32 and is
    # intentionally outside this first study. A published WSE-2 FMAC
    # microbenchmark models pure FP32 at two cycles per scalar element; its
    # use here is a provisional WSE-3 assumption, not a vendor FP32 roof.
    "fp32": Precision(
        "fp32",
        bytes_per_element=4,
        fmac_lanes_per_pe=1,
        cycles_per_vector_fmac=2.0,
    ),
}


# TSC → seconds. Not the SDK bandwidth-test 850 MHz (WSE-2) figure.
NOMINAL_CLOCK_MHZ = 1400.0


@dataclass(frozen=True)
class HardwareSpec:
    """Published and literature-derived WSE-3 constants."""

    nominal_frequency_hz: float = NOMINAL_CLOCK_MHZ * 1e6
    advertised_active_pes: int = 900_000
    aggregate_sram_bandwidth_Bps: float = 21e15
    aggregate_fabric_bandwidth_bps: float = 214e15
    advertised_system_io_bps: float = 1.2e12
    advertised_sparse_ai_flops: float = 125e15
    sram_bytes_per_pe: int = 48 * 1024

    @property
    def sram_bandwidth_per_pe_Bps(self) -> float:
        return self.aggregate_sram_bandwidth_Bps / self.advertised_active_pes


WSE3 = HardwareSpec()

# CS-3 wafer used by the compile launchers: --fabric-dims=762,1172
# --fabric-offsets=4,1. memcpy reserves 3 columns west of the kernel (plus
# the extra offset column), 3 east (2+1), 1 row north, and 1 south. See
# benchmarks/vendor/bandwidth-test/run.py. A square kernel therefore needs
# P+7 ≤ 762 and P+2 ≤ 1172, so P ≤ 755. 850 fits the tall axis only.
CS3_FABRIC_WIDTH = 762
CS3_FABRIC_HEIGHT = 1172
CS3_FABRIC_DIMS = f"{CS3_FABRIC_WIDTH},{CS3_FABRIC_HEIGHT}"
CS3_FABRIC_OFFSETS = "4,1"
MEMCPY_WEST_COLUMNS = 4
MEMCPY_EAST_COLUMNS = 3
MEMCPY_NORTH_ROWS = 1
MEMCPY_SOUTH_ROWS = 1
MAX_CS3_SQUARE_P = min(
    CS3_FABRIC_WIDTH - MEMCPY_WEST_COLUMNS - MEMCPY_EAST_COLUMNS,
    CS3_FABRIC_HEIGHT - MEMCPY_NORTH_ROWS - MEMCPY_SOUTH_ROWS,
)  # 755
# Two-hop MeshGEMM uses P/2 steps; keep the max square even.
MAX_EVEN_SQUARE_P = MAX_CS3_SQUARE_P - (MAX_CS3_SQUARE_P % 2)  # 754
# P=850 is requested sometimes but is not a legal CS-3 square (850 > 762).
SQUARE_GRID_WHITELIST = (64, 128, 256, 512, 640, MAX_EVEN_SQUARE_P)
# 16K-class on P=754: T must be even for FP16 u32 packing (T=19 odd fails).
# T=20 → N=15080; T=22 → N=16588. Canonical is T=22 (even tile, nearest 16K).
CANONICAL_P754_TILE = 22
CANONICAL_P754_N = MAX_EVEN_SQUARE_P * CANONICAL_P754_TILE  # 16588
GRID_SIZE_HELP = (
    f"Square PE grid {SQUARE_GRID_WHITELIST}. "
    f"850×850 does not fit CS-3 (762×1172, memcpy max even P={MAX_EVEN_SQUARE_P}); "
    f"canonical 16K-class is N={CANONICAL_P754_N} (T={CANONICAL_P754_TILE})."
)
PE_MAP_FORBIDDEN_MIN_P = 640


def max_cs3_memcpy_extent(p: int) -> tuple[int, int]:
    """Kernel P×P plus memcpy halo, in fabric columns and rows."""

    return (
        p + MEMCPY_WEST_COLUMNS + MEMCPY_EAST_COLUMNS,
        p + MEMCPY_NORTH_ROWS + MEMCPY_SOUTH_ROWS,
    )


def validate_cs3_square_placement(p: int) -> None:
    """Reject a square that cannot sit on CS-3 with the memcpy halo."""

    if p <= 0:
        raise ValueError("grid size must be positive")
    need_w, need_h = max_cs3_memcpy_extent(p)
    if need_w <= CS3_FABRIC_WIDTH and need_h <= CS3_FABRIC_HEIGHT:
        return
    extra = ""
    if p == 850:
        extra = (
            " P=850 would fit only as a non-square height on the tall "
            f"{CS3_FABRIC_HEIGHT} axis "
            f"(850+{MEMCPY_NORTH_ROWS + MEMCPY_SOUTH_ROWS}<={CS3_FABRIC_HEIGHT}), "
            f"not on the {CS3_FABRIC_WIDTH}-wide axis. Largest even square is "
            f"{MAX_EVEN_SQUARE_P} (two-hop P/2={MAX_EVEN_SQUARE_P // 2}). "
            f"Canonical 16K-class size is N=K=M={CANONICAL_P754_N} "
            f"(T={CANONICAL_P754_TILE}, even tile for FP16 packing); "
            f"T=20 → N={MAX_EVEN_SQUARE_P * 20}."
        )
    raise ValueError(
        f"P={p} square does not fit CS-3 fabric "
        f"{CS3_FABRIC_WIDTH}×{CS3_FABRIC_HEIGHT} with memcpy halo "
        f"(need {need_w}×{need_h})."
        + extra
    )


def parse_grid_size(value: str | int) -> int:
    """CLI type for --grid-size. Rejects P=850 with a fabric-fit message."""

    p = int(value)
    if p == 850:
        validate_cs3_square_placement(p)
    if p not in SQUARE_GRID_WHITELIST:
        raise ValueError(
            f"unsupported --grid-size {p}; allowed {SQUARE_GRID_WHITELIST}. "
            f"P=850 is not a legal CS-3 square (max even P={MAX_EVEN_SQUARE_P})."
        )
    return p


def reject_large_pe_map(grid_size: int, dump_pe_map: bool) -> None:
    if dump_pe_map and grid_size >= PE_MAP_FORBIDDEN_MIN_P:
        raise ValueError(
            f"--dump-pe-map is forbidden on P={grid_size} "
            f"(>={PE_MAP_FORBIDDEN_MIN_P})"
        )


@dataclass(frozen=True)
class Grid:
    """Rectangular PE allocation.

    The reference SUMMA implementation requires a square allocation, but the
    generic roof calculations remain valid for a rectangle.
    """

    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("grid dimensions must be positive")

    @property
    def active_pes(self) -> int:
        return self.width * self.height

    @property
    def is_square(self) -> bool:
        return self.width == self.height

    @property
    def summa_steps(self) -> int:
        if not self.is_square:
            raise ValueError("the reference SUMMA kernel requires a square PE grid")
        return self.width


DEFAULT_GRID = Grid(64, 64)


@dataclass(frozen=True)
class GemmShape:
    """Shape of C=A@B using the user's n-by-k and k-by-m notation."""

    n: int
    k: int
    m: int

    def __post_init__(self) -> None:
        if self.n <= 0 or self.k <= 0 or self.m <= 0:
            raise ValueError("all GEMM dimensions must be positive")

    @property
    def flops(self) -> int:
        """Conventional dense GEMM operation count."""

        return 2 * self.n * self.k * self.m

    def h2d_bytes(self, precision: Precision) -> int:
        """Bytes for cold A and B inputs."""

        return precision.bytes_per_element * (self.n * self.k + self.k * self.m)

    def d2h_bytes(self, precision: Precision) -> int:
        """Bytes for the C output."""

        return precision.bytes_per_element * self.n * self.m

    def host_wire_bytes(self, precision: Precision) -> int:
        return self.h2d_bytes(precision) + self.d2h_bytes(precision)

    def host_operational_intensity(self, precision: Precision) -> float:
        """FLOPs per byte crossing the host/WSE boundary once."""

        return self.flops / self.host_wire_bytes(precision)

    def local_tiles(self, grid: Grid = DEFAULT_GRID) -> tuple[int, int, int]:
        """Return (Mt, Kt, Nt) for the square SUMMA decomposition."""

        if not grid.is_square:
            raise ValueError("SUMMA local tiles require a square grid")
        p = grid.width
        if self.n % p or self.k % p or self.m % p:
            raise ValueError(f"n, k, and m must be divisible by P={p}")
        return self.n // p, self.k // p, self.m // p


def dense_compute_ceiling_flops(
    precision: Precision,
    grid: Grid = DEFAULT_GRID,
    *,
    frequency_hz: float = WSE3.nominal_frequency_hz,
) -> float:
    """Instruction-derived dense ceiling for the selected PE rectangle.

    An FMA counts as two FLOPs.
    """

    return (
        grid.active_pes
        * precision.fmac_lanes_per_pe
        * 2
        * frequency_hz
        / precision.cycles_per_vector_fmac
    )


def scaled_sram_bandwidth_Bps(
    grid: Grid = DEFAULT_GRID,
    *,
    hardware: HardwareSpec = WSE3,
) -> float:
    """Scale the published aggregate SRAM bandwidth by active PE count."""

    return grid.active_pes * hardware.sram_bandwidth_per_pe_Bps


def theoretical_device_roof_flops(
    operational_intensity_flops_per_byte: float,
    precision: Precision,
    grid: Grid = DEFAULT_GRID,
    *,
    frequency_hz: float = WSE3.nominal_frequency_hz,
    hardware: HardwareSpec = WSE3,
) -> float:
    """Classic two-roof bound min(compute, SRAM bandwidth * OI)."""

    if operational_intensity_flops_per_byte < 0:
        raise ValueError("operational intensity cannot be negative")
    return min(
        dense_compute_ceiling_flops(precision, grid, frequency_hz=frequency_hz),
        scaled_sram_bandwidth_Bps(grid, hardware=hardware) * operational_intensity_flops_per_byte,
    )


def device_ridge_point_flops_per_byte(
    precision: Precision,
    grid: Grid = DEFAULT_GRID,
    *,
    frequency_hz: float = WSE3.nominal_frequency_hz,
    hardware: HardwareSpec = WSE3,
) -> float:
    return dense_compute_ceiling_flops(
        precision, grid, frequency_hz=frequency_hz
    ) / scaled_sram_bandwidth_Bps(grid, hardware=hardware)


def summa_data_footprint_bytes(
    mt: int,
    kt: int,
    nt: int,
    precision: Precision,
) -> int:
    """Per-PE matrix storage in the reference five-buffer SUMMA kernel.

    Storage comprises A, an A receive buffer, B, a B receive buffer, and C.
    Code, stack, timing words, and memcpy runtime state are not included.
    """

    if min(mt, kt, nt) <= 0:
        raise ValueError("local tile dimensions must be positive")
    elements = 2 * mt * kt + 2 * kt * nt + mt * nt
    return precision.bytes_per_element * elements


def validate_summa_configuration(
    shape: GemmShape,
    precision: Precision,
    grid: Grid = DEFAULT_GRID,
    *,
    data_budget_bytes: int = 40 * 1024,
) -> tuple[int, int, int]:
    """Validate divisibility, FP16 packing, and a conservative SRAM budget."""

    if grid.is_square:
        validate_cs3_square_placement(grid.width)
    mt, kt, nt = shape.local_tiles(grid)
    if precision.name == "fp16" and ((mt * kt) % 2 or (kt * nt) % 2):
        raise ValueError("FP16 collective tile sizes must be even for u32 packing")
    footprint = summa_data_footprint_bytes(mt, kt, nt, precision)
    if footprint > data_budget_bytes:
        raise ValueError(
            f"matrix buffers need {footprint} bytes/PE, above the "
            f"{data_budget_bytes}-byte data budget"
        )
    return mt, kt, nt


def summa_local_fmac_sram_bytes(
    shape: GemmShape,
    precision: Precision,
    grid: Grid = DEFAULT_GRID,
    *,
    include_collective_receiver_writes: bool = False,
) -> int:
    """Approximate implementation-level SRAM traffic per PE.

    Every vector-scalar FMAC reads A and C and writes C (three element
    transfers), plus one scalar B read per DSD invocation. Optionally count one
    receiver write for each A and B tile delivered during every SUMMA step.
    This is not the compulsory minimum traffic and therefore is kept distinct
    from host-wire OI.
    """

    mt, kt, nt = shape.local_tiles(grid)
    steps = grid.summa_steps
    s = precision.bytes_per_element
    fmac_traffic = steps * s * (3 * mt * kt * nt + kt * nt)
    if include_collective_receiver_writes:
        fmac_traffic += steps * s * (mt * kt + kt * nt)
    return fmac_traffic


def summa_implementation_sram_oi(
    shape: GemmShape,
    precision: Precision,
    grid: Grid = DEFAULT_GRID,
    *,
    include_collective_receiver_writes: bool = False,
) -> float:
    local_flops = shape.flops // grid.active_pes
    return local_flops / summa_local_fmac_sram_bytes(
        shape,
        precision,
        grid,
        include_collective_receiver_writes=include_collective_receiver_writes,
    )


def packed_collective_words(elements: int, precision: Precision) -> int:
    """Number of 32-bit wavelets needed to carry a tile."""

    if elements <= 0:
        raise ValueError("elements must be positive")
    return ceil(elements * precision.bytes_per_element / 4)


@dataclass(frozen=True)
class SummaCoefficients:
    """Calibratable WSE-3 SUMMA cycle model.

    Per step:
      a * (Mt*Kt*Nt) + b * (Kt*Nt)
      + c * (packed A words + packed B words) + d

    The terms represent vector FMAC work, per-DSD-invocation overhead,
    collective payload cost, and fixed step latency.
    """

    cycles_per_fmac_element: float
    cycles_per_fmac_call: float
    cycles_per_collective_word: float
    cycles_per_step: float

    def __post_init__(self) -> None:
        if (
            min(
                self.cycles_per_fmac_element,
                self.cycles_per_fmac_call,
                self.cycles_per_collective_word,
                self.cycles_per_step,
            )
            < 0
        ):
            raise ValueError("SUMMA cycle coefficients cannot be negative")

    def predict_cycles(
        self,
        shape: GemmShape,
        precision: Precision,
        grid: Grid = DEFAULT_GRID,
    ) -> float:
        mt, kt, nt = shape.local_tiles(grid)
        a_words = packed_collective_words(mt * kt, precision)
        b_words = packed_collective_words(kt * nt, precision)
        per_step = (
            self.cycles_per_fmac_element * mt * kt * nt
            + self.cycles_per_fmac_call * kt * nt
            + self.cycles_per_collective_word * (a_words + b_words)
            + self.cycles_per_step
        )
        return grid.summa_steps * per_step


@dataclass(frozen=True)
class TransferFit:
    """Affine transfer model t = latency + bytes / bandwidth."""

    latency_s: float
    bandwidth_Bps: float

    def __post_init__(self) -> None:
        if self.latency_s < 0:
            raise ValueError("latency cannot be negative")
        if self.bandwidth_Bps <= 0:
            raise ValueError("bandwidth must be positive")

    def predict_seconds(self, byte_count: int, *, calls: int = 1) -> float:
        if byte_count < 0:
            raise ValueError("byte_count cannot be negative")
        if calls <= 0:
            raise ValueError("calls must be positive")
        return calls * self.latency_s + byte_count / self.bandwidth_Bps


def effective_host_io_bandwidth_Bps(h2d: TransferFit, d2h: TransferFit) -> float:
    """Single host-wire bandwidth for a square cold GEMM.

    A and B are two thirds of the bytes and C is one third, so
    beta = 3 / (2 / beta_H2D + 1 / beta_D2H). The naive roofline then
    uses one bandwidth, as in Williams et al.
    """

    return 3.0 / (2.0 / h2d.bandwidth_Bps + 1.0 / d2h.bandwidth_Bps)


def predict_cold_e2e_seconds(
    shape: GemmShape,
    precision: Precision,
    h2d: TransferFit,
    d2h: TransferFit,
    *,
    kernel_seconds: float,
    aggregate_h2d_calls: bool = True,
) -> float:
    """Predict H2D(A+B) + kernel + D2H(C), excluding setup and packing."""

    if kernel_seconds < 0:
        raise ValueError("kernel_seconds cannot be negative")
    h2d_calls = 1 if aggregate_h2d_calls else 2
    return (
        h2d.predict_seconds(shape.h2d_bytes(precision), calls=h2d_calls)
        + kernel_seconds
        + d2h.predict_seconds(shape.d2h_bytes(precision))
    )


def throughput_flops(flops: int, seconds: float) -> float:
    if flops < 0:
        raise ValueError("flops cannot be negative")
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    return flops / seconds
