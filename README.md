# Ozaki GEMM on Cerebras CS-3

Parameterized dense GEMM measurements on one ALCF CS-3 (WSE-3), SDK 2.10.
The current kernels are fused Ozaki MeshGEMM variants and a native MeshGEMM
FP32/FP16 baseline. Local tile `T = N / P` for a square problem
`N = K = M`.

Recorded wafer runs are in `results/raw/`. Cluster addresses are not in this
tree. Compiled artifacts are not either: `scripts/measure.py` rebuilds them
under `artifacts/`.

## Run one measurement

From this directory, with the Cerebras SDK on `PATH` (`cs_python`, SDK 2.10)
and Python 3.11:

```bash
python scripts/measure.py --list-methods

# On-wafer kernel TSC. P=512, T=32, N=K=M=16384, ozIMMU phi=1.
# This is the 2026-10-05 4-slice triangle + TwoSum run.
python scripts/measure.py \
  --method 4slice-tri --grid-size 512 --tile 32 --timing kernel

# Same shape, native FP32 baseline and the 2-slice skip / TwoSum sweep.
python scripts/measure.py --method fp32 --grid-size 512 --n 16384 --timing kernel
python scripts/measure.py --method skip-a1b1 --grid-size 512 --tile 32 --timing kernel
python scripts/measure.py --method twosum --grid-size 512 --tile 32 --timing kernel
python scripts/measure.py --method twosum-every5 --grid-size 512 --tile 32 --timing kernel
python scripts/measure.py --method twosum-every10 --grid-size 512 --tile 32 --timing kernel
python scripts/measure.py --method twosum-every32 --grid-size 512 --tile 32 --timing kernel
python scripts/measure.py --method twosum-every128 --grid-size 512 --tile 32 --timing kernel

# Cold host-to-device E2E. No fabric start barrier.
python scripts/measure.py --method 4slice-tri --grid-size 512 --tile 32 --timing e2e

# Print the compile and launch commands without submitting a job.
python scripts/measure.py --dry-run \
  --method twosum-fp64 --grid-size 512 --tile 32 --timing kernel
```

`--timing kernel` passes `--start-barrier` to compile and launch. That mode
is only for on-wafer kernel TSC. `--timing e2e` does not. `--dump-pe-map` is
rejected for `P >= 640`. Device time uses max-PE TSC at 1.4 GHz.

`--n` and `--tile` may be passed together only when `N = T * P` (and `K` and
`M` match that tile). `--k` and `--m` default to `N`. `--input-kind` defaults
to `ozimmu` and `--phi` defaults to `1`, which is the October campaign.
`scaled` and `full-mantissa` are the other input generators.

Wafer workers keep seed 7. `--seed` applies to the host check below.

## Methods

| Method | What it runs |
| --- | --- |
| `fp32`, `fp16` | Native MeshGEMM `@fmacs` / `@fmach` |
| `skip-a1b1` | 2-slice triangle, 3 products, skip A1B1 |
| `twosum` | `skip-a1b1` plus Knuth TwoSum on every product |
| `twosum-every5`, `twosum-every10`, `twosum-every32`, `twosum-every128` | TwoSum after that many products |
| `twosum-a0b0`, `twosum-cross` | TwoSum only on A0B0, or only on the cross terms |
| `twosum-fp64` | TwoSum with an on-wafer IEEE fp64 merge |
| `4slice-tri` | 4-slice triangle, 10 products, TwoSum |
| `two-slice`, `twosum-two-slice`, `3slice-tri`, `3slice`, `4slice`, `fp16x5`, `a0b0`, `cross-f16`, `fp32-ozaki` | Other fused Ozaki product sets |
| `int`, `u16x2` | Integer-slice and packed uint16 paths |
| `host-skip` | Host numeric check. No CS-3 and no compile |

```bash
python scripts/measure.py --method host-skip --n 64
```

## Where the October numbers are

- `results/raw/gemm-ozaki-direct.jsonl` — fused Ozaki samples, through 2026-10-05
- `results/raw/gemm-mesh-direct.jsonl` — MeshGEMM FP32/FP16 baseline
- `results/ozaki-phi1-skip-vs-fp32-p512.md` — P=512 ozIMMU φ=1 skip vs FP32
- `results/ozaki-twosum-vs-fp32.md` — P=64 TwoSum vs native FP32
- `results/figures/` — period sweep, error/latency, and roofline figures

Plots that read those JSONL files:

```bash
python scripts/plot_twosum_period_tradeoff.py
python scripts/plot_twosum_period_table.py
python scripts/plot_error_latency_tradeoff_cs3.py
python scripts/plot_fp32_vs_skip_table.py
```

## Earlier SUMMA roofline

`benchmarks/gemm_summa/`, `benchmarks/bandwidth/`, and `scripts/run_suite.py`
are the September 2026 FP16/FP32 SUMMA and host-transfer study. Definitions
are in `docs/methodology.md`. Measured points from that study are in
`docs/results.md`. Vendor files under `benchmarks/vendor/` come from
Cerebras sdk-examples (Apache-2.0); see `benchmarks/NOTICE.md`.

```bash
python scripts/run_suite.py --dry-run wafer-direct-full
```

## Layout

Host-only checks need NumPy. Wafer compile and launch need `cerebras_sdk` and
`cerebras_appliance` 2.10, as installed on the ALCF user node. This repo does
not vendor those wheels.

```bash
PYTHONPATH=src python tests/test_campaign.py
PYTHONPATH=src python scripts/host_skip_twosum_sim.py --n 64 --phi 1 --check
```
