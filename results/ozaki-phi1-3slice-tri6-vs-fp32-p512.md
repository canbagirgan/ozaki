# Ozaki 3-slice triangle (6 products) vs FP32 and 2-slice skip, P=512, ozIMMU φ=1

Square `N=K=M`, seed 7. Triangle is fused 3-slice Ozaki, `n_products=6` (`i+j<3`: A0B0 + A0B1 + A0B2 + A1B0 + A1B1 + A2B0; skip A1B2, A2B1, A2B2). Hops stay packed A0|A1|A2 (1.5× FP32 bytes). Single fp32 C, no TwoSum. Rel Frob is vs `float64(A) @ float64(B)`. Kernel times are start-barrier max-PE TSC at 1.4 GHz (median `device_kernel_s` over 3 reps). `T = N/512`.

Baseline FP32 / skip numbers are from `results/ozaki-phi1-skip-vs-fp32-p512.md`.

| | | FP32 | | 2-slice skip | | 3-slice tri6 | | vs FP32 | | vs skip | |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| N | T | Rel | Time | Rel | Time | Rel | Time | Acc worse | Speed | Acc | Time |
| 2048 | 4 | 8.10e-7 | 0.181 ms | 1.35e-6 | 0.292 ms | 1.35e-6 | 0.511 ms | 1.67× | −182% | 1.00× | 1.75× |
| 8192 | 16 | 1.62e-6 | 3.54 ms | 2.64e-6 | 3.40 ms | 2.64e-6 | 6.70 ms | 1.63× | −89% | 1.00× | 1.97× |
| 16384 | 32 | 2.29e-6 | 25.5 ms | 3.69e-6 | 21.3 ms | 3.69e-6 | 41.9 ms | 1.61× | −64% | 1.00× | 1.97× |

vs FP32 accuracy worse = tri6 Rel / FP32 Rel. vs FP32 speed = (t_FP32 − t_tri6) / t_FP32; negative means slower. vs skip Acc = tri6 Rel / skip Rel. vs skip Time = t_tri6 / t_skip.

On ozIMMU φ=1, device Rel Frob matches 2-slice skip to three figures (A2 is almost all zero). Host six-product oracle vs f64 is ~3.2e-7–3.5e-7, so the extra products exist on the host; the single fp32 C on-wafer stays skip-class. Kernel is ~1.75× (T=4) to ~1.97× (T≥16) skip, and slower than FP32 at all three N.

JSONL: `results/raw/gemm-ozaki-direct.jsonl` (`n_products=6`, ozIMMU φ=1, P=512, start-barrier).

Exact median `device_kernel_s` and Rel Frob:

| N | Rel Frob vs f64 | median `device_kernel_s` | host oracle vs f64 |
| ---: | ---: | ---: | ---: |
| 2048 | 1.349337e-6 | 0.0005107064285714285 | 3.151280e-7 |
| 8192 | 2.643320e-6 | 0.006695872142857143 | 3.322701e-7 |
| 16384 | 3.687040e-6 | 0.04194584071428571 | 3.457230e-7 |

run_ids (2026-10-02):

- `ozaki-fused-3slice-tri-ozimmu-phi1-p512-2048-2048-2048-1790936880845803471`
- `ozaki-fused-3slice-tri-ozimmu-phi1-p512-8192-8192-8192-1790937016197861189`
- `ozaki-fused-3slice-tri-ozimmu-phi1-p512-16384-16384-16384-1790937068759775281`

Compile artifacts: `ozaki-fused-3slice-tri-p512-n{N}-k{N}-m{N}-start-barrier` (`n_products: 6`, `n_slices: 3`, `twosum: false`, `start_barrier: true`).
