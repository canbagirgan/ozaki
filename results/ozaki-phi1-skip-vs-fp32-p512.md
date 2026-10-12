# Ozaki 2-slice skip vs native MeshGEMM FP32, P=512, ozIMMU φ=1

Square `N=K=M`, seed 7. Skip is fused 2-slice Ozaki, `n_products=3` (A0B0 + A0B1 + A1B0; skip A1B1). TwoSum is the same skip plus Knuth TwoSum into `(C_hi, C_lo)`, host `float64(hi)+float64(lo)`. Rel Frob is vs `float64(A) @ float64(B)`. Kernel times are start-barrier max-PE TSC at 1.4 GHz (`device_kernel_s`). `T = N/512`.

| | | FP32 (baseline) | | 2-slice Ozaki | | | | 2-slice Ozaki + TwoSum | | | |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| N | T | Accuracy | Time | Accuracy | Time | Acc / FP32 | × faster | Accuracy | Time | × better | Time / FP32 |
| 2048 | 4 | 8.10e-7 | 0.181 ms | 1.35e-6 | 0.292 ms | 1.67× | 1.61× | 7.98e-8 | 4.66 ms | 10.2× | 25.7× |
| 8192 | 16 | 1.62e-6 | 3.54 ms | 2.64e-6 | 3.40 ms | 1.63× | 1.04× | 8.37e-8 | 89.4 ms | 19.4× | 25.3× |
| 16384 | 32 | 2.29e-6 | 25.5 ms | 3.69e-6 | 21.3 ms | 1.61× | 1.20× | 7.98e-8 | 455 ms | 28.7× | 17.8× |
| 20480 | 40 | 2.56e-6 | 49.0 ms | 4.10e-6 | 39.7 ms | 1.60× | 1.23× | 8.37e-8 | 784 ms | 30.6× | 16.0× |

Skip Acc / FP32 = Rel / Rel_FP32 (`>1` worse). Skip × faster = t_FP32 / t. TwoSum × better = Rel_FP32 / Rel. TwoSum Time / FP32 = t / t_FP32. N=22528 dropped.

JSONL: `results/raw/gemm-mesh-direct.jsonl` (meshgemm, ozIMMU φ=1, P=512); `results/raw/gemm-ozaki-direct.jsonl` (n_products=3 skip, ozIMMU φ=1, P=512).

New run_ids (2026-10-02):

- `meshgemm-direct-ozimmu-phi1-p512-fp32-2048-2048-2048-1790933156773046060` (`device_kernel_s` = 0.00018126357142857143)
- `ozaki-fused-skip-a1b1-ozimmu-phi1-p512-2048-2048-2048-1790933217626731063` (`device_kernel_s` = 0.0002921)
- `meshgemm-direct-ozimmu-phi1-p512-fp32-8192-8192-8192-1790933099079551473` (`device_kernel_s` ≈ 0.00354253)
- `ozaki-fused-skip-a1b1-ozimmu-phi1-p512-8192-8192-8192-1790933045547957289` (`device_kernel_s` ≈ 0.00340362)
- `ozaki-fused-skip-a1b1-twosum-ozimmu-phi1-p512-8192-8192-8192-1790941144516276493` (`device_kernel_s` ≈ 0.08941138)
- `ozaki-fused-skip-a1b1-twosum-ozimmu-phi1-p512-20480-20480-20480-1790942396717922143` (`device_kernel_s` = 0.7837982514285714)
