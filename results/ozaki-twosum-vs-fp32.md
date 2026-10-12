# Ozaki TwoSum vs MeshGEMM FP32

P=64, N=K=M=2560, seed 7. Same float32 A and B bits. Error is relative Frobenius against `C_ref = float64(A) @ float64(B)`. Kernel times are start-barrier `device_kernel_s`.

- **P** = 64
- **N = K = M** = 2560
- **seed** 7
- **start-barrier** on (on-wafer kernel TSC)

## Headline

TwoSum did **not** beat native MeshGEMM FP32. Covering is 2×fp16 (22 bits of mantissa) versus float32’s 24 bits. TwoSum improved fused Ozaki (1.98e-6 → 1.231e-6) but stayed worse than MeshGEMM FP32 (9.06e-7). The TwoSum kernel is 131.46 ms versus 6.15 ms, about **21× slower**.

| Kernel | Rel. Frobenius | Start-barrier kernel |
| --- | ---: | ---: |
| MeshGEMM FP32 `@fmacs` | 9.06e-7 | 6.15 ms |
| Ozaki TwoSum, 4-product | 1.231e-6 | 131.46 ms |
| Ozaki fused, single fp32 C | 1.98e-6 | — |

## Relative Frobenius vs float64 matmul

Lower is better. Same A and B bits. TwoSum reconstructs `C_hi + C_lo` as f64 on the host.

| Kernel | Rel. Frobenius | Note |
| --- | ---: | --- |
| MeshGEMM FP32 | 9.06e-7 | Native `@fmacs` |
| Ozaki TwoSum, 4-product | 1.231e-6 | A0B0 → A0B1 → A1B0 → A1B1; Knuth TwoSum into C_hi / C_lo |
| Ozaki fused, single C | 1.98e-6 | Four products, one fp32 C |

## Start-barrier kernel time

Fresh paired run. Both kernels used the fabric start barrier, then max PE TSC at 1.4 GHz.

| Kernel | Rel. Frobenius | Start-barrier kernel | vs FP32 |
| --- | ---: | ---: | ---: |
| MeshGEMM FP32 | 9.06e-7 | 6.15 ms | 1× |
| Ozaki TwoSum, 4-product | 1.231e-6 | 131.46 ms | ~21× slower |

## Why TwoSum loses

Two times fp16 covering is 22 bits, not 24, so TwoSum cannot recover the bits the split already dropped. Extra Knuth TwoSum work (four products into `C_hi` / `C_lo`) costs ~21× kernel time and still does not match native 24-bit FP32.

## Three-slice follow-up (P=64, N=K=M=2048, T=32)

One start-barrier launch, nine products, Knuth TwoSum, seed 7. SRAM is 33536 bytes/PE (`32 T² + 24 T`). Relative Frobenius vs `float64(A) @ float64(B)` is **1.102e-6**; vs host SGEMM (`against_numpy_float32_gemm`) **1.147e-6**. Kernel `device_kernel_s` is **170.09 ms**. That does **not** beat the ~9e-7 MeshGEMM FP32 class. At `1/sqrt(K)` Gaussian scale the leftover after two fp16 words is below the fp16 min subnormal, so **A2 and B2 were identically zero** (`extra_slice_zero_fraction = 1`). Three-slice covering could not add bits the two-slice split had already flushed; the FP32 gap is not fixed by a third fp16 word on this distribution. T=40 / N=2560 is forbidden for this kernel (52160 bytes/PE).
