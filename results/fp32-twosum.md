# FP32 two-slice GEMM with periodic TwoSum

SDK 2.10.1 WSE-3 simulator; N=128, K=8, M=8; 4x4 PEs; local Mt=32, Kt=2, Nt=2. ozIMMU phi=1, seed 7; one warmup, two measured repetitions; no start barrier.

**Reference:** original FP64 `A @ B`, before splitting. The device uses two FP32 slices per operand and all four slice products. Its two output words are reconstructed as `float64(res) + float64(res_lo)` before comparison.

| TwoSum period (slice products) | Cycles, repeat 1 | Cycles, repeat 2 | Relative Frobenius error vs FP64 | Max elementwise relative error vs FP64 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 76929 | 76932 | 2.25291948e-08 | 7.11980627e-06 |
| 10 | 56606 | 56604 | 4.62850019e-08 | 1.24528576e-05 |

Multiplications are scalar FP32 operations; bucket additions and Knuth TwoSum use vector operations. Four products count per K position. Counters span mesh steps and partial buckets are flushed at the end. The period is configurable from 1 to 32767. TwoSum compensates accumulation rounding, not scalar multiplication rounding.

Both measured repetitions returned bitwise identical results. The SDK emitted both complete measured samples, then repeated a stopping message during shutdown; processes were terminated after collecting the outputs. These are simulator kernel cycles, excluding host splitting, transfers, and host output reconstruction.

Worker copies enabled grid size 4, disabled simulator tracing, and recorded output hashes and max elementwise relative error. That error is `max(abs(C-Cref)/abs(Cref))`; the JSON also includes `relative_max_error = max(abs(C-Cref))/max(abs(Cref))`.

```sh
python scripts/measure.py --method fp32-ozaki-twosum --twosum-period 10 --grid-size 512 --tile 28 --timing kernel
```

Machine-readable results: [fp32-twosum.json](fp32-twosum.json). No comparisons to other local experiments are retained.
