# CS-3 preliminary roofline and grid-scaling results

Measured 2026-09-02 on an ALCF CS-3/WSE-3 with 16 I/O channels and SDK
2.10 kernels. Device times convert max-PE TSC at **1.4 GHz**, not the SDK
WSE-2 850 MHz example. Host H2D/D2H/E2E times are wall-clock and do not
depend on that assumption. The fixed problem is `n = k = m = 2560`. The
64-by-64 grid uses 40-by-40 local tiles; the 128-by-128 grid uses 20-by-20
local tiles. Each reported primary point is the median of two independent
worker/WSE allocations, with two warmups and nine retained samples per
allocation.

## Host/WSE transfer fits

The payload-size transfer fit was measured on the 64-by-64 rectangle:

- FP16 useful payload: 9.93 GB/s H2D and 1.58 GB/s D2H.
- FP32: 18.53 GB/s H2D and 3.23 GB/s D2H.
- FP16 host-container normalization: 19.86 GB/s H2D and 3.15 GB/s D2H.

The SDK exposes one 32-bit host container per FP16 or FP32 element. The
container-normalized rates show that the apparent useful-payload precision
difference is byte accounting, not a faster FP32 link.

## Primary worker/WSE strong-scaling comparison

For pure FP16:

- 64-by-64: 2.204 ms H2D, 8.645 ms device makespan, 7.643 ms isolated
  D2H, and 28.460 ms cold E2E.
- 128-by-128: 4.798 ms H2D, 5.193 ms device makespan, 6.338 ms isolated
  D2H, and 21.610 ms cold E2E.
- Kernel speedup is 1.67x and E2E speedup is 1.31x. Relative to the ideal
  fourfold speedup from four times as many PEs, efficiencies are 41.8% and
  32.7%.

For pure FP32:

- 64-by-64: 2.598 ms H2D, 13.129 ms device makespan, 7.554 ms isolated
  D2H, and 38.066 ms cold E2E.
- 128-by-128: 4.668 ms H2D, 5.574 ms device makespan, 6.024 ms isolated
  D2H, and 23.564 ms cold E2E.
- Kernel speedup is 2.36x and E2E speedup is 1.62x. Strong-scaling
  efficiencies are 58.9% and 40.4%.

H2D becomes slower on the larger rectangle despite unchanged total matrix
bytes because the same 16 channels distribute smaller fragments over four
times as many PEs. This offsets part of the kernel improvement. SUMMA also
doubles from 64 to 128 stages while per-PE arithmetic falls by four, so a
fourfold kernel speedup is not expected.

## Per-PE device times

The official device result remains the worst PE, because GEMM finishes only
when every PE finishes. Fastest and typical (median) PE times are retained
to show how much of that result is a tail:

| Grid | Precision | Fastest PE | Typical PE | Worst PE (reported) |
| --- | --- | ---: | ---: | ---: |
| 64x64 | FP16 | 3.552 ms | 3.626 ms | 8.645 ms |
| 64x64 | FP32 | 8.744 ms | 8.912 ms | 13.129 ms |
| 128x128 | FP16 | 1.667 ms | 2.341 ms | 5.193 ms |
| 128x128 | FP32 | 2.955 ms | 3.287 ms | 5.574 ms |

On 64x64 the typical PE already separates the precisions (3.6 ms vs 8.9 ms).
On 128x128 the typical PEs remain separated (2.3 ms vs 3.3 ms), but the
slowest-PE tails are closer (5.2 ms vs 5.6 ms). That tail, not equal
arithmetic speed, is why the reported 128x128 kernels look similar.

The arithmetic mean across PEs was stored only for one later 64x64 FP16
allocation: 4.731 ms, between its 3.626 ms median PE and 8.645 ms worst PE.
Newer runs emit that mean; the primary comparison above uses the statistics
available on every allocation.

## Reproducibility and correctness

The two worker allocations agree closely:

- 128-by-128 FP16 device medians are 5.131 and 5.253 ms.
- 128-by-128 FP32 device medians are 5.530 and 5.619 ms.
- The corresponding 64-by-64 allocation ranges are 8.621--8.756 ms and
  13.118--13.140 ms.

Both grids pass the same randomized projection validation. Relative residuals
are 0.00729 for pure FP16 and 1.04e-6 for pure FP32.

## Roofline interpretation

Both figures use the naive Williams roofline \(P=\min(\pi,\beta I)\) with
\(I=W/Q\). The two plots use different \(Q\).

The E2E plot uses host-wire \(Q=s(nk+km+nm)\) and a single worker
\(\beta\) that combines the measured H2D and D2H fits in a 2:1 byte ratio.
The device-kernel plot uses SUMMA SRAM traffic as \(Q\) and the vendor
21 PB/s SRAM number scaled to 4,096 PEs (95.57 TB/s). That SRAM \(\beta\)
is a nameplate scaling, not a measured application bandwidth.

At 128-by-128, allocation-median device throughput is 6.462 TFLOP/s FP16 and
6.019 TFLOP/s FP32. The FP16 point reaches 5.3% of its applicable analytical
SRAM roof. The FP32 point reaches 26.2% of the provisional 22.938 TFLOP/s
compute roof. The FP32 roofs remain literature-derived assumptions because
Cerebras does not publish a dense WSE-3 FP32 peak.

## Cold E2E size sweep (64-by-64)

The same worker/WSE path was repeated for square \(N\in\{128,256,512,1024,2048,2560\}\).
Host-wire OI is \(2N/(3s)\). Worker-median E2E throughputs:

| N | FP16 OI | FP16 E2E TFLOP/s | FP32 OI | FP32 E2E TFLOP/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 42.7 | 0.0096 | 21.3 | 0.0088 |
| 256 | 85.3 | 0.066 | 42.7 | 0.055 |
| 512 | 170.7 | 0.221 | 85.3 | 0.192 |
| 1024 | 341.3 | 0.452 | 170.7 | 0.398 |
| 2048 | 682.7 | 0.970 | 341.3 | 0.754 |
| 2560 | 853.3 | 1.181 | 426.7 | 0.879 |

Host-wire OI grows with \(N\), so the E2E points walk right along the
\(\beta I\) diagonal. They stay under the I/O roof and far below the 91.8
TFLOP/s FP16 compute roof. The FP32 ridge is about 800 FLOP/byte
(\(N\sim 4.8{\times}10^3\)); \(N=2560\) is still left of that knee.

The device-kernel plot uses SRAM intensity, which is almost flat in \(N\)
(~0.33 FP16, ~0.16 FP32). Those points therefore stack in \(x\) under the
SRAM/compute roofs:

| N | FP16 kernel TFLOP/s | FP32 kernel TFLOP/s |
| ---: | ---: | ---: |
| 128 | 0.096 | 0.062 |
| 256 | 0.384 | 0.307 |
| 512 | 0.883 | 0.837 |
| 1024 | 1.627 | 1.382 |
| 2048 | 3.230 | 2.269 |
| 2560 | 3.866 | 2.555 |

Without H2D/D2H, FP32 reaches 2.56 TFLOP/s at \(N=2560\), about 45% of its
5.73 TFLOP/s compute roof. FP16 reaches 3.87 TFLOP/s, still only 4.2% of
91.8 TFLOP/s. Figures: `results/figures/e2e-roofline.png` and
`results/figures/device-kernel-roofline.png`.