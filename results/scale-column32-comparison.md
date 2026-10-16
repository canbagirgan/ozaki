# Scaling comparison with 32-element product columns

Measured on 2026-10-15 using the Cerebras SDK 2.10.1 WSE-3 simulator.

Moving rescaling out of the K loop reduces kernel cycles from **13978**
to **7882** (**43.61% fewer cycles**). The final-merge
version has **23.02%** overhead versus the unscaled baseline.

| Implementation | Max-PE cycles, repeat 1 | Max-PE cycles, repeat 2 | Relative Frobenius error |
| --- | ---: | ---: | ---: |
| Unscaled | 6408 | 6406 | 7.63243559e-08 |
| Scaled, final-only vector | 7882 | 7882 | 7.37986042e-08 |
| Scaled, final-only fscale | 9000 | 9000 | 7.37986042e-08 |
| Scaled, per-contribution vector (previous) | 13978 | 13978 | 7.16423477e-08 |
| Scaled, per-contribution scalar (previous) | 30522 | 30520 | 7.16423477e-08 |

## Final-merge implementation

The main product A0B0 and the combined scaled cross products A0B1s+A1sB0
have independent FP32 buckets and two-word accumulators. Each bucket flushes
after 10 of its own contributions, plus a final partial bucket. Cross terms
are never rescaled during the K loop or at bucket flushes. After all K values
have been processed, both cross-sum words are rescaled by 2^-16 in one final
stage, then combined with A0B0 using a compensated merge. The dead main bucket
is reused as integer-mask scratch. Descriptor setup is outside the timed kernel.

The different grouping of contributions changes rounding: final-merge relative
error is 7.37986042e-08, versus
7.16423477e-08 with per-contribution rescaling.
The extra compensated state requires three more FP32 output-sized tiles
(12*Mt*Nt bytes) than the previous compensated path. Estimated square-T=32
storage is 41,728 bytes, above the default 40 KiB budget; T=28 is 32,032 bytes.

## Configuration and validation

- A is 128x8; B is 8x8; 4x4 PE grid; local Mt=32, Kt=2, Nt=2.
- All paths compute three slice products with TwoSum period 10. Final merge uses
  independent main/cross schedules; earlier paths use one shared bucket.
- ozIMMU phi=1, seed 7; one warmup and two retained samples per implementation.
- One memcpy channel, no fabric start barrier, simulator tracing suppressed.
- Reported cycles are max-PE kernel TSC, not host emulation wall time.
- Input checksums match across all variants. Final-merge repetitions have equal
  output checksums. The two earlier scaled implementations match bit-for-bit.
- An additional final-merge test at N=128, K=12, M=8 exercises full buckets and
  tails in both streams: relative error
  9.10191745e-08.
- The updated unscaled branch also compiles successfully.

Machine-readable measurements and source hashes are in
[scale-column32-comparison.json](scale-column32-comparison.json).
These are 32-element-column tests, not full square-tile-32 or wafer timings.
The unscaled runtime was terminated after its samples and validation were
collected because SDK shutdown did not return; its kernel measurements were
emitted before shutdown.

## Final-only vector versus fscale

Both variants scale the completed cross accumulator once at the end, then merge
with A0B0. Vector takes 7,882 cycles; scalar SDK `math.fscale_f32` takes 9,000
cycles in both repetitions. Vector uses 12.42% fewer cycles (1.142x speedup).
The high and low output arrays are bit-for-bit identical on this input; both
have relative Frobenius error 7.37986042e-08. The fscale runtime exited normally.
Select `scaled-residual-final-vector` or `scaled-residual-final-fscale` in the
campaign runner. The existing scaled TwoSum-every10 method defaults to vector.
