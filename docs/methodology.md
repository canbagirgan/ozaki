# Methodology and source notes

## Scope

This baseline answers three questions using 64-by-64 and 128-by-128 PE
rectangles with 16 memcpy channels at two host boundaries: the appliance
worker adjacent to the CS system, and the `cer-usn-01` submission process:

1. What useful-payload bandwidth does the appliance SDK deliver for H2D and
   D2H transfers?
2. How long do pure-FP16 and pure-FP32 SUMMA GEMMs take on the WSE?
3. What is the cold-input operational time after adding H2D(A, B), device
   kernel, and D2H(C)?

It does not benchmark mixed accumulation, sparsity, multiple wafers, or Ozaki.

## Primary sources

- [WSE-3 datasheet](https://8968533.fs1.hubspotusercontent-na2.net/hubfs/8968533/Datasheets/WSE-3%20Datasheet.pdf):
  vendor specifications for SRAM, aggregate SRAM bandwidth, fabric bandwidth,
  system I/O, PE count, and sparse-AI throughput.
- [CSL SIMD appendix](https://sdk.cerebras.ai/csl/language/appendix):
  WSE-3 FP16 and mixed-precision SIMD widths and SRAM-bank constraints.
- [SDK release notes](https://sdk.cerebras.ai/sdk-release-notes/sdk-rel-notes-cumulative):
  memcpy channel scaling and the documented H2D/D2H asymmetry.
- [SDK examples](https://sdk.cerebras.ai/csl/sdk-examples):
  provenance for `bandwidth-test` and `gemm-collectives_2d`.
- [Miyajima et al., 2026](https://www.jstage.jst.go.jp/article/ipsjjip/34/0/34_132/_pdf):
  pure-FP32 SUMMA and FMAC cycle models on WSE-2. In particular, the measured
  model is \(T_{\mathrm{FMAC}}=2I+10\) cycles and the broadcast model is
  \(T_{\mathrm{bcast}}=1.6I+250\) cycles.
- [Belli and De Sensi, 2026](https://arxiv.org/html/2605.07954v1):
  WSE-3 scientific-computing measurements. This repo converts TSC at 1.4 GHz,
  not that paper's 875 MHz figure or the SDK WSE-2 850 MHz example.

Marketing throughput is not substituted for dense arithmetic throughput. The
125 PFLOP/s WSE-3 figure is explicitly sparse and reduced precision.

## Transfer experiment

Separate CSL artifacts expose 16-bit and 32-bit buffers over the 64-by-64
rectangle. Each size/direction pair has one warmup and at least five retained
samples. The payload-size transfer fit was not repeated on 128-by-128; that
grid records the H2D and D2H phases of the fixed GEMM itself.

The primary transfer sweep uses `SdkLauncher` to stage and run an ordinary
`SdkRuntime` host program directly on the appliance worker. This is the
execution mode Cerebras recommends for large transfers. Its blocking memcpy
wall time measures the worker/WSE boundary.

A supplemental sweep uses the deprecated appliance `SdkRuntime` Python
binding from `cer-usn-01`. Two measurements are retained there:

- synchronized WSE timestamp-counter makespan, following the SDK
  `bandwidth-test`;
- wall time of a separate blocking `memcpy_h2d` or `memcpy_d2h` API call.

The first characterizes the interval visible to WSE timestamp markers. The
second characterizes the complete user-node API path and is used for the
user-node cold E2E model. This separation prevents the timestamp marker RPC
from being fitted as memcpy latency. It also exposes client/RPC bottlenecks
that are outside the direct worker/WSE data path.

FP16 is represented in the appliance Python API by one 32-bit host container
per 16-bit logical value, as required by `SdkRuntime`. Reported bandwidth uses
useful logical payload bytes, while `host_container_bytes` is retained in the
raw records. The summary includes a second fit against host-container bytes so
the FP16 packing effect is explicit rather than mistaken for a precision
dependent link rate.

For each precision, direction, and host boundary, medians are fitted to

\[
t(b)=\alpha+\frac{b}{\beta},
\]

where \(\alpha\) is the per-call latency and \(\beta\) is sustained useful
payload bandwidth. Negative OLS intercepts are clamped to zero and the slope
is then refitted through the origin.

## GEMM experiment

The kernels are adaptations of SDK 2.10 `gemm-collectives_2d`:

- FP16 stores A, B, and C as IEEE `f16` and invokes `@fmach`;
- FP32 stores A, B, and C as `f32` and invokes pure `@fmacs`.

The retained performance case is one 2560-by-2560-by-2560 square GEMM.
The 64-by-64 grid uses 40-by-40 local tiles and the 128-by-128 grid uses
20-by-20 local tiles. This is a fixed-problem strong-scaling comparison. It
cannot support an empirical size-scaling curve or a fitted SUMMA cycle model.

The same operation sequence is measured directly on an appliance worker via
`SdkLauncher` and, as a supplemental site-level result, from the user node via
the appliance `SdkRuntime` binding. Median-PE device cycles should agree; the
global makespan can vary with the slowest PE in each placement. H2D, D2H, and
host-observed E2E can also differ because the measurements include different
host/RPC paths.

Every measured iteration performs:

1. untimed device reset of C and the step counter;
2. timed H2D(A) plus H2D(B);
3. timed dispatch of all 64 SUMMA steps;
4. timed primary D2H(C), which is also the host synchronization point and ends
   the measured operational E2E interval;
5. untimed repeat D2H(C) after the result is ready, to isolate transfer time
   from outstanding kernel work;
6. untimed timer readback and host validation.

Each allocation uses two warmups and nine retained samples. For each grid and
precision, the raw data contain two independent worker/WSE allocations and one
supplemental user-node allocation. Raw samples retain a `run_id`, allowing
device results to be reduced to one median per allocation before
cross-allocation aggregation.

The operational E2E interval is steps 2–4. The exported CSL entry point
dispatches asynchronous PE tasks, so its host API return is not treated as
kernel completion. Device kernel time is the maximum per-PE elapsed TSC
duration, and the isolated D2H component comes from step 5. The measured E2E
and the sum of isolated H2D, device-kernel, and D2H components are both
retained; small differences reflect host dispatch and synchronization
semantics. Per-PE minimum (fastest), arithmetic mean, median, maximum
(slowest/makespan), and spread are retained. The maximum remains the reported
kernel result because GEMM is complete only when every PE is complete.

Inputs are generated with standard deviation \(1/\sqrt{k}\) to reduce pure
FP16 overflow risk. The first measured result is checked with a randomized
projection; matrices up to dimension 512 also receive a full FP32 host
reference comparison.

## Roofline axes

The host-wire point uses

\[
OI_{\mathrm{host}} =
\frac{2nkm}{s(nk+km+nm)}
\]

and throughput \(2nkm/T_{\mathrm{e2e}}\).

The device point uses an implementation-level SRAM intensity. For each local
FMAC, it counts A and C reads plus a C write, scalar B reads, and collective
receiver writes. This is intentionally distinct from compulsory host bytes.

The SRAM roof scales the vendor's 21 PB/s aggregate figure by active PE count:
\(4096/900000\) or \(16384/900000\). The FP16 compute roof uses eight FMAC
lanes per PE per cycle.
The provisional pure-FP32 roof uses one scalar FMAC element per two cycles,
following a published WSE-2 pure-FP32 FMAC microbenchmark. Cerebras does not
publish a dense WSE-3 FP32 system roof, so the extrapolation is explicitly
provisional. The measured point is shown against that assumption without
claiming to validate its slope.

## Clock treatment

Raw cycles are authoritative. Plots convert them at 1.4 GHz. The SDK's older
bandwidth example hard-codes 850 MHz for WSE-2; that value is not used here.
Results retain cycles and the frequency assumption so device-time quantities
can be regenerated if the CS-3 clock is later confirmed to differ.

## Interpretation limits

- The 64-by-64 and 128-by-128 allocations measure selected rectangles, not
  whole-wafer throughput.
- SUMMA is a transparent reference implementation, not an optimized GEMM
  library or a hardware maximum.
- Host API time includes the appliance data path observed from `cer-usn-01`,
  so it is specific to this ALCF software/network configuration.
- Simulator wall time is used only for correctness. No simulator timing is
  included in performance fits.
- Two primary GEMM allocations plus one supplemental allocation per grid and
  precision expose placement variability but are not enough to infer
  long-term system variability.
