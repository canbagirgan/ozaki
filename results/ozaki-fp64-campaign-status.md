# Ozaki FP64-class CS-3 campaign

**Updated:** 2026-09-09 18:06 UTC  
**Headline:** Pareto point is **old-CSL 10-product TwoSum + covering**. Best large-N: **N=15360 T=30 kernel 1308 ms**, subsample **1.77e-11**. N=14336: **1105 ms**, **1.63e-11**. N=1792 ozIMMU **5.69e-12** / gauss **2.34e-12**, **138 ms**. Scheme F T=32 stays **1.81e-4** — do not use. Misses 1e-12 and 500 ms.

## Executive summary

| Item | Status |
| --- | --- |
| Best scheme so far | **A:** 4-slice `i+j<4` + per-FMA TwoSum + **fp16 covering scale** (host). Weight-split dual-C without TwoSum is **not** 1e-12 (prior 1.261e-6). |
| Rel Frob vs high-prec GEMM | **Wafer (old CSL, N=1792, 64×K×64 block):** **5.69e-12**. Scheme F N=512 full matrix: **1.81e-4** (broken copy). Host ozIMMU 1.98e-12 (K=32) → 5.47e-12 (K=2048). |
| Rel Frob vs f64 GEMM | Wafer N=1792 subsample **5.69e-12**. F N=512 **1.81e-4**. |
| Kernel ms at N=16384 | Trusted kernel does not exist at T=32. Closest: **N=15360 T=30 = 1308 ms** (1.832e9 cyc). **N=14336 T=28 = 1105 ms**. Skip-A1B1 N=16384: **21.4 ms**. |
| ≤1e-12 and ≤500 ms | **Neither met.** Closest accuracy: wafer gauss N=1792 **2.34e-12**; ozIMMU N=1792 **5.69e-12**. Closest large-N speed: **1105 ms** (over 500). |
| Blocker | Scheme F (needed for T=32) is wrong (1.81e-4). TwoSum compute ~1.1–1.3 s at N~15K. |

## Scheme cards (research → CS-3)

CS-3 facts: no f64 FPU, no INT8 tensor core. `@fmachs` = f16×f16+f32 (SIMD-4). An f16×f16 product is **exact** in f32 (22 ≤ 24 bits) if it does not overflow. The missing piece vs ozIMMU’s exact INT8 products is the **K-reduction**: that must be Knuth TwoSum’d into `(C_hi, C_lo)` (~48-bit sum). Host reconstructs `float64(hi)+float64(lo)`.

Sources for 1e-12 must be generated in the host wide type (this node: **80-bit x87 long double**, 64-bit significand, NumPy `float128` — not IEEE binary128). Headline Rel Frob = device C vs **wide GEMM(A,B)**. `float64(A)@float64(B)` is secondary (~K·2^{-52}).

### A. 4-slice triangle + per-FMA TwoSum (`n_products=10`) — **implement first**

- **Products:** 10, `i+j<4` (A0B0…A0B3, A1B0…A1B2, A2B0–A2B1, A3B0). Not 11 (11 is not a weight class). Next class is 13 (`i+j≤4`).
- **Covering:** 4×11 = 44 significand bits (1e-12 needs ~40). Short of binary64’s 53.
- **SRAM (unpacked, double-buffer):** 16 T² A + 16 T² B f16 + 8 T² C + 24T TwoSum scratch. T=32: **41728 B** (84.9% of 48 KiB; 41 KiB budget). T=28: 32032 B (fits 40 KiB).
- **Expected error:** 1e-12–1e-14 if covering is enough and products do not overflow. ozIMMU φ=1 is the hard distribution (wide exponents).
- **Expected time vs 500 ms:** Prior 4-product TwoSum N=2560 P=64: 131 ms; 9-product N=2048: 170 ms. Scale PE work `P·n_prod·T³` → N=16384 P=512 T=32 n=10 ≈ **1.3–1.5 s**. Likely **over** 500 ms. Accuracy first; then cut products/TwoSum if needed.
- **Implement?** **Yes.** CSL body is present; CLI/host high-prec validation were not wired.

### B. 5-slice triangle + TwoSum (`n_products=15` **with** TwoSum)

- **Products:** 15, `i+j<5`. Covering 55>53.
- **SRAM + TwoSum + dbl-buf:** 48 T² + 24T. T=32: **49920 B > 48 KiB**. T=28: 38304 B. **Cannot do N=16384 (P=512,T=32)** without dropping double-buffer (then 28 T²+24T = 29440 B).
- **Expected error:** Better than A if 4-slice covering loses bits on φ=1. Weight-split **without** TwoSum was **1.261e-6** (P=64 T=28 N=1792) — do not sell that as 1e-12.
- **Expected time:** ~1.5× A, worse vs 500 ms.
- **Implement?** Only if A misses 1e-12. Then T=28 (N≠16384) or drop dbl-buf for T=32.

### C. 3-slice + TwoSum (`n_products=9 --twosum`)

- **Products:** 9. Covering 33 bits — likely miss 1e-12 on wide sources.
- **SRAM:** 32 T² + 24T = 33536 B at T=32.
- **Prior wafer (scaled 1/√K, A2 flushed):** Rel Frob vs f64 **1.102e-6**, kernel **170.09 ms** (P=64 N=2048). Artifact exists: `ozaki-fused-twosum-3slice-p64-n2048-…-start-barrier`. Also `ozaki-fused-3slice-p512-n16384-…-start-barrier` (no TwoSum).
- **Implement?** Measure on ozIMMU/high sources. Do not expect 1e-12.

### D. Integer / exact product (`run_int.py`, `run_u16x2.py`)

- **Idea:** Shared-exponent i16 slices, scalar `i16*i16` → i32. Exact K-reduction if `α=7`, K=16384 (`127²·16384 < 2³¹`). Host scales products in f64. No TwoSum required when i32 does not overflow.
- **Products:** Sequential launches, triangle `s(s+1)/2`. s=8 → 36 kernels; s=6 (42-bit cover) → 21 kernels.
- **SRAM:** 12 T² (one pair at a time).
- **Expected error:** Can beat `@fmachs`+TwoSum if integer reduction stays exact.
- **Expected time:** ~21–36 × one MeshGEMM (~40–65 ms skip/FP32 at N~20K) → **1–2 s**, over 500 ms.
- **Implement?** Host oracle + small wafer if A misses accuracy. Existing artifact: `ozaki-int-p64-n3584`.

### E. K-blocking / superblock / distillation

- TwoSum of full K can still lose if the two-word f32 pair is not enough for K·u. Split K, TwoSum blocks, distill.
- **Implement?** Only if A–D miss 1e-12.

### F. Packed hops / drop double-buffer — **now required for N=16384**

- Packing 4 slices into one hop does **not** cut SRAM if still double-buffered (still 16+16+8 T² + 24T = 41728).
- T=32 n=10 TwoSum **failed to link** (PE SRAM, not the 41 KiB data-budget check). 16-product T=32 (36864, no TwoSum) already links; C_lo + TwoSum code is the straw.
- **Implemented:** A2/A3/B2/B3 single-buffer; recv aliases A1/B1 send; `@fmovh` copy. **33536 B** at T=32. P=64 T=32 and **P=512 T=32 N=16384 linked**.
- Lose overlap on those hops; TwoSum still dominates time. Not yet wafer-validated (queue).

### Rejected as 1e-12

- Weight-split dual C **without** per-FMA TwoSum (n_products=15 as currently compiled).
- 2×fp16 TwoSum vs native FP32 (22 vs 24 bits; prior 1.231e-6 vs 9.06e-7).

### Papers / notes (what transfers)

| Source | Transfer to CS-3 |
| --- | --- |
| Ozaki–Ogita–Oishi–Rump 2012 EFT | Split high type → low-prec words; GEMM = sum of error-free products. Triangle drops `i+j≥s`. |
| Mukunoki et al. 2020 (DGEMM via FP16 TC) | Same split idea; they use Tensor Cores. We use `@fmachs`. |
| Ootomo–Ozaki–Yokota ozIMMU (INT8 IMMU) | INT8 products are exact. **Analog:** exact f16×f16 in f32 + TwoSum of the reduction. φ=1 is the study distribution. More slices for wider exponents. |
| Ozaki scheme II (CRT / modular, 2025) | Fewer products via modular GEMM. No INT8 TC / no modular unit on WSE-3. **Skip.** |
| FP8 TC Ozaki (2025) | No FP8 tensor cores. **Skip.** |
| Compensated Horner / distillation | Scheme E if long-K TwoSum loses. |
| Repo `results/ozaki-twosum-vs-fp32.md`, `docs/` | TwoSum helps fused Ozaki but 2-slice covering loses to native FP32. 3-slice flushed A2 on 1/√K. Clock 1.4 GHz. |

---

## Log

### 2026-09-09 16:05 UTC — kickoff

- **Tried:** Read CSL, `split.py`, `high_precision.py`, compile/launch/run, prior JSONL/artifacts, Ozaki/ozIMMU/Ozaki-II notes.
- **Worked:** `high_precision.py` looks complete. `ten_product_twosum_fp64` exists. CSL `n_products=10` hops, align A0–A3, triangle TwoSum, `tri4` forces `use_twosum`/`res_lo` clear. Layout already exports A3/res_lo.
- **Failed / gaps:** CLI `choices` omit 10. Config would have collided with default 4-product `ozaki-fused-p…`. Worker does not generate wide sources or call `compare_device_vs_high`. Tests omit 10-product and high-prec headline. `high_precision.residual_split_fp16_high` imports `ozaki_roofline.split` (breaks appliance staging).
- **Latest best rel-Frob:** prior 15-product weight-split **1.261e-6** vs f64 (not this campaign’s headline).
- **Latest kernel ms:** prior 9-product TwoSum **170.09 ms** at N=2048 P=64 (not 16384).
- **Next:** Wire CLI + tests + host oracles; compile P=64 T=28 N=1792 start-barrier; then P=512 N=16384.
- **Blockers:** none yet (ssh not tried this session).

### 2026-09-09 16:39 UTC — host oracles + CLI

- **Tried:** Wired `--n-products=10` (TwoSum forced, name `ozaki-fused-twosum-4slice-tri-...`). Host covering-scale + high-prec GEMM in `run_direct.py`. Unittest `tests.test_high_precision` + `tests.test_ozaki_split` (43 OK on login-node CPython 3.9). `scripts/host_ozaki_fp64_oracle.py` → `results/raw/ozaki-fp64-host-oracle.jsonl` (20 records).
- **Worked:** Covering scale is the accuracy unlock. Unscaled 4-slice TwoSum ~1e-8 (A3 flushes). Extra products 13/15/16 do not help once covering is the limit. Integer 7-slice host hits 1e-12 on ozIMMU (backup D).
- **Failed:** Naive residual `astype(fp16)` without scale. Weight-split dual still ~1e-7–1e-8.
- **Latest best rel-Frob (host vs wide GEMM):** covering-scale 10-product **8.91e-13** (scaled-gauss 8×32×8); ozIMMU **1.98e-12** / **2.66e-12**.
- **Latest kernel ms:** still no new wafer timing.
- **Next:** P=64 N=1792 start-barrier launch; P=512 T=32 compile.
- **Blockers:** login-node `.venv` is a broken python3.11 symlink; host tests used `/usr/bin/python3`. Remote venv on `cer-usn-01` is fine.

### 2026-09-09 16:46–16:52 UTC — compiles + hung launch

- **Tried:** Compile P=64 N=1792 T=28 start-barrier + E2E (OK, 32032 B). Compile P=512 N=16384 T=32 `--data-budget-kib 41` (linker **ran out of PE memory**). Compile P=512 N=14336 T=28 start-barrier (**OK**). T=30 N=15360 still compiling (~25 min). Compile P=64 N=512 start-barrier + E2E (OK). Launch P=64 N=1792 `--high-precision` killed after 25 min **no stdout** (likely x87 GEMM 1792³ on the worker, ~10¹⁰ FLOP).
- **Worked:** T=28 kernels link at P=64 and P=512. N=512 accuracy artifact ready.
- **Failed:** T=32 n=10 TwoSum does not fit 48 KiB total PE SRAM. N=1792 validated launch produced no record.
- **Latest best rel-Frob:** host numbers above; wafer still pending.
- **Latest kernel ms:** none this campaign. Prior skip-A1B1 N=16384 = **21.4 ms**; 4-product fused N=16384 = **28.0 ms**; 15-product weight-split N=1792 = **9.0–14.5 ms**.
- **Next:** N=512 ozIMMU validated launch (in flight). Then N=1792 `--skip-validation` for kernel time, N=14336 `--skip-validation`, scheme F for T=32.
- **Blockers:** T=32 link; N≥1792 high-prec host GEMM too slow for in-worker validation.

### 2026-09-09 16:55 UTC — T=30 links; K-scaling; papers

- **Tried:** Host covering-scale at K=256/512; K-block TwoSum (32/64); 5-slice covering TwoSum; integer 6/7/8-slice. Compile P=512 T=30 N=15360 start-barrier. Papers: Ozaki 2012 EFT; Mukunoki FP16-TC DGEMM; ozIMMU INT8 (Ootomo 2024 / IJHPCA 2025); HPCwire 2025 explainer; ADP/ESC (SCA/HPCAsia 2026); Ozaki II CRT / FP4 (skip on WSE-3).
- **Worked:** **T=30 N=15360 linked** (`ozaki-fused-twosum-4slice-tri-p512-n15360-…-start-barrier`). Covering-scale error grows slowly with K (ozIMMU 1.98e-12 @K=32 → 3.62e-12 @K=512). Integer **8-slice** hits 1e-12 at K=512 ozIMMU (**8.9e-14**). α=7 K-reduction exact at K=16384. CS-3 analog of ozIMMU exact INT8 products = exact f16×f16 in f32 + TwoSum (or exact i16×i16 in i32).
- **Failed / skip:** 5-slice covering TwoSum **does not beat** 4-slice (same ~2e-12 on ozIMMU) — extra word is not the limiter once rows/cols are scaled into the fp16 range. K-block 32/64 **does not recover** 1e-12 (4.56e-12, slightly worse). Integer 7-slice at K=512 ozIMMU is **1.34e-11** (misses). Ozaki II / FP8 / FP4: no those units.
- **Latest best rel-Frob:** host covering 10-product **8.91e-13** (gauss K=32); ozIMMU **1.98e-12** (K=32) / **3.62e-12** (K=512). Integer 8-slice **7.5e-15** (K=32) / **8.9e-14** (K=512).
- **Latest kernel ms:** still no new wafer sample. N=512 launch ~4 min in, no stdout yet (SdkLauncher often silent until worker returns; prior startup 41–941 s).
- **Next:** Finish N=512 accuracy; skip-val timing at N=1792, 14336, 15360; scheme F (~8 KiB) to try T=32.
- **Blockers:** T=32 still unlinkable without SRAM cut. N=16384 exact size still missing.

### 2026-09-09 17:14 UTC — scheme F links T=32

- **Tried:** Single-buffer A2/A3/B2/B3 for `n_products=10`: hop recv aliases A1/B1 send, `@fmovh` copy into the working tile. Data = 12+12+8 T² + 24T = **33536 B** at T=32 (was 41728). Compile P=64 N=2048 T=32 start-barrier. Started P=512 N=16384 compile.
- **Worked:** **P=64 T=32 linked** (`ozaki-fused-twosum-4slice-tri-p64-n2048-…-start-barrier`). `@fmovh` two-DSD copy accepted by CSL.
- **Failed / pending:** N=512 ozIMMU launch still silent at **~24 min** (process alive, ~0.6% CPU — cluster/SdkLauncher wait, not GEMM). E2E artifacts for N=512/1792/14336/15360 are ready (old CSL for those sizes).
- **Latest best rel-Frob:** host numbers unchanged; wafer still pending.
- **Latest kernel ms:** none new.
- **Next:** Confirm P=512 T=32 N=16384 links. Keep N=512 launch. Then skip-val N=16384 + N=2048 accuracy on the F kernel.
- **Blockers:** Wafer queue / silent SdkLauncher. F kernel not yet correctness-tested on wafer.

### 2026-09-09 17:33 UTC — N=16384 compiled; wafer queued

- **Tried:** Compile P=512 T=32 N=16384 start-barrier + E2E (scheme F). `csctl job` to explain silent launches. Host covering-scale at K=2048.
- **Worked:** **Both N=16384 artifacts exist.** Manifest: P=512, T=32, `n_products:10`, `twosum:1`, `start_barrier:1`, **fused_buffer_bytes=33536**, SDK 2.10.0. E2E twin: `ozaki-fused-twosum-4slice-tri-p512-n16384-…json` (`start_barrier` compile option 0). Host K=2048 covering: ozIMMU **5.47e-12**, gauss **2.85e-12**.
- **Failed / wait:** First N=512 launch killed after 31 min no stdout (was queued). Second N=512 F-kernel launch is **QUEUED** behind timnoack (4 RUNNING + 2 QUEUED). Do not submit more execute jobs (would line up behind ourselves).
- **Latest best rel-Frob:** host covering 10-product **8.91e-13** (gauss K=32); ozIMMU **1.98e-12** (K=32) … **5.47e-12** (K=2048). Integer 8-slice still the only host path that cleanly beats 1e-12 on ozIMMU at K≥32.
- **Latest kernel ms:** none new on wafer.
- **Next:** Wait for `wsjob-38mhkjgwjss7jjpwqfuv6j` (N=512 accuracy). Then N=16384 `--skip-validation` start-barrier + E2E. If F is wrong, fall back to T=28/T=30 old-CSL artifacts already compiled.
- **Blockers:** Shared CS-3 queue.

### 2026-09-09 17:49 UTC — first wafer execute crashed (mpmath)

- **Tried:** N=512 F-kernel launch after ~24 min queue. Worker started (`wsjob-38mhkjgwjss7jjpwqfuv6j`).
- **Worked:** Queue did drain; execute reached the appliance.
- **Failed:** `high_precision_info()` preferred **mpmath** (installed on the worker) over x87 `float128`. `ozimmu_matrix_high` built object arrays; `residual_split_fp16_high` then `promote_to_high(float16)` → `TypeError: cannot create mpf from 416.2`. No Rel Frob recorded.
- **Fix:** Prefer NumPy wide type over mpmath; `mpf(float(...))` fallback. Relaunch N=512 (same artifact, restaged `high_precision.py`).
- **Latest best rel-Frob / kernel ms:** unchanged (no successful wafer sample).
- **Next:** N=512 accuracy, then N=16384 skip-val + subsample 64.
- **Blockers:** timnoack still holds 3/4 systems; one slot should be free.

### 2026-09-09 17:54 UTC — first good wafer number; F is wrong

- **Tried:** N=512 F-kernel (full high-prec). N=1792 old-CSL `--skip-validation --validate-subsample 64`. Replaced `@fmovh(dst,src)` with `@faddh(dst,src,0)`.
- **Worked:** **Old CSL N=1792 P=64 T=28:** subsample 64×1792×64 Rel Frob vs high **5.691e-12** (vs f64 same). Kernel **138.13 ms** (193,387,724 cycles, 1.4 GHz). SRAM 32032 B. Covering+TwoSum is real on wafer. Sources for that run were f64 (skip-val), block promoted to float128.
- **Failed:** F-kernel N=512 full matrix Rel Frob **1.814e-4**, kernel 8.22 ms. `@fmovh` second arg is a scalar, not a tile copy — A2/A3 hops garbage. Do not trust any F T=32 artifact until `faddh` rebuild + re-measure.
- **Latest best rel-Frob:** **5.69e-12** wafer subsample (old CSL). Host gauss 8.91e-13 (K=32).
- **Latest kernel ms:** **138.13 ms** at N=1792. Scale ~1.1 s at N=14336 T=28, ~1.6 s at N=16384 T=32.
- **Next:** N=14336 old-CSL timing+subsample (in flight). Recompile F with faddh; re-check N=512; then N=16384.
- **Blockers:** 1e-12 missed by ~6× on ozIMMU K=1792 block. 500 ms missed by scale. Integer 8-slice still the accuracy hammer if we need <1e-12.

### 2026-09-09 18:10 UTC — large-N Pareto + E2E; F still broken

- **Tried:** N=14336 start-barrier subsample; N=15360 start-barrier subsample; N=14336 E2E (no barrier); N=1792 scaled-gauss subsample; three F N=512 variants (fmovh / faddh / scalar copy) — all **identical 1.814e-4**.
- **Worked (trusted old-CSL TwoSum+covering):**
  - N=1792 P=64 T=28 ozIMMU: Rel Frob **5.69e-12**, kernel **138.13 ms**, 193,387,724 cyc, 32032 B
  - N=1792 scaled-gauss: Rel Frob **2.34e-12**, kernel **138.13 ms**
  - N=14336 P=512 T=28 ozIMMU: Rel Frob **1.63e-11** (64-block), kernel **1105 ms**, 1.547e9 cyc, E2E barrier-run 5357 ms; **E2E no-barrier 5349 ms**, TSC 1115 ms
  - N=15360 P=512 T=30 ozIMMU: Rel Frob **1.77e-11**, kernel **1308 ms**, 1.832e9 cyc, 36720 B, E2E 6266 ms
- **Failed:** Scheme F T=32 (all copy methods) **1.81e-4** — alias/align bug, not the copy builtin. **N=16384 has no trusted kernel.**
- **Latest best rel-Frob:** wafer gauss N=1792 **2.34e-12**; ozIMMU N=1792 **5.69e-12**.
- **Latest kernel ms:** large-N **1105 ms** (N=14336) / **1308 ms** (N=15360).
- **Next if more time:** dedicated recv scratch (do not alias A1) for T=32; integer 8-slice wafer; superblock for 1e-12 at K=16K.
- **Blockers:** F broken; TwoSum >500 ms at N~15K; ozIMMU covering ~2–6e-12 at modest K, ~2e-11 at K=14K.
