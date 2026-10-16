# Integer FP32 exponent scaling simulator check

Run from the repository root inside the Cerebras SDK environment:

```sh
cslc tests/sdk_scale_fp32/layout.csl --arch=wse3 \
  --fabric-dims=8,3 --fabric-offsets=4,1 --memcpy --channels=1 \
  -o subtraction-scale-probe
cs_python tests/sdk_scale_fp32/run.py --name subtraction-scale-probe
```

This checks the same vector helper imported by the fused GEMM kernel.
The kernel calls it only at the final merge, after the cross-term reduction. It compares
256 finite FP32 bit patterns against exact `ldexp(x, -16)`, flushing magnitudes below
FP32 min-normal to positive zero before rounding. Cases include signed zeros, negative values,
subnormal inputs, both sides of the exponent-16/17 boundary, maximal finite
values, and random mantissas. It repeats the transfer and kernel twice.
NaN/Inf are outside this helper's contract. Finite FP16 products cannot
produce NaN/Inf or an FP32 subnormal result after scaling down by 2^16.
