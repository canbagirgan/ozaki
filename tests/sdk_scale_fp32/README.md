# Integer FP32 exponent scaling simulator check

Run from the repository root inside the Cerebras SDK environment:

```sh
cslc tests/sdk_scale_fp32/layout.csl --arch=wse3 \
  --fabric-dims=8,3 --fabric-offsets=4,1 --memcpy --channels=1 \
  -o subtraction-scale-probe
cs_python tests/sdk_scale_fp32/run.py --name subtraction-scale-probe
```

This checks the standalone vector helper in
`benchmarks/gemm_mesh/src/fp32_product_scale.csl`. It compares
256 finite FP32 bit patterns against exact `ldexp(x, -16)`, flushing magnitudes below
FP32 min-normal to positive zero before rounding. Cases include signed zeros, negative values,
subnormal inputs, both sides of the exponent-16/17 boundary, maximal finite
values, and random mantissas. It repeats the transfer and kernel twice.
NaN/Inf are outside this helper's contract. Scaling is independent of the FP32
TwoSum GEMM and is not enabled in that kernel.
