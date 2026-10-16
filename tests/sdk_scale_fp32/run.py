"""Simulator test of integer exponent scaling, including exponent boundaries."""
import argparse
import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
args = parser.parse_args()
rng = np.random.default_rng(7)
bits = rng.integers(0, 2**32, size=256, dtype=np.uint32)
# Exclude NaN/Inf, which are outside the finite-input helper contract.
bits[(bits & 0x7f800000) == 0x7f800000] &= np.uint32(0xff7fffff)
bits[:20] = np.array([
    0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
    0x00800000, 0x80800000, 0x087fffff, 0x887fffff,
    0x08800000, 0x88800000, 0x08800001, 0x88800001,
    0x3f800001, 0xbf800001, 0x7f7fffff, 0xff7fffff,
    0x27800000, 0xa7800000,  # +/- 2^-48, smallest nonzero FP16 product.
], dtype=np.uint32)
values = bits.view(np.float32)
# Decide flushing from the exact scaled magnitude, before FP32 rounding.
# The largest exponent-16 value would otherwise round up to min-normal.
expected_wide = np.ldexp(values.astype(np.float64), -16)
expected_wide[np.abs(expected_wide) < float(np.finfo(np.float32).tiny)] = 0.0
expected = expected_wide.astype(np.float32)
runner = SdkRuntime(args.name)
runner.load()
runner.run()
try:
    symbol = runner.get_id("values")
    for repetition in range(2):
        runner.memcpy_h2d(symbol, values, 0, 0, 1, 1, 256,
                          streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                          order=MemcpyOrder.COL_MAJOR, nonblock=False)
        runner.launch("scale", nonblock=False)
        actual = np.zeros(256, dtype=np.float32)
        runner.memcpy_d2h(actual, symbol, 0, 0, 1, 1, 256,
                          streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                          order=MemcpyOrder.COL_MAJOR, nonblock=False)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    print("PASS: 256 finite lanes match exact ldexp(-16), with subnormal magnitudes flushed to +0 before rounding; two repetitions")
finally:
    runner.stop()
