# Third-party benchmark provenance

The files under `vendor/` were extracted without modification from the ALCF
installation:

`/software/cerebras/cs_sdk-2.10/csl-extras-202604101435-6-d2f7d96e.tar.gz`

The relevant upstream project is
[Cerebras/sdk-examples](https://github.com/Cerebras/sdk-examples), distributed
under the Apache License 2.0.

The synchronization module under `bandwidth/src/sync/` is an unmodified copy
from that bundle's `benchmarks/bandwidth-test/src/sync/` directory. The other
files in `bandwidth/` and `gemm_summa/` are experiment-specific adaptations
and carry SPDX attribution in their source headers.
