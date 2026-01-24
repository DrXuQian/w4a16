Standalone SM80 fpA_intB GEMM (FP16 x INT4, GPTQ)
===================================================

This directory contains a minimal, standalone build of TensorRT-LLM's fpA_intB
(weight-only) GEMM for SM80 (Ampere). It supports:

- FP16 activations
- INT4 weights (GPTQ-style, scale + zero)
- CUTLASS fpA_intB path
- CUDA kernel fallback for small M (best performance for M=1)

What I extracted
----------------
- fpA_intB CUTLASS runner (`fpA_intB_gemm_template.h` + instantiation)
- CUTLASS config selection via TRT-LLM profiling-style timing
- weight-only CUDA kernel fallback (`weightOnlyBatchedGemv`)
- C++ weight preprocessing (`cutlass_preprocessors.*`)
- Standalone CLI test (`test_fpA_intB_gemm.cu`)

What is NOT included (limitations)
----------------------------------
- SM90/SM100 paths (SM80 only)
- BF16/FP8/FP4 or INT8 weights
- bias/alpha/act-scale fusion (GPTQ scale+zero only)
- non-groupwise quantization (group_size must be 64 or 128)

Config selection (aligned with TRT-LLM)
---------------------------------------
The selector uses TRT-LLM's profiling method:

- Generate candidate CUTLASS configs via `get_candidate_configs` (weight-only).
- Include the CUDA kernel as an extra "config" with `enableCudaKernel=true`.
- Time each candidate with CUDA events (warmup + runs) and pick the fastest.
- M<16 is where the CUDA fallback typically wins.

You can list configs or force a config via CLI.

Input/weight layout assumptions
-------------------------------
- A: FP16 row-major, shape [M, K].
- B: INT4 packed (2 values per byte), then preprocessed with
  `preprocess_weights_for_mixed_gemm(..., QuantType::W4_A16, force_interleave=true)`.
- Scales and zeros: FP16 row-major, shape [K / group_size, N].
- GPTQ zeros are passed as `zero_x_scale = (-qzeros + 7) * scale` (same as TRT-LLM).

Build
-----
From the repo root:

```
cmake -S fpA_intB_standalone -B fpA_intB_standalone/build
cmake --build fpA_intB_standalone/build -j8
```

Run
---
Example:

```
fpA_intB_standalone/build/test_fpA_intB_gemm \
  --m=1 --n=4096 --k=4096 --group_size=128
```

List configs:

```
fpA_intB_standalone/build/test_fpA_intB_gemm --list_configs
```

Force CUDA kernel:

```
fpA_intB_standalone/build/test_fpA_intB_gemm \
  --m=1 --n=4096 --k=4096 --group_size=128 \
  --config=cuda
```

Force a CUTLASS config (tile_m,tile_n,tile_k,stages,split_k):

```
fpA_intB_standalone/build/test_fpA_intB_gemm \
  --m=1 --n=4096 --k=4096 --group_size=128 \
  --config=16x128x64x3x1
```

CPU reference (small shapes only):

```
fpA_intB_standalone/build/test_fpA_intB_gemm \
  --m=1 --n=128 --k=128 --group_size=128 --verify
```

Notes
-----
- K must be a multiple of 64 (SM80 fpA_intB requirement) and group_size.
- The CUDA fallback only supports M in [1, 15].
- CPU reference is a simple sanity check (not a full correctness suite).
