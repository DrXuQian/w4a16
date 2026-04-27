Standalone fpA_intB GEMM (FP16 x INT4, GPTQ) for SM80 + SM90
============================================================

This directory contains a minimal, standalone build of TensorRT-LLM's fpA_intB
(weight-only) GEMM. It supports:

- FP16 activations
- INT4 weights (GPTQ-style, scale + zero)
- CUTLASS fpA_intB path
- CUDA kernel fallback for small M (best performance for M=1)
- SM80 (Ampere) and SM90 (Hopper) CUTLASS paths (build-arch dependent)

What I extracted
----------------
- fpA_intB CUTLASS runner (`fpA_intB_gemm_template.h` + instantiation)
- CUTLASS config selection via TRT-LLM profiling-style timing
- weight-only CUDA kernel fallback (`weightOnlyBatchedGemv`)
- C++ weight preprocessing (`cutlass_preprocessors.*`)
- Standalone CLI test (`test_fpA_intB_gemm.cu`)

What is NOT included (limitations)
----------------------------------
- SM100/SM110/SM120 specialized CUTLASS paths
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
  `preprocess_weights_for_mixed_gemm(..., QuantType::W4_A16, force_interleave=false)`.
  (If you need to force the SM80 interleaved layout on Hopper, pass `force_interleave=true`.)
- Scales and zeros: FP16 row-major, shape [K / group_size, N].
- GPTQ zeros are passed as `zero_x_scale = (-qzeros + 7) * scale` (same as TRT-LLM).

Build
-----
From the repo root:

```
cmake -S fpA_intB_standalone -B fpA_intB_standalone/build_cmake_release \
  -DGPU_ARCH=sm_90a \
  -DCUTLASS_DIR=$PWD/../../third_party/cutlass \
  -DCMAKE_BUILD_TYPE=Release
cmake --build fpA_intB_standalone/build_cmake_release \
  --target test_fpA_intB_gemm -j$(nproc)
```

The output binary is
`fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm`.
Use `-DGPU_ARCH=sm_80` for an Ampere build.

Run
---
Example:

```
fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm \
  --m=1 --n=4096 --k=4096 --group_size=128
```

List configs:

```
fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm --list_configs
```

Force CUDA kernel:

```
fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm \
  --m=1 --n=4096 --k=4096 --group_size=128 \
  --config=cuda
```

Force an SM90 TMA CUTLASS config:

```
fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm \
  --m=3823 --n=12288 --k=3072 --group_size=128 \
  --config=sm90:128x256x128:2x1x1
```

Force an SM80 CUTLASS config:

```
fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm \
  --m=1 --n=2048 --k=3584 --group_size=128 \
  --config=sm80:128x128x64:3:7
```

Notes:
- `split_k` must divide K and keep `K / split_k` a multiple of 64 (weight-only fpA_intB requirement).
  For example `K=2048` cannot use `split_k=7` (not divisible).
- The old SM80 spelling `--config=128x128x64x3x7` is still accepted for compatibility, but new commands should use
  `--config=sm80:128x128x64:3:7`.

Interpreting `--list_configs` output
------------------------------------
`--list_configs` prints values that can be passed back directly to `--config=`:

```
0: cuda
1: sm90:64x16x128:1x1x1
2: sm90:128x256x128:2x1x1
```

On an SM80 build the CUTLASS entries use the same aligned spelling:

```
sm80:128x128x64:3:1
```

Debug profile logging (config search)
-------------------------------------
Set `FPA_INTB_PROFILE_LOG=1` to print each candidate config, its timing, and failure reasons:

```
FPA_INTB_PROFILE_LOG=1 \
fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm \
  --m=1 --n=4096 --k=4096 --group_size=128
```

Traverse all configs without search
-----------------------------------
Use the helper script to run every candidate configuration directly:

```
fpA_intB_standalone/scripts/run_all_configs.sh \
  --bin fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm \
  --m 1 --n 4096 --k 4096 --group 128 --warmup 10 --iters 100 --out results.txt
```

Skip the CUDA fallback:

```
fpA_intB_standalone/scripts/run_all_configs.sh \
  --bin fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm \
  --m 1 --n 4096 --k 4096 --group 128 --warmup 10 --iters 100 --skip-cuda
```

CPU reference (small shapes only):

```
fpA_intB_standalone/build_cmake_release/test_fpA_intB_gemm \
  --m=1 --n=128 --k=128 --group_size=128 --verify
```

Notes
-----
- K must be a multiple of 64 (SM80 fpA_intB requirement) and group_size.
- The CUDA fallback only supports M in [1, 15].
- CPU reference is a simple sanity check (not a full correctness suite).

SM110/SM120 behavior
--------------------
This standalone extraction includes SM80 and SM90 CUTLASS fpA_intB kernel paths (plus the CUDA fallback).
On SM110/SM120 GPUs, the runner intentionally dispatches to the SM80 implementation (i.e. no SM100/SM110/SM120
specialized CUTLASS kernels are included here).
