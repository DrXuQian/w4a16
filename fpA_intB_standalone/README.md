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
From the repo root, use the direct `nvcc` Makefile as the default build path.

```
make -f fpA_intB_standalone/Makefile.nvcc \
  GPU_ARCH=sm_90a \
  CUTLASS_DIR=$PWD/../../third_party/cutlass
```

The output binary is `fpA_intB_standalone/build_nvcc/test_fpA_intB_gemm`.
Use `GPU_ARCH=sm_80` for an Ampere build.

If `nvcc` is a Clang/PPU wrapper and fails with an `fpclassify` host/device
constexpr overload error, enable the Clang CUDA compatibility flag:

```
make -f fpA_intB_standalone/Makefile.nvcc \
  GPU_ARCH=sm_90a \
  CUTLASS_DIR=$PWD/../../third_party/cutlass \
  CLANG_CUDA_COMPAT=1
```

Run
---
Example:

```
fpA_intB_standalone/build_nvcc/test_fpA_intB_gemm \
  --m=1 --n=4096 --k=4096 --group_size=128
```

List configs:

```
fpA_intB_standalone/build_nvcc/test_fpA_intB_gemm --list_configs
```

Force CUDA kernel:

```
fpA_intB_standalone/build_nvcc/test_fpA_intB_gemm \
  --m=1 --n=4096 --k=4096 --group_size=128 \
  --config=cuda
```

Force a CUTLASS config (tile_m,tile_n,tile_k,stages,split_k):

```
fpA_intB_standalone/build_nvcc/test_fpA_intB_gemm \
  --m=1 --n=4096 --k=4096 --group_size=128 \
  --config=16x128x64x3x1
```

Interpreting `--list_configs` output (`tile_enum/stages/split_k`)
-----------------------------------------------------------------
`--list_configs` prints CUTLASS candidates in the form:

- `tile_enum=<int> stages=<int> split_k=<int>`

Meaning:

- `tile_enum`: the underlying integer value of `cutlass_extensions::CutlassTileConfig`
  (see `fpA_intB_standalone/cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/gemm_configs.h`).
  The enum name encodes the CTA tile shape, e.g. `CtaShape128x128x64_...` means `tile_m=128, tile_n=128, tile_k=64`.
- `stages`: CUTLASS mainloop pipeline stages.
- `split_k`: serial split-K factor (when `> 1`).

Example: `tile_enum=11 stages=3 split_k=7` corresponds to:

- `tile_enum=11` -> `CtaShape128x128x64_WarpShape128x32x64` -> `tile_m=128 tile_n=128 tile_k=64`
- so the CLI form is:

```
fpA_intB_standalone/build_nvcc/test_fpA_intB_gemm \
  --m=1 --n=2048 --k=3584 --group_size=128 \
  --config=128x128x64x3x7
```

Notes:
- `split_k` must divide K and keep `K / split_k` a multiple of 64 (weight-only fpA_intB requirement).
  For example `K=2048` cannot use `split_k=7` (not divisible).
- The standalone `--config=tile_m,tile_n,tile_k,stages,split_k` interface selects by tile shape; it does not accept
  `tile_enum` directly.

NCU mode (profile one iteration)
--------------------------------
`--ncu` disables the separate warmup loop and sets a larger default `--iters` so you can use Nsight Compute
to capture a single steady-state launch via `--launch-skip/--launch-count`.

Note: `--ncu` requires `--config=...` to avoid profiling-style config search.

Example:

```
fpA_intB_standalone/build_nvcc/test_fpA_intB_gemm \
  --m=1 --n=2048 --k=2048 --group_size=128 \
  --ncu --config=cuda --iters=2000
```

Debug profile logging (config search)
-------------------------------------
Set `FPA_INTB_PROFILE_LOG=1` to print each candidate config, its timing, and failure reasons:

```
FPA_INTB_PROFILE_LOG=1 \
fpA_intB_standalone/build_nvcc/test_fpA_intB_gemm \
  --m=1 --n=4096 --k=4096 --group_size=128
```

Traverse all configs without search
-----------------------------------
Use the helper script to run every candidate configuration directly:

```
fpA_intB_standalone/scripts/run_all_configs.sh \
  --bin fpA_intB_standalone/build_nvcc/test_fpA_intB_gemm \
  --m 1 --n 4096 --k 4096 --group 128 --warmup 10 --iters 100 --out results.txt
```

Skip the CUDA fallback:

```
fpA_intB_standalone/scripts/run_all_configs.sh \
  --bin fpA_intB_standalone/build_nvcc/test_fpA_intB_gemm \
  --m 1 --n 4096 --k 4096 --group 128 --warmup 10 --iters 100 --skip-cuda
```

CPU reference (small shapes only):

```
fpA_intB_standalone/build_nvcc/test_fpA_intB_gemm \
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
