# w4a16

This repo is a scratchpad to compare/extract W4A16 kernels (FP16 activations, INT4 weights).

Included projects
-----------------
- `fpA_intB_standalone/`: TensorRT-LLM extracted fpA_intB (FP16 x INT4, GPTQ scale+zero)
  - CUTLASS path + CUDA fallback for small M (e.g. M=1)
  - TRT-LLM-style config profiling (`--list_configs`, `--config=...`, `FPA_INTB_PROFILE_LOG=1`)
  - Tactic caching (`--tactic=<file>`) to skip profiling on repeated runs
- `moe_w4a16_standalone/`: TensorRT-LLM extracted MoE W4A16 grouped GEMM
  - FP16/BF16 activations with INT4 weights
  - Standalone tactic caching for Qwen MoE shapes
- `homemade_marlin/`: standalone Marlin kernel variants used for performance comparison
  - `MARLIN_USE_CREATEPOLICY=0` by default to avoid `createpolicy` illegal-instruction issues on some setups

## Build

```bash
make -f fpA_intB_standalone/Makefile.nvcc \
  GPU_ARCH=sm_90a \
  CUTLASS_DIR=$PWD/../../third_party/cutlass

make -f moe_w4a16_standalone/Makefile.nvcc \
  GPU_ARCH=sm_90 \
  CUTLASS_DIR=$PWD/../../third_party/cutlass
```

If `fpA_intB` is built with a Clang/PPU `nvcc` wrapper and hits an
`fpclassify` host/device constexpr overload error, add
`CLANG_CUDA_COMPAT=1` to the `fpA_intB` make command.

## Usage

```bash
# Basic run (profiles all configs on first call, slow ~5s)
fpA_intB_standalone/build_nvcc/test_fpA_intB_gemm --m=1 --n=12288 --k=3072 --group_size=128

# With tactic cache (recommended):
#   First run: profiles and saves best config to file
#   Subsequent runs with same (m,n,k,gs): loads directly, no profiling
fpA_intB_standalone/build_nvcc/test_fpA_intB_gemm --m=1 --n=12288 --k=3072 --group_size=128 \
    --tactic=tactics.cache --warmup=10 --iters=100

# Force a specific config (skip profiling entirely)
fpA_intB_standalone/build_nvcc/test_fpA_intB_gemm --m=1 --n=12288 --k=3072 --group_size=128 --config=cuda
fpA_intB_standalone/build_nvcc/test_fpA_intB_gemm --m=3823 --n=12288 --k=3072 --group_size=128 --config=64,16,64,4,1

# List all available configs
fpA_intB_standalone/build_nvcc/test_fpA_intB_gemm --list_configs

# Debug profiling (print per-config timing)
FPA_INTB_PROFILE_LOG=1 fpA_intB_standalone/build_nvcc/test_fpA_intB_gemm --m=3823 --n=4096 --k=4096 --group_size=128

# Correctness verification (small shapes only)
fpA_intB_standalone/build_nvcc/test_fpA_intB_gemm --m=2 --n=128 --k=128 --group_size=128 --verify

# Single inference, no profiling, no warmup (GPU only runs the GEMM kernel)
fpA_intB_standalone/build_nvcc/test_fpA_intB_gemm --m=1 --n=12288 --k=3072 --group_size=128 \
    --tactic=tactics.cache --warmup=0 --iters=1
```

### Tactic cache

The `--tactic=<file>` option saves/loads kernel configs to skip online profiling:

- **First run** with a new (m,n,k,gs): profiles all candidate configs (~5s), saves best to file
- **Subsequent runs** with same shape: loads config directly, zero overhead
- Pre-profiled cache for H800 PCIe included: `tactics_h800.cache`

File format (portable across builds, human-readable):

```
3823,12288,3072,128|cuda=0,tma=1,sm=90,tile90=128256128,ml=0,el=0,cl=2001001
1,12288,3072,128|cuda=1
```

Fields: `m,n,k,gs|` followed by config key-value pairs:
- `cuda=1`: CUDA core GEMV path (M < 16)
- `tma=1,sm=90,tile90=...,ml=...,el=...,cl=...`: SM90 TMA WGMMA path
- `tma=0,tile80=...,stages=...,splitk=...`: SM80 CUTLASS path

Delete the cache file to force re-profiling (e.g. when switching GPU arch).

## Notes

- Build directories are intentionally ignored (see `.gitignore`).
- `fpA_intB_standalone` defaults to `GPU_ARCH=sm_90a` for Hopper TMA kernels.
- `moe_w4a16_standalone` defaults to `GPU_ARCH=sm_90` because this extraction
  uses the SM80 grouped GEMM fallback on Hopper.
- Pass `GPU_ARCH=sm_80` for Ampere.
