# w4a16

This repo is a scratchpad to compare/extract W4A16 kernels (FP16 activations, INT4 weights).

Included projects
-----------------
- `fpA_intB_standalone/`: TensorRT-LLM extracted fpA_intB (FP16 x INT4, GPTQ scale+zero)
  - CUTLASS path + CUDA fallback for small M (e.g. M=1)
  - TRT-LLM-style config profiling (`--list_configs`, `--config=...`, `FPA_INTB_PROFILE_LOG=1`)
  - Tactic caching (`--tactic=<file>`) to skip profiling on repeated runs
- `homemade_marlin/`: standalone Marlin kernel variants used for performance comparison
  - `MARLIN_USE_CREATEPOLICY=0` by default to avoid `createpolicy` illegal-instruction issues on some setups

## Build

```bash
cd fpA_intB_standalone
mkdir -p build && cd build
cmake .. -DCUTLASS_DIR=<path_to_cutlass> -DCMAKE_CUDA_ARCHITECTURES="90a-real"
make -j$(nproc)
```

## Usage

```bash
cd fpA_intB_standalone/build

# Basic run (profiles all configs on first call, slow ~5s)
./test_fpA_intB_gemm --m=1 --n=12288 --k=3072 --group_size=128

# With tactic cache (recommended):
#   First run: profiles and saves best config to file
#   Subsequent runs with same (m,n,k,gs): loads directly, no profiling
./test_fpA_intB_gemm --m=1 --n=12288 --k=3072 --group_size=128 \
    --tactic=tactics.cache --warmup=10 --iters=100

# Force a specific config (skip profiling entirely)
./test_fpA_intB_gemm --m=1 --n=12288 --k=3072 --group_size=128 --config=cuda
./test_fpA_intB_gemm --m=3823 --n=12288 --k=3072 --group_size=128 --config=64,16,64,4,1

# List all available configs
./test_fpA_intB_gemm --list_configs

# Debug profiling (print per-config timing)
FPA_INTB_PROFILE_LOG=1 ./test_fpA_intB_gemm --m=3823 --n=4096 --k=4096 --group_size=128

# Correctness verification (small shapes only)
./test_fpA_intB_gemm --m=2 --n=128 --k=128 --group_size=128 --verify

# Nsight Compute profiling (requires --config to avoid profiling loop)
ncu --set full ./test_fpA_intB_gemm --m=1 --n=4096 --k=4096 --group_size=128 --config=cuda --ncu
```

### Tactic cache file format

The `--tactic=<file>` option saves/loads kernel configs as a simple CSV:

```
# m,n,k,group_size,config_index
1,12288,3072,128,21
3823,12288,3072,128,18
```

- `config_index` is the index into the candidate config list (from `--list_configs`)
- The file is append-only: new shapes are added automatically
- Delete the file to force re-profiling (e.g. after recompiling with different arch)

## Notes

- Build directories are intentionally ignored (see `.gitignore`).
- Each subproject defaults to a target arch in its own `CMakeLists.txt`; override with
  `-DCMAKE_CUDA_ARCHITECTURES=...` when building on a different GPU.
- SM90 (Hopper) requires `90a-real` to enable WGMMA instructions.
