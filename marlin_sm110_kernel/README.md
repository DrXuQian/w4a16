# Marlin SM110 Kernel (Extracted from vLLM)

This is a standalone extraction of the vLLM Marlin kernel specialized for INT4 GPTQ and compiled for SM110.

## What is included
- Kernel headers copied from `vllm/csrc/quantization/gptq_marlin/`
- Minimal scalar type header copied from `vllm/csrc/core/`
- A stub `torch/library.h` to satisfy `TORCH_CHECK` in `scalar_type.hpp`
- GPTQ INT4 kernel instantiations for the same config grid as vLLM
- Heuristic selector (shape-based) that mirrors vLLM `gptq_marlin` selection
- Standalone test CLI with configurable `M/N/K/group_size`

## Build
```
cmake -S . -B build
cmake --build build -j
```

By default this builds for SM110. Override when building on a different GPU, e.g.:

```
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j
```

This produces `libmarlin_sm110_kernel.a` and the test executable.

## Test
```
./build/test_marlin_sm110 --m=1 --n=2048 --k=2048 --group_size=128
```

List compiled configs:
```
./build/test_marlin_sm110 --list_configs
```
