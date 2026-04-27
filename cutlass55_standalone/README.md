CUTLASS Example 55 Standalone
=============================

This directory contains a standalone copy of CUTLASS example 55 for Hopper
FP16/BF16 activation x INT4 weight GEMM. The kernel configuration is the
example-55 shape:

- `TileShape=128x128x64` for FP16/BF16
- `ClusterShape=1x1x1`
- `KernelTmaWarpSpecializedCooperative`
- `TmaWarpSpecializedCooperative` epilogue
- optional `LayoutB_Reordered` value shuffle

The wrapper adds profiler-friendly controls to skip setup/reference kernels and
to launch exactly one target GEMM kernel.

Build
-----

```
cmake -S cutlass55_standalone -B cutlass55_standalone/build_cmake_release \
  -DGPU_ARCH=sm_90a \
  -DCUTLASS_DIR=$PWD/../../third_party/cutlass \
  -DCMAKE_BUILD_TYPE=Release

cmake --build cutlass55_standalone/build_cmake_release \
  --target cutlass55_fp16_gemm cutlass55_bf16_gemm -j$(nproc)
```

Direct `nvcc` build for FP16:

```
CUTLASS_DIR=$PWD/../../third_party/cutlass
nvcc -O3 -std=c++17 -arch=sm_90a --expt-relaxed-constexpr \
  -DCUTLASS_ARCH_MMA_SM90_SUPPORTED=1 \
  -DCUTLASS_USE_FP16=1 \
  -Icutlass55_standalone \
  -I$CUTLASS_DIR/include \
  -I$CUTLASS_DIR/tools/util/include \
  -I$CUTLASS_DIR/examples/common \
  -I$CUTLASS_DIR/examples/55_hopper_mixed_dtype_gemm \
  cutlass55_standalone/cutlass55_gemm.cu \
  -o /tmp/cutlass55_fp16_gemm
```

Run
---

Steady-state GEMM-only timing. This skips random fill, dequantize, reorder,
the correctness warmup GEMM, reference GEMM, and comparison kernel:

```
cutlass55_standalone/build_cmake_release/cutlass55_fp16_gemm \
  --m=4096 --n=4096 --k=4096 --g=128 \
  --mode=1 --shuffle=true \
  --skip_setup_kernels --skip_verify \
  --warmup=100 --iterations=1000
```

The skipped-setup modes intentionally leave input buffers uninitialized. They
are for profiler isolation only; output values are not meaningful.

Launch exactly one target GEMM kernel:

```
cutlass55_standalone/build_cmake_release/cutlass55_fp16_gemm \
  --m=4096 --n=4096 --k=4096 --g=128 \
  --mode=1 --shuffle=true \
  --single_kernel --profile_gemm_only
```

`--single_kernel` implies `--skip_setup_kernels`, `--skip_verify`,
`--warmup=0`, and `--iterations=1`.
