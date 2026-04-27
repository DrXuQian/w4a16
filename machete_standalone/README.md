Standalone vLLM Machete GEMM for SM90
=====================================

This directory contains a minimal, standalone extraction of vLLM's Machete
weight-only GEMM path. It keeps the Machete prepacked weight layout, SM90 TMA
warp-specialized CUTLASS kernel, and the H100/H800 heuristic schedule selector
from vLLM.

Source
------
- vLLM tree used for extraction:
  `/root/autodl-tmp/awesome-cute/Vibe-CUTE/3rdparty/vllm`
- Source commit observed locally:
  `e3126cd107460444d7fd9a1445b8d4f4393a06b2`
- Main extracted files:
  - `csrc/quantization/machete/*`
  - `csrc/cutlass_extensions/*`

Scope
-----
Supported:

- SM90/SM90a Hopper builds
- FP16 and BF16 activations
- INT4 weights:
  - GPTQ-style `u4b8` with group scales
  - AWQ-style `u4` with group scales and group zeros
- vLLM Machete prepack kernel
- vLLM H100 heuristic schedule selection

Not included:

- PyTorch/ATen bindings
- u8/u8b128 kernels
- channel scales or token scales
- SM80/SM100+ specializations

Build
-----
From this directory's parent repo:

```
cmake -S machete_standalone -B machete_standalone/build_cmake_release \
  -DGPU_ARCH=sm_90a \
  -DCUTLASS_DIR=$PWD/../../third_party/cutlass \
  -DCMAKE_BUILD_TYPE=Release

cmake --build machete_standalone/build_cmake_release \
  --target test_machete_gemm -j$(nproc)
```

If the machine has multiple CUDA/PPU SDKs in `PATH`, configure CMake with the
intended compiler explicitly:

```
CUDA_ROOT=/path/to/CUDA_SDK
CUDACXX=$CUDA_ROOT/bin/nvcc \
cmake -S machete_standalone -B machete_standalone/build_cmake_release \
  -DCUDAToolkit_ROOT=$CUDA_ROOT \
  -DGPU_ARCH=sm_90a \
  -DCUTLASS_DIR=$PWD/../../third_party/cutlass \
  -DCMAKE_BUILD_TYPE=Release
```

Run
---
Default run, using heuristic schedule selection:

```
machete_standalone/build_cmake_release/test_machete_gemm \
  --m=4096 --n=4096 --k=4096 --group_size=128 \
  --act=fp16 --quant=gptq_u4b8 \
  --warmup=100 --iters=1000
```

AWQ-style u4:

```
machete_standalone/build_cmake_release/test_machete_gemm \
  --m=4096 --n=4096 --k=4096 --group_size=128 \
  --act=bf16 --quant=awq_u4 \
  --warmup=100 --iters=1000
```

List schedules:

```
machete_standalone/build_cmake_release/test_machete_gemm --list_schedules
```

Force a schedule:

```
machete_standalone/build_cmake_release/test_machete_gemm \
  --m=128 --n=4096 --k=4096 --group_size=128 \
  --schedule=128x32_2x1x1_TmaMI_TmaCoop_streamK
```

Run all schedules:

```
machete_standalone/scripts/run_all_schedules.sh \
  --bin machete_standalone/build_cmake_release/test_machete_gemm \
  --m 4096 --n 4096 --k 4096 --group 128 \
  --act fp16 --quant gptq_u4b8 \
  --warmup 100 --iters 1000 --out /tmp/machete_4096_all_schedules.txt
```

H800 SM90 performance: 4096x4096x4096
-------------------------------------
Measured on H800 PCIe with `build_cmake_release`, `group_size=128`,
`warmup=100`, and `iters=1000`. The reported time is CUDA event time around the
timed GEMM loop; Machete prepack is run before timing. Effective dense TFLOPS is
`2*M*N*K / time`.

Command:

```
machete_standalone/build_cmake_release/test_machete_gemm \
  --m=4096 --n=4096 --k=4096 --group_size=128 \
  --act=fp16 --quant=gptq_u4b8 \
  --warmup=100 --iters=1000
```

| Activation | Quant | Selected schedule | Avg time (us) | Effective TFLOPS |
|---|---|---|---:|---:|
| FP16 | GPTQ `u4b8` | `128x128_2x1x1_TmaMI_TmaCoop_streamK` | 354.820 | 387.3 |
| FP16 | AWQ `u4` | `128x128_2x1x1_TmaMI_TmaCoop_streamK` | 370.910 | 370.5 |
| BF16 | GPTQ `u4b8` | `128x128_2x1x1_TmaMI_TmaCoop_streamK` | 358.621 | 383.2 |
| BF16 | AWQ `u4` | `128x128_2x1x1_TmaMI_TmaCoop_streamK` | 360.375 | 381.4 |

CUTLASS example 55 comparison
-----------------------------
For reference, CUTLASS example 55 was measured on the same shape with
`--g=128 --l=1 --shuffle=true --warmup=100 --iterations=1000`.

| Kernel | Mode | Avg time (us) | Effective TFLOPS |
|---|---|---:|---:|
| example 55 BF16 | mode 1, group scale | 438.524 | 313.4 |
| example 55 BF16 | mode 2, group scale + zero | 470.285 | 292.2 |
| example 55 FP16 | mode 1, group scale | 439.630 | 312.6 |
| example 55 FP16 | mode 2, group scale + zero | 477.292 | 288.0 |

Machete and example 55 do not use the same kernel even when both have a
`128x128x64` threadblock tile:

- vLLM's heuristic selects `128x128_2x1x1_TmaMI_TmaCoop_streamK` for 4096^3,
  while example 55 is hard-coded to `TileShape=128x128x64` and
  `ClusterShape=1x1x1`.
- Machete uses `PrepackedLayoutBTemplate` and `MacheteCollectiveMma`, so the
  int4 weights are prepacked to match the WGMMA register-load pattern more
  directly.
- Example 55 uses CUTLASS's generic mixed-input `CollectiveBuilder` with
  `LayoutB_Reordered` and `ValueShuffle`. Its `--shuffle=true` path is an
  offline value reorder, but it is still not Machete's prepacked layout or
  custom mainloop.
- Machete launches the transposed formulation internally (`Y^T = W^T X^T`) to
  feed the decompressed weight operand through the GMMA register-sourced side.
  This changes the memory layout and schedule interpretation relative to the
  example 55 benchmark.

Heuristic
---------
This is copied from vLLM `csrc/quantization/machete/generate.py`.
The condition order matters.

| Condition | Schedule |
|---|---|
| `M > 256 && K <= 16384 && N <= 4096` | `128x128_2x1x1_TmaMI_TmaCoop_streamK` |
| `M > 256` | `128x256_2x1x1_TmaMI_TmaCoop_streamK` |
| `M > 128 && K <= 4096 && N <= 4096` | `128x64_2x1x1_TmaMI_TmaCoop_streamK` |
| `M > 128 && K <= 8192 && N <= 8192` | `128x128_2x1x1_TmaMI_TmaCoop_streamK` |
| `M > 128` | `128x256_2x1x1_TmaMI_TmaCoop_streamK` |
| `M > 64 && K <= 4069 && N <= 4069` | `128x32_2x1x1_TmaMI_TmaCoop_streamK` |
| `M > 64 && K <= 4069 && N <= 8192` | `128x64_2x1x1_TmaMI_TmaCoop_streamK` |
| `M > 64 && K >= 8192 && N >= 12288` | `256x128_2x1x1_TmaMI_TmaCoop_streamK` |
| `M > 64` | `128x128_2x1x1_TmaMI_TmaCoop_streamK` |
| `M > 32 && K <= 6144 && N <= 6144` | `128x16_1x1x1_TmaMI_TmaCoop_streamK` |
| `M > 32 && K >= 16384 && N >= 12288` | `256x64_2x1x1_TmaMI_TmaCoop_streamK` |
| `M > 32` | `128x64_2x1x1_TmaMI_TmaCoop_streamK` |
| `M > 16 && K <= 12288 && N <= 8192` | `128x32_2x1x1_TmaMI_TmaCoop_streamK` |
| `M > 16` | `256x32_2x1x1_TmaMI_TmaCoop_streamK` |
| `N >= 26624` | `256x16_1x1x1_TmaMI_TmaCoop_streamK` |
| fallback | `128x16_1x1x1_TmaMI_TmaCoop_streamK` |

Supported schedule set:

```
128x128_2x1x1_TmaMI_TmaCoop_streamK
128x256_2x1x1_TmaMI_TmaCoop_streamK
128x64_2x1x1_TmaMI_TmaCoop_streamK
128x32_2x1x1_TmaMI_TmaCoop_streamK
256x128_2x1x1_TmaMI_TmaCoop_streamK
128x16_1x1x1_TmaMI_TmaCoop_streamK
256x64_2x1x1_TmaMI_TmaCoop_streamK
256x32_2x1x1_TmaMI_TmaCoop_streamK
256x16_1x1x1_TmaMI_TmaCoop_streamK
```

Smoke Test
----------
Verified locally on H800 PCIe, compute capability 9.0:

```
test_machete_gemm --m=128 --n=128 --k=128 --group_size=128 --warmup=1 --iters=1
```

The four combinations below launch successfully:

- `--act=fp16 --quant=gptq_u4b8`
- `--act=bf16 --quant=gptq_u4b8`
- `--act=fp16 --quant=awq_u4`
- `--act=bf16 --quant=awq_u4`

For `4096x4096x4096`, the heuristic selects
`128x128_2x1x1_TmaMI_TmaCoop_streamK`.
