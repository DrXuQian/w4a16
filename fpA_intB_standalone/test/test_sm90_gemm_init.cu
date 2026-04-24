/*
 * Minimal test: reproduce CUTLASS SM90 mixed-input GEMM initialize() failure.
 * Single file, only needs CUTLASS headers + this repo's cutlass_extensions.
 *
 * Build (from fpA_intB_standalone/):
 *   nvcc -O2 -std=c++17 -arch=sm_90a --expt-relaxed-constexpr \
 *     -I cpp/include -I cpp -I cpp/tensorrt_llm -I cpp/tensorrt_llm/cutlass_extensions/include \
 *     -I <CUTLASS>/include \
 *     test/test_sm90_gemm_init.cu -o test_sm90_gemm_init
 *
 * Run:
 *   ./test_sm90_gemm_init
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>

// CUTLASS
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/util/packed_stride.hpp"

// TRT-LLM cutlass extensions
#include "cutlass_extensions/gemm/collective/collective_builder_interleaved.hpp"
#include "cutlass_extensions/epilogue_helpers.h"

using namespace cute;

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf("Max shared mem per block (optin): %zu bytes\n\n", prop.sharedMemPerBlockOptin);

    // Exactly matches the failing config: tile=128x256x128, cluster=2x1x1
    using ActivationType = cutlass::half_t;
    using WeightType = cutlass::uint4b_t;
    using ScaleType = cutlass::half_t;
    using OutputType = cutlass::half_t;
    using ElementAccumulator = float;

    using ArchTag = cutlass::arch::Sm90;
    using OperatorClass = cutlass::arch::OpClassTensorOp;

    // The failing tile + cluster from the error log
    using TileShape = Shape<_128, _256, cute::Int<128>>;
    using ClusterShape = Shape<_2, _1, _1>;

    using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;

    constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ActivationType>::value;  // 8
    constexpr int AlignmentB = 128 / cutlass::sizeof_bits<WeightType>::value;      // 32

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutA_T = cutlass::layout::ColumnMajor;
    using LayoutB_T = cutlass::layout::RowMajor;

    constexpr int epi_tile_M = 128;
    constexpr int epi_tile_N = 32;
    using EpilogueTileType = Shape<cute::Int<epi_tile_M>, cute::Int<epi_tile_N>>;

    // Simple epilogue: just alpha * acc (no bias)
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, TileShape, ClusterShape, EpilogueTileType,
        ElementAccumulator, ElementAccumulator,
        void, LayoutA_T, AlignmentA,  // C (unused)
        OutputType, LayoutA_T, AlignmentA,  // D
        EpilogueSchedule>::CollectiveOp;

    // Mixed-input mainloop with interleaved weights (same as fpA_intB)
    using PackedScale = cute::tuple<WeightType, ScaleType>;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilderInterleaved<
        ArchTag, OperatorClass,
        PackedScale, LayoutB_T, AlignmentB,
        ActivationType, LayoutA_T, AlignmentA,
        ElementAccumulator, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
        cutlass::gemm::PersistentScheduler>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    // Print kernel shared memory requirement
    int smem_size = static_cast<int>(sizeof(typename GemmKernel::SharedStorage));
    printf("Kernel SharedStorage size: %d bytes\n", smem_size);
    printf("Device max shared mem:     %zu bytes\n", prop.sharedMemPerBlockOptin);
    printf("Fits: %s\n\n", smem_size <= (int)prop.sharedMemPerBlockOptin ? "YES" : "NO *** WILL FAIL ***");

    // Test cudaFuncSetAttribute
    printf("Testing cudaFuncSetAttribute for %d bytes...\n", smem_size);
    auto attr_err = cudaFuncSetAttribute(
        cutlass::device_kernel<GemmKernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    printf("Result: %s\n\n", attr_err == cudaSuccess ? "OK" : cudaGetErrorString(attr_err));

    // Allocate minimal buffers and test initialize()
    int M = 128, N = 256, K = 128;
    printf("Testing gemm.initialize() with M=%d N=%d K=%d...\n", M, N, K);

    half *d_A, *d_C;
    uint8_t *d_B;
    half *d_scales;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N / 2);
    cudaMalloc(&d_C, M * N * sizeof(half));
    cudaMalloc(&d_scales, (K / 128) * N * sizeof(half));

    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideD = typename GemmKernel::StrideD;

    // Note: A and B are swapped in the kernel (transpose trick)
    StrideA stride_B = cutlass::make_cute_packed_stride(StrideA{}, {N, K, 1});
    StrideB stride_A = cutlass::make_cute_packed_stride(StrideB{}, {M, K, 1});
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {N, M, 1});

    int group_size = 128;
    int groups_per_k = K / group_size;
    using StrideS = typename CollectiveMainloop::StrideScale;
    StrideS stride_S = cutlass::make_cute_packed_stride(StrideS{}, {groups_per_k, N, 1});

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {N, M, K, 1},
        {reinterpret_cast<WeightType const*>(d_B), stride_B,
         reinterpret_cast<ActivationType const*>(d_A), stride_A,
         reinterpret_cast<ScaleType const*>(d_scales), stride_S, group_size},
        {{1.0f}, {}, {}, reinterpret_cast<OutputType*>(d_C), stride_D}
    };

    Gemm gemm;

    auto can = gemm.can_implement(args);
    printf("can_implement: %s\n", cutlassGetStatusString(can));

    size_t ws_size = gemm.get_workspace_size(args);
    printf("workspace_size: %zu\n", ws_size);
    void* d_ws = nullptr;
    if (ws_size > 0) cudaMalloc(&d_ws, ws_size);

    auto init = gemm.initialize(args, d_ws);
    printf("initialize: %s\n", cutlassGetStatusString(init));

    if (init == cutlass::Status::kSuccess) {
        auto run = gemm.run();
        cudaDeviceSynchronize();
        printf("run: %s\n", cutlassGetStatusString(run));
        cudaError_t err = cudaGetLastError();
        printf("cuda after run: %s\n", err == cudaSuccess ? "OK" : cudaGetErrorString(err));
        printf("\nPASS\n");
    } else {
        cudaError_t err = cudaGetLastError();
        printf("cuda error: %s\n", err == cudaSuccess ? "none" : cudaGetErrorString(err));
        printf("\nFAIL\n");
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_scales);
    if (d_ws) cudaFree(d_ws);
    return 0;
}
