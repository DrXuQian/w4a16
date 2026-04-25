/*
 * Minimal SM90 fpA_intB CUTLASS kernel-attribute repro.
 *
 * This intentionally does not link libfpA_intB_gemm and does not run the test
 * harness. It instantiates one concrete Hopper TMA fp16 x int4 GEMM kernel
 * and calls the same runtime API that CUTLASS initialize() uses:
 *
 *   cudaFuncSetAttribute(cutlass::device_kernel<GemmKernel>,
 *       cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size)
 */

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "cutlass_extensions/epilogue_helpers.h"
#include "cutlass_extensions/gemm/collective/collective_builder_interleaved.hpp"
#include "cutlass_extensions/gemm_configs.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define FPA_INTB_REPRO_STRINGIZE_DETAIL(x) #x
#define FPA_INTB_REPRO_STRINGIZE(x) FPA_INTB_REPRO_STRINGIZE_DETAIL(x)

namespace
{

using namespace cute;

using CutlassActivationType = cutlass::half_t;
using CutlassWeightType = cutlass::uint4b_t;
using CutlassScaleZeroType = cutlass::half_t;
using CutlassBiasType = cutlass::half_t;
using CutlassOutputType = cutlass::half_t;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutBias = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<CutlassActivationType>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<CutlassWeightType>::value;
constexpr int AlignmentBias = 128 / cutlass::sizeof_bits<CutlassBiasType>::value;
constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<CutlassOutputType>::value;

using TileShape = Shape<_128, _256, _64>;
using ClusterShape = Shape<_2, _1, _1>;
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;

using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;
using ElementZero = CutlassScaleZeroType;
using ElementScale = CutlassScaleZeroType;
using ElementAccumulator = float;
using ElementCompute = float;
using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;

constexpr int EpiTileM = cute::min(shape<0>(TileShape{}), 128);
constexpr int EpiTileN = cute::min(shape<1>(TileShape{}), 32);
using EpilogueTileType = Shape<Int<EpiTileM>, Int<EpiTileN>>;
constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;

using EvtBiasAddition = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<cutlass::homogeneous_multiply_add, CutlassOutputType, ElementCompute,
        RoundStyle>,
    cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementAccumulator>, cutlass::epilogue::fusion::Sm90AccFetch,
    cutlass::epilogue::fusion::Sm90ColBroadcast<0, TileShape, CutlassBiasType, CutlassBiasType, Stride<_1, _0, _0>,
        AlignmentBias>>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<ArchTag, OperatorClass,
    TileShape, ClusterShape, EpilogueTileType, ElementAccumulator, ElementAccumulator, void,
    typename cutlass::layout::LayoutTranspose<LayoutBias>::type, AlignmentBias, CutlassOutputType,
    typename cutlass::layout::LayoutTranspose<LayoutOutput>::type, AlignmentOutput, EpilogueSchedule,
    EvtBiasAddition>::CollectiveOp;

using PackedScaleZero = cute::tuple<CutlassWeightType, ElementScale, ElementZero>;
using ElementBCollectiveInfo = PackedScaleZero;

using CollectiveMainloopInterleaved = typename cutlass::gemm::collective::CollectiveBuilderInterleaved<ArchTag, OperatorClass,
    ElementBCollectiveInfo, LayoutB_Transpose, AlignmentB, CutlassActivationType, LayoutA_Transpose, AlignmentA,
    ElementAccumulator, TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule>::CollectiveOp;

using CollectiveMainloopUpstream = typename cutlass::gemm::collective::CollectiveBuilder<ArchTag, OperatorClass,
    ElementBCollectiveInfo, LayoutB_Transpose, AlignmentB, CutlassActivationType, LayoutA_Transpose, AlignmentA,
    ElementAccumulator, TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule>::CollectiveOp;

#if defined(FPA_INTB_REPRO_USE_UPSTREAM_BUILDER)
using CollectiveMainloop = CollectiveMainloopUpstream;
char const* const kBuilderName = "upstream_cutlass_collective_builder";
#else
using CollectiveMainloop = CollectiveMainloopInterleaved;
char const* const kBuilderName = "trtllm_collective_builder_interleaved";
#endif

using TileScheduler = cutlass::gemm::StreamKScheduler;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop,
    CollectiveEpilogue, TileScheduler>;

void print_compile_info()
{
#if defined(__CUDACC_VER_MAJOR__)
    std::printf("nvcc=%d.%d.%d\n", __CUDACC_VER_MAJOR__,
#if defined(__CUDACC_VER_MINOR__)
        __CUDACC_VER_MINOR__,
#else
        0,
#endif
#if defined(__CUDACC_VER_BUILD__)
        __CUDACC_VER_BUILD__
#else
        0
#endif
    );
#else
    std::printf("nvcc=<not reported>\n");
#endif

#if defined(__CUDA_ARCH_LIST__)
    std::printf("__CUDA_ARCH_LIST__=%s\n", FPA_INTB_REPRO_STRINGIZE(__CUDA_ARCH_LIST__));
#else
    std::printf("__CUDA_ARCH_LIST__=<not defined>\n");
#endif
}

void print_cuda_info()
{
    int runtime_version = 0;
    int driver_version = 0;
    cudaRuntimeGetVersion(&runtime_version);
    cudaDriverGetVersion(&driver_version);
    std::printf("cuda runtime=%d driver=%d\n", runtime_version, driver_version);

    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);
    std::printf("device=%d name=\"%s\" cc=%d.%d sms=%d\n", device, prop.name, prop.major, prop.minor,
        prop.multiProcessorCount);

    int optin_smem = 0;
    int smem_per_block = 0;
    int smem_per_sm = 0;
    cudaDeviceGetAttribute(&optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    cudaDeviceGetAttribute(&smem_per_block, cudaDevAttrMaxSharedMemoryPerBlock, device);
    cudaDeviceGetAttribute(&smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
    std::printf("device_smem: per_block=%d optin_per_block=%d per_sm=%d\n", smem_per_block, optin_smem,
        smem_per_sm);
}

int parse_smem_arg(int argc, char** argv, int default_smem)
{
    for (int i = 1; i < argc; ++i)
    {
        char const* prefix = "--smem=";
        size_t const prefix_len = std::strlen(prefix);
        if (std::strncmp(argv[i], prefix, prefix_len) == 0)
        {
            return std::atoi(argv[i] + prefix_len);
        }
    }
    return default_smem;
}

} // namespace

int main(int argc, char** argv)
{
    cudaFree(nullptr);

    print_compile_info();
    print_cuda_info();

    int const computed_smem = static_cast<int>(sizeof(typename GemmKernel::SharedStorage));
    int const requested_smem = parse_smem_arg(argc, argv, computed_smem);

    std::printf("kernel: builder=%s tile=128x256x64 cluster=2x1x1 mainloop=cooperative epilogue=cooperative\n",
        kBuilderName);
    std::printf("kernel_traits: shared_storage=%d requested_smem=%d max_threads=%d load_wg=%d mma_wg=%d\n",
        computed_smem, requested_smem, GemmKernel::MaxThreadsPerBlock, GemmKernel::NumLoadWarpGroups,
        GemmKernel::NumMmaWarpGroups);

    cudaError_t err = cudaFuncSetAttribute(cutlass::device_kernel<GemmKernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, requested_smem);
    std::printf("cudaFuncSetAttribute max_dynamic_smem=%d: %s (%d)\n", requested_smem, cudaGetErrorString(err),
        static_cast<int>(err));

    cudaError_t last = cudaGetLastError();
    std::printf("cudaGetLastError: %s (%d)\n", cudaGetErrorString(last), static_cast<int>(last));

    return err == cudaSuccess ? 0 : 1;
}
