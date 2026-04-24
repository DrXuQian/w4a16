/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifdef __GNUC__ // Check if the compiler is GCC or Clang
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif // __GNUC__

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/packed_stride.hpp"

#include "cutlass_extensions/compute_occupancy.h"
#include "cutlass_extensions/epilogue_helpers.h"
#include "cutlass_extensions/gemm_configs.h"

#include "cutlass_extensions/gemm/collective/collective_builder_interleaved.hpp"

#ifdef __GNUC__ // Check if the compiler is GCC or Clang
#pragma GCC diagnostic pop
#endif          // __GNUC__

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_type_conversion.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/launchers/fpA_intB_launcher_sm90.h"

#include <typeinfo>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace cutlass_kernels_oss
{
using namespace tensorrt_llm::kernels::cutlass_kernels;
namespace tk = tensorrt_llm::common;
namespace tkc = tensorrt_llm::cutlass_extensions;

using namespace cute;

#define FPA_INTB_STRINGIZE_DETAIL(x) #x
#define FPA_INTB_STRINGIZE(x) FPA_INTB_STRINGIZE_DETAIL(x)

namespace
{

inline void fpAIntBSm90PrintCompileInfo()
{
#if defined(__CUDACC_VER_MAJOR__)
    std::fprintf(stderr, "[fpA_intB SM90 diag] nvcc=%d.%d.%d\n", __CUDACC_VER_MAJOR__,
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
    std::fprintf(stderr, "[fpA_intB SM90 diag] nvcc=<not reported>\n");
#endif

#if defined(__CUDA_ARCH_LIST__)
    std::fprintf(stderr, "[fpA_intB SM90 diag] __CUDA_ARCH_LIST__=%s\n", FPA_INTB_STRINGIZE(__CUDA_ARCH_LIST__));
#else
    std::fprintf(stderr, "[fpA_intB SM90 diag] __CUDA_ARCH_LIST__=<not defined>\n");
#endif

#if defined(COMPILE_HOPPER_TMA_GEMMS)
    std::fprintf(stderr, "[fpA_intB SM90 diag] COMPILE_HOPPER_TMA_GEMMS=1\n");
#else
    std::fprintf(stderr, "[fpA_intB SM90 diag] COMPILE_HOPPER_TMA_GEMMS=0\n");
#endif
}

inline void fpAIntBSm90PrintDeviceInfo()
{
    int runtime_version = 0;
    int driver_version = 0;
    cudaRuntimeGetVersion(&runtime_version);
    cudaDriverGetVersion(&driver_version);
    std::fprintf(stderr, "[fpA_intB SM90 diag] cuda runtime=%d driver=%d\n", runtime_version, driver_version);

    int device = -1;
    cudaError_t device_err = cudaGetDevice(&device);
    if (device_err != cudaSuccess)
    {
        std::fprintf(stderr, "[fpA_intB SM90 diag] cudaGetDevice FAILED: %s (%d)\n", cudaGetErrorString(device_err),
            static_cast<int>(device_err));
        return;
    }

    cudaDeviceProp prop{};
    cudaError_t prop_err = cudaGetDeviceProperties(&prop, device);
    if (prop_err != cudaSuccess)
    {
        std::fprintf(stderr, "[fpA_intB SM90 diag] cudaGetDeviceProperties FAILED: %s (%d)\n",
            cudaGetErrorString(prop_err), static_cast<int>(prop_err));
        return;
    }

    int optin_smem = 0;
    int block_smem = 0;
    int smem_per_sm = 0;
    int max_blocks_per_sm = 0;
    cudaDeviceGetAttribute(&optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    cudaDeviceGetAttribute(&block_smem, cudaDevAttrMaxSharedMemoryPerBlock, device);
    cudaDeviceGetAttribute(&smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
    cudaDeviceGetAttribute(&max_blocks_per_sm, cudaDevAttrMaxBlocksPerMultiprocessor, device);

    std::fprintf(stderr,
        "[fpA_intB SM90 diag] device=%d name=\"%s\" cc=%d.%d sms=%d max_threads_per_block=%d\n", device,
        prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.maxThreadsPerBlock);
    std::fprintf(stderr,
        "[fpA_intB SM90 diag] device_smem: per_block=%d optin_per_block=%d per_sm=%d max_blocks_per_sm=%d\n",
        block_smem, optin_smem, smem_per_sm, max_blocks_per_sm);
}

template <typename GemmKernel>
bool fpAIntBSm90DiagnoseKernel(char const* where, int m, int n, int k, int group_size, size_t workspace_needed,
    size_t workspace_bytes, void* workspace, dim3 grid, dim3 block)
{
    auto kernel = cutlass::device_kernel<GemmKernel>;
    auto kernel_ptr = reinterpret_cast<void const*>(kernel);
    int const requested_smem = static_cast<int>(sizeof(typename GemmKernel::SharedStorage));
    int device = -1;
    int optin_smem = 0;
    if (cudaGetDevice(&device) == cudaSuccess)
    {
        cudaDeviceGetAttribute(&optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    }

    std::fprintf(stderr, "[fpA_intB SM90 diag] ===== %s =====\n", where);
    fpAIntBSm90PrintCompileInfo();
    fpAIntBSm90PrintDeviceInfo();
    std::fprintf(stderr,
        "[fpA_intB SM90 diag] problem: m=%d n=%d k=%d group=%d workspace_needed=%zu workspace_provided=%zu "
        "workspace=%p\n",
        m, n, k, group_size, workspace_needed, workspace_bytes, workspace);
    std::fprintf(stderr,
        "[fpA_intB SM90 diag] launch_shape: grid=(%u,%u,%u) block=(%u,%u,%u) kernel_ptr=%p\n", grid.x, grid.y,
        grid.z, block.x, block.y, block.z, kernel_ptr);
    std::fprintf(stderr,
        "[fpA_intB SM90 diag] kernel_traits: shared_storage=%d max_threads=%d load_wg=%d mma_wg=%d type=%s\n",
        requested_smem, GemmKernel::MaxThreadsPerBlock, GemmKernel::NumLoadWarpGroups, GemmKernel::NumMmaWarpGroups,
        typeid(GemmKernel).name());

    cudaFuncAttributes attr{};
    cudaError_t get_attr = cudaFuncGetAttributes(&attr, kernel);
    std::fprintf(stderr, "[fpA_intB SM90 diag] cudaFuncGetAttributes: %s (%d)\n", cudaGetErrorString(get_attr),
        static_cast<int>(get_attr));
    if (get_attr == cudaSuccess)
    {
        std::fprintf(stderr,
            "[fpA_intB SM90 diag] func_attr: binary=%d ptx=%d static_smem=%zu max_dyn_smem=%d const=%zu local=%zu "
            "regs=%d max_threads=%d cacheModeCA=%d preferred_carveout=%d\n",
            attr.binaryVersion, attr.ptxVersion, static_cast<size_t>(attr.sharedSizeBytes),
            attr.maxDynamicSharedSizeBytes, static_cast<size_t>(attr.constSizeBytes),
            static_cast<size_t>(attr.localSizeBytes), attr.numRegs, attr.maxThreadsPerBlock, attr.cacheModeCA,
            attr.preferredShmemCarveout);
    }

    cudaError_t carveout_err = cudaFuncSetAttribute(
        kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    std::fprintf(stderr, "[fpA_intB SM90 diag] cudaFuncSetAttribute carveout=max_shared: %s (%d)\n",
        cudaGetErrorString(carveout_err), static_cast<int>(carveout_err));

    cudaError_t set_smem = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, requested_smem);
    std::fprintf(stderr, "[fpA_intB SM90 diag] cudaFuncSetAttribute max_dynamic_smem=%d: %s (%d)\n",
        requested_smem, cudaGetErrorString(set_smem), static_cast<int>(set_smem));

    int active_blocks_zero_smem = -1;
    cudaError_t occ_zero = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_zero_smem, kernel, GemmKernel::MaxThreadsPerBlock, 0);
    std::fprintf(stderr, "[fpA_intB SM90 diag] occupancy smem=0: %s (%d), active_blocks=%d\n",
        cudaGetErrorString(occ_zero), static_cast<int>(occ_zero), active_blocks_zero_smem);

    int active_blocks_requested_smem = -1;
    cudaError_t occ_requested = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_requested_smem, kernel, GemmKernel::MaxThreadsPerBlock, requested_smem);
    std::fprintf(stderr, "[fpA_intB SM90 diag] occupancy smem=%d: %s (%d), active_blocks=%d\n", requested_smem,
        cudaGetErrorString(occ_requested), static_cast<int>(occ_requested), active_blocks_requested_smem);

    if (get_attr != cudaSuccess)
    {
        std::fprintf(stderr,
            "[fpA_intB SM90 diag] diagnosis: kernel function is not loadable/registered in this binary.\n");
    }
    else if (requested_smem > attr.maxDynamicSharedSizeBytes)
    {
        std::fprintf(stderr,
            "[fpA_intB SM90 diag] diagnosis: requested dynamic smem exceeds this kernel's maxDynamicSharedSizeBytes.\n");
    }
    else if (optin_smem > 0 && requested_smem + static_cast<int>(attr.sharedSizeBytes) > optin_smem)
    {
        std::fprintf(stderr,
            "[fpA_intB SM90 diag] diagnosis: requested dynamic smem plus static smem exceeds device opt-in limit.\n");
    }
    else if (set_smem != cudaSuccess)
    {
        std::fprintf(stderr,
            "[fpA_intB SM90 diag] diagnosis: cuda rejected a valid-looking smem opt-in for this concrete CUTLASS "
            "kernel; suspect device image metadata or runtime/toolchain handling.\n");
    }
    else
    {
        std::fprintf(stderr,
            "[fpA_intB SM90 diag] diagnosis: kernel attributes and smem opt-in look valid before CUTLASS initialize.\n");
    }

    if (get_attr != cudaSuccess || carveout_err != cudaSuccess || set_smem != cudaSuccess || occ_zero != cudaSuccess
        || occ_requested != cudaSuccess)
    {
        cudaError_t cleared = cudaGetLastError();
        std::fprintf(stderr, "[fpA_intB SM90 diag] cudaGetLastError(clear): %s (%d)\n", cudaGetErrorString(cleared),
            static_cast<int>(cleared));
    }

    std::fprintf(stderr, "[fpA_intB SM90 diag] ===== end %s =====\n", where);
    return set_smem == cudaSuccess;
}

template <typename GemmKernel, typename Arguments>
void fpAIntBSm90DiagnoseInitializeSteps(Arguments const& args, void* workspace, cudaStream_t stream)
{
    std::fprintf(stderr, "[fpA_intB SM90 diag] ----- initialize step replay -----\n");

    cutlass::Status workspace_status = GemmKernel::initialize_workspace(args, workspace, stream, nullptr);
    std::fprintf(stderr, "[fpA_intB SM90 diag] init_step initialize_workspace: %s\n",
        cutlassGetStatusString(workspace_status));
    cudaError_t workspace_cuda_err = cudaGetLastError();
    std::fprintf(stderr, "[fpA_intB SM90 diag] init_step initialize_workspace cudaGetLastError: %s (%d)\n",
        cudaGetErrorString(workspace_cuda_err), static_cast<int>(workspace_cuda_err));

    if (workspace_status != cutlass::Status::kSuccess)
    {
        std::fprintf(stderr,
            "[fpA_intB SM90 diag] init_step diagnosis: initialize_workspace failed before kernel attribute setup.\n");
        std::fprintf(stderr, "[fpA_intB SM90 diag] ----- end initialize step replay -----\n");
        return;
    }

    auto params = GemmKernel::to_underlying_arguments(args, workspace);
    dim3 const params_grid = GemmKernel::get_grid_shape(params);
    dim3 const params_block = GemmKernel::get_block_shape();
    std::fprintf(stderr,
        "[fpA_intB SM90 diag] init_step to_underlying_arguments: OK, grid=(%u,%u,%u) block=(%u,%u,%u)\n",
        params_grid.x, params_grid.y, params_grid.z, params_block.x, params_block.y, params_block.z);

    int const smem_size = static_cast<int>(sizeof(typename GemmKernel::SharedStorage));
    cudaError_t set_smem = cudaFuncSetAttribute(
        cutlass::device_kernel<GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    std::fprintf(stderr, "[fpA_intB SM90 diag] init_step cudaFuncSetAttribute max_dynamic_smem=%d: %s (%d)\n",
        smem_size, cudaGetErrorString(set_smem), static_cast<int>(set_smem));
    cudaError_t set_smem_last = cudaGetLastError();
    std::fprintf(stderr, "[fpA_intB SM90 diag] init_step cudaFuncSetAttribute cudaGetLastError: %s (%d)\n",
        cudaGetErrorString(set_smem_last), static_cast<int>(set_smem_last));

    if (set_smem != cudaSuccess)
    {
        std::fprintf(stderr,
            "[fpA_intB SM90 diag] init_step diagnosis: CUTLASS initialize fails at dynamic shared-memory attribute "
            "setup.\n");
    }
    else
    {
        std::fprintf(stderr,
            "[fpA_intB SM90 diag] init_step diagnosis: replayed initialize steps succeeded; original failure is not "
            "from initialize_workspace or smem attribute setup.\n");
    }

    std::fprintf(stderr, "[fpA_intB SM90 diag] ----- end initialize step replay -----\n");
}

} // namespace

template <typename ActivationType, typename WeightType, typename ScaleZeroType, typename BiasType, typename OutputType,
    cutlass::WeightOnlyQuantOp QuantOp, typename EpilogueTag, typename CTAShape, typename ClusterShape,
    typename MainloopScheduleType, typename EpilogueScheduleType>
void sm90_generic_mixed_gemm_kernelLauncher(ActivationType const* A, WeightType const* B,
    ScaleZeroType const* weight_scales, ScaleZeroType const* weight_zero_points, BiasType const* biases,
    float const alpha, OutputType* C, int m, int n, int k, int const group_size, tkc::CutlassGemmConfig gemm_config,
    char* workspace, size_t workspace_bytes, cudaStream_t stream, int* occupancy)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);

#ifdef COMPILE_HOPPER_TMA_GEMMS
    using CutlassActivationType = typename TllmToCutlassTypeAdapter<ActivationType>::type;

    if constexpr (!should_filter_tma_warp_specialized_gemm_problem_shape_v<cutlass::arch::Sm90, CTAShape, ClusterShape,
                      false, ActivationType>)
    {
        using CutlassWeightType = typename TllmToCutlassTypeAdapter<WeightType>::type;

        using CutlassScaleZeroType = typename TllmToCutlassTypeAdapter<ScaleZeroType>::type;
        using CutlassBiasType = typename TllmToCutlassTypeAdapter<BiasType>::type;
        using CutlassOutputType = typename TllmToCutlassTypeAdapter<OutputType>::type;

        static_assert(std::is_same_v<CutlassActivationType, cutlass::half_t>
                || std::is_same_v<CutlassActivationType, cutlass::bfloat16_t>
                || std::is_same_v<CutlassActivationType, cutlass::float_e4m3_t>
                || std::is_same_v<CutlassActivationType, cutlass::float_e5m2_t>,
            "Activation type must be bfloat16, half, FP8");

        static_assert(std::is_same_v<CutlassWeightType, uint8_t> || std::is_same_v<CutlassWeightType, cutlass::uint4b_t>
                || std::is_same_v<CutlassWeightType, cutlass::float_e4m3_t>
                || std::is_same_v<CutlassWeightType, cutlass::float_e5m2_t>,
            "Weight type must be fp8, uint8_t or uint4_t");

        static_assert(!std::is_same_v<CutlassActivationType, cutlass::float_e4m3_t>
                || std::is_same_v<CutlassScaleZeroType, cutlass::half_t>,
            "Scale/Zero type must be half for fp8 activation");

        using LayoutA = cutlass::layout::RowMajor; // Layout type for A matrix operand
        constexpr int AlignmentA = 128 / cutlass::sizeof_bits<CutlassActivationType>::value;

        using LayoutB = cutlass::layout::ColumnMajor; // Layout type for B matrix operand
        constexpr int AlignmentB = 128 / cutlass::sizeof_bits<CutlassWeightType>::value;

        // This example manually swaps and transposes, so keep transpose of input layouts
        using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
        using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

        using ElementZero = CutlassScaleZeroType;
        using ElementScale = CutlassScaleZeroType;

        // C/D matrix configuration. We reuse the C operand for the bias and set the stride for broadcast.
        using LayoutBias = cutlass::layout::RowMajor;
        constexpr int AlignmentBias = 128 / cutlass::sizeof_bits<CutlassBiasType>::value;

        // D matrix configuration
        using LayoutOutput = cutlass::layout::RowMajor;
        constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<CutlassOutputType>::value;

        // Core kernel configurations
        using ElementAccumulator = float;    // Element type for internal accumulation
        using ElementCompute = float;        // Element type for epilogue computation
        using ArchTag = cutlass::arch::Sm90; // Tag indicating the minimum SM that supports the intended feature
        using OperatorClass = cutlass::arch::OpClassTensorOp; // Operator class tag
        using TileShape = CTAShape;                           // Threadblock-level tile size
        using KernelSchedule = MainloopScheduleType;
        using EpilogueSchedule = EpilogueScheduleType;

        // Shrink the N dimension to match CTA_N if needed
        constexpr int epi_tile_M = cute::min(shape<0>(TileShape{}), 128); // 64 or 128
        constexpr int epi_tile_N = cute::min(shape<1>(TileShape{}), 32);  // Allow this to be 16 for some small N tiles.
        using EpilogueTileType = cute::Shape<cute::Int<epi_tile_M>, cute::Int<epi_tile_N>>;

        static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
        static_assert(std::is_same_v<EpilogueTag, tensorrt_llm::cutlass_extensions::EpilogueOpBias>, "");
        using EVT_bias_addition = cutlass::epilogue::fusion::Sm90EVT<
            cutlass::epilogue::fusion::Sm90Compute<cutlass::homogeneous_multiply_add, CutlassOutputType, ElementCompute,
                RoundStyle>,                                                    // alpha * acc + bias
            cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementAccumulator>, // alpha
            cutlass::epilogue::fusion::Sm90AccFetch,                            // acc
            cutlass::epilogue::fusion::Sm90ColBroadcast<0, TileShape, CutlassBiasType, CutlassBiasType,
                Stride<_1, _0, _0>,
                AlignmentBias> // bias
            >;

        using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<ArchTag, OperatorClass,
            TileShape, ClusterShape, EpilogueTileType, ElementAccumulator, ElementAccumulator,
            // Transpose layout of D here since we use the explicit swap + transpose trick
            // Void C since we don't use it. Prevents smem allocation.
            void, typename cutlass::layout::LayoutTranspose<LayoutBias>::type, AlignmentBias, CutlassOutputType,
            typename cutlass::layout::LayoutTranspose<LayoutOutput>::type, AlignmentOutput, EpilogueSchedule,
            EVT_bias_addition>::CollectiveOp;

        using PackedScaleZero = cute::tuple<CutlassWeightType, ElementScale, ElementZero>;
        using PackedScale = cute::tuple<CutlassWeightType, ElementScale>;
        using ElementBCollectiveInfo = std::conditional_t<cutlass::hasZero(QuantOp), PackedScaleZero, PackedScale>;

        // We swap A and B operands to the builder here
        using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilderInterleaved<ArchTag,
            OperatorClass, ElementBCollectiveInfo, LayoutB_Transpose, AlignmentB, CutlassActivationType,
            LayoutA_Transpose, AlignmentA, ElementAccumulator, TileShape, ClusterShape,
            cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
                sizeof(typename CollectiveEpilogue::SharedStorage))>,
            KernelSchedule>::CollectiveOp;

        using TileScheduler = cute::conditional_t<size<0>(CTAShape{}) == Int<64>{}, cutlass::gemm::PersistentScheduler,
            cutlass::gemm::StreamKScheduler>;

        using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, // Indicates ProblemShape
            CollectiveMainloop, CollectiveEpilogue, TileScheduler>;

        if (occupancy != nullptr)
        {
            *occupancy = tensorrt_llm::cutlass_extensions::compute_occupancy_for_kernel<GemmKernel, true>();
            return;
        }

        using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

        using StrideA = typename GemmKernel::StrideA;
        using StrideB = typename GemmKernel::StrideB;
        using StrideC = typename GemmKernel::StrideC;
        using StrideD = typename GemmKernel::StrideD;
        using StrideS = typename CollectiveMainloop::StrideScale;

        if (weight_scales == nullptr)
        {
            throw std::runtime_error("Weight scales must always be set to a non-null value.");
        }

        if constexpr (cutlass::isFinegrained(QuantOp))
        {
            int cta_shape_k = cute::size<2>(TileShape{});
            if (group_size % cta_shape_k != 0)
            {
                std::string err_msg = "The group size must a multiple of " + std::to_string(cta_shape_k);
                throw std::runtime_error("[TensorRT LLM Error][fpA_intB Runner]" + err_msg);
            }

            if constexpr (QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY)
            {
                if (weight_zero_points != nullptr)
                {
                    throw std::runtime_error("Weight zero pointer must be a nullptr for scale only fine grained");
                }
            }
            else if constexpr (QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS)
            {
                if (weight_zero_points == nullptr)
                {
                    throw std::runtime_error("Weight zero pointer must be valid for scale and bias fine grained");
                }
            }
        }
        else
        {
            if (group_size != k)
            {
                throw std::runtime_error("Invalid group size for per column scaling kernels.");
            }

            if (weight_zero_points != nullptr)
            {
                throw std::runtime_error("Weight zero-points must be null when running per column scaling");
            }
        }

        auto cutlass_scale_k = (k + group_size - 1) / group_size;
        StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
        StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
        StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(n, m, 1));
        StrideS stride_S = cutlass::make_cute_packed_stride(StrideS{}, cute::make_shape(n, cutlass_scale_k, 1));

        // Use the output as the bias to avoid making a tma descriptor with a nullptr.
        auto output_as_bias_type = reinterpret_cast<CutlassBiasType const*>(C);

        typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, {n, m, k, 1},
            {reinterpret_cast<CutlassWeightType const*>(B), stride_B, reinterpret_cast<CutlassActivationType const*>(A),
                stride_A, reinterpret_cast<ElementScale const*>(weight_scales), stride_S, group_size,
                reinterpret_cast<ElementZero const*>(weight_zero_points)},
            {{}, output_as_bias_type, stride_D, reinterpret_cast<CutlassOutputType*>(C), stride_D}};

        args.epilogue.thread = {
            {alpha},                                                                  // alpha args
            {},                                                                       // accumulator
            {reinterpret_cast<CutlassBiasType const*>(biases), CutlassBiasType(0.f)}, // bias args
            {}                                                                        // end multiply_add
        };

        Gemm gemm;

        size_t const needed_ws = gemm.get_workspace_size(args);
        dim3 const diag_grid = Gemm::get_grid_shape(args, workspace);
        dim3 const diag_block = GemmKernel::get_block_shape();

        bool const smem_attr_ok = fpAIntBSm90DiagnoseKernel<GemmKernel>(
            "before initialize", m, n, k, group_size, needed_ws, workspace_bytes, workspace, diag_grid, diag_block);

        if (needed_ws > workspace_bytes)
        {
            TLLM_LOG_ERROR("[TensorRT LLM Error][fpA_intB Runner] given workspace size insufficient.");
        }

        auto can_implement = gemm.can_implement(args);
        if (can_implement != cutlass::Status::kSuccess)
        {
            std::string err_msg = "fpA_intB cutlass kernel will fail for params. Error: "
                + std::string(cutlassGetStatusString(can_implement));
            std::fprintf(stderr, "[fpA_intB SM90] can_implement FAILED: %s\n", err_msg.c_str());
            std::fprintf(stderr, "[fpA_intB SM90] workspace=%p workspace_size=%zu needed=%zu\n",
                workspace, workspace_bytes, needed_ws);
            throw std::runtime_error("[TensorRT LLM Error][fpA_intB Runner] " + err_msg);
        }

        auto init_status = gemm.initialize(args, workspace, stream);
        if (init_status != cutlass::Status::kSuccess)
        {
            std::string err_msg = "Failed to initialize cutlass fpA_intB gemm. Error: "
                + std::string(cutlassGetStatusString(init_status));
            std::fprintf(stderr, "[fpA_intB SM90] initialize FAILED: %s\n", err_msg.c_str());
            fpAIntBSm90DiagnoseInitializeSteps<GemmKernel>(args, workspace, stream);
            if (smem_attr_ok)
            {
                fpAIntBSm90DiagnoseKernel<GemmKernel>(
                    "after initialize failure", m, n, k, group_size, needed_ws, workspace_bytes, workspace, diag_grid,
                    diag_block);
            }

            // Try to get more info from CUDA
            cudaError_t cuda_err = cudaGetLastError();
            if (cuda_err != cudaSuccess)
            {
                std::fprintf(stderr, "  - CUDA error: %s (%d)\n",
                    cudaGetErrorString(cuda_err), static_cast<int>(cuda_err));
            }
            throw std::runtime_error("[TensorRT LLM Error][fpA_intB Runner] " + err_msg);
        }

        auto run_status = gemm.run(stream);
        if (run_status != cutlass::Status::kSuccess)
        {
            std::string err_msg
                = "Failed to run cutlass fpA_intB gemm. Error: " + std::string(cutlassGetStatusString(run_status));
            std::fprintf(stderr, "[fpA_intB SM90] run FAILED: %s\n", err_msg.c_str());
            throw std::runtime_error("[TensorRT LLM Error][fpA_intB Runner] " + err_msg);
        }
    }
    else
    {
        std::stringstream ss;
        ss << "[TensorRT LLM Error][fpA_intB Runner] Config (" << (int64_t) cute::size<0>(CTAShape{}) << ","
           << (int64_t) cute::size<1>(CTAShape{}) << "," << (int64_t) cute::size<2>(CTAShape{}) << ") ("
           << (int64_t) cute::size<0>(ClusterShape{}) << "," << (int64_t) cute::size<1>(ClusterShape{}) << ","
           << (int64_t) cute::size<2>(ClusterShape{}) << ") not compiled with FAST_BUILD.";

        throw std::runtime_error(ss.str());
    }

#else  // COMPILE_HOPPER_TMA_GEMMS
    throw std::runtime_error(
        "[TensorRT LLM Error][fpA_intB Runner] Please recompile with support for hopper by passing 90-real as an arch "
        "to build_wheel.py.");
#endif // COMPILE_HOPPER_TMA_GEMMS
}

} // namespace cutlass_kernels_oss
} // namespace kernels

TRTLLM_NAMESPACE_END
