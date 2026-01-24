/*
 * Standalone wrappers for SM80 fpA_intB GEMM (FP16 x INT4, GPTQ).
 */

#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/launchers/fpA_intB_gemm_sm80_wrappers.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/kernelDispatcher.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <limits>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::cutlass_kernels_oss
{
namespace
{
constexpr int kSplitKLimit = 7;

using Runner = kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<half, cutlass::uint4b_t,
    cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>;

Runner& get_runner()
{
    static Runner runner;
    return runner;
}

std::vector<CutlassGemmConfig> const& get_candidate_configs_cached(bool enable_cuda_fallback)
{
    static std::vector<CutlassGemmConfig> cached_with_cuda;
    static std::vector<CutlassGemmConfig> cached_no_cuda;
    if (enable_cuda_fallback)
    {
        if (cached_with_cuda.empty())
        {
            int const sm = tensorrt_llm::common::getSMVersion();
            auto config_type_param = static_cast<cutlass_extensions::CutlassGemmConfig::CandidateConfigTypeParam>(
                cutlass_extensions::CutlassGemmConfig::CandidateConfigTypeParam::HOPPER
                | cutlass_extensions::CutlassGemmConfig::CandidateConfigTypeParam::WEIGHT_ONLY);
            cached_with_cuda = kernels::cutlass_kernels::get_candidate_configs(sm, kSplitKLimit, config_type_param);
        }
        return cached_with_cuda;
    }

    if (cached_no_cuda.empty())
    {
        auto const& base = get_candidate_configs_cached(true);
        cached_no_cuda.reserve(base.size());
        for (auto const& cfg : base)
        {
            if (!cfg.enableCudaKernel)
            {
                cached_no_cuda.push_back(cfg);
            }
        }
    }
    return cached_no_cuda;
}

bool same_config(CutlassGemmConfig const& a, CutlassGemmConfig const& b)
{
    return a.enableCudaKernel == b.enableCudaKernel && a.tile_config_sm80 == b.tile_config_sm80
        && a.split_k_style == b.split_k_style && a.split_k_factor == b.split_k_factor && a.stages == b.stages;
}

__global__ void fill_half_kernel(half* data, size_t n, float scale)
{
    size_t const idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float const val = static_cast<float>(idx % 97) * scale;
        data[idx] = __float2half_rn(val);
    }
}

void fill_device_half(half* data, size_t n, float scale, cudaStream_t stream)
{
    int const block = 256;
    int const grid = static_cast<int>((n + block - 1) / block);
    fill_half_kernel<<<grid, block, 0, stream>>>(data, n, scale);
}

float profile_tactic(int m, int n, int k, int group_size, CutlassGemmConfig const& config, half const* d_a,
    int8_t const* d_b, half const* d_scales, half const* d_zeros, half* d_c, void* workspace, size_t workspace_bytes,
    cudaStream_t stream, int sm)
{
    (void) sm;
    constexpr int warmup = 5;
    constexpr int runs = 10;

    auto run_once = [&]() {
        if (config.enableCudaKernel)
        {
            kernels::weight_only::Params params{d_a, nullptr, d_b, d_scales, d_zeros, nullptr, d_c, 1.0f, m, n, k,
                group_size, kernels::weight_only::KernelType::FP16Int4Groupwise, false};
            kernels::weight_only::select_gs<true,
                kernels::weight_only::KernelDetails<kernels::weight_only::FP16DetailsA,
                    kernels::weight_only::Int4DetailsW, kernels::weight_only::ColumnMajorInterleaved, true, 64>>(
                params, stream);
        }
        else
        {
            get_runner().gemm(d_a, reinterpret_cast<cutlass::uint4b_t const*>(d_b), d_scales, d_zeros, nullptr, 1.0f,
                d_c, m, n, k, group_size, config, reinterpret_cast<char*>(workspace), workspace_bytes, stream);
        }
    };

    for (int i = 0; i < warmup; ++i)
    {
        run_once();
    }

    cudaEvent_t start{};
    cudaEvent_t stop{};
    tensorrt_llm::common::check_cuda_error(cudaEventCreate(&start));
    tensorrt_llm::common::check_cuda_error(cudaEventCreate(&stop));
    tensorrt_llm::common::check_cuda_error(cudaEventRecord(start, stream));
    for (int i = 0; i < runs; ++i)
    {
        run_once();
    }
    tensorrt_llm::common::check_cuda_error(cudaEventRecord(stop, stream));
    tensorrt_llm::common::check_cuda_error(cudaEventSynchronize(stop));

    float elapsed = 0.0f;
    tensorrt_llm::common::check_cuda_error(cudaEventElapsedTime(&elapsed, start, stop));
    tensorrt_llm::common::check_cuda_error(cudaEventDestroy(start));
    tensorrt_llm::common::check_cuda_error(cudaEventDestroy(stop));

    return elapsed / runs;
}

struct ProfileBuffers
{
    half* d_a = nullptr;
    int8_t* d_b = nullptr;
    half* d_scales = nullptr;
    half* d_zeros = nullptr;
    half* d_c = nullptr;
    void* d_workspace = nullptr;
    size_t workspace_bytes = 0;
};

ProfileBuffers allocate_profile_buffers(int m, int n, int k, int group_size, cudaStream_t stream)
{
    ProfileBuffers buf;
    size_t const a_elems = static_cast<size_t>(m) * k;
    size_t const b_bytes = static_cast<size_t>(k) * n / 2;
    size_t const scale_elems = static_cast<size_t>(k / group_size) * n;
    size_t const c_elems = static_cast<size_t>(m) * n;

    tensorrt_llm::common::check_cuda_error(cudaMalloc(&buf.d_a, a_elems * sizeof(half)));
    tensorrt_llm::common::check_cuda_error(cudaMalloc(&buf.d_b, b_bytes));
    tensorrt_llm::common::check_cuda_error(cudaMalloc(&buf.d_scales, scale_elems * sizeof(half)));
    tensorrt_llm::common::check_cuda_error(cudaMalloc(&buf.d_zeros, scale_elems * sizeof(half)));
    tensorrt_llm::common::check_cuda_error(cudaMalloc(&buf.d_c, c_elems * sizeof(half)));

    buf.workspace_bytes = get_runner().getWorkspaceSize(m, n, k);
    if (buf.workspace_bytes > 0)
    {
        tensorrt_llm::common::check_cuda_error(cudaMalloc(&buf.d_workspace, buf.workspace_bytes));
    }

    fill_device_half(buf.d_a, a_elems, 0.01f, stream);
    tensorrt_llm::common::check_cuda_error(cudaMemsetAsync(buf.d_b, 0, b_bytes, stream));
    fill_device_half(buf.d_scales, scale_elems, 0.01f, stream);
    tensorrt_llm::common::check_cuda_error(cudaMemsetAsync(buf.d_zeros, 0, scale_elems * sizeof(half), stream));
    tensorrt_llm::common::check_cuda_error(cudaMemsetAsync(buf.d_c, 0, c_elems * sizeof(half), stream));

    return buf;
}

void free_profile_buffers(ProfileBuffers& buf)
{
    if (buf.d_a)
    {
        tensorrt_llm::common::check_cuda_error(cudaFree(buf.d_a));
    }
    if (buf.d_b)
    {
        tensorrt_llm::common::check_cuda_error(cudaFree(buf.d_b));
    }
    if (buf.d_scales)
    {
        tensorrt_llm::common::check_cuda_error(cudaFree(buf.d_scales));
    }
    if (buf.d_zeros)
    {
        tensorrt_llm::common::check_cuda_error(cudaFree(buf.d_zeros));
    }
    if (buf.d_c)
    {
        tensorrt_llm::common::check_cuda_error(cudaFree(buf.d_c));
    }
    if (buf.d_workspace)
    {
        tensorrt_llm::common::check_cuda_error(cudaFree(buf.d_workspace));
    }
}
} // namespace

size_t fpA_intB_get_all_configs(CutlassGemmConfig const** configs)
{
    auto const& list = get_candidate_configs_cached(true);
    if (configs != nullptr)
    {
        *configs = list.data();
    }
    return list.size();
}

bool fpA_intB_is_supported_config(CutlassGemmConfig const& config)
{
    auto const& list = get_candidate_configs_cached(true);
    for (auto const& cfg : list)
    {
        if (same_config(cfg, config))
        {
            return true;
        }
    }
    return false;
}

bool fpA_intB_select_config_fp16_int4_gptq(CutlassGemmConfig& out_config, int m, int n, int k, int group_size,
    int /*multi_processor_count*/, bool enable_cuda_fallback)
{
    if (m <= 0 || n <= 0 || k <= 0)
    {
        return false;
    }
    if (group_size != 64 && group_size != 128)
    {
        return false;
    }
    if ((k % 64) != 0)
    {
        return false;
    }
    if ((k % group_size) != 0)
    {
        return false;
    }

    int const sm = tensorrt_llm::common::getSMVersion();
    auto const& configs = get_candidate_configs_cached(enable_cuda_fallback);
    if (configs.empty())
    {
        return false;
    }

    cudaStream_t stream{};
    tensorrt_llm::common::check_cuda_error(cudaStreamCreate(&stream));
    ProfileBuffers buf = allocate_profile_buffers(m, n, k, group_size, stream);

    float best_time = std::numeric_limits<float>::max();
    bool found = false;
    CutlassGemmConfig best{};

    for (auto const& cfg : configs)
    {
        if (cfg.enableCudaKernel && m >= 16)
        {
            continue;
        }
        try
        {
            float const time = profile_tactic(m, n, k, group_size, cfg, buf.d_a, buf.d_b, buf.d_scales, buf.d_zeros,
                buf.d_c, buf.d_workspace, buf.workspace_bytes, stream, sm);
            if (time < best_time)
            {
                best_time = time;
                best = cfg;
                found = true;
            }
        }
        catch (...)
        {
            cudaGetLastError();
            continue;
        }
    }

    free_profile_buffers(buf);
    tensorrt_llm::common::check_cuda_error(cudaStreamDestroy(stream));

    if (!found)
    {
        return false;
    }
    out_config = best;
    return true;
}

size_t fpA_intB_get_workspace_size(int m, int n, int k)
{
    return get_runner().getWorkspaceSize(m, n, k);
}

void fpA_intB_gemm_fp16_int4_gptq_with_config(half const* A, int8_t const* B, half const* weight_scales,
    half const* weight_zero_points, half* C, int m, int n,
    int k, int group_size, cudaStream_t stream, void* workspace, size_t workspace_bytes,
    CutlassGemmConfig const& config)
{
    TLLM_CHECK_WITH_INFO(A != nullptr, "A must not be null");
    TLLM_CHECK_WITH_INFO(B != nullptr, "B must not be null");
    TLLM_CHECK_WITH_INFO(weight_scales != nullptr, "weight_scales must not be null");
    TLLM_CHECK_WITH_INFO(C != nullptr, "C must not be null");

    int const sm = tensorrt_llm::common::getSMVersion();
    if (config.enableCudaKernel)
    {
        kernels::weight_only::Params params{A, nullptr, B, weight_scales, weight_zero_points, nullptr, C, 1.0f, m, n, k,
            group_size, kernels::weight_only::KernelType::FP16Int4Groupwise, false};
        kernels::weight_only::select_gs<true,
            kernels::weight_only::KernelDetails<kernels::weight_only::FP16DetailsA, kernels::weight_only::Int4DetailsW,
                kernels::weight_only::ColumnMajorInterleaved, true, 64>>(params, stream);
        return;
    }

    get_runner().gemm(A, reinterpret_cast<cutlass::uint4b_t const*>(B), weight_scales, weight_zero_points, nullptr,
        1.0f, C, m, n, k, group_size, config, reinterpret_cast<char*>(workspace), workspace_bytes, stream);
}

void fpA_intB_gemm_fp16_int4_gptq(half const* A, int8_t const* B, half const* weight_scales,
    half const* weight_zero_points, half* C, int m, int n, int k, int group_size,
    cudaStream_t stream, void* workspace, size_t workspace_bytes, CutlassGemmConfig* selected_config)
{
    CutlassGemmConfig config{};
    bool ok = fpA_intB_select_config_fp16_int4_gptq(config, m, n, k, group_size, 0, true);
    TLLM_CHECK_WITH_INFO(ok, "No valid fpA_intB config found for the given shape.");
    fpA_intB_gemm_fp16_int4_gptq_with_config(
        A, B, weight_scales, weight_zero_points, C, m, n, k, group_size, stream, workspace, workspace_bytes, config);
    if (selected_config != nullptr)
    {
        *selected_config = config;
    }
}
} // namespace kernels::cutlass_kernels_oss

TRTLLM_NAMESPACE_END
