/*
 * Minimal test: reproduce fpA_intB SM90 initialize() failure.
 * Uses the existing fpA_intB library (no need to duplicate CUTLASS types).
 *
 * Build (from fpA_intB_standalone/build/):
 *   Already built as part of cmake. Or manually:
 *   nvcc -O2 -std=c++17 -arch=sm_90a --expt-relaxed-constexpr \
 *     -I ../cpp/include -I ../cpp -I ../cpp/tensorrt_llm \
 *     -I ../cpp/tensorrt_llm/cutlass_extensions/include \
 *     -I <CUTLASS>/include \
 *     ../test/test_sm90_gemm_init.cu -L. -lfpA_intB_gemm -lcudart -o test_sm90_gemm_init
 *
 * Run:
 *   FPA_INTB_PROFILE_LOG=1 ./test_sm90_gemm_init
 */

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/launchers/fpA_intB_gemm_sm80_wrappers.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main() {
    // Print device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf("Max shared mem per block (optin): %zu bytes\n", prop.sharedMemPerBlockOptin);
    printf("CUDA driver: ");
    int driver = 0, runtime = 0;
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);
    printf("%d.%d, runtime: %d.%d\n\n", driver/1000, (driver%100)/10, runtime/1000, (runtime%100)/10);

    int M = 128, N = 256, K = 128, group_size = 128;
    printf("Test shape: M=%d N=%d K=%d group_size=%d\n\n", M, N, K, group_size);

    // Quantize dummy weights
    std::vector<half> h_a(M * K);
    std::vector<half> h_w(K * N);
    for (size_t i = 0; i < h_a.size(); i++) h_a[i] = __float2half_rn(0.01f);
    for (size_t i = 0; i < h_w.size(); i++) h_w[i] = __float2half_rn(0.02f);

    int groups = K / group_size;
    std::vector<half> h_scales(groups * N);
    std::vector<half> h_zeros(groups * N);
    std::vector<int8_t> h_packed(K * N / 2);
    for (auto& v : h_scales) v = __float2half_rn(0.01f);
    for (auto& v : h_zeros) v = __float2half_rn(0.0f);
    for (auto& v : h_packed) v = 0x55;  // nibbles = 5,5

    // Preprocess weights
    std::vector<int8_t> h_preprocessed(h_packed.size());
    std::vector<size_t> shape = {(size_t)K, (size_t)N};
    tensorrt_llm::kernels::cutlass_kernels::preprocess_weights_for_mixed_gemm(
        h_preprocessed.data(), h_packed.data(), shape,
        tensorrt_llm::kernels::cutlass_kernels::QuantType::W4_A16, false);

    // Device alloc
    half *d_a, *d_scales, *d_zeros, *d_c;
    int8_t *d_b;
    cudaMalloc(&d_a, M * K * sizeof(half));
    cudaMalloc(&d_b, h_preprocessed.size());
    cudaMalloc(&d_scales, groups * N * sizeof(half));
    cudaMalloc(&d_zeros, groups * N * sizeof(half));
    cudaMalloc(&d_c, M * N * sizeof(half));

    cudaMemcpy(d_a, h_a.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_preprocessed.data(), h_preprocessed.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, h_scales.data(), groups * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_zeros, h_zeros.data(), groups * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_c, 0, M * N * sizeof(half));

    size_t ws_bytes = tensorrt_llm::kernels::cutlass_kernels_oss::fpA_intB_get_workspace_size(M, N, K);
    void* d_ws = nullptr;
    if (ws_bytes > 0) cudaMalloc(&d_ws, ws_bytes);
    printf("Workspace: %zu bytes\n\n", ws_bytes);

    // List all configs
    using Config = tensorrt_llm::cutlass_extensions::CutlassGemmConfig;
    Config const* configs = nullptr;
    size_t count = tensorrt_llm::kernels::cutlass_kernels_oss::fpA_intB_get_all_configs(&configs);
    printf("Total configs: %zu\n\n", count);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Try each config one by one
    int pass = 0, fail = 0;
    for (size_t i = 0; i < count; i++) {
        auto const& cfg = configs[i];

        const char* path = "unknown";
        if (cfg.enableCudaKernel) path = "cuda_kernel";
        else if (cfg.is_tma_warp_specialized) path = "tma_ws";
        else path = "sm80";

        printf("Config %2zu [%s]: ", i, path);
        if (cfg.is_tma_warp_specialized) {
            printf("tile=%d ml=%d el=%d cl=%d ",
                cfg.getTileConfigAsInt(),
                (int)cfg.mainloop_schedule, (int)cfg.epilogue_schedule,
                (int)cfg.cluster_shape);
        }

        try {
            tensorrt_llm::kernels::cutlass_kernels_oss::fpA_intB_gemm_fp16_int4_gptq_with_config(
                d_a, d_b, d_scales, d_zeros, d_c,
                M, N, K, group_size, stream, d_ws, ws_bytes, cfg);
            cudaStreamSynchronize(stream);
            cudaError_t err = cudaGetLastError();
            if (err == cudaSuccess) {
                printf("PASS\n");
                pass++;
            } else {
                printf("CUDA ERROR: %s\n", cudaGetErrorString(err));
                fail++;
                // Reset error state
                cudaGetLastError();
            }
        } catch (std::exception const& e) {
            printf("EXCEPTION: %s\n", e.what());
            fail++;
            cudaGetLastError();  // clear
        }
    }

    printf("\n========================================\n");
    printf("Results: %d PASS, %d FAIL out of %zu configs\n", pass, fail, count);
    printf("========================================\n");

    cudaStreamDestroy(stream);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_scales); cudaFree(d_zeros); cudaFree(d_c);
    if (d_ws) cudaFree(d_ws);
    return fail > 0 ? 1 : 0;
}
