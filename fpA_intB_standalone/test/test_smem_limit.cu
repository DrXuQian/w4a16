/*
 * Test: find exact shared memory limit where cudaFuncSetAttribute fails.
 *
 * Build:
 *   nvcc -O2 -std=c++17 -arch=sm_90a test_smem_limit.cu -o test_smem_limit
 * Run:
 *   ./test_smem_limit
 */
#include <cuda_runtime.h>
#include <cstdio>

// Dummy kernel that uses dynamic shared memory
__global__ void dummy_kernel(char* out) {
    extern __shared__ char smem[];
    smem[threadIdx.x] = threadIdx.x;
    __syncthreads();
    if (threadIdx.x == 0) out[0] = smem[0];
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf("sharedMemPerBlock:        %zu\n", prop.sharedMemPerBlock);
    printf("sharedMemPerBlockOptin:   %zu\n", prop.sharedMemPerBlockOptin);
    printf("sharedMemPerMultiprocessor: %zu\n\n", prop.sharedMemPerMultiprocessor);

    int driver = 0, runtime = 0;
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);
    printf("CUDA driver: %d.%d, runtime: %d.%d\n", driver/1000, (driver%100)/10, runtime/1000, (runtime%100)/10);
    printf("nvcc compiled for: sm_90a\n\n");

    char* d_out;
    cudaMalloc(&d_out, 1);

    // Test cudaFuncSetAttribute at various sizes around the limit
    int sizes[] = {
        200 * 1024,  // 200 KB
        210 * 1024,  // 210 KB
        220 * 1024,  // 220 KB
        225 * 1024,  // 225 KB
        226 * 1024,  // 226 KB
        227 * 1024,  // 227 KB
        (int)prop.sharedMemPerBlockOptin - 1024,
        (int)prop.sharedMemPerBlockOptin,
        (int)prop.sharedMemPerBlockOptin + 1024,
        230 * 1024,  // 230 KB
        231 * 1024,
        232 * 1024,
        233 * 1024,
        234 * 1024,
        240 * 1024,
    };

    printf("%-12s %-20s %-20s\n", "SmemSize", "FuncSetAttribute", "Launch+Sync");
    printf("%-12s %-20s %-20s\n", "--------", "----------------", "----------");

    for (int smem : sizes) {
        // Reset error state
        cudaGetLastError();

        cudaError_t attr_err = cudaFuncSetAttribute(
            dummy_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem);

        const char* attr_str = (attr_err == cudaSuccess) ? "OK" : cudaGetErrorString(attr_err);

        const char* launch_str = "skipped";
        if (attr_err == cudaSuccess) {
            dummy_kernel<<<1, 32, smem>>>(d_out);
            cudaError_t launch_err = cudaDeviceSynchronize();
            launch_str = (launch_err == cudaSuccess) ? "OK" : cudaGetErrorString(launch_err);
            if (launch_err != cudaSuccess) cudaGetLastError();
        } else {
            cudaGetLastError();
        }

        printf("%8d KB  %-20s %-20s\n", smem / 1024, attr_str, launch_str);
    }

    cudaFree(d_out);
    return 0;
}
