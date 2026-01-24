#include "marlin_runner.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

struct Args {
  int m = 1;
  int n = 2048;
  int k = 2048;
  int group_size = 128;
  int warmup = 10;
  int iters = 100;
  bool list_configs = false;
};

void print_usage(const char* name) {
  std::printf(
      "Usage: %s [--m=N] [--n=N] [--k=N] [--group_size=N] [--warmup=N] [--iters=N] [--list_configs]\n",
      name);
}

bool parse_int(const char* arg, const char* key, int& out) {
  size_t len = std::strlen(key);
  if (std::strncmp(arg, key, len) != 0) {
    return false;
  }
  out = std::strtol(arg + len, nullptr, 10);
  return true;
}

void parse_args(int argc, char** argv, Args& args) {
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--help") == 0) {
      print_usage(argv[0]);
      std::exit(0);
    }
    if (parse_int(argv[i], "--m=", args.m)) {
      continue;
    }
    if (parse_int(argv[i], "--n=", args.n)) {
      continue;
    }
    if (parse_int(argv[i], "--k=", args.k)) {
      continue;
    }
    if (parse_int(argv[i], "--group_size=", args.group_size)) {
      continue;
    }
    if (parse_int(argv[i], "--warmup=", args.warmup)) {
      continue;
    }
    if (parse_int(argv[i], "--iters=", args.iters)) {
      continue;
    }
    if (std::strcmp(argv[i], "--list_configs") == 0) {
      args.list_configs = true;
      continue;
    }
    std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
    print_usage(argv[0]);
    std::exit(1);
  }
}

void check_cuda(cudaError_t status, const char* msg) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(status));
    std::exit(1);
  }
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  parse_args(argc, argv, args);

  if (args.list_configs) {
    marlin::print_gptq_fp16_int4_configs();
    return 0;
  }

  if (args.m <= 0 || args.n <= 0 || args.k <= 0) {
    std::fprintf(stderr, "M/N/K must be > 0.\n");
    return 1;
  }
  if (args.warmup < 0 || args.iters <= 0) {
    std::fprintf(stderr, "warmup must be >= 0 and iters must be > 0.\n");
    return 1;
  }
  if (args.k % 16 != 0) {
    std::fprintf(stderr, "K must be divisible by 16.\n");
    return 1;
  }
  if (args.n % 64 != 0) {
    std::fprintf(stderr, "N must be divisible by 64.\n");
    return 1;
  }
  if (args.group_size != -1) {
    if (args.group_size <= 0 || (args.k % args.group_size) != 0) {
      std::fprintf(stderr, "group_size must be -1 or a positive divisor of K.\n");
      return 1;
    }
  }

  int sms = 0;
  check_cuda(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0),
      "cudaDeviceGetAttribute");

  int pack_factor = 8;  // int4
  int b_rows = args.k / 16;
  int b_cols = (args.n / pack_factor) * 16;
  size_t b_ints = static_cast<size_t>(b_rows) * b_cols;

  int num_groups = args.group_size == -1 ? 1 : args.k / args.group_size;
  size_t scales_elems = static_cast<size_t>(num_groups) * args.n;

  half* d_A = nullptr;
  uint32_t* d_B = nullptr;
  half* d_scales = nullptr;
  half* d_C = nullptr;
  int* d_workspace = nullptr;

  check_cuda(cudaMalloc(&d_A, sizeof(half) * args.m * args.k), "cudaMalloc A");
  check_cuda(cudaMalloc(&d_B, sizeof(uint32_t) * b_ints), "cudaMalloc B");
  check_cuda(cudaMalloc(&d_scales, sizeof(half) * scales_elems), "cudaMalloc scales");
  check_cuda(cudaMalloc(&d_C, sizeof(half) * args.m * args.n), "cudaMalloc C");
  check_cuda(cudaMalloc(&d_workspace, sizeof(int) * sms), "cudaMalloc workspace");

  check_cuda(cudaMemset(d_A, 0, sizeof(half) * args.m * args.k), "memset A");
  check_cuda(cudaMemset(d_B, 0, sizeof(uint32_t) * b_ints), "memset B");
  check_cuda(cudaMemset(d_scales, 0, sizeof(half) * scales_elems), "memset scales");
  check_cuda(cudaMemset(d_C, 0, sizeof(half) * args.m * args.n), "memset C");
  check_cuda(cudaMemset(d_workspace, 0, sizeof(int) * sms), "memset workspace");

  marlin::MarlinConfig config{};
  for (int i = 0; i < args.warmup; ++i) {
    if (!marlin::run_gptq_fp16_int4(d_A, d_B, d_scales, d_C, args.m, args.n,
            args.k, args.group_size, args.k, d_workspace, 0, nullptr)) {
      std::fprintf(stderr, "Warmup failed.\n");
      return 1;
    }
  }

  cudaEvent_t start{};
  cudaEvent_t stop{};
  check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
  check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");
  check_cuda(cudaEventRecord(start, 0), "cudaEventRecord start");
  bool ok = true;
  for (int i = 0; i < args.iters; ++i) {
    ok = marlin::run_gptq_fp16_int4(d_A, d_B, d_scales, d_C, args.m, args.n,
        args.k, args.group_size, args.k, d_workspace, 0, &config);
    if (!ok) {
      break;
    }
  }
  check_cuda(cudaEventRecord(stop, 0), "cudaEventRecord stop");
  check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
  float ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
  check_cuda(cudaEventDestroy(start), "cudaEventDestroy start");
  check_cuda(cudaEventDestroy(stop), "cudaEventDestroy stop");

  if (!ok) {
    std::fprintf(stderr, "Failed to select or launch a Marlin config.\n");
  }

  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  std::printf("Avg kernel time: %.3f us (%d iters, %d warmup)\n",
      (ms * 1000.0f) / static_cast<float>(args.iters), args.iters, args.warmup);
  std::printf(
      "Selected config: thread_k=%d thread_n=%d threads=%d thread_m_blocks=%d "
      "m_block_size_8=%d group_blocks=%d stages=%d\n",
      config.thread_k, config.thread_n, config.num_threads,
      config.thread_m_blocks, config.m_block_size_8 ? 1 : 0,
      config.group_blocks, config.stages);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_scales);
  cudaFree(d_C);
  cudaFree(d_workspace);

  return ok ? 0 : 1;
}
