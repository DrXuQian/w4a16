#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace marlin {

struct MarlinConfig {
  int thread_k;
  int thread_n;
  int num_threads;
  int thread_m_blocks;
  bool m_block_size_8;
  int group_blocks;
  int stages;
};

bool select_gptq_fp16_int4_config(MarlinConfig& out, int m, int n, int k,
    int group_size, int max_shared_mem, int sms);

void print_gptq_fp16_int4_configs();

bool run_gptq_fp16_int4(const half* A, const uint32_t* B, const half* scales,
    half* C, int m, int n, int k, int group_size, int lda, int* workspace,
    cudaStream_t stream, MarlinConfig* selected_config);

}  // namespace marlin
