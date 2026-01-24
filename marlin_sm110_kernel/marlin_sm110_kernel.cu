#include "kernel.h"
#include "marlin_config_list.h"
#include "marlin_template.h"

namespace marlin {

using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

#define MARLIN_INSTANTIATE(threads, thread_m_blocks, thread_n_blocks, thread_k_blocks, m_block_size_8, stages, group_blocks) \
  template __global__ void Marlin< \
      vllm::kFloat16.id(), \
      vllm::kU4B8.id(), \
      vllm::kFloat16.id(), \
      vllm::kFloat16.id(), \
      threads, thread_m_blocks, thread_n_blocks, thread_k_blocks, \
      m_block_size_8, stages, group_blocks, false>(MARLIN_KERNEL_PARAMS);

MARLIN_GPTQ_CONFIGS(MARLIN_INSTANTIATE)

#undef MARLIN_INSTANTIATE

MarlinFuncPtr get_marlin_kernel(int thread_m_blocks, int thread_n_blocks,
    int thread_k_blocks, bool m_block_size_8, int group_blocks, int threads,
    int stages) {
  auto kernel = static_cast<MarlinFuncPtr>(nullptr);

#define MARLIN_SELECT(cfg_threads, cfg_thread_m_blocks, cfg_thread_n_blocks, cfg_thread_k_blocks, cfg_m_block_size_8, cfg_stages, cfg_group_blocks) \
  if (threads == cfg_threads && thread_m_blocks == cfg_thread_m_blocks && \
      thread_n_blocks == cfg_thread_n_blocks && thread_k_blocks == cfg_thread_k_blocks && \
      m_block_size_8 == cfg_m_block_size_8 && stages == cfg_stages && \
      group_blocks == cfg_group_blocks) { \
    return Marlin< \
        vllm::kFloat16.id(), vllm::kU4B8.id(), vllm::kFloat16.id(), \
        vllm::kFloat16.id(), cfg_threads, cfg_thread_m_blocks, \
        cfg_thread_n_blocks, cfg_thread_k_blocks, cfg_m_block_size_8, \
        cfg_stages, cfg_group_blocks, false>; \
  }

  MARLIN_GPTQ_CONFIGS(MARLIN_SELECT)

#undef MARLIN_SELECT

  return kernel;
}

}  // namespace marlin
