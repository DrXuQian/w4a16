#include "marlin_runner.h"

#include "kernel.h"
#include "marlin_config_list.h"

#include <algorithm>
#include <cstdio>

namespace marlin {

using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

MarlinFuncPtr get_marlin_kernel(int thread_m_blocks, int thread_n_blocks,
    int thread_k_blocks, bool m_block_size_8, int group_blocks, int threads,
    int stages);

namespace {

struct thread_config_t {
  int thread_k;
  int thread_n;
  int num_threads;
};

struct exec_config_t {
  int blocks_per_sm;
  thread_config_t tb_cfg;
};

thread_config_t small_batch_thread_configs[] = {
    {128, 128, 256},
    {64, 128, 128},
    {128, 64, 128},
};

thread_config_t large_batch_thread_configs[] = {
    {64, 256, 256},
    {64, 128, 128},
    {128, 64, 128},
};

int get_scales_cache_size(thread_config_t const& th_config, int prob_m,
    int prob_n, int prob_k, int num_bits, int group_size, bool has_act_order,
    bool is_k_full, int stages) {
  bool cache_scales_chunk = has_act_order && !is_k_full;

  int tb_n = th_config.thread_n;
  int tb_k = th_config.thread_k;

  int tb_groups;
  if (group_size == -1) {
    tb_groups = 1;
  } else if (group_size == 0) {
    tb_groups = div_ceil(tb_k, 32);
  } else {
    tb_groups = div_ceil(tb_k, group_size);
  }

  if (cache_scales_chunk) {
    int load_groups = tb_groups * stages * 2;
    load_groups = std::max(load_groups, 32);
    return load_groups * tb_n * 2;
  } else {
    int tb_scales = tb_groups * tb_n * 2;
    return tb_scales * stages;
  }
}

int get_kernel_cache_size(thread_config_t const& th_config, int thread_m_blocks,
    int prob_m, int prob_n, int prob_k, int num_bits, int group_size,
    bool has_act_order, bool is_k_full, int has_zp, bool is_zp_float,
    bool is_a_8bit, int stages) {
  int pack_factor = 32 / num_bits;

  int tb_k = th_config.thread_k;
  int tb_n = th_config.thread_n;
  int tb_m = thread_m_blocks * 16;
  int sh_a_size = stages * (tb_m * tb_k) * (is_a_8bit ? 1 : 2);
  int sh_b_size = stages * (tb_k * tb_n / pack_factor) * 4;
  int sh_red_size = tb_m * (tb_n + 8) * 2;
  int sh_bias_size = tb_n * 2;
  int tmp_size =
      (sh_b_size > sh_red_size ? sh_red_size : sh_b_size) + sh_bias_size;
  tmp_size = std::max(std::max(sh_b_size, sh_red_size), tmp_size);

  int sh_s_size = get_scales_cache_size(th_config, prob_m, prob_n, prob_k,
      num_bits, group_size, has_act_order, is_k_full, stages);
  int sh_g_idx_size = has_act_order && !is_k_full ? stages * tb_k / 4 : 0;
  int sh_zp_size = 0;
  if (has_zp) {
    if (is_zp_float) {
      sh_zp_size = sh_s_size;
    } else if (num_bits == 4) {
      sh_zp_size = sh_s_size / 4;
    } else if (num_bits == 8) {
      sh_zp_size = sh_s_size / 2;
    }
  }

  int total_size =
      tmp_size + sh_a_size + sh_s_size + sh_zp_size + sh_g_idx_size;
  return total_size;
}

bool is_valid_config(thread_config_t const& th_config, int thread_m_blocks,
    int prob_m, int prob_n, int prob_k, int num_bits, int group_size,
    bool has_act_order, bool is_k_full, int has_zp, bool is_zp_float,
    bool is_a_8bit, int stages, int max_shared_mem) {
  if (th_config.thread_k == -1 || th_config.thread_n == -1 ||
      th_config.num_threads == -1) {
    return false;
  }

  if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) {
    return false;
  }

  if (th_config.thread_n < min_thread_n || th_config.thread_k < min_thread_k) {
    return false;
  }

  if (th_config.num_threads < 128) {
    return false;
  }

  int cache_size = get_kernel_cache_size(th_config, thread_m_blocks, prob_m,
      prob_n, prob_k, num_bits, group_size, has_act_order, is_k_full, has_zp,
      is_zp_float, is_a_8bit, stages);
  return cache_size <= max_shared_mem;
}

exec_config_t determine_exec_config(int prob_m, int prob_n, int prob_k,
    int thread_m_blocks, bool m_block_size_8, int num_bits, int group_size,
    bool has_act_order, bool is_k_full, bool has_zp, bool is_zp_float,
    int is_a_8bit, int stages, int max_shared_mem, int sms) {
  exec_config_t exec_cfg = exec_config_t{1, thread_config_t{-1, -1, -1}};
  thread_config_t* thread_configs = thread_m_blocks > 1
      ? large_batch_thread_configs
      : small_batch_thread_configs;
  int thread_configs_size = thread_m_blocks > 1
      ? static_cast<int>(sizeof(large_batch_thread_configs) /
          sizeof(thread_config_t))
      : static_cast<int>(sizeof(small_batch_thread_configs) /
          sizeof(thread_config_t));

  for (int i = 0; i < thread_configs_size; ++i) {
    thread_config_t th_config = thread_configs[i];

    if (!is_valid_config(th_config, thread_m_blocks, prob_m, prob_n, prob_k,
            num_bits, group_size, has_act_order, is_k_full, has_zp,
            is_zp_float, is_a_8bit, stages, max_shared_mem - 512)) {
      continue;
    }

    int group_blocks = 0;
    if (!has_act_order) {
      group_blocks = group_size == -1 ? -1 : group_size / 16;
    }

    auto kernel = get_marlin_kernel(thread_m_blocks,
        th_config.thread_n / 16, th_config.thread_k / 16, m_block_size_8,
        group_blocks, th_config.num_threads, stages);

    if (kernel == nullptr) {
      continue;
    }

    return {1, th_config};
  }

  return exec_cfg;
}

bool select_config_for_problem(MarlinConfig& out, int prob_m, int prob_n,
    int prob_k, int group_size, int max_shared_mem, int sms) {
  constexpr int num_bits = 4;
  constexpr bool has_act_order = false;
  constexpr bool is_k_full = true;
  constexpr bool has_zp = false;
  constexpr bool is_zp_float = false;
  constexpr bool is_a_8bit = false;
  constexpr int stages = 4;

  int thread_m_blocks = std::min(div_ceil(prob_m, 16), 4);
  bool m_block_size_8 = prob_m <= 8;

  exec_config_t exec_cfg = determine_exec_config(prob_m, prob_n, prob_k,
      thread_m_blocks, m_block_size_8, num_bits, group_size, has_act_order,
      is_k_full, has_zp, is_zp_float, is_a_8bit, stages, max_shared_mem, sms);

  thread_config_t th = exec_cfg.tb_cfg;
  if (th.thread_k == -1) {
    return false;
  }

  out.thread_k = th.thread_k;
  out.thread_n = th.thread_n;
  out.num_threads = th.num_threads;
  out.thread_m_blocks = thread_m_blocks;
  out.m_block_size_8 = m_block_size_8;
  out.group_blocks = group_size == -1 ? -1 : group_size / 16;
  out.stages = stages;
  return true;
}

}  // namespace

bool select_gptq_fp16_int4_config(MarlinConfig& out, int m, int n, int k,
    int group_size, int max_shared_mem, int sms) {
  return select_config_for_problem(out, m, n, k, group_size, max_shared_mem,
      sms);
}

void print_gptq_fp16_int4_configs() {
  int idx = 0;
  std::printf("GPTQ INT4 configs (SM80-style list):\n");
#define PRINT_CFG(threads, thread_m_blocks, thread_n_blocks, thread_k_blocks, m_block_size_8, stages, group_blocks) \
  std::printf("  %d: threads=%d m_blocks=%d m8=%d n_blocks=%d k_blocks=%d stages=%d group_blocks=%d\n", \
      idx++, threads, thread_m_blocks, m_block_size_8 ? 1 : 0, \
      thread_n_blocks, thread_k_blocks, stages, group_blocks);
  MARLIN_GPTQ_CONFIGS(PRINT_CFG)
#undef PRINT_CFG
}

bool run_gptq_fp16_int4(const half* A, const uint32_t* B, const half* scales,
    half* C, int m, int n, int k, int group_size, int lda, int* workspace,
    cudaStream_t stream, MarlinConfig* selected_config) {
  constexpr int num_bits = 4;
  constexpr bool has_act_order = false;
  constexpr bool is_k_full = true;
  constexpr bool has_zp = false;
  constexpr bool is_zp_float = false;
  constexpr bool is_a_8bit = false;
  constexpr int stages = 4;

  if (group_size != -1 && (group_size <= 0 || (group_size % 16) != 0)) {
    return false;
  }

  int sms = 0;
  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  cudaDeviceGetAttribute(&max_shared_mem,
      cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  if (sms <= 0 || max_shared_mem <= 0) {
    return false;
  }

  int group_blocks = group_size == -1 ? -1 : group_size / 16;
  int num_groups = group_size == -1 ? 1 : (k / group_size);

  const int4* A_ptr = reinterpret_cast<const int4*>(A);
  const int4* B_ptr = reinterpret_cast<const int4*>(B);
  int4* C_ptr = reinterpret_cast<int4*>(C);
  int4* C_tmp_ptr = nullptr;
  const int4* bias_ptr = nullptr;
  const float* a_s_ptr = nullptr;
  const int4* b_s_ptr = reinterpret_cast<const int4*>(scales);
  const uint16_t* g_s_ptr = nullptr;
  const int4* zp_ptr = nullptr;
  const int* g_idx_ptr = nullptr;

  int* locks = workspace;

  int max_par = 16;
  if (n <= 4096) {
    max_par = 16 * 8;
  }

  int rest_m = m;
  int max_thread_m_blocks = 4;
  while (rest_m) {
    int par_count = rest_m / (max_thread_m_blocks * 16);
    if (par_count > max_par) {
      par_count = max_par;
    }
    int prob_m_split =
        par_count > 0 ? (par_count * (max_thread_m_blocks * 16)) : rest_m;

    int thread_m_blocks =
        std::min(div_ceil(prob_m_split, 16), max_thread_m_blocks);
    bool m_block_size_8 = prob_m_split <= 8;

    exec_config_t exec_cfg = determine_exec_config(prob_m_split, n, k,
        thread_m_blocks, m_block_size_8, num_bits, group_size, has_act_order,
        is_k_full, has_zp, is_zp_float, is_a_8bit, stages, max_shared_mem, sms);
    thread_config_t th = exec_cfg.tb_cfg;

    if (th.thread_k == -1 && max_thread_m_blocks > 1) {
      max_thread_m_blocks--;
      continue;
    }

    if (th.thread_n != -1) {
      if (n / th.thread_n * div_ceil(prob_m_split, thread_m_blocks * 16) * 4 <=
          sms) {
        if (is_valid_config({128, 64, 128}, thread_m_blocks, prob_m_split, n,
                k, num_bits, group_size, has_act_order, is_k_full, has_zp,
                is_zp_float, is_a_8bit, stages, max_shared_mem)) {
          th = {128, 64, 128};
          exec_cfg = {1, th};
        }
      }
    }

    int num_threads = th.num_threads;
    int thread_k = th.thread_k;
    int thread_n = th.thread_n;
    int blocks = sms * exec_cfg.blocks_per_sm;

    int thread_k_blocks = thread_k / 16;
    int thread_n_blocks = thread_n / 16;

    if (!is_valid_config(th, thread_m_blocks, prob_m_split, n, k, num_bits,
            group_size, has_act_order, is_k_full, has_zp, is_zp_float,
            is_a_8bit, stages, max_shared_mem)) {
      return false;
    }

    auto kernel = get_marlin_kernel(thread_m_blocks, thread_n_blocks,
        thread_k_blocks, m_block_size_8, group_blocks, num_threads, stages);
    if (kernel == nullptr) {
      return false;
    }

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        max_shared_mem);

    kernel<<<blocks, num_threads, max_shared_mem, stream>>>(
        A_ptr, B_ptr, C_ptr, C_tmp_ptr, bias_ptr, a_s_ptr, b_s_ptr, g_s_ptr,
        zp_ptr, g_idx_ptr, num_groups, prob_m_split, n, k, lda, locks, false,
        false, false, max_shared_mem);

    if (selected_config != nullptr) {
      selected_config->thread_k = thread_k;
      selected_config->thread_n = thread_n;
      selected_config->num_threads = num_threads;
      selected_config->thread_m_blocks = thread_m_blocks;
      selected_config->m_block_size_8 = m_block_size_8;
      selected_config->group_blocks = group_blocks;
      selected_config->stages = stages;
    }

    A_ptr += prob_m_split * (lda / 8);
    C_ptr += prob_m_split * (n / 8);
    rest_m -= prob_m_split;
  }

  return true;
}

}  // namespace marlin
