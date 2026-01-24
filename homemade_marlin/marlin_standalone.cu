/*
 * Standalone Marlin INT4 GEMM Kernel
 * Extracted from Marlin with support for group_size=128
 *
 * Features:
 * - INT4 weight quantization with group size = 128
 * - Optimized for m=1 (inference batch size 1)
 * - Tensor Core acceleration with m16n8k16
 * - 4-stage async pipeline
 * - Interleaved weight layout for optimal memory access
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstring>

// Disable createpolicy cache hints by default for broader arch compatibility.
#ifndef MARLIN_USE_CREATEPOLICY
#define MARLIN_USE_CREATEPOLICY 0
#endif

// ============================================================================
// Utility Functions
// ============================================================================
__host__ __device__ constexpr int ceildiv(int a, int b) {
  return (a + b - 1) / b;
}

// Vector type for organizing registers
template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) {
    return elems[i];
  }
};

// ============================================================================
// Matrix Fragment Types
// ============================================================================

using I4 = Vec<int, 4>;           // 4x int32 = 128 bits = 32x INT4 values
using FragA = Vec<half2, 4>;      // Input A fragment: 4x half2 = 16x FP16
using FragB = Vec<half2, 2>;      // Weight B fragment: 2x half2 = 4x FP16
using FragC = Vec<float, 4>;      // Accumulator fragment: 4x float
using FragS = Vec<half2, 1>;      // Scale fragment: 1x half2 = 2x FP16

// ============================================================================
// Async Memory Operations
// ============================================================================

// Predicated async global->shared copy for A matrix
__device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr, bool pred = true) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "{\n"
    "   .reg .pred p;\n"
    "   setp.ne.b32 p, %0, 0;\n"
    "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
    "}\n" :: "r"((int) pred), "r"(smem), "l"(glob_ptr), "n"(BYTES)
  );
}

// Async global->shared copy with evict_first hint for B matrix
__device__ inline void cp_async4_stream(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
#if MARLIN_USE_CREATEPOLICY
  asm volatile(
    "{\n"
    "   .reg .b64 p;\n"
    "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;"
    "   cp.async.cg.shared.global.L2::cache_hint [%0], [%1], %2, p;\n"
    "}\n" :: "r"(smem), "l"(glob_ptr), "n"(BYTES)
  );
#else
  asm volatile(
    "cp.async.cg.shared.global [%0], [%1], %2;\n" :: "r"(smem), "l"(glob_ptr), "n"(BYTES)
  );
#endif
}

__device__ inline void cp_async_fence() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int n>
__device__ inline void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
}

// ============================================================================
// Tensor Core MMA Operations
// ============================================================================

// m16n8k16 tensor core MMA instruction
__device__ inline void mma(const FragA& a_frag, const FragB& frag_b, FragC& frag_c) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  float* c = reinterpret_cast<float*>(&frag_c);
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
    :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),  "r"(b[0]),  "r"(b[1]),
       "f"(c[0]),  "f"(c[1]),  "f"(c[2]),  "f"(c[3])
  );
}

// m16n16k16 using two m16n8k16 instructions
__device__ inline void mma_m16n16k16(const FragA& a_frag, const FragB& frag_b0, const FragB& frag_b1, FragC& frag_c0, FragC& frag_c1) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
  const uint32_t* b0 = reinterpret_cast<const uint32_t*>(&frag_b0);
  const uint32_t* b1 = reinterpret_cast<const uint32_t*>(&frag_b1);
  float* c0 = reinterpret_cast<float*>(&frag_c0);
  float* c1 = reinterpret_cast<float*>(&frag_c1);

  // First m16n8k16: output columns 0-7
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(c0[0]), "=f"(c0[1]), "=f"(c0[2]), "=f"(c0[3])
    :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),  "r"(b0[0]),  "r"(b0[1]),
       "f"(c0[0]),  "f"(c0[1]),  "f"(c0[2]),  "f"(c0[3])
  );

  // Second m16n8k16: output columns 8-15
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(c1[0]), "=f"(c1[1]), "=f"(c1[2]), "=f"(c1[3])
    :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),  "r"(b1[0]),  "r"(b1[1]),
       "f"(c1[0]),  "f"(c1[1]),  "f"(c1[2]),  "f"(c1[3])
  );
}

// Load 16x16 A matrix fragment from shared memory
__device__ inline void ldsm4(FragA& frag_a, const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(smem)
  );
}

// ============================================================================
// Dequantization Operations
// ============================================================================

// LOP3 instruction for efficient bit manipulation
template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile(
    "lop3.b32 %0, %1, %2, %3, %4;\n"
    : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut)
  );
  return res;
}

// Dequantize INT4 to FP16
// Input: int32 containing 8x INT4 values (4 bits each)
// Output: FragB with 4x FP16 values
__device__ inline FragB dequant(int q) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;

  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);

  // Symmetric INT4 with zero point at -8
  const int SUB = 0x64086408;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd480d480;

  FragB frag_b;
  frag_b[0] = __hsub2(
    *reinterpret_cast<half2*>(&lo),
    *reinterpret_cast<const half2*>(&SUB)
  );
  frag_b[1] = __hfma2(
    *reinterpret_cast<half2*>(&hi),
    *reinterpret_cast<const half2*>(&MUL), *reinterpret_cast<const half2*>(&ADD)
  );
  return frag_b;
}

// Apply quantization scale (for grouped quantization)
__device__ inline void scale(FragB& frag_b, FragS& frag_s, int i) {
  half2 s = __half2half2(reinterpret_cast<__half*>(&frag_s)[i]);
  frag_b[0] = __hmul2(frag_b[0], s);
  frag_b[1] = __hmul2(frag_b[1], s);
}

// ============================================================================
// Barrier Synchronization
// ============================================================================

__device__ inline void barrier_acquire(int* lock, int count) {
  if (threadIdx.x == 0) {
    int state = -1;
    do
      asm volatile ("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
    while (state != count);
  }
  __syncthreads();
}

__device__ inline void barrier_release(int* lock, bool reset = false) {
  __syncthreads();
  if (threadIdx.x == 0) {
    if (reset) {
      lock[0] = 0;
      return;
    }
    int val = 1;
    asm volatile ("fence.acq_rel.gpu;\n");
    asm volatile ("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(lock), "r"(val));
  }
}

// ============================================================================
// Main Marlin Kernel
// ============================================================================

template <
  const int threads,           // 256 threads per block
  const int thread_m_blocks,   // M dimension blocks per threadblock
  const int thread_n_blocks,   // N dimension blocks per threadblock
  const int thread_k_blocks,   // K dimension blocks per threadblock
  const int stages,            // Pipeline stages (4)
  const int group_blocks       // Blocks per quantization group (8 for groupsize=128)
>
__global__ void Marlin(
  const int4* __restrict__ A,  // FP16 input [m, k]
  const int4* __restrict__ B,  // INT4 quantized weights [k/16, n*16/8]
        int4* __restrict__ C,  // FP16 output [m, n]
  const int4* __restrict__ s,  // FP16 scales [(k/groupsize), n]
  int  prob_m,                 // Batch size
  int  prob_n,                 // Output dimension
  int  prob_k,                 // Reduction dimension
  int* locks                   // Barrier synchronization
) {
  // Compute tile dimensions
  int parallel = 1;
  if (prob_m > 16 * thread_m_blocks) {
    parallel = prob_m / (16 * thread_m_blocks);
    prob_m = 16 * thread_m_blocks;
  }

  int k_tiles = prob_k / 16 / thread_k_blocks;
  int n_tiles = prob_n / 16 / thread_n_blocks;
  int iters = ceildiv(k_tiles * n_tiles * parallel, gridDim.x);

  // Ensure iterations align with group boundaries
  if (group_blocks != -1)
    iters = (group_blocks / thread_k_blocks) * ceildiv(iters, (group_blocks / thread_k_blocks));

  // Compute slice information
  int slice_row = (iters * blockIdx.x) % k_tiles;
  int slice_col_par = (iters * blockIdx.x) / k_tiles;
  int slice_col = slice_col_par;
  int slice_iters;
  int slice_count = 0;
  int slice_idx;

  // Handle parallel execution
  if (slice_col_par >= n_tiles) {
    A += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_k / 8;
    C += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_n / 8;
    locks += (slice_col_par / n_tiles) * n_tiles;
    slice_col = slice_col_par % n_tiles;
  }

  // Initialize slice parameters
  auto init_slice = [&] () {
    slice_iters = iters * (blockIdx.x + 1) - (k_tiles * slice_col_par + slice_row);
    if (slice_iters < 0 || slice_col_par >= n_tiles * parallel)
      slice_iters = 0;
    if (slice_iters == 0)
      return;
    if (slice_row + slice_iters > k_tiles)
      slice_iters = k_tiles - slice_row;
    slice_count = 1;
    slice_idx = 0;
    int col_first = iters * ceildiv(k_tiles * slice_col_par, iters);
    if (col_first <= k_tiles * (slice_col_par + 1)) {
      int col_off = col_first - k_tiles * slice_col_par;
      slice_count = ceildiv(k_tiles - col_off, iters);
      if (col_off > 0)
        slice_count++;
      int delta_first = iters * blockIdx.x - col_first;
      if (delta_first < 0 || (col_off == 0 && delta_first == 0))
        slice_idx = slice_count - 1;
      else {
        slice_idx = slice_count - 1 - delta_first / iters;
        if (col_off > 0)
          slice_idx--;
      }
    }
    if (slice_col == n_tiles) {
      A += 16 * thread_m_blocks * prob_k / 8;
      C += 16 * thread_m_blocks * prob_n / 8;
      locks += n_tiles;
      slice_col = 0;
    }
  };
  init_slice();

  // ========================================================================
  // Memory Access Parameters - A Matrix
  // ========================================================================
  int a_gl_stride = prob_k / 8;
  constexpr int a_sh_stride = 16 * thread_k_blocks / 8;
  constexpr int a_gl_rd_delta_o = 16 * thread_k_blocks / 8;
  int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o);
  constexpr int a_sh_wr_delta = a_sh_stride * (threads / a_gl_rd_delta_o);
  constexpr int a_sh_rd_delta_o = 2 * ((threads / 32) / (thread_n_blocks / 4));
  constexpr int a_sh_rd_delta_i = a_sh_stride * 16;
  constexpr int a_sh_stage = a_sh_stride * (16 * thread_m_blocks);
  constexpr int a_sh_wr_iters = ceildiv(a_sh_stage, a_sh_wr_delta);

  // ========================================================================
  // Memory Access Parameters - B Matrix (Weights)
  // ========================================================================
  int b_gl_stride = 16 * prob_n / 32;
  constexpr int b_sh_stride = 32 * thread_n_blocks / 4;
  int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks;
  int b_gl_rd_delta_i = b_gl_stride * (threads / b_sh_stride);
  constexpr int b_sh_wr_delta = threads;
  constexpr int b_sh_rd_delta = threads;
  constexpr int b_sh_stage = b_sh_stride * thread_k_blocks;
  constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;

  // ========================================================================
  // Memory Access Parameters - Scales
  // ========================================================================
  int s_gl_stride = prob_n / 8;
  constexpr int s_sh_stride = 16 * thread_n_blocks / 8;
  constexpr int s_sh_stage = s_sh_stride;
  int s_gl_rd_delta = s_gl_stride;

  // ========================================================================
  // Compute Thread-Specific Indices
  // ========================================================================

  // A matrix indices
  int a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
  a_gl_rd += a_gl_rd_delta_o * slice_row;
  int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
  int a_sh_rd = a_sh_stride * ((threadIdx.x % 32) % 16) + (threadIdx.x % 32) / 16;
  a_sh_rd += 2 * ((threadIdx.x / 32) / (thread_n_blocks / 4));

  // B matrix indices
  int b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);
  b_gl_rd += b_sh_stride * slice_col;
  b_gl_rd += b_gl_rd_delta_o * slice_row;
  int b_sh_wr = threadIdx.x;
  int b_sh_rd = threadIdx.x;

  // Scale indices
  int s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) + s_sh_stride * slice_col + threadIdx.x;
  int s_sh_wr = threadIdx.x;
  int s_sh_rd;
  if (group_blocks != -1)
    s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + (threadIdx.x % 32) / 4;
  else
    s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + (threadIdx.x % 32) % 4;

  bool a_sh_wr_pred[a_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_pred[i] = a_sh_wr_delta * i + a_sh_wr < a_sh_stride * prob_m;
  bool s_sh_wr_pred = threadIdx.x < s_sh_stride;

  // XOR-based layout to avoid bank conflicts
  auto transform_a = [&] (int i) {
    int row = i / a_gl_rd_delta_o;
    return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ row;
  };

  int a_sh_wr_trans[a_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);

  int a_sh_rd_trans[b_sh_wr_iters][thread_m_blocks];
  #pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++) {
    #pragma unroll
    for (int j = 0; j < thread_m_blocks; j++)
      a_sh_rd_trans[i][j] = transform_a(a_sh_rd_delta_o * i + a_sh_rd_delta_i * j + a_sh_rd);
  }

  // B matrix pointers
  const int4* B_ptr[b_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++)
    B_ptr[i] = B + b_gl_rd_delta_i * i + b_gl_rd;

  // ========================================================================
  // Shared Memory and Register Allocation
  // ========================================================================
  extern __shared__ int4 sh[];
  int4* sh_a = sh;
  int4* sh_b = sh_a + (stages * a_sh_stage);
  int4* sh_s = sh_b + (stages * b_sh_stage);

  FragA frag_a[2][thread_m_blocks];
  I4 frag_b_quant[2];
  FragC frag_c[thread_m_blocks][4][2];
  FragS frag_s[2][4];

  // ========================================================================
  // Lambda Functions
  // ========================================================================

  // Zero accumulators
  auto zero_accums = [&] () {
    #pragma unroll
    for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++)
      reinterpret_cast<float*>(frag_c)[i] = 0;
  };

  // Fetch tile to shared memory
  auto fetch_to_shared = [&] (int pipe, int a_off, bool pred = true) {
    if (pred) {
      int4* sh_a_stage = sh_a + a_sh_stage * pipe;
      #pragma unroll
      for (int i = 0; i < a_sh_wr_iters; i++) {
        cp_async4_pred(
          &sh_a_stage[a_sh_wr_trans[i]],
          &A[a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * a_off],
          a_sh_wr_pred[i]
        );
      }
      int4* sh_b_stage = sh_b + b_sh_stage * pipe;
      #pragma unroll
      for (int i = 0; i < b_sh_wr_iters; i++) {
        cp_async4_stream(&sh_b_stage[b_sh_wr_delta * i + b_sh_wr], B_ptr[i]);
        B_ptr[i] += b_gl_rd_delta_o;
      }
      // Fetch scales at group boundaries
      if (group_blocks != -1 && pipe % (group_blocks / thread_k_blocks) == 0) {
        int4* sh_s_stage = sh_s + s_sh_stage * pipe;
        if (s_sh_wr_pred)
          cp_async4_stream(&sh_s_stage[s_sh_wr], &s[s_gl_rd]);
        s_gl_rd += s_gl_rd_delta;
      }
    }
    cp_async_fence();
  };

  // Wait for stage to complete
  auto wait_for_stage = [&] () {
    cp_async_wait<stages - 2>();
    __syncthreads();
  };

  // Fetch from shared to registers
  auto fetch_to_registers = [&] (int k, int pipe) {
    if (group_blocks != -1) {
      int4* sh_s_stage = sh_s + s_sh_stage * ((group_blocks / thread_k_blocks) * (pipe / (group_blocks / thread_k_blocks)));
      reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_stage[s_sh_rd];
    }
    int4* sh_a_stage = sh_a + a_sh_stage * pipe;
    #pragma unroll
    for (int i = 0; i < thread_m_blocks; i++)
      ldsm4(frag_a[k % 2][i], &sh_a_stage[a_sh_rd_trans[k % b_sh_wr_iters][i]]);
    int4* sh_b_stage = sh_b + b_sh_stage * pipe;
    frag_b_quant[k % 2] = *reinterpret_cast<I4*>(&sh_b_stage[b_sh_rd_delta * (k % b_sh_wr_iters) + b_sh_rd]);
  };

  // Execute matmul
  auto matmul = [&] (int k) {
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      int b_quant = frag_b_quant[k % 2][j];
      int b_quant_shift = b_quant >> 8;

      FragB frag_b0 = dequant(b_quant);
      if (group_blocks != -1)
        scale(frag_b0, frag_s[k % 2][j], 0);

      FragB frag_b1 = dequant(b_quant_shift);
      if (group_blocks != -1)
        scale(frag_b1, frag_s[k % 2][j], 1);

      #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        mma(frag_a[k % 2][i], frag_b0, frag_c[i][j][0]);
        mma(frag_a[k % 2][i], frag_b1, frag_c[i][j][1]);
      }
    }
  };

  // Threadblock reduction
  auto thread_block_reduce = [&] () {
    constexpr int red_off = threads / b_sh_stride / 2;
    if (red_off >= 1) {
      int red_idx = threadIdx.x / b_sh_stride;
      constexpr int red_sh_stride = b_sh_stride * 4 * 2;
      constexpr int red_sh_delta = b_sh_stride;
      int red_sh_rd = red_sh_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);

      #pragma unroll
      for (int m_block = 0; m_block < thread_m_blocks; m_block++) {
        #pragma unroll
        for (int i = red_off; i > 0; i /= 2) {
          if (i <= red_idx && red_idx < 2 * i) {
            #pragma unroll
            for (int j = 0; j < 4 * 2; j++) {
              int red_sh_wr = red_sh_delta * j + (red_sh_rd - red_sh_stride * i);
              if (i < red_off) {
                float* c_rd = reinterpret_cast<float*>(&sh[red_sh_delta * j + red_sh_rd]);
                float* c_wr = reinterpret_cast<float*>(&sh[red_sh_wr]);
                #pragma unroll
                for (int k = 0; k < 4; k++)
                  reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + j][k] += c_rd[k] + c_wr[k];
              }
              sh[red_sh_wr] = reinterpret_cast<int4*>(&frag_c)[4 * 2 * m_block + j];
            }
          }
          __syncthreads();
        }
        if (red_idx == 0) {
          #pragma unroll
          for (int i = 0; i < 4 * 2; i++) {
            float* c_rd = reinterpret_cast<float*>(&sh[red_sh_delta * i + red_sh_rd]);
            #pragma unroll
            for (int j = 0; j < 4; j++)
              reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + i][j] += c_rd[j];
          }
        }
        __syncthreads();
      }
    }
  };

  // Global reduction
  auto global_reduce = [&] (bool first = false, bool last = false) {
    constexpr int active_threads = 32 * thread_n_blocks / 4;
    if (threadIdx.x < active_threads) {
      int c_gl_stride = prob_n / 8;
      int c_gl_wr_delta_o = 8 * c_gl_stride;
      int c_gl_wr_delta_i = 4 * (active_threads / 32);
      int c_gl_wr = c_gl_stride * ((threadIdx.x % 32) / 4) + 4 * (threadIdx.x / 32) + threadIdx.x % 4;
      c_gl_wr += (2 * thread_n_blocks) * slice_col;
      constexpr int c_sh_wr_delta = active_threads;
      int c_sh_wr = threadIdx.x;

      int row = (threadIdx.x % 32) / 4;

      if (!first) {
        #pragma unroll
        for (int i = 0; i < thread_m_blocks * 4; i++) {
          cp_async4_pred(
            &sh[c_sh_wr + c_sh_wr_delta * i],
            &C[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2)],
            i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m
          );
        }
        cp_async_fence();
        cp_async_wait<0>();
      }

      #pragma unroll
      for (int i = 0; i < thread_m_blocks * 4; i++) {
        if (i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m) {
          if (!first) {
            int4 c_red = sh[c_sh_wr + i * c_sh_wr_delta];
            #pragma unroll
            for (int j = 0; j < 2 * 4; j++) {
              reinterpret_cast<float*>(&frag_c)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)] += __half2float(
                reinterpret_cast<__half*>(&c_red)[j]
              );
            }
          }
          if (!last) {
            int4 c;
            #pragma unroll
            for (int j = 0; j < 2 * 4; j++) {
              reinterpret_cast<__half*>(&c)[j] = __float2half(
                reinterpret_cast<float*>(&frag_c)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)]
              );
            }
            C[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2)] = c;
          }
        }
      }
    }
  };

  // Write final result
  auto write_result = [&] () {
    int c_gl_stride = prob_n / 8;
    constexpr int c_sh_stride = 2 * thread_n_blocks + 1;
    int c_gl_wr_delta = c_gl_stride * (threads / (2 * thread_n_blocks));
    constexpr int c_sh_rd_delta = c_sh_stride * (threads / (2 * thread_n_blocks));

    int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));
    c_gl_wr += (2 * thread_n_blocks) * slice_col;
    int c_sh_wr = (4 * c_sh_stride) * ((threadIdx.x % 32) / 4) + (threadIdx.x % 32) % 4;
    c_sh_wr += 32 * (threadIdx.x / 32);
    int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));

    int c_gl_wr_end = c_gl_stride * prob_m;

    auto write = [&] (int idx, float c0, float c1, FragS& s) {
      half2 res = __halves2half2(__float2half(c0), __float2half(c1));
      if (group_blocks == -1)
        res = __hmul2(res, s[0]);
      ((half2*) sh)[idx] = res;
    };

    if (threadIdx.x / 32 < thread_n_blocks / 4) {
      #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
          int wr = c_sh_wr + 8 * j;
          write(wr + (4 * c_sh_stride) * 0 + 0, frag_c[i][j][0][0], frag_c[i][j][0][1], frag_s[j / 2][2 * (j % 2) + 0]);
          write(wr + (4 * c_sh_stride) * 8 + 0, frag_c[i][j][0][2], frag_c[i][j][0][3], frag_s[j / 2][2 * (j % 2) + 0]);
          write(wr + (4 * c_sh_stride) * 0 + 4, frag_c[i][j][1][0], frag_c[i][j][1][1], frag_s[j / 2][2 * (j % 2) + 1]);
          write(wr + (4 * c_sh_stride) * 8 + 4, frag_c[i][j][1][2], frag_c[i][j][1][3], frag_s[j / 2][2 * (j % 2) + 1]);
        }
        c_sh_wr += 16 * (4 * c_sh_stride);
      }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ceildiv(16 * thread_m_blocks, threads / (2 * thread_n_blocks)); i++) {
      if (c_gl_wr < c_gl_wr_end) {
        C[c_gl_wr] = sh[c_sh_rd];
        c_gl_wr += c_gl_wr_delta;
        c_sh_rd += c_sh_rd_delta;
      }
    }
  };

  // ========================================================================
  // Pipeline Initialization and Main Loop
  // ========================================================================

  // Start pipelines
  auto start_pipes = [&] () {
    #pragma unroll
    for (int i = 0; i < stages - 1; i++)
      fetch_to_shared(i, i, i < slice_iters);
    zero_accums();
    wait_for_stage();
    fetch_to_registers(0, 0);
    a_gl_rd += a_gl_rd_delta_o * (stages - 1);
  };
  start_pipes();

  // Main loop
  while (slice_iters) {
    #pragma unroll
    for (int pipe = 0; pipe < stages;) {
      #pragma unroll
      for (int k = 0; k < b_sh_wr_iters; k++) {
        fetch_to_registers(k + 1, pipe % stages);
        if (k == b_sh_wr_iters - 2) {
          fetch_to_shared((pipe + stages - 1) % stages, pipe, slice_iters >= stages);
          pipe++;
          wait_for_stage();
        }
        matmul(k);
      }
      slice_iters--;
      if (slice_iters == 0)
        break;
    }
    a_gl_rd += a_gl_rd_delta_o * stages;

    if (slice_iters == 0) {
      cp_async_wait<0>();
      bool last = slice_idx == slice_count - 1;

      if (group_blocks == -1 && last) {
        if (s_sh_wr_pred)
          cp_async4_stream(&sh_s[s_sh_wr], &s[s_gl_rd]);
        cp_async_fence();
      }

      thread_block_reduce();

      if (group_blocks == -1 && last) {
        cp_async_wait<0>();
        __syncthreads();
        if (threadIdx.x / 32 < thread_n_blocks / 4) {
          reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd + 0];
          reinterpret_cast<int4*>(&frag_s)[1] = sh_s[s_sh_rd + 4];
        }
      }

      if (slice_count > 1) {
        barrier_acquire(&locks[slice_col], slice_idx);
        global_reduce(slice_idx == 0, last);
        barrier_release(&locks[slice_col], last);
      }

      if (last)
        write_result();

      slice_row = 0;
      slice_col_par++;
      slice_col++;
      init_slice();

      if (slice_iters) {
        a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
        #pragma unroll
        for (int i = 0; i < b_sh_wr_iters; i++)
          B_ptr[i] += b_sh_stride - b_gl_rd_delta_o * k_tiles;
        if (slice_col == 0) {
          #pragma unroll
          for (int i = 0; i < b_sh_wr_iters; i++)
            B_ptr[i] -= b_gl_stride;
        }
        s_gl_rd = s_sh_stride * slice_col + threadIdx.x;
        start_pipes();
      }
    }
  }
}

// ============================================================================
// Kernel Launch Configuration
// ============================================================================

const int THREADS = 256;
const int STAGES = 4;
const int SHARED_MEM = 96 * 1024;  // 96 KB shared memory

#define CALL_IF(THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, GROUP_BLOCKS) \
  else if ( \
    thread_m_blocks == THREAD_M_BLOCKS && thread_n_blocks == THREAD_N_BLOCKS && thread_k_blocks == THREAD_K_BLOCKS && \
    group_blocks == GROUP_BLOCKS \
  ) { \
    cudaFuncSetAttribute( \
      Marlin<THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, STAGES, GROUP_BLOCKS>, \
      cudaFuncAttributeMaxDynamicSharedMemorySize, \
      SHARED_MEM \
    ); \
    Marlin< \
      THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, STAGES, GROUP_BLOCKS \
    ><<<blocks, THREADS, SHARED_MEM, stream>>>( \
      A_ptr, B_ptr, C_ptr, s_ptr, \
      prob_m, prob_n, prob_k, \
      locks \
    ); \
  }

const int ERR_PROB_SHAPE = 1;
const int ERR_KERN_SHAPE = 2;

int marlin_cuda(
  const void* A,
  const void* B,
        void* C,
        void* s,
  int prob_m,
  int prob_n,
  int prob_k,
  void* workspace,
  int groupsize = -1,
  int dev = 0,
  cudaStream_t stream = 0,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 16
) {
  int tot_m = prob_m;
  int tot_m_blocks = ceildiv(tot_m, 16);
  int pad = 16 * tot_m_blocks - tot_m;

  if (sms == -1)
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  if (thread_k == -1 || thread_n == -1) {
    if (prob_m <= 16) {
      thread_k = 128;
      thread_n = 128;
    } else {
      thread_k = 64;
      thread_n = 256;
    }
  }

  int thread_k_blocks = thread_k / 16;
  int thread_n_blocks = thread_n / 16;
  int group_blocks = (groupsize == -1) ? -1 : groupsize / 16;
  int blocks = sms;

  if (prob_n % thread_n != 0 || prob_k % thread_k != 0 || (group_blocks != -1 && prob_k % group_blocks != 0))
    return ERR_PROB_SHAPE;
  if (prob_m == 0 || prob_n == 0 || prob_k == 0)
    return 0;

  const int4* A_ptr = (const int4*) A;
  const int4* B_ptr = (const int4*) B;
  int4* C_ptr = (int4*) C;
  const int4* s_ptr = (const int4*) s;

  int* locks = (int*) workspace;

  int ret = 0;
  for (int i = 0; i < tot_m_blocks; i += 4) {
    int thread_m_blocks = tot_m_blocks - i;
    prob_m = tot_m - 16 * i;
    int par = 1;
    if (thread_m_blocks > 4) {
      par = (16 * thread_m_blocks - pad) / 64;
      if (par > max_par)
        par = max_par;
      prob_m = 64 * par;
      i += 4 * (par - 1);
      thread_m_blocks = 4;
    }

    if (false) {}
    CALL_IF(1,  8,  8, -1)
    CALL_IF(1,  8,  8,  8)
    CALL_IF(1, 16,  4, -1)
    CALL_IF(1, 16,  4,  8)
    CALL_IF(2, 16,  4, -1)
    CALL_IF(2, 16,  4,  8)
    CALL_IF(3, 16,  4, -1)
    CALL_IF(3, 16,  4,  8)
    CALL_IF(4, 16,  4, -1)
    CALL_IF(4, 16,  4,  8)
    else
      ret = ERR_KERN_SHAPE;

    A_ptr += 16 * thread_m_blocks * (prob_k / 8) * par;
    C_ptr += 16 * thread_m_blocks * (prob_n / 8) * par;
  }

  return ret;
}

void print_usage(const char* prog_name) {
  printf("Usage: %s [OPTIONS]\n", prog_name);
  printf("\nOptions:\n");
  printf("  -m <M>          Batch size (default: 128)\n");
  printf("  -n <N>          Output dimension (default: 256)\n");
  printf("  -k <K>          Input dimension (default: 512)\n");
  printf("  -g <groupsize>  Quantization group size, -1 for per-column (default: -1)\n");
  printf("  -s <sms>        Number of SMs to use, -1 for auto (default: -1)\n");
  printf("  -w <warmup>     Warmup iterations (default: 10)\n");
  printf("  -i <iters>      Timed iterations (default: 100)\n");
  printf("  -h              Show this help message\n");
  printf("\nExample:\n");
  printf("  %s -m 128 -n 4096 -k 4096 -g 128 -s 108\n", prog_name);
}

int main(int argc, char* argv[]) {
  // Default parameters
  int M = 128;           // Batch size
  int N = 256;           // Output dimension
  int K = 512;           // Input dimension
  int groupsize = -1;    // -1 for per-column quantization
  int num_sms = -1;      // -1 for auto-detect
  int warmup = 10;
  int iters = 100;

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
      M = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
      N = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
      K = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-g") == 0 && i + 1 < argc) {
      groupsize = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
      num_sms = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
      warmup = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
      iters = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      print_usage(argv[0]);
      return 0;
    } else {
      printf("Unknown option: %s\n", argv[i]);
      print_usage(argv[0]);
      return 1;
    }
  }

  // Validate parameters
  if (M <= 0 || N <= 0 || K <= 0) {
    printf("Error: M, N, K must be positive integers\n");
    return 1;
  }
  if (warmup < 0 || iters <= 0) {
    printf("Error: warmup must be >= 0 and iters must be > 0\n");
    return 1;
  }

  // Print configuration
  printf("========================================\n");
  printf("Marlin INT4 GEMM Configuration\n");
  printf("========================================\n");
  printf("Matrix dimensions:\n");
  printf("  M (batch size):      %d\n", M);
  printf("  N (output dim):      %d\n", N);
  printf("  K (input dim):       %d\n", K);
  if (groupsize == -1) {
    printf("  Group size:          per-column\n");
  } else {
    printf("  Group size:          %d\n", groupsize);
  }
  if (num_sms == -1) {
  printf("  Number of SMs:       auto-detect\n");
  } else {
    printf("  Number of SMs:       %d\n", num_sms);
  }
  printf("  Warmup iterations:   %d\n", warmup);
  printf("  Timed iterations:    %d\n", iters);
  printf("========================================\n\n");

  // Allocate device memory
  void *d_A, *d_B, *d_C, *d_s, *d_workspace;
  size_t A_size = M * K * sizeof(half);
  size_t B_size = (K * N) / 2;  // 4-bit packed weights
  size_t C_size = M * N * sizeof(half);
  size_t s_size = (groupsize == -1) ? N * sizeof(half) : (K / groupsize) * N * sizeof(half);
  size_t workspace_size = (N / 128) * 16 * sizeof(int);

  printf("Allocating device memory...\n");
  printf("  A matrix:      %.2f MB\n", A_size / (1024.0 * 1024.0));
  printf("  B matrix:      %.2f MB\n", B_size / (1024.0 * 1024.0));
  printf("  C matrix:      %.2f MB\n", C_size / (1024.0 * 1024.0));
  printf("  Scales:        %.2f MB\n", s_size / (1024.0 * 1024.0));
  printf("  Workspace:     %.2f KB\n\n", workspace_size / 1024.0);

  cudaMalloc(&d_A, A_size);
  cudaMalloc(&d_B, B_size);
  cudaMalloc(&d_C, C_size);
  cudaMalloc(&d_s, s_size);
  cudaMalloc(&d_workspace, workspace_size);

  // Initialize with random data
  cudaMemset(d_A, 0, A_size);
  cudaMemset(d_B, 0, B_size);
  cudaMemset(d_s, 1, s_size);
  cudaMemset(d_workspace, 0, workspace_size);

  printf("Running kernel...\n");

  for (int i = 0; i < warmup; ++i) {
    marlin_cuda(
      d_A, d_B, d_C, d_s,
      M, N, K,
      d_workspace,
      groupsize,
      0,           // device 0
      0,           // default stream
      -1,          // auto thread_k
      -1,          // auto thread_n
      num_sms,     // custom number of SMs
      16           // max_par
    );
    cudaError_t warmup_err = cudaGetLastError();
    if (warmup_err != cudaSuccess) {
      printf("Warmup launch failed: %s\n", cudaGetErrorString(warmup_err));
      return 1;
    }
  }
  cudaDeviceSynchronize();

  cudaEvent_t start{};
  cudaEvent_t stop{};
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  int result = 0;
  for (int i = 0; i < iters; ++i) {
    result = marlin_cuda(
      d_A, d_B, d_C, d_s,
      M, N, K,
      d_workspace,
      groupsize,
      0,           // device 0
      0,           // default stream
      -1,          // auto thread_k
      -1,          // auto thread_n
      num_sms,     // custom number of SMs
      16           // max_par
    );
    if (result != 0) {
      break;
    }
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
      printf("Timed launch failed: %s\n", cudaGetErrorString(launch_err));
      result = launch_err;
      break;
    }
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  if (result == 0) {
    printf("\n");
    printf("========================================\n");
    printf("SUCCESS!\n");
    printf("========================================\n");
    printf("Kernel executed successfully!\n");
    printf("Matrix multiplication: (%d x %d) * (%d x %d) = (%d x %d)\n", M, K, K, N, M, N);
    printf("Avg kernel time: %.3f us (%d iters, %d warmup)\n",
           (ms * 1000.0f) / (float)iters, iters, warmup);
    printf("========================================\n");
  } else if (result == ERR_PROB_SHAPE) {
    printf("\n");
    printf("========================================\n");
    printf("ERROR: Problem shape incompatible\n");
    printf("========================================\n");
    printf("The problem dimensions are not compatible with kernel constraints.\n");
    printf("Ensure:\n");
    printf("  - N is divisible by thread_n (128 or 256)\n");
    printf("  - K is divisible by thread_k (64 or 128)\n");
    printf("  - If using group quantization, K is divisible by groupsize\n");
    printf("========================================\n");
  } else if (result == ERR_KERN_SHAPE) {
    printf("\n");
    printf("========================================\n");
    printf("ERROR: No kernel implementation\n");
    printf("========================================\n");
    printf("No kernel implementation available for these parameters.\n");
    printf("Try different M, N, K, or groupsize values.\n");
    printf("========================================\n");
  }

  // Cleanup
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_s);
  cudaFree(d_workspace);

  return result;
}
