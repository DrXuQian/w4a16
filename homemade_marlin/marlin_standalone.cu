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

#include <cuda.h>  // include cuda.h
#include <cuda_fp16.h>  // include cuda_fp16.h
#include <cuda_runtime.h>  // include cuda_runtime.h
#include <iostream>  // include iostream
#include <cstdlib>  // include cstdlib
#include <cstdio>  // include cstdio
#include <cstring>  // include cstring

// Disable createpolicy cache hints by default for broader arch compatibility.
#ifndef MARLIN_USE_CREATEPOLICY  // cfg
#define MARLIN_USE_CREATEPOLICY 0  // macro
#endif  // cfg

// Optional device-side tracing via printf (slow; intended for understanding/debug only).
// Enabled at runtime via the standalone CLI flag `--trace`.
struct MarlinTraceConfig {  // type
  int enabled;  // enable device-side printf tracing
  int block;  // only trace this CTA (blockIdx.x)
  int thread;  // only trace this thread (threadIdx.x)
};  // end MarlinTraceConfig

__device__ __managed__ MarlinTraceConfig g_marlin_trace{0, 0, 0};  // runtime trace controls (managed)

// ============================================================================
// Utility Functions
// ============================================================================
__host__ __device__ constexpr int ceildiv(int a, int b) {  // host/device fn
  return (a + b - 1) / b;  // return
}  // scope

// Vector type for organizing registers
template <typename T, int n>  // template
struct Vec {  // type
  T elems[n];  // stmt
  __device__ T& operator[](int i) {  // device fn
    return elems[i];  // return
  }  // scope
};  // scope

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
__device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr, bool pred = true) {  // device fn
  const int BYTES = 16;  // init
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));  // init
  asm volatile(  // ptx asm
    "{\n"  // stmt
    "   .reg .pred p;\n"  // stmt
    "   setp.ne.b32 p, %0, 0;\n"  // stmt
    "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"  // stmt
    "}\n" :: "r"((int) pred), "r"(smem), "l"(glob_ptr), "n"(BYTES)  // stmt
  );  // scope
}  // scope

// Async global->shared copy with evict_first hint for B matrix
__device__ inline void cp_async4_stream(void* smem_ptr, const void* glob_ptr) {  // device fn
  const int BYTES = 16;  // init
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));  // init
#if MARLIN_USE_CREATEPOLICY  // cfg
  asm volatile(  // ptx asm
    "{\n"  // stmt
    "   .reg .b64 p;\n"  // stmt
    "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;"  // stmt
    "   cp.async.cg.shared.global.L2::cache_hint [%0], [%1], %2, p;\n"  // stmt
    "}\n" :: "r"(smem), "l"(glob_ptr), "n"(BYTES)  // stmt
  );  // scope
#else  // cfg
  asm volatile(  // ptx asm
    "cp.async.cg.shared.global [%0], [%1], %2;\n" :: "r"(smem), "l"(glob_ptr), "n"(BYTES)  // stmt
  );  // scope
#endif  // cfg
}  // scope

__device__ inline void cp_async_fence() {  // device fn
  asm volatile("cp.async.commit_group;\n" ::);  // ptx asm
}  // scope

template <int n>  // template
__device__ inline void cp_async_wait() {  // device fn
  asm volatile("cp.async.wait_group %0;\n" :: "n"(n));  // ptx asm
}  // scope

// ============================================================================
// Tensor Core MMA Operations
// ============================================================================

// m16n8k16 tensor core MMA instruction
__device__ inline void mma(const FragA& a_frag, const FragB& frag_b, FragC& frag_c) {  // device fn
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);  // init
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);  // init
  float* c = reinterpret_cast<float*>(&frag_c);  // init
  asm volatile(  // ptx asm
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "  // mma
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"  // stmt
    : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])  // stmt
    :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),  "r"(b[0]),  "r"(b[1]),  // stmt
       "f"(c[0]),  "f"(c[1]),  "f"(c[2]),  "f"(c[3])  // stmt
  );  // scope
}  // scope

// m16n16k16 using two m16n8k16 instructions
__device__ inline void mma_m16n16k16(const FragA& a_frag, const FragB& frag_b0, const FragB& frag_b1, FragC& frag_c0, FragC& frag_c1) {  // device fn
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);  // init
  const uint32_t* b0 = reinterpret_cast<const uint32_t*>(&frag_b0);  // init
  const uint32_t* b1 = reinterpret_cast<const uint32_t*>(&frag_b1);  // init
  float* c0 = reinterpret_cast<float*>(&frag_c0);  // init
  float* c1 = reinterpret_cast<float*>(&frag_c1);  // init

  // First m16n8k16: output columns 0-7
  asm volatile(  // ptx asm
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "  // mma
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"  // stmt
    : "=f"(c0[0]), "=f"(c0[1]), "=f"(c0[2]), "=f"(c0[3])  // stmt
    :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),  "r"(b0[0]),  "r"(b0[1]),  // stmt
       "f"(c0[0]),  "f"(c0[1]),  "f"(c0[2]),  "f"(c0[3])  // stmt
  );  // scope

  // Second m16n8k16: output columns 8-15
  asm volatile(  // ptx asm
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "  // mma
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"  // stmt
    : "=f"(c1[0]), "=f"(c1[1]), "=f"(c1[2]), "=f"(c1[3])  // stmt
    :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),  "r"(b1[0]),  "r"(b1[1]),  // stmt
       "f"(c1[0]),  "f"(c1[1]),  "f"(c1[2]),  "f"(c1[3])  // stmt
  );  // scope
}  // scope

// Load 16x16 A matrix fragment from shared memory
__device__ inline void ldsm4(FragA& frag_a, const void* smem_ptr) {  // device fn
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);  // init
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));  // init
  asm volatile(  // ptx asm
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"  // ldmatrix
    : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(smem)  // stmt
  );  // scope
}  // scope

// ============================================================================
// Dequantization Operations
// ============================================================================

// LOP3 instruction for efficient bit manipulation
template <int lut>  // template
__device__ inline int lop3(int a, int b, int c) {  // device fn
  int res;  // stmt
  asm volatile(  // ptx asm
    "lop3.b32 %0, %1, %2, %3, %4;\n"  // stmt
    : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut)  // stmt
  );  // scope
  return res;  // return
}  // scope

// Dequantize INT4 to FP16
// Input: int32 containing 8x INT4 values (4 bits each)
// Output: FragB with 4x FP16 values
__device__ inline FragB dequant(int q) {  // device fn
  const int LO = 0x000f000f;  // init
  const int HI = 0x00f000f0;  // init
  const int EX = 0x64006400;  // init

  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);  // init
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);  // init

  // Symmetric INT4 with zero point at -8
  const int SUB = 0x64086408;  // init
  const int MUL = 0x2c002c00;  // init
  const int ADD = 0xd480d480;  // init

  FragB frag_b;  // stmt
  frag_b[0] = __hsub2(  // stmt
    *reinterpret_cast<half2*>(&lo),  // stmt
    *reinterpret_cast<const half2*>(&SUB)  // stmt
  );  // scope
  frag_b[1] = __hfma2(  // stmt
    *reinterpret_cast<half2*>(&hi),  // stmt
    *reinterpret_cast<const half2*>(&MUL), *reinterpret_cast<const half2*>(&ADD)  // stmt
  );  // scope
  return frag_b;  // return
}  // scope

// Apply quantization scale (for grouped quantization)
__device__ inline void scale(FragB& frag_b, FragS& frag_s, int i) {  // device fn
  half2 s = __half2half2(reinterpret_cast<__half*>(&frag_s)[i]);  // init
  frag_b[0] = __hmul2(frag_b[0], s);  // init
  frag_b[1] = __hmul2(frag_b[1], s);  // init
}  // scope

// ============================================================================
// Barrier Synchronization
// ============================================================================

__device__ inline void barrier_acquire(int* lock, int count) {  // device fn
  if (threadIdx.x == 0) {  // branch
    int state = -1;  // init
    do  // stmt
      asm volatile ("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));  // ptx asm
    while (state != count);  // loop
  }  // scope
  __syncthreads();  // stmt
}  // scope

__device__ inline void barrier_release(int* lock, bool reset = false) {  // device fn
  __syncthreads();  // stmt
  if (threadIdx.x == 0) {  // branch
    if (reset) {  // branch
      lock[0] = 0;  // init
      return;  // return
    }  // scope
    int val = 1;  // init
    asm volatile ("fence.acq_rel.gpu;\n");  // ptx asm
    asm volatile ("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(lock), "r"(val));  // ptx asm
  }  // scope
}  // scope

// ============================================================================
// Main Marlin Kernel
// ============================================================================

template <  // template
  const int threads,           // 256 threads per block
  const int thread_m_blocks,   // M dimension blocks per threadblock
  const int thread_n_blocks,   // N dimension blocks per threadblock
  const int thread_k_blocks,   // K dimension blocks per threadblock
  const int stages,            // Pipeline stages (4)
  const int group_blocks       // Blocks per quantization group (8 for groupsize=128)
>  // end template params
__global__ void Marlin(  // kernel
  const int4* __restrict__ A,  // FP16 input [m, k]
  const int4* __restrict__ B,  // INT4 quantized weights [k/16, n*16/8]
        int4* __restrict__ C,  // FP16 output [m, n]
  const int4* __restrict__ s,  // FP16 scales [(k/groupsize), n]
  int  prob_m,                 // Batch size
  int  prob_n,                 // Output dimension
  int  prob_k,                 // Reduction dimension
  int* locks                   // Barrier synchronization
) {  // kernel entry
  // Compute tile dimensions
  int parallel = 1;  // problem-parallel factor (M slicing)
  if (prob_m > 16 * thread_m_blocks) {  // branch
    parallel = prob_m / (16 * thread_m_blocks);  // problem-parallel factor (M slicing)
    prob_m = 16 * thread_m_blocks;  // init
  }  // scope

  int k_tiles = prob_k / 16 / thread_k_blocks;  // num K-tiles (each 128 K elems here)
  int n_tiles = prob_n / 16 / thread_n_blocks;  // num N-tiles (each 128 cols here)
  int iters = ceildiv(k_tiles * n_tiles * parallel, gridDim.x);  // timed iters

  // Ensure iterations align with group boundaries
  if (group_blocks != -1)  // branch
    iters = (group_blocks / thread_k_blocks) * ceildiv(iters, (group_blocks / thread_k_blocks));  // timed iters

  // Compute slice information
  int slice_row = (iters * blockIdx.x) % k_tiles;  // tile row in K grid (idx%k_tiles)
  int slice_col_par = (iters * blockIdx.x) / k_tiles;  // tile col in N grid (idx/k_tiles, incl parallel)
  int slice_col = slice_col_par;  // tile col (N tile index)
  int slice_iters;  // K-tiles this CTA will process for current N tile
  int slice_count = 0;  // num blocks contributing to this col
  int slice_idx;  // this CTA's index within slice_count (barrier order)

  // Handle parallel execution
  if (slice_col_par >= n_tiles) {  // branch
    A += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_k / 8;  // init
    C += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_n / 8;  // init
    locks += (slice_col_par / n_tiles) * n_tiles;  // init
    slice_col = slice_col_par % n_tiles;  // tile col (N tile index)
  }  // scope

  // Initialize slice parameters
  auto init_slice = [&] () {  // compute slice_iters/slice_count/slice_idx for this CTA
    slice_iters = iters * (blockIdx.x + 1) - (k_tiles * slice_col_par + slice_row);  // tiles this block does in current col
    if (slice_iters < 0 || slice_col_par >= n_tiles * parallel)  // branch
      slice_iters = 0;  // tiles this block does in current col
    if (slice_iters == 0)  // branch
      return;  // return
    if (slice_row + slice_iters > k_tiles)  // branch
      slice_iters = k_tiles - slice_row;  // tiles this block does in current col
    slice_count = 1;  // num blocks contributing to this col
    slice_idx = 0;  // this block order within slice (for barrier)
    int col_first = iters * ceildiv(k_tiles * slice_col_par, iters);  // init
    if (col_first <= k_tiles * (slice_col_par + 1)) {  // branch
      int col_off = col_first - k_tiles * slice_col_par;  // init
      slice_count = ceildiv(k_tiles - col_off, iters);  // num blocks contributing to this col
      if (col_off > 0)  // branch
        slice_count++;  // account for partial first segment
      int delta_first = iters * blockIdx.x - col_first;  // init
      if (delta_first < 0 || (col_off == 0 && delta_first == 0))  // branch
        slice_idx = slice_count - 1;  // this block order within slice (for barrier)
      else {  // branch
        slice_idx = slice_count - 1 - delta_first / iters;  // this block order within slice (for barrier)
        if (col_off > 0)  // branch
          slice_idx--;  // adjust for non-zero col_off
      }  // scope
    }  // scope
    if (slice_col == n_tiles) {  // branch
      A += 16 * thread_m_blocks * prob_k / 8;  // init
      C += 16 * thread_m_blocks * prob_n / 8;  // init
      locks += n_tiles;  // init
      slice_col = 0;  // tile col (N tile index)
    }  // scope
  };  // end init_slice
  init_slice();  // initialize slice state

  if (g_marlin_trace.enabled && blockIdx.x == g_marlin_trace.block && threadIdx.x == g_marlin_trace.thread) {  // branch
    printf("[marlin_trace] block=%d thread=%d\\n", (int) blockIdx.x, (int) threadIdx.x);  // init
    printf("[marlin_trace] tm=%d tn=%d tk=%d stages=%d group_blocks=%d threads=%d\\n",  // stmt
      thread_m_blocks, thread_n_blocks, thread_k_blocks, stages, group_blocks, threads);  // stmt
    printf("[marlin_trace] prob_m=%d prob_n=%d prob_k=%d parallel=%d\\n", prob_m, prob_n, prob_k, parallel);  // init
    printf("[marlin_trace] k_tiles=%d n_tiles=%d iters=%d\\n", k_tiles, n_tiles, iters);  // num K-tiles (each 128 K elems here)
    if (slice_iters == 0) {  // branch
      printf("[marlin_trace] slice: <inactive>\\n");  // stmt
    } else {  // stmt
      printf("[marlin_trace] slice_row=%d slice_col_par=%d slice_col=%d slice_iters=%d slice_count=%d slice_idx=%d\\n",  // stmt
        slice_row, slice_col_par, slice_col, slice_iters, slice_count, slice_idx);  // stmt
    }  // scope
  }  // scope

  // ========================================================================
  // Memory Access Parameters - A Matrix
  // ========================================================================
  int a_gl_stride = prob_k / 8;  // A row stride in int4 (K/8)
  constexpr int a_sh_stride = 16 * thread_k_blocks / 8;  // A shared stride per row in int4
  constexpr int a_gl_rd_delta_o = 16 * thread_k_blocks / 8;  // A global delta (int4) per tid group
  int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o);  // A global delta (int4) within a tile
  constexpr int a_sh_wr_delta = a_sh_stride * (threads / a_gl_rd_delta_o);  // A shared write delta (int4) per iter
  constexpr int a_sh_rd_delta_o = 2 * ((threads / 32) / (thread_n_blocks / 4));  // A shared read delta across warps
  constexpr int a_sh_rd_delta_i = a_sh_stride * 16;  // A shared read delta across m-blocks
  constexpr int a_sh_stage = a_sh_stride * (16 * thread_m_blocks);  // A shared bytes per stage (in int4)
  constexpr int a_sh_wr_iters = ceildiv(a_sh_stage, a_sh_wr_delta);  // A shared write iters per stage

  // ========================================================================
  // Memory Access Parameters - B Matrix (Weights)
  // ========================================================================
  int b_gl_stride = 16 * prob_n / 32;  // B stride in int4 (layout-specific)
  constexpr int b_sh_stride = 32 * thread_n_blocks / 4;  // B shared stride in int4
  int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks;  // B global delta (int4) per k-tile
  int b_gl_rd_delta_i = b_gl_stride * (threads / b_sh_stride);  // B global delta (int4) per load-iter
  constexpr int b_sh_wr_delta = threads;  // B shared write delta (int4)
  constexpr int b_sh_rd_delta = threads;  // B shared read delta (int4)
  constexpr int b_sh_stage = b_sh_stride * thread_k_blocks;  // B shared int4 per stage
  constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;  // B shared write iters per stage

  // ========================================================================
  // Memory Access Parameters - Scales
  // ========================================================================
  int s_gl_stride = prob_n / 8;  // scale row stride in int4 (N/8)
  constexpr int s_sh_stride = 16 * thread_n_blocks / 8;  // scale shared stride in int4
  constexpr int s_sh_stage = s_sh_stride;  // scale shared int4 per stage
  int s_gl_rd_delta = s_gl_stride;  // scale global delta per group

  // ========================================================================
  // Compute Thread-Specific Indices
  // ========================================================================

  // A matrix indices
  int a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);  // A gmem read index (int4)
  a_gl_rd += a_gl_rd_delta_o * slice_row;  // A gmem read index (int4)
  int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);  // A smem write index (int4)
  int a_sh_rd = a_sh_stride * ((threadIdx.x % 32) % 16) + (threadIdx.x % 32) / 16;  // A smem read index (int4)
  a_sh_rd += 2 * ((threadIdx.x / 32) / (thread_n_blocks / 4));  // A smem read index (int4)

  // B matrix indices
  int b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);  // B gmem read base (int4)
  b_gl_rd += b_sh_stride * slice_col;  // B gmem read base (int4)
  b_gl_rd += b_gl_rd_delta_o * slice_row;  // B gmem read base (int4)
  int b_sh_wr = threadIdx.x;  // B smem write base (int4)
  int b_sh_rd = threadIdx.x;  // B smem read base (int4)

  // Scale indices
  int s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) + s_sh_stride * slice_col + threadIdx.x;  // scale gmem read index (int4)
  int s_sh_wr = threadIdx.x;  // scale smem write index (int4)
  int s_sh_rd;  // scale smem read index (int4)
  if (group_blocks != -1)  // branch
    s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + (threadIdx.x % 32) / 4;  // scale smem read index (int4)
  else  // branch
    s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + (threadIdx.x % 32) % 4;  // scale smem read index (int4)

  bool a_sh_wr_pred[a_sh_wr_iters];  // per-iter pred: A gmem loads are in-bounds
  #pragma unroll  // unroll pred init
  for (int i = 0; i < a_sh_wr_iters; i++)  // loop
    a_sh_wr_pred[i] = a_sh_wr_delta * i + a_sh_wr < a_sh_stride * prob_m;  // init
  bool s_sh_wr_pred = threadIdx.x < s_sh_stride;  // init

  if (g_marlin_trace.enabled && blockIdx.x == g_marlin_trace.block && threadIdx.x == g_marlin_trace.thread) {  // branch
    printf("[marlin_trace] a_gl_stride=%d a_sh_stride=%d\\n", a_gl_stride, a_sh_stride);  // A row stride in int4 (K/8)
    printf("[marlin_trace] a_gl_rd_delta_o=%d a_gl_rd_delta_i=%d a_sh_wr_delta=%d\\n",  // stmt
      a_gl_rd_delta_o, a_gl_rd_delta_i, a_sh_wr_delta);  // stmt
    printf("[marlin_trace] a_sh_rd_delta_o=%d a_sh_rd_delta_i=%d a_sh_stage=%d a_sh_wr_iters=%d\\n",  // stmt
      a_sh_rd_delta_o, a_sh_rd_delta_i, a_sh_stage, a_sh_wr_iters);  // stmt
    printf("[marlin_trace] a_gl_rd=%d a_sh_wr=%d a_sh_rd=%d a_sh_wr_pred0=%d\\n",  // stmt
      a_gl_rd, a_sh_wr, a_sh_rd, (int) a_sh_wr_pred[0]);  // stmt

    printf("[marlin_trace] b_gl_stride=%d b_sh_stride=%d\\n", b_gl_stride, b_sh_stride);  // B stride in int4 (layout-specific)
    printf("[marlin_trace] b_gl_rd_delta_o=%d b_gl_rd_delta_i=%d b_sh_stage=%d b_sh_wr_iters=%d\\n",  // stmt
      b_gl_rd_delta_o, b_gl_rd_delta_i, b_sh_stage, b_sh_wr_iters);  // stmt
    printf("[marlin_trace] b_gl_rd=%d b_sh_wr=%d b_sh_rd=%d\\n", b_gl_rd, b_sh_wr, b_sh_rd);  // B gmem read base (int4)

    printf("[marlin_trace] s_gl_stride=%d s_sh_stride=%d s_gl_rd=%d s_sh_rd=%d s_sh_wr_pred=%d\\n",  // stmt
      s_gl_stride, s_sh_stride, s_gl_rd, s_sh_rd, (int) s_sh_wr_pred);  // stmt
  }  // scope

  // XOR-based layout to avoid bank conflicts
  auto transform_a = [&] (int i) {  // apply XOR swizzle for A shared layout
    int row = i / a_gl_rd_delta_o;  // init
    return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ row;  // return
  };  // end transform_a

  int a_sh_wr_trans[a_sh_wr_iters];  // A smem write indices after swizzle
  #pragma unroll  // unroll swizzle init
  for (int i = 0; i < a_sh_wr_iters; i++)  // loop
    a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);  // init

  int a_sh_rd_trans[b_sh_wr_iters][thread_m_blocks];  // A smem read indices [k-subtile][m_block] after swizzle
  #pragma unroll  // unroll swizzle init
  for (int i = 0; i < b_sh_wr_iters; i++) {  // loop
    #pragma unroll  // unroll m-block init
    for (int j = 0; j < thread_m_blocks; j++)  // loop
      a_sh_rd_trans[i][j] = transform_a(a_sh_rd_delta_o * i + a_sh_rd_delta_i * j + a_sh_rd);  // init
  }  // scope

  // B matrix pointers
  const int4* B_ptr[b_sh_wr_iters];  // per-iter B gmem pointers for cp.async
  #pragma unroll  // unroll pointer init
  for (int i = 0; i < b_sh_wr_iters; i++)  // loop
    B_ptr[i] = B + b_gl_rd_delta_i * i + b_gl_rd;  // init

  // ========================================================================
  // Shared Memory and Register Allocation
  // ========================================================================
  extern __shared__ int4 sh[];  // shared slab: A|B|S pipelines and temp reduce buffer
  int4* sh_a = sh;  // shared A pipeline base
  int4* sh_b = sh_a + (stages * a_sh_stage);  // shared B pipeline base
  int4* sh_s = sh_b + (stages * b_sh_stage);  // shared scale pipeline base

  FragA frag_a[2][thread_m_blocks];  // regs: A fragments (ping-pong over k)
  I4 frag_b_quant[2];  // regs: packed INT4 B fragments (ping-pong over k)
  FragC frag_c[thread_m_blocks][4][2];  // regs: accumulators [m_block][n_subtile][b_half]
  FragS frag_s[2][4];  // regs: scale fragments (ping-pong over k)

  // ========================================================================
  // Lambda Functions
  // ========================================================================

  // Zero accumulators
  auto zero_accums = [&] () {  // zero f32 accumulators in frag_c
    #pragma unroll  // unroll flat store
    for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++)  // loop
      reinterpret_cast<float*>(frag_c)[i] = 0;  // clear one f32 lane
  };  // end zero_accums

  // Fetch tile to shared memory
  auto fetch_to_shared = [&] (int pipe, int a_off, bool pred = true) {  // cp.async A/B(/S) into smem stage=pipe
    if (pred) {  // branch
      int4* sh_a_stage = sh_a + a_sh_stage * pipe;  // A stage base in smem
      #pragma unroll  // unroll A cp.async
      for (int i = 0; i < a_sh_wr_iters; i++) {  // loop
        cp_async4_pred(  // cp.async
          &sh_a_stage[a_sh_wr_trans[i]],  // smem dst (swizzled)
          &A[a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * a_off],  // gmem src
          a_sh_wr_pred[i]  // in-bounds pred
        );  // end cp_async4_pred
      }  // scope
      int4* sh_b_stage = sh_b + b_sh_stage * pipe;  // B stage base in smem
      #pragma unroll  // unroll B cp.async
      for (int i = 0; i < b_sh_wr_iters; i++) {  // loop
        cp_async4_stream(&sh_b_stage[b_sh_wr_delta * i + b_sh_wr], B_ptr[i]);  // cp.async
        B_ptr[i] += b_gl_rd_delta_o;  // advance B gmem ptr by one k-tile
      }  // scope
      // Fetch scales at group boundaries
      if (group_blocks != -1 && pipe % (group_blocks / thread_k_blocks) == 0) {  // branch
        int4* sh_s_stage = sh_s + s_sh_stage * pipe;  // S stage base in smem
        if (s_sh_wr_pred)  // branch
          cp_async4_stream(&sh_s_stage[s_sh_wr], &s[s_gl_rd]);  // cp.async
        s_gl_rd += s_gl_rd_delta;  // scale gmem read index (int4)
      }  // scope
    }  // scope
    cp_async_fence();  // cp.async
  };  // end fetch_to_shared

  // Wait for stage to complete
  auto wait_for_stage = [&] () {  // wait until the needed cp.async group is visible in smem
    cp_async_wait<stages - 2>();  // cp.async
    __syncthreads();  // make stage ready for ldmatrix/loads
  };  // end wait_for_stage

  // Fetch from shared to registers
  auto fetch_to_registers = [&] (int k, int pipe) {  // load A/B(/S) for k-subtile into regs
    if (group_blocks != -1) {  // branch
      int4* sh_s_stage = sh_s + s_sh_stage * ((group_blocks / thread_k_blocks) * (pipe / (group_blocks / thread_k_blocks)));  // group-aligned S stage
      reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_stage[s_sh_rd];  // load 2x half2 scales
    }  // scope
    int4* sh_a_stage = sh_a + a_sh_stage * pipe;  // A stage base in smem
    #pragma unroll  // unroll ldmatrix
    for (int i = 0; i < thread_m_blocks; i++)  // loop
      ldsm4(frag_a[k % 2][i], &sh_a_stage[a_sh_rd_trans[k % b_sh_wr_iters][i]]);  // ldmatrix
    int4* sh_b_stage = sh_b + b_sh_stage * pipe;  // B stage base in smem
    frag_b_quant[k % 2] = *reinterpret_cast<I4*>(&sh_b_stage[b_sh_rd_delta * (k % b_sh_wr_iters) + b_sh_rd]);  // load packed INT4 weights
  };  // end fetch_to_registers

  // Execute matmul
  auto matmul = [&] (int k) {  // dequant + optional scale + mma for one k-subtile
    #pragma unroll  // unroll N subtiles
    for (int j = 0; j < 4; j++) {  // loop
      int b_quant = frag_b_quant[k % 2][j];  // 2x packed INT4 (low 8 nibbles)
      int b_quant_shift = b_quant >> 8;  // 2x packed INT4 (high 8 nibbles)

      FragB frag_b0 = dequant(b_quant);  // dequant low half
      if (group_blocks != -1)  // branch
        scale(frag_b0, frag_s[k % 2][j], 0);  // apply scale[0]

      FragB frag_b1 = dequant(b_quant_shift);  // dequant high half
      if (group_blocks != -1)  // branch
        scale(frag_b1, frag_s[k % 2][j], 1);  // apply scale[1]

      #pragma unroll  // unroll M blocks
      for (int i = 0; i < thread_m_blocks; i++) {  // loop
        mma(frag_a[k % 2][i], frag_b0, frag_c[i][j][0]);  // mma
        mma(frag_a[k % 2][i], frag_b1, frag_c[i][j][1]);  // mma
      }  // scope
    }  // scope
  };  // end matmul

  // Threadblock reduction
  auto thread_block_reduce = [&] () {  // stmt
    constexpr int red_off = threads / b_sh_stride / 2;  // init
    if (red_off >= 1) {  // branch
      int red_idx = threadIdx.x / b_sh_stride;  // init
      constexpr int red_sh_stride = b_sh_stride * 4 * 2;  // init
      constexpr int red_sh_delta = b_sh_stride;  // init
      int red_sh_rd = red_sh_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);  // init

      #pragma unroll  // stmt
      for (int m_block = 0; m_block < thread_m_blocks; m_block++) {  // loop
        #pragma unroll  // stmt
        for (int i = red_off; i > 0; i /= 2) {  // loop
          if (i <= red_idx && red_idx < 2 * i) {  // branch
            #pragma unroll  // stmt
            for (int j = 0; j < 4 * 2; j++) {  // loop
              int red_sh_wr = red_sh_delta * j + (red_sh_rd - red_sh_stride * i);  // init
              if (i < red_off) {  // branch
                float* c_rd = reinterpret_cast<float*>(&sh[red_sh_delta * j + red_sh_rd]);  // init
                float* c_wr = reinterpret_cast<float*>(&sh[red_sh_wr]);  // init
                #pragma unroll  // stmt
                for (int k = 0; k < 4; k++)  // loop
                  reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + j][k] += c_rd[k] + c_wr[k];  // init
              }  // scope
              sh[red_sh_wr] = reinterpret_cast<int4*>(&frag_c)[4 * 2 * m_block + j];  // init
            }  // scope
          }  // scope
          __syncthreads();  // stmt
        }  // scope
        if (red_idx == 0) {  // branch
          #pragma unroll  // stmt
          for (int i = 0; i < 4 * 2; i++) {  // loop
            float* c_rd = reinterpret_cast<float*>(&sh[red_sh_delta * i + red_sh_rd]);  // init
            #pragma unroll  // stmt
            for (int j = 0; j < 4; j++)  // loop
              reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + i][j] += c_rd[j];  // init
          }  // scope
        }  // scope
        __syncthreads();  // stmt
      }  // scope
    }  // scope
  };  // scope

  // Global reduction
  auto global_reduce = [&] (bool first = false, bool last = false) {  // stmt
    constexpr int active_threads = 32 * thread_n_blocks / 4;  // init
    if (threadIdx.x < active_threads) {  // branch
      int c_gl_stride = prob_n / 8;  // init
      int c_gl_wr_delta_o = 8 * c_gl_stride;  // init
      int c_gl_wr_delta_i = 4 * (active_threads / 32);  // init
      int c_gl_wr = c_gl_stride * ((threadIdx.x % 32) / 4) + 4 * (threadIdx.x / 32) + threadIdx.x % 4;  // init
      c_gl_wr += (2 * thread_n_blocks) * slice_col;  // init
      constexpr int c_sh_wr_delta = active_threads;  // init
      int c_sh_wr = threadIdx.x;  // init

      int row = (threadIdx.x % 32) / 4;  // init

      if (!first) {  // branch
        #pragma unroll  // stmt
        for (int i = 0; i < thread_m_blocks * 4; i++) {  // loop
          cp_async4_pred(  // cp.async
            &sh[c_sh_wr + c_sh_wr_delta * i],  // stmt
            &C[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2)],  // stmt
            i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m  // stmt
          );  // scope
        }  // scope
        cp_async_fence();  // cp.async
        cp_async_wait<0>();  // cp.async
      }  // scope

      #pragma unroll  // stmt
      for (int i = 0; i < thread_m_blocks * 4; i++) {  // loop
        if (i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m) {  // branch
          if (!first) {  // branch
            int4 c_red = sh[c_sh_wr + i * c_sh_wr_delta];  // init
            #pragma unroll  // stmt
            for (int j = 0; j < 2 * 4; j++) {  // loop
              reinterpret_cast<float*>(&frag_c)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)] += __half2float(  // stmt
                reinterpret_cast<__half*>(&c_red)[j]  // stmt
              );  // scope
            }  // scope
          }  // scope
          if (!last) {  // branch
            int4 c;  // stmt
            #pragma unroll  // stmt
            for (int j = 0; j < 2 * 4; j++) {  // loop
              reinterpret_cast<__half*>(&c)[j] = __float2half(  // stmt
                reinterpret_cast<float*>(&frag_c)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)]  // stmt
              );  // scope
            }  // scope
            C[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2)] = c;  // init
          }  // scope
        }  // scope
      }  // scope
    }  // scope
  };  // scope

  // Write final result
  auto write_result = [&] () {  // stmt
    int c_gl_stride = prob_n / 8;  // init
    constexpr int c_sh_stride = 2 * thread_n_blocks + 1;  // init
    int c_gl_wr_delta = c_gl_stride * (threads / (2 * thread_n_blocks));  // init
    constexpr int c_sh_rd_delta = c_sh_stride * (threads / (2 * thread_n_blocks));  // init

    int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));  // init
    c_gl_wr += (2 * thread_n_blocks) * slice_col;  // init
    int c_sh_wr = (4 * c_sh_stride) * ((threadIdx.x % 32) / 4) + (threadIdx.x % 32) % 4;  // init
    c_sh_wr += 32 * (threadIdx.x / 32);  // init
    int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));  // init

    int c_gl_wr_end = c_gl_stride * prob_m;  // init

    auto write = [&] (int idx, float c0, float c1, FragS& s) {  // stmt
      half2 res = __halves2half2(__float2half(c0), __float2half(c1));  // init
      if (group_blocks == -1)  // branch
        res = __hmul2(res, s[0]);  // init
      ((half2*) sh)[idx] = res;  // init
    };  // scope

    if (threadIdx.x / 32 < thread_n_blocks / 4) {  // branch
      #pragma unroll  // stmt
      for (int i = 0; i < thread_m_blocks; i++) {  // loop
        #pragma unroll  // stmt
        for (int j = 0; j < 4; j++) {  // loop
          int wr = c_sh_wr + 8 * j;  // init
          write(wr + (4 * c_sh_stride) * 0 + 0, frag_c[i][j][0][0], frag_c[i][j][0][1], frag_s[j / 2][2 * (j % 2) + 0]);  // stmt
          write(wr + (4 * c_sh_stride) * 8 + 0, frag_c[i][j][0][2], frag_c[i][j][0][3], frag_s[j / 2][2 * (j % 2) + 0]);  // stmt
          write(wr + (4 * c_sh_stride) * 0 + 4, frag_c[i][j][1][0], frag_c[i][j][1][1], frag_s[j / 2][2 * (j % 2) + 1]);  // stmt
          write(wr + (4 * c_sh_stride) * 8 + 4, frag_c[i][j][1][2], frag_c[i][j][1][3], frag_s[j / 2][2 * (j % 2) + 1]);  // stmt
        }  // scope
        c_sh_wr += 16 * (4 * c_sh_stride);  // init
      }  // scope
    }  // scope
    __syncthreads();  // stmt

    #pragma unroll  // stmt
    for (int i = 0; i < ceildiv(16 * thread_m_blocks, threads / (2 * thread_n_blocks)); i++) {  // loop
      if (c_gl_wr < c_gl_wr_end) {  // branch
        C[c_gl_wr] = sh[c_sh_rd];  // init
        c_gl_wr += c_gl_wr_delta;  // init
        c_sh_rd += c_sh_rd_delta;  // init
      }  // scope
    }  // scope
  };  // scope

  // ========================================================================
  // Pipeline Initialization and Main Loop
  // ========================================================================

  // Start pipelines
  auto start_pipes = [&] () {  // stmt
    #pragma unroll  // stmt
    for (int i = 0; i < stages - 1; i++)  // loop
      fetch_to_shared(i, i, i < slice_iters);  // stmt
    zero_accums();  // stmt
    wait_for_stage();  // stmt
    fetch_to_registers(0, 0);  // stmt
    a_gl_rd += a_gl_rd_delta_o * (stages - 1);  // A gmem read index (int4)
  };  // scope
  start_pipes();  // stmt

  // Main loop
  while (slice_iters) {  // loop
    #pragma unroll  // stmt
    for (int pipe = 0; pipe < stages;) {  // loop
      #pragma unroll  // stmt
      for (int k = 0; k < b_sh_wr_iters; k++) {  // loop
        fetch_to_registers(k + 1, pipe % stages);  // stmt
        if (k == b_sh_wr_iters - 2) {  // branch
          fetch_to_shared((pipe + stages - 1) % stages, pipe, slice_iters >= stages);  // stmt
          pipe++;  // stmt
          wait_for_stage();  // stmt
        }  // scope
        matmul(k);  // stmt
      }  // scope
      slice_iters--;  // stmt
      if (slice_iters == 0)  // branch
        break;  // break
    }  // scope
    a_gl_rd += a_gl_rd_delta_o * stages;  // A gmem read index (int4)

    if (slice_iters == 0) {  // branch
      cp_async_wait<0>();  // cp.async
      bool last = slice_idx == slice_count - 1;  // stmt

      if (group_blocks == -1 && last) {  // branch
        if (s_sh_wr_pred)  // branch
          cp_async4_stream(&sh_s[s_sh_wr], &s[s_gl_rd]);  // cp.async
        cp_async_fence();  // cp.async
      }  // scope

      thread_block_reduce();  // stmt

      if (group_blocks == -1 && last) {  // branch
        cp_async_wait<0>();  // cp.async
        __syncthreads();  // stmt
        if (threadIdx.x / 32 < thread_n_blocks / 4) {  // branch
          reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd + 0];  // scale fragments (regs)
          reinterpret_cast<int4*>(&frag_s)[1] = sh_s[s_sh_rd + 4];  // scale fragments (regs)
        }  // scope
      }  // scope

      if (slice_count > 1) {  // branch
        barrier_acquire(&locks[slice_col], slice_idx);  // stmt
        global_reduce(slice_idx == 0, last);  // stmt
        barrier_release(&locks[slice_col], last);  // stmt
      }  // scope

      if (last)  // branch
        write_result();  // stmt

      slice_row = 0;  // tile row in K grid (idx%k_tiles)
      slice_col_par++;  // stmt
      slice_col++;  // stmt
      init_slice();  // stmt

      if (slice_iters) {  // branch
        a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);  // A gmem read index (int4)
        #pragma unroll  // stmt
        for (int i = 0; i < b_sh_wr_iters; i++)  // loop
          B_ptr[i] += b_sh_stride - b_gl_rd_delta_o * k_tiles;  // init
        if (slice_col == 0) {  // branch
          #pragma unroll  // stmt
          for (int i = 0; i < b_sh_wr_iters; i++)  // loop
            B_ptr[i] -= b_gl_stride;  // init
        }  // scope
        s_gl_rd = s_sh_stride * slice_col + threadIdx.x;  // scale gmem read index (int4)
        start_pipes();  // stmt
      }  // scope
    }  // scope
  }  // scope
}  // scope

// ============================================================================
// Kernel Launch Configuration
// ============================================================================

const int THREADS = 256;  // threads per CTA
const int STAGES = 4;  // cp.async pipeline stages
const int SHARED_MEM = 96 * 1024;  // 96 KB shared memory

#define CALL_IF(THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, GROUP_BLOCKS) /* macro */ \
  else if ( /* branch */ \
    thread_m_blocks == THREAD_M_BLOCKS && thread_n_blocks == THREAD_N_BLOCKS && thread_k_blocks == THREAD_K_BLOCKS && /* stmt */ \
    group_blocks == GROUP_BLOCKS /* stmt */ \
  ) { /* stmt */ \
    cudaFuncSetAttribute( /* CUDA runtime */ \
      Marlin<THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, STAGES, GROUP_BLOCKS>, /* stmt */ \
      cudaFuncAttributeMaxDynamicSharedMemorySize, /* CUDA runtime */ \
      SHARED_MEM /* stmt */ \
    ); /* stmt */ \
    Marlin< /* stmt */ \
      THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, STAGES, GROUP_BLOCKS /* stmt */ \
    ><<<blocks, THREADS, SHARED_MEM, stream>>>( /* stmt */ \
      A_ptr, B_ptr, C_ptr, s_ptr, /* stmt */ \
      prob_m, prob_n, prob_k, /* stmt */ \
      locks /* stmt */ \
    ); /* stmt */ \
  }  // scope

const int ERR_PROB_SHAPE = 1;  // init
const int ERR_KERN_SHAPE = 2;  // init

int marlin_cuda(  // CUDA runtime
  const void* A,  // stmt
  const void* B,  // stmt
        void* C,  // stmt
        void* s,  // stmt
  int prob_m,  // stmt
  int prob_n,  // stmt
  int prob_k,  // stmt
  void* workspace,  // stmt
  int groupsize = -1,  // stmt
  int dev = 0,  // stmt
  cudaStream_t stream = 0,  // CUDA runtime
  int thread_k = -1,  // stmt
  int thread_n = -1,  // stmt
  int sms = -1,  // stmt
  int max_par = 16  // stmt
) {  // stmt
  int tot_m = prob_m;  // init
  int tot_m_blocks = ceildiv(tot_m, 16);  // init
  int pad = 16 * tot_m_blocks - tot_m;  // init

  if (sms == -1)  // branch
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);  // CUDA runtime
  if (thread_k == -1 || thread_n == -1) {  // branch
    if (prob_m <= 16) {  // branch
      thread_k = 128;  // init
      thread_n = 128;  // init
    } else {  // stmt
      thread_k = 64;  // init
      thread_n = 256;  // init
    }  // scope
  }  // scope

  int thread_k_blocks = thread_k / 16;  // init
  int thread_n_blocks = thread_n / 16;  // init
  int group_blocks = (groupsize == -1) ? -1 : groupsize / 16;  // stmt
  int blocks = sms;  // init

  if (prob_n % thread_n != 0 || prob_k % thread_k != 0 || (group_blocks != -1 && prob_k % group_blocks != 0))  // branch
    return ERR_PROB_SHAPE;  // return
  if (prob_m == 0 || prob_n == 0 || prob_k == 0)  // branch
    return 0;  // return

  const int4* A_ptr = (const int4*) A;  // init
  const int4* B_ptr = (const int4*) B;  // init
  int4* C_ptr = (int4*) C;  // init
  const int4* s_ptr = (const int4*) s;  // init

  int* locks = (int*) workspace;  // init

  int ret = 0;  // init
  for (int i = 0; i < tot_m_blocks; i += 4) {  // loop
    int thread_m_blocks = tot_m_blocks - i;  // init
    prob_m = tot_m - 16 * i;  // init
    int par = 1;  // init
    if (thread_m_blocks > 4) {  // branch
      par = (16 * thread_m_blocks - pad) / 64;  // init
      if (par > max_par)  // branch
        par = max_par;  // init
      prob_m = 64 * par;  // init
      i += 4 * (par - 1);  // init
      thread_m_blocks = 4;  // init
    }  // scope

    if (false) {}  // branch
    CALL_IF(1,  8,  8, -1)  // stmt
    CALL_IF(1,  8,  8,  8)  // stmt
    CALL_IF(1, 16,  4, -1)  // stmt
    CALL_IF(1, 16,  4,  8)  // stmt
    CALL_IF(2, 16,  4, -1)  // stmt
    CALL_IF(2, 16,  4,  8)  // stmt
    CALL_IF(3, 16,  4, -1)  // stmt
    CALL_IF(3, 16,  4,  8)  // stmt
    CALL_IF(4, 16,  4, -1)  // stmt
    CALL_IF(4, 16,  4,  8)  // stmt
    else  // branch
      ret = ERR_KERN_SHAPE;  // init

    A_ptr += 16 * thread_m_blocks * (prob_k / 8) * par;  // init
    C_ptr += 16 * thread_m_blocks * (prob_n / 8) * par;  // init
  }  // scope

  return ret;  // return
}  // scope

void print_usage(const char* prog_name) {  // stmt
  printf("Usage: %s [OPTIONS]\n", prog_name);  // stmt
  printf("\nOptions:\n");  // stmt
  printf("  -m <M>          Batch size (default: 128)\n");  // stmt
  printf("  -n <N>          Output dimension (default: 256)\n");  // stmt
  printf("  -k <K>          Input dimension (default: 512)\n");  // stmt
  printf("  -g <groupsize>  Quantization group size, -1 for per-column (default: -1)\n");  // stmt
  printf("  -s <sms>        Number of SMs to use, -1 for auto (default: -1)\n");  // stmt
  printf("  -w <warmup>     Warmup iterations (default: 10)\n");  // stmt
  printf("  -i <iters>      Timed iterations (default: 100)\n");  // stmt
  printf("  --ncu           Nsight Compute mode: warmup=0 and default iters=1000\n");  // warmup iters
  printf("  --trace         Print a small device-side trace (printf) for one block/thread (implies warmup=0 iters=1)\n");  // warmup iters
  printf("  --trace_block B Select which CTA (blockIdx.x) to trace (default: 0)\n");  // stmt
  printf("  --trace_thread T Select which threadIdx.x to trace (default: 0)\n");  // stmt
  printf("  -h              Show this help message\n");  // stmt
  printf("\nExample:\n");  // stmt
  printf("  %s -m 128 -n 4096 -k 4096 -g 128 -s 108\n", prog_name);  // stmt
}  // scope

int main(int argc, char* argv[]) {  // stmt
  // Default parameters
  int M = 128;           // Batch size
  int N = 256;           // Output dimension
  int K = 512;           // Input dimension
  int groupsize = -1;    // -1 for per-column quantization
  int num_sms = -1;      // -1 for auto-detect
  int warmup = 10;  // warmup iters
  int iters = 100;  // timed iters
  bool iters_set = false;  // init
  bool ncu_mode = false;  // init
  bool trace = false;  // enable device printf trace
  int trace_block = 0;  // CTA to trace
  int trace_thread = 0;  // thread to trace

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {  // loop
    if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {  // branch
      M = atoi(argv[++i]);  // GEMM M
    } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {  // stmt
      N = atoi(argv[++i]);  // GEMM N
    } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {  // stmt
      K = atoi(argv[++i]);  // GEMM K
    } else if (strcmp(argv[i], "-g") == 0 && i + 1 < argc) {  // stmt
      groupsize = atoi(argv[++i]);  // quant group size
    } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {  // stmt
      num_sms = atoi(argv[++i]);  // SM count used for launch
    } else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {  // stmt
      warmup = atoi(argv[++i]);  // warmup iters
    } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {  // stmt
      iters = atoi(argv[++i]);  // timed iters
      iters_set = true;  // init
    } else if (strcmp(argv[i], "--ncu") == 0) {  // stmt
      ncu_mode = true;  // init
    } else if (strcmp(argv[i], "--trace") == 0) {  // stmt
      trace = true;  // enable device printf trace
    } else if (strcmp(argv[i], "--trace_block") == 0 && i + 1 < argc) {  // stmt
      trace_block = atoi(argv[++i]);  // CTA to trace
    } else if (strcmp(argv[i], "--trace_thread") == 0 && i + 1 < argc) {  // stmt
      trace_thread = atoi(argv[++i]);  // thread to trace
    } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {  // stmt
      print_usage(argv[0]);  // stmt
      return 0;  // return
    } else {  // stmt
      printf("Unknown option: %s\n", argv[i]);  // stmt
      print_usage(argv[0]);  // stmt
      return 1;  // return
    }  // scope
  }  // scope

  if (ncu_mode) {  // branch
    // No separate warmup loop; use ncu --launch-skip/--launch-count to pick a single iteration.
    warmup = 0;  // warmup iters
    if (!iters_set) {  // branch
      iters = 1000;  // timed iters
    }  // scope
  }  // scope
  if (trace) {  // branch
    // Device printf is very slow; keep this to a single launch by default.
    warmup = 0;  // warmup iters
    iters = 1;  // timed iters
    g_marlin_trace.enabled = 1;  // init
    g_marlin_trace.block = trace_block;  // init
    g_marlin_trace.thread = trace_thread;  // init
  } else {  // stmt
    g_marlin_trace.enabled = 0;  // init
  }  // scope

  // Validate parameters
  if (M <= 0 || N <= 0 || K <= 0) {  // branch
    printf("Error: M, N, K must be positive integers\n");  // stmt
    return 1;  // return
  }  // scope
  if (warmup < 0 || iters <= 0) {  // branch
    printf("Error: warmup must be >= 0 and iters must be > 0\n");  // stmt
    return 1;  // return
  }  // scope

  // Print configuration
  printf("========================================\n");  // stmt
  printf("Marlin INT4 GEMM Configuration\n");  // stmt
  printf("========================================\n");  // stmt
  printf("Matrix dimensions:\n");  // stmt
  printf("  M (batch size):      %d\n", M);  // stmt
  printf("  N (output dim):      %d\n", N);  // stmt
  printf("  K (input dim):       %d\n", K);  // stmt
  if (groupsize == -1) {  // branch
    printf("  Group size:          per-column\n");  // stmt
  } else {  // stmt
    printf("  Group size:          %d\n", groupsize);  // stmt
  }  // scope
  if (num_sms == -1) {  // branch
  printf("  Number of SMs:       auto-detect\n");  // stmt
  } else {  // stmt
    printf("  Number of SMs:       %d\n", num_sms);  // stmt
  }  // scope
  printf("  Warmup iterations:   %d\n", warmup);  // stmt
  printf("  Timed iterations:    %d\n", iters);  // stmt
  printf("========================================\n\n");  // stmt

  // Allocate device memory
  void *d_A, *d_B, *d_C, *d_s, *d_workspace;  // stmt
  size_t A_size = M * K * sizeof(half);  // A bytes
  size_t B_size = (K * N) / 2;  // 4-bit packed weights
  size_t C_size = M * N * sizeof(half);  // C bytes
  size_t s_size = (groupsize == -1) ? N * sizeof(half) : (K / groupsize) * N * sizeof(half);  // stmt
  size_t workspace_size = (N / 128) * 16 * sizeof(int);  // locks bytes

  printf("Allocating device memory...\n");  // stmt
  printf("  A matrix:      %.2f MB\n", A_size / (1024.0 * 1024.0));  // stmt
  printf("  B matrix:      %.2f MB\n", B_size / (1024.0 * 1024.0));  // stmt
  printf("  C matrix:      %.2f MB\n", C_size / (1024.0 * 1024.0));  // stmt
  printf("  Scales:        %.2f MB\n", s_size / (1024.0 * 1024.0));  // stmt
  printf("  Workspace:     %.2f KB\n\n", workspace_size / 1024.0);  // stmt

  cudaMalloc(&d_A, A_size);  // CUDA runtime
  cudaMalloc(&d_B, B_size);  // CUDA runtime
  cudaMalloc(&d_C, C_size);  // CUDA runtime
  cudaMalloc(&d_s, s_size);  // CUDA runtime
  cudaMalloc(&d_workspace, workspace_size);  // CUDA runtime

  // Initialize with random data
  cudaMemset(d_A, 0, A_size);  // CUDA runtime
  cudaMemset(d_B, 0, B_size);  // CUDA runtime
  cudaMemset(d_s, 1, s_size);  // CUDA runtime
  cudaMemset(d_workspace, 0, workspace_size);  // CUDA runtime

  printf("Running kernel...\n");  // stmt

  for (int i = 0; i < warmup; ++i) {  // loop
    marlin_cuda(  // CUDA runtime
      d_A, d_B, d_C, d_s,  // stmt
      M, N, K,  // stmt
      d_workspace,  // stmt
      groupsize,  // stmt
      0,           // device 0
      0,           // default stream
      -1,          // auto thread_k
      -1,          // auto thread_n
      num_sms,     // custom number of SMs
      16           // max_par
    );  // scope
    cudaError_t warmup_err = cudaGetLastError();  // CUDA runtime
    if (warmup_err != cudaSuccess) {  // branch
      printf("Warmup launch failed: %s\n", cudaGetErrorString(warmup_err));  // CUDA runtime
      return 1;  // return
    }  // scope
  }  // scope
  cudaDeviceSynchronize();  // CUDA runtime

  cudaEvent_t start{};  // CUDA runtime
  cudaEvent_t stop{};  // CUDA runtime
  cudaEventCreate(&start);  // CUDA runtime
  cudaEventCreate(&stop);  // CUDA runtime
  cudaEventRecord(start, 0);  // CUDA runtime
  int result = 0;  // init
  for (int i = 0; i < iters; ++i) {  // loop
    result = marlin_cuda(  // CUDA runtime
      d_A, d_B, d_C, d_s,  // stmt
      M, N, K,  // stmt
      d_workspace,  // stmt
      groupsize,  // stmt
      0,           // device 0
      0,           // default stream
      -1,          // auto thread_k
      -1,          // auto thread_n
      num_sms,     // custom number of SMs
      16           // max_par
    );  // scope
    if (result != 0) {  // branch
      break;  // break
    }  // scope
    cudaError_t launch_err = cudaGetLastError();  // CUDA runtime
    if (launch_err != cudaSuccess) {  // branch
      printf("Timed launch failed: %s\n", cudaGetErrorString(launch_err));  // CUDA runtime
      result = launch_err;  // init
      break;  // break
    }  // scope
  }  // scope
  cudaEventRecord(stop, 0);  // CUDA runtime
  cudaEventSynchronize(stop);  // CUDA runtime
  float ms = 0.0f;  // init
  cudaEventElapsedTime(&ms, start, stop);  // CUDA runtime
  cudaEventDestroy(start);  // CUDA runtime
  cudaEventDestroy(stop);  // CUDA runtime

  if (result == 0) {  // branch
    printf("\n");  // stmt
    printf("========================================\n");  // stmt
    printf("SUCCESS!\n");  // stmt
    printf("========================================\n");  // stmt
    printf("Kernel executed successfully!\n");  // stmt
    printf("Matrix multiplication: (%d x %d) * (%d x %d) = (%d x %d)\n", M, K, K, N, M, N);  // init
    printf("Avg kernel time: %.3f us (%d iters, %d warmup)\n",  // stmt
           (ms * 1000.0f) / (float)iters, iters, warmup);  // stmt
    printf("========================================\n");  // stmt
  } else if (result == ERR_PROB_SHAPE) {  // stmt
    printf("\n");  // stmt
    printf("========================================\n");  // stmt
    printf("ERROR: Problem shape incompatible\n");  // stmt
    printf("========================================\n");  // stmt
    printf("The problem dimensions are not compatible with kernel constraints.\n");  // stmt
    printf("Ensure:\n");  // stmt
    printf("  - N is divisible by thread_n (128 or 256)\n");  // stmt
    printf("  - K is divisible by thread_k (64 or 128)\n");  // stmt
    printf("  - If using group quantization, K is divisible by groupsize\n");  // stmt
    printf("========================================\n");  // stmt
  } else if (result == ERR_KERN_SHAPE) {  // stmt
    printf("\n");  // stmt
    printf("========================================\n");  // stmt
    printf("ERROR: No kernel implementation\n");  // stmt
    printf("========================================\n");  // stmt
    printf("No kernel implementation available for these parameters.\n");  // stmt
    printf("Try different M, N, K, or groupsize values.\n");  // stmt
    printf("========================================\n");  // stmt
  }  // scope

  // Cleanup
  cudaFree(d_A);  // CUDA runtime
  cudaFree(d_B);  // CUDA runtime
  cudaFree(d_C);  // CUDA runtime
  cudaFree(d_s);  // CUDA runtime
  cudaFree(d_workspace);  // CUDA runtime

  return result;  // return
}  // scope
