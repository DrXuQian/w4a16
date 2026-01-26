# Marlin kernel (FP16 x INT4) 逐行理解笔记: M=1, N=2048, K=2048, group=128

本文目标: 以 **M=1, N=2048, K=2048, groupsize=128 (GPTQ/AWQ 常见)** 为例, 帮你读懂 Marlin 的 CUDA kernel, 让你能改.

建议你一边打开源码一边看:
- 核心 kernel: `homemade_marlin/marlin/marlin_cuda_kernel.cu`
- 可直接跑的 standalone(我加了 `--trace` 打印): `homemade_marlin/marlin_standalone.cu`

> 说明: 下面很多变量名/公式与 `marlin_cuda_kernel.cu` 完全一致, 你可以用它当作“变量字典”.

## 代码导航(带行号, 以 `marlin_cuda_kernel.cu` 为准)

把这几段当作“目录”去跳转看源码:

- L54: `cp_async4_pred` (A 的 predicated cp.async)
- L69: `cp_async4_stream` (B 的 cp.async + L2 evict hint)
- L93: `mma` (m16n8k16)
- L107: `mma_m16n16k16` (用两个 m16n8k16 拼成 n=16)
- L134: `ldsm4` (ldmatrix.x4, 直接读 A fragment)
- L158: `dequant` (int4 -> fp16 fragment 的 bit trick)
- L189: `barrier_acquire/release` (global reduction 的锁)
- L224: `__global__ void Marlin<...>()` (主 kernel)
  - L274: `init_slice` (stripe 切分/同步参数)
  - L300-380: A/B/s 的 stride/index/predication 计算
  - L414: `fetch_to_shared` (cp.async pipeline)
  - L452: `fetch_to_registers` (shared -> regs)
  - L469: `matmul` (dequant + mma)
  - L493: `thread_block_reduce`
  - L541: `global_reduce`
  - L600: `write_result`
- L758: `int marlin_cuda(...)` (host 侧 config 选择 + kernel launch)

---

## 0. 这个 case 会走哪个 kernel 配置?

入口在 `marlin_cuda(...)`(同文件末尾). 对于 `prob_m <= 16` 的小 batch, Marlin 选择:

- `thread_k = 128`
- `thread_n = 128`

因此:

- `thread_k_blocks = thread_k / 16 = 8`
- `thread_n_blocks = thread_n / 16 = 8`
- groupsize=128 -> `group_blocks = groupsize / 16 = 8`
- M=1 -> `tot_m_blocks = ceildiv(1,16)=1` -> `thread_m_blocks=1`

最终会命中这一项:

- `CALL_IF(1, 8, 8, 8)` -> `Marlin<threads=256, thread_m_blocks=1, thread_n_blocks=8, thread_k_blocks=8, stages=4, group_blocks=8>`

你可以把它记成一个“CTA tile”:

- CTA-M: `16 * thread_m_blocks = 16`
- CTA-N: `16 * thread_n_blocks = 128`
- CTA-K: `16 * thread_k_blocks = 128`

即每个 threadblock(CTA) 负责: `C[16x128] += A[16x128] * B[128x128]` 的一部分.

---

## 1. 数据类型与打包(先把“单位”统一)

### 1.1 代码里用的 `int4` 不是 int4 weight, 是 16-byte vector

在 CUDA 里:
- `int4` 是 `int` 的 vector: `{int x,y,z,w}` -> **16 bytes**

Marlin 把 A/B/C/s 都强行当作 `int4*` 来做 16-byte 的向量化 load/store.

直觉上你可以记:

- **一个 `int4` = 16B**
- 对于 FP16: 16B = 8 个 half
- 对于 INT4: 16B = 32 个 int4 weight (因为 1 byte=2个int4)

ASCII:

```
int4 (16B) 作为 FP16 视角:
  [half0 half1 half2 half3 half4 half5 half6 half7]

int4 (16B) 作为 INT4 视角:
  [w0 w1 w2 ... w31]   (每个 w 是 4-bit)
```

### 1.2 关键结构体 Vec/Frag

`marlin_cuda_kernel.cu` 里有:

- `Vec<T,n>`: 只是“寄存器数组”容器, 用来保证编译期常量索引(配合 `#pragma unroll`)
- `FragA = Vec<half2,4>`: A fragment, 供 ldmatrix + mma 使用
- `FragB = Vec<half2,2>`: B fragment, dequant 后喂给 mma
- `FragC = Vec<float,4>`: 累加器 fragment (fp32 accumulate)
- `FragS = Vec<half2,1>`: scale fragment (group quant)

---

## 2. Tile 切分: 2048x2048 被拆成什么网格?

对我们的 case:

- `k_tiles = prob_k / 16 / thread_k_blocks = 2048 / 16 / 8 = 16`
- `n_tiles = prob_n / 16 / thread_n_blocks = 2048 / 16 / 8 = 16`

也就是说整张 GEMM 被拆成一个 `K-tile x N-tile` 的 16x16 网格(每格=一个 128x128 的 B tile).

```
         N tiles (each 128 columns)
        0   1   2  ...  15
K  0   [ ] [ ] [ ]     [ ]
t  1   [ ] [ ] [ ]     [ ]
i  2   [ ] [ ] [ ]     [ ]
l  ...                  ...
e 15   [ ] [ ] [ ]     [ ]
s

每个格子 = 一个 (slice_row, slice_col) 组合:
  slice_row = K tile index (0..15)
  slice_col = N tile index (0..15)
```

注意: M=1 但 kernel 的基本 tile 仍是 16 行, 多出来的 15 行靠 predication 不读不写.

---

## 2.1 这张卡(48 SM)上, 每个 block(你可以近似当成每个 SM)干了哪些 tile?

重要前提:

- 这个 kernel 启动时 `blocks = sms` (host 侧 `marlin_cuda(...)` 里这么写的)
- 在 RTX 5070 上查询到 `sms=48`
- 一次 launch 的 `gridDim.x = 48`
- 总 tiles = `k_tiles * n_tiles = 16 * 16 = 256`
- `iters = ceildiv(256, 48) = 6`

调度方式(可把每个 block 视为领取一个连续区间):

```
block b 负责 tile_linear_idx in [6*b, 6*(b+1))
tile_linear_idx -> (slice_col, slice_row):
  slice_col = idx / 16
  slice_row = idx % 16
```

所以每个 block 的任务如下(只列出有工作的 block0..block42; block43..47 会是 inactive):

```
block0:  col0 rows0-5
block1:  col0 rows6-11
block2:  col0 rows12-15; col1 rows0-1
block3:  col1 rows2-7
block4:  col1 rows8-13
block5:  col1 rows14-15; col2 rows0-3
block6:  col2 rows4-9
block7:  col2 rows10-15
block8:  col3 rows0-5
block9:  col3 rows6-11
block10: col3 rows12-15; col4 rows0-1
block11: col4 rows2-7
block12: col4 rows8-13
block13: col4 rows14-15; col5 rows0-3
block14: col5 rows4-9
block15: col5 rows10-15
block16: col6 rows0-5
block17: col6 rows6-11
block18: col6 rows12-15; col7 rows0-1
block19: col7 rows2-7
block20: col7 rows8-13
block21: col7 rows14-15; col8 rows0-3
block22: col8 rows4-9
block23: col8 rows10-15
block24: col9 rows0-5
block25: col9 rows6-11
block26: col9 rows12-15; col10 rows0-1
block27: col10 rows2-7
block28: col10 rows8-13
block29: col10 rows14-15; col11 rows0-3
block30: col11 rows4-9
block31: col11 rows10-15
block32: col12 rows0-5
block33: col12 rows6-11
block34: col12 rows12-15; col13 rows0-1
block35: col13 rows2-7
block36: col13 rows8-13
block37: col13 rows14-15; col14 rows0-3
block38: col14 rows4-9
block39: col14 rows10-15
block40: col15 rows0-5
block41: col15 rows6-11
block42: col15 rows12-15
```

读法:

- `colX` 对应输出 `C` 的列范围: `[128*X, 128*X + 127]`
- `rowsR0-R1` 对应 `K` 的范围(每个 row 是 128 个 K):
  - row r -> K range `[128*r, 128*r+127]`
  - rows r0-r1 -> K range `[128*r0, 128*(r1+1)-1]`

---

## 2.2 对单个 tile(一个 col, 一个 row), 它到底算了多少? 对应哪个输出?

对任意 tile `(slice_col=c, slice_row=r)`:

- 逻辑上算:
  - `C_tile[16x128] += A_tile[16x128] * B_tile[128x128]` (K=128)
- 这对应的输出区域(全局, row-major):
  - 输出行: `m in [0..15]` (但 M=1 时只有 m=0 有效)
  - 输出列: `n in [128*c .. 128*c+127]`
- 这个 tile 的 MAC 数:
  - `16 * 128 * 128 = 262,144` MAC (kernel 仍然会做满, 只是多余行不写回)

---

## 2.3 (重点) 这个 tile 的 A/B/s/C 在 gmem 上的“地址”(用 index/byte offset 表示)

下面全部用 kernel 里实际访问的单位: `int4` (16 bytes) 来描述.
你可以把它转换成 byte offset: `byte_offset = int4_index * 16`.

### A (FP16 activations), 仅 row0 有效

kernel 里 `A` 是 `const int4*` (把 `half*` cast 成 `int4*`), 因此:

- A 的 row stride (int4) = `a_gl_stride = K/8 = 2048/8 = 256`
- 一个 K tile(r) 覆盖 128 个 half = 16 个 int4

对于 tile row=r, row0 的 A 数据是:

```
A int4 indices: [r*16 + 0 ... r*16 + 15]
byte offsets:   16*(r*16 + t), t=0..15
half columns:   [128*r + 8*t ... 128*r + 8*t + 7]
```

并且 M=1 时, 只有 `threadIdx.x in [0..15]` 会真的去 cp.async A:

```
tid=t (0..15) loads A[r*16 + t] (int4, 16 bytes) into shared
```

### s (scales, group=128)

scale 矩阵逻辑 shape = `(K/groupsize, N)` = `(16, 2048)` (FP16).
在 kernel 里同样 cast 成 `int4*`:

- s 的 row stride (int4) = `s_gl_stride = N/8 = 2048/8 = 256`
- 一个 N tile(c) 覆盖 128 个 half = 16 个 int4

tile (c,r) 对应的 scale 行就是 group 行 r:

```
s int4 indices: [r*256 + c*16 + 0 ... r*256 + c*16 + 15]
```

同样只有 `tid in [0..15]` 会写 s 到 shared.

### B (INT4 weights, offline 预排布后的 gmem 布局)

B 的 layout 是 Marlin 的关键, 它不是简单的 row-major int4 pack.
但你不用先理解 python packing 才能“跟踪地址”, 因为 kernel 的访问模式非常明确.

对 tile (c,r), kernel 每个线程会 load **两次** `int4` (因为 `b_sh_wr_iters=2`):

先定义几个常量(本 case):

```
b_gl_stride      = 16*N/32 = 1024           (int4)
b_gl_rd_delta_o  = b_gl_stride * thread_k_blocks = 1024*8 = 8192   (int4)
b_gl_rd_delta_i  = b_gl_stride * (threads/b_sh_stride) = 1024*4 = 4096 (int4)
b_sh_stride      = 32*thread_n_blocks/4 = 64
```

对任意线程 `tid=threadIdx.x`, 令:

```
k_group = tid / 64         (0..3)   // 4 组, 对应 CTA 内 split-K
lane    = tid % 64         (0..63)
```

那么 tile(c,r) 的 B 访问 int4 index 是:

```
base = 8192*r + 64*c + 1024*k_group + lane

load0: B[ base + 0   ]
load1: B[ base + 4096]    // 因为 b_gl_rd_delta_i=4096
```

所以整块 tile 一共读了:
- 256 threads * 2 loads * 16B = 8192B
- 对应 128x128 个 int4 weight (8192B 正好)

### C (FP16 output), tile 写回的位置

输出 C 也是 `half*` cast 成 `int4*`.

- C 的 row stride(int4) = `c_gl_stride = N/8 = 256`
- tile(c) 覆盖 128 列 = 16 个 int4

row0 的最终输出 tile 写回位置:

```
C int4 indices: [c*16 + 0 ... c*16 + 15]
byte offsets:   16*(c*16 + j) = 256*c + 16*j
```

注意: 因为 split-K, 多个 block 会先把 partial sum“累加”到同一个 C tile 上(在 L2 里做 global_reduce),
最终由该 slice 的最后一个 block 写出最终结果.

---

## 2.4 同一个 tile 在 shared memory 里的位置(精确到 int4 index/byte offset)

kernel 里 shared 是一个动态数组:

```cpp
extern __shared__ int4 sh[];
int4* sh_a = sh;
int4* sh_b = sh_a + (stages * a_sh_stage);
int4* sh_s = sh_b + (stages * b_sh_stage);
```

对本 case 的常量:

```
stages=4
a_sh_stage=256 int4  (=4096B)
b_sh_stage=512 int4  (=8192B)
s_sh_stage=16  int4  (=256B)
```

所以 shared 的分段布局是(单位: int4):

```
sh:
  A0 [0 .. 255]
  A1 [256 .. 511]
  A2 [512 .. 767]
  A3 [768 .. 1023]

  B0 [1024 .. 1535]
  B1 [1536 .. 2047]
  B2 [2048 .. 2559]
  B3 [2560 .. 3071]

  S0 [3072 .. 3087]
  S1 [3088 .. 3103]
  S2 [3104 .. 3119]
  S3 [3120 .. 3135]
```

换成 byte offset(每个 int4=16B):

```
A stage p base = 16 * (p*256)    = p*4096 bytes
B stage p base = 16 * (1024 + p*512) = 16384 + p*8192 bytes
S stage p base = 16 * (3072 + p*16)  = 49152 + p*256 bytes
```

### 写入 shared 时的 index(谁写了哪里)

当 `fetch_to_shared(pipe=p, a_off=...)` 被调用:

#### A stage p

代码(写 A)在:

```cpp
int4* sh_a_stage = sh_a + a_sh_stage * p;
cp_async4_pred(&sh_a_stage[a_sh_wr_trans[i]], &A[...], a_sh_wr_pred[i]);
```

对 M=1:
- `a_sh_wr_iters=1` 所以只有 `i=0`
- 只有 `tid=0..15` 满足 `a_sh_wr_pred[0]=true`
- 并且对 row0, XOR swizzle 退化为 identity: `a_sh_wr_trans[0] == a_sh_wr == tid`

因此:

```
tid=t (0..15): sh_a_stage[t] = A_tile_int4[t]   (16B each)
```

也就是 A stage p 的 `[0..15]` 这 16 个 int4 保存了这一个 tile 的 A(row0)数据.

#### B stage p

代码(写 B)在:

```cpp
int4* sh_b_stage = sh_b + b_sh_stage * p;
cp_async4_stream(&sh_b_stage[256*i + tid], B_ptr[i]);  // i=0..1
```

所以:

```
tid=0..255:
  sh_b_stage[tid]       <- B load #0 (k-subtile 0)
  sh_b_stage[256 + tid] <- B load #1 (k-subtile 1)
```

#### S stage p (group=128)

本 case `group_blocks==thread_k_blocks`, 所以每个 tile 都会加载 scales:

```cpp
int4* sh_s_stage = sh_s + s_sh_stage * p;
if (tid < 16) sh_s_stage[tid] <- s[...]
```

所以:

```
tid=t (0..15): sh_s_stage[t] = s_tile_int4[t]
```

### 从 shared 读回寄存器时的 index(谁读了哪里)

#### B: 每个 sub-k 对应 shared 的哪一半?

代码:

```cpp
frag_b_quant[k % 2] =
  *reinterpret_cast<I4*>(&sh_b_stage[256 * (k % 2) + tid]);
```

因此:

```
k=0 -> 读 sh_b_stage[tid]        (对应本 tile 的 K[0..63] 那一半)
k=1 -> 读 sh_b_stage[256 + tid]  (对应本 tile 的 K[64..127] 那一半)
```

#### A: `ldmatrix` 的读地址

代码:

```cpp
ldsm4(frag_a[k%2][0], &sh_a_stage[a_sh_rd_trans[k][0]]);
```

这里 `a_sh_rd_trans[...]` 是为了保证 shared bank conflict free 的 XOR layout.
对 M=1 你可以先把它理解为:

```
ldmatrix 从 sh_a_stage 的那 16 个 int4 里, 按 tensorcore 需要的 fragment layout 取数据
```

如果你想看到具体数值, 用我加的 `--trace` 先把 `a_sh_rd` / `a_sh_rd_trans` 打出来再对照.

---

## 3. 为什么需要 global reduction? (Split-K across CTAs)

如果 M 很小(比如 1), 只按 N tile 切分, 你只有 16 个 CTA 可用(每个 CTA 做一个 16x128 输出 tile).
但 GPU 可能有几十/上百个 SM, 16 个 CTA 不够填满.

Marlin 的做法: **把 K tiles 也分给不同 CTA**(相当于 split-K), 让更多 CTA 同时工作.
多个 CTA 会对同一 `(slice_col)` 的输出 tile 做部分 K 段累加, 最后需要归约:

- CTA0: 负责 K tiles [0..x] 的 partial sum
- CTA1: 负责 K tiles [x..y] 的 partial sum
- ...
- 最后把 partial sums 在全局上累加得到最终 C

这就是代码里:
- `thread_block_reduce()` (CTA 内部的 reduction)
- `global_reduce()` + `barrier_acquire/release()` (CTA 之间的 reduction)

---

## 4. Stripe 调度: blockIdx.x 怎么映射到 (slice_row, slice_col)?

核心在 kernel 开头这几行(以 `marlin_cuda_kernel.cu` 为准, 大约 240+ 行开始):

- `iters = ceildiv(k_tiles * n_tiles * parallel, gridDim.x)`
- `slice_row = (iters * blockIdx.x) % k_tiles`
- `slice_col_par = (iters * blockIdx.x) / k_tiles`
- `slice_col = slice_col_par`
- `init_slice()` 会算出:
  - `slice_iters`: 这个 CTA 在当前 slice_col 上要做多少个连续的 K tiles
  - `slice_count`: 当前 slice_col 一共有多少个 CTA 会参与(所以需不需要 global reduce)
  - `slice_idx`: 当前 CTA 在这组 CTA 里的“归约顺序编号”(从下到上)

你可以用 standalone 的 `--trace` 看一个 CTA 的这些变量:

```
./marlin_standalone_sm80 -m 1 -n 2048 -k 2048 -g 128 --trace --trace_block 0 --trace_thread 0
```

它会打印:
- k_tiles/n_tiles/iters
- slice_row/slice_col/slice_iters/slice_count/slice_idx

这一步的直觉图(“stripe”):

```
把 16x16 的网格拉直成一条长度 256 的 list(按 K 先变, N 后变):
  idx = slice_col * k_tiles + slice_row

每个 CTA 领取 iters 个连续 idx:
  start = iters * blockIdx.x
  end   = start + iters

但一个 CTA 在一个 slice_col 里不能跨列, 所以 init_slice() 会把 iters 切成若干段:
  (slice_col=0, slice_row=...)
  (slice_col=1, slice_row=0..)
  ...
```

---

## 5. 共享内存布局(用我们的 1x2048x2048 case 算具体数)

kernel 里:

```cpp
extern __shared__ int4 sh[];
int4* sh_a = sh;
int4* sh_b = sh_a + (stages * a_sh_stage);
int4* sh_s = sh_b + (stages * b_sh_stage);
```

对 `thread_m_blocks=1, thread_n_blocks=8, thread_k_blocks=8, stages=4`:

### 5.1 A tile shared footprint

- `a_gl_stride = prob_k / 8 = 2048/8 = 256` (单位: int4)
- `a_sh_stride = 16 * thread_k_blocks / 8 = 16` (单位: int4)
- `a_sh_stage = a_sh_stride * (16 * thread_m_blocks) = 16 * 16 = 256` (单位: int4)

所以:
- A 每个 stage = 256 * 16B = 4096B

### 5.2 B tile shared footprint

- `b_gl_stride = 16 * prob_n / 32 = 1024` (单位: int4)
- `b_sh_stride = 32 * thread_n_blocks / 4 = 64` (单位: int4)
- `b_sh_stage = b_sh_stride * thread_k_blocks = 64 * 8 = 512` (单位: int4)

所以:
- B 每个 stage = 512 * 16B = 8192B

这正好对应 128x128 个 int4 weight:
- 128*128 weights * 4 bits = 65536 bits = 8192 bytes

### 5.3 Scale tile shared footprint (group=128)

- `s_sh_stride = 16 * thread_n_blocks / 8 = 16` (单位: int4)
- `s_sh_stage = s_sh_stride = 16` (单位: int4)

所以:
- s 每个 stage = 16 * 16B = 256B

### 5.4 总 shared

每个 stage: 4096 + 8192 + 256 = 12544B
4 stages: 50176B (约 49KB), fits.

ASCII 内存图:

```
sh (shared)
  stage0: [A0 4096B][B0 8192B][S0 256B]
  stage1: [A1 4096B][B1 8192B][S1 256B]
  stage2: [A2 4096B][B2 8192B][S2 256B]
  stage3: [A3 4096B][B3 8192B][S3 256B]
```

---

## 6. A/B/s 的线程级索引: 逐个变量解释(含本 case 的具体值)

这一段是理解的核心(对应 `marlin_cuda_kernel.cu` 大约 300-380 行).

### 6.1 A 索引

```cpp
int a_gl_rd = a_gl_stride * (tid / a_gl_rd_delta_o) + (tid % a_gl_rd_delta_o);
a_gl_rd += a_gl_rd_delta_o * slice_row;

int a_sh_wr = a_sh_stride * (tid / a_gl_rd_delta_o) + (tid % a_gl_rd_delta_o);
...
bool a_sh_wr_pred[i] = (a_sh_wr < a_sh_stride * prob_m);
```

对本 case:

- `a_gl_rd_delta_o = 16 * thread_k_blocks / 8 = 16`
- `threads / a_gl_rd_delta_o = 256/16 = 16`
- `a_gl_rd_delta_i = a_gl_stride * 16 = 256*16 = 4096` (单位: int4)
- `a_sh_wr_delta = a_sh_stride * 16 = 16*16 = 256` (单位: int4)
- `a_sh_wr_iters = ceildiv(a_sh_stage, a_sh_wr_delta) = ceildiv(256,256)=1`

因此只有一轮写 shared A, 且 predication 变成:

- `a_sh_wr_pred[0] = (a_sh_wr < 16 * prob_m)`
- M=1 -> `a_sh_wr < 16`

结论(非常关键):

> **只有 16 个线程会真的去 load A**, 刚好对应 1 行(128 个 FP16)所需的 16 个 `int4`(每个 int4=8 half).

### 6.2 B 索引

```cpp
int b_gl_rd = b_gl_stride * (tid / b_sh_stride) + (tid % b_sh_stride);
b_gl_rd += b_sh_stride * slice_col;
b_gl_rd += b_gl_rd_delta_o * slice_row;
int b_sh_wr = tid;
```

对本 case:

- `b_sh_stride = 64`
- `tid/b_sh_stride` 取值 0..3, 每组 64 线程
- `b_sh_stage=512`, `b_sh_wr_delta=256` -> `b_sh_wr_iters=2`

结论:

> 每个 stage, 每个线程做 2 次 `cp.async`(16B), 总共搬 256*2*16B=8192B, 恰好是一个 128x128 int4 tile.

### 6.3 Scale 索引 (group=128)

```cpp
int s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks)
            + s_sh_stride * slice_col + tid;
```

对本 case:
- `thread_k_blocks == group_blocks == 8`, 所以 `((8*slice_row)/8) == slice_row`
- `s_gl_stride = prob_n/8 = 256`
- `s_sh_stride = 16`

所以:
- `s_gl_rd = 256*slice_row + 16*slice_col + tid` (单位: int4)

以及:
- `s_sh_wr_pred = tid < s_sh_stride` -> 只有前 16 个线程写 scales (因为一行 scales=128 half=16 int4)

---

## 7. cp.async pipeline: stages=4 是怎么工作的?

关键函数(在 kernel 里是 lambda):

- `fetch_to_shared(pipe, ...)`:
  - A: `cp_async4_pred` (带 pred)
  - B: `cp_async4_stream` (带 L2 evict hint, 避免污染 L2)
  - s: 只在 group 边界抓一次
  - 最后 `cp_async_fence()` 提交

- `wait_for_stage()`:
  - `cp_async_wait<stages-2>()` + `__syncthreads()`
  - `stages-2` 的原因: double-buffering + 保证不会覆盖还没完成的 stage

直觉图:

```
time --->

pipe=0: [cp.async A0,B0,S0] [compute on stage0]
pipe=1:           [cp.async A1,B1,S1] [compute on stage1]
pipe=2:                     [cp.async A2,B2,S2] [compute on stage2]
pipe=3:                               [cp.async A3,B3,S3] [compute on stage3]

wait_group(stages-2)=wait_group(2):
  保证最老的那一批 async copy 已经完成, 这样我们才能回绕复用 shared buffer.
```

---

## 8. 从 shared 到寄存器, 再到 tensorcore

每个 K 子迭代:

1) `fetch_to_registers(k, pipe)`:
   - `ldsm4(frag_a, ...)` 把 shared A 直接按 MMA fragment layout 读到寄存器
   - 从 shared B 读出 `frag_b_quant` (I4=4个int)
   - (group quant) 把 scale tile 读入 `frag_s`

2) `matmul(k)`:
   - 对每个 `j=0..3`:
     - 取出 `b_quant` 与 `b_quant >> 8`
     - `dequant(...)` 得到 `FragB` (half2x2)
     - `scale(...)` (group quant)
     - `mma_m16n16k16(...)` 或两个 `mma(...)` 完成累加到 `FragC`

这一步的要点是:
- **dequant 与 mma 在同一 inner loop** 以便指令级 overlap(尽量让 tensorcore pipe 与 int pipe 都忙).

---

## 9. Reduction: CTA 内 + CTA 间

### 9.1 CTA 内 reduction: `thread_block_reduce()`

因为 kernel 为了增加 warp 数量但不把 N tile 做太宽, 会让多个 warp 计算同一输出 tile 的不同 K 子片段.
最后要在 shared 中把它们加起来.

本 case 的一个快速 sanity:
- `b_sh_stride=64`
- `red_off = threads / b_sh_stride / 2 = 256/64/2 = 2`
说明存在 reduction 分支(>=1), 也就是确实有跨 warp-group 的 partial sum 需要归约.

### 9.2 CTA 间 reduction: `global_reduce()` + locks

当 `slice_count > 1` 时才需要(同一 slice_col 有多个 CTA).

流程:
1) `barrier_acquire(&locks[slice_col], slice_idx)`
2) `global_reduce(first=(slice_idx==0), last=last)`
3) `barrier_release(&locks[slice_col], last)`

实现细节:
- 归约直接在输出 C buffer 上进行(尽量吃 L2 cache)
- 中间会把 fp32 accum downcast 成 fp16 存回 C, 下一 CTA 再读回来转 fp32 继续累加

---

## 10. `write_result()`: 最终写回 C 的布局重排

`mma` 的 fragment layout 不是 row-major 的 `[m,n]` 排列.
所以最后要先把 `FragC` 的结果按一定规则写到 shared, 再用 coalesced 的方式写回全局 C.

对 M=1:
- 只有 row0 是有效行
- 其它行的 store 会被 predication 跳过

---

## 11. 如何用 standalone 的 `--trace` 帮你对照理解

我在 `homemade_marlin/marlin_standalone.cu` 里加了:

- `--trace`: 开启 device printf(默认 warmup=0 iters=1)
- `--trace_block B`, `--trace_thread T`: 选择打印哪个 CTA/线程

示例:

```
./marlin_standalone_sm80 -m 1 -n 2048 -k 2048 -g 128 --trace --trace_block 0 --trace_thread 0
```

输出会包含:
- tile 相关: k_tiles/n_tiles/iters, slice_row/slice_col/slice_iters/...
- A/B/s 的 stride 与每线程 index: a_gl_rd/b_gl_rd/s_gl_rd 等
- predication: a_sh_wr_pred0 (对 M=1 应该为 1)

把 trace 输出和源码逐行对照, 你会非常快建立“变量->含义->数值”的直觉.

---

## 12. 你最可能改哪些地方? (改动指南)

按“对性能/正确性影响最大”的顺序:

1) **kernel config 列表**(`CALL_IF(...)`):
   - 想加新的 tile: 增加新的 `CALL_IF(thread_m_blocks, thread_n_blocks, thread_k_blocks, group_blocks)`
   - 注意 `prob_n % thread_n == 0` 和 `prob_k % thread_k == 0`

2) **thread_k/thread_n 的选择策略**(marlin_cuda 里的 if/else):
   - 小 M 倾向更好的 partitioning: `128x128`
   - 大 M 倾向更高吞吐: `64x256`

3) **group quant 路径**:
   - `group_blocks == -1` 走 per-column scale
   - `group_blocks != -1` 走 grouped scale
   - 任何更换 groupsize 都必须同时改:
     - `s_gl_rd` 的寻址逻辑
     - `fetch_to_shared` 里 “何时加载 scale”

4) **global reduction 的条件与锁**:
   - 想减少 reduction: 需要让 stripe 分配尽量让每个 slice_col 的 slice_count 变小(理想=1)
   - 但这和 SM utilization 是 trade-off

---

## 13. 下一步建议(我建议你怎么学得更快)

1) 先跑一次 `--trace` 只看 index/stride/predication, 确认“单位”正确.
2) 然后把 `slice_row/slice_col/slice_iters` 改成 trace 不同 blockIdx.x(例如 0,1,2,3...), 看 stripe 的变化.
3) 最后再去啃 `dequant()` 的 bit trick(LOP3 + 常量), 这部分是最“像魔法”的.
