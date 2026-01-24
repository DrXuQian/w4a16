# Marlin CUDA Kernel 技术深度分析

## 目录
1. [核心架构](#核心架构)
2. [内存访问模式](#内存访问模式)
3. [B矩阵Interleaving格式](#b矩阵interleaving格式)
4. [Threadblock计算规模](#threadblock计算规模)
5. [代码链接分析](#代码链接分析)

## 核心架构

### 配置参数（m=1, k=11008, n=2048）

| 参数 | 值 | 说明 |
|------|-----|------|
| `thread_m_blocks` | 1 | M维度：1个16×16块 |
| `thread_n_blocks` | 8 | N维度：8个16×16块 |
| `thread_k_blocks` | 8 | K维度：8个16×16块（单次迭代）|
| `threads` | 256 | 每个threadblock 256个线程 |
| `group_blocks` | 8 | groupsize/16 = 128/16 |

### Tile计算

- **k_tiles** = 11008 / (16×8) = 86
- **n_tiles** = 2048 / (16×8) = 16
- **iters** = 86 / 2 = 43（Split-K=2）

## 内存访问模式

### A矩阵参数

| 参数 | 计算公式 | 值 | 说明 |
|------|----------|-----|------|
| `a_gl_stride` | prob_k / 8 | 1376 | 每行的8元素单位数 |
| `a_sh_stride` | 16 × thread_k_blocks / 8 | 16 | Shared memory行stride |
| `a_sh_wr_iters` | 1 | 1 | 只需1次迭代（m=1） |

### B矩阵参数（INT4量化）

| 参数 | 计算公式 | 值 | 说明 |
|------|----------|-----|------|
| `b_gl_stride` | 16 × prob_n / 32 | 1024 | INT4量化后的行stride |
| `b_sh_stride` | 32 × thread_n_blocks / 4 | 64 | Shared memory stride |
| `b_sh_wr_iters` | b_sh_stage / b_sh_wr_delta | 2 | K维度分2次处理 |

### 关键发现：b_sh_wr_iters = 2

- 每个warp一次加载8行INT4数据
- 通过2次子迭代完成K=128的计算
- 解量化后提供完整的K=16数据给MMA指令

## B矩阵Interleaving格式

### 离线预处理（Python端）

**文件**: `/home/qianxu/marlin/marlin/__init__.py`

#### 1. 维度重排（Line 129-131）
```python
w = w.reshape((self.k // tile, tile, self.n // tile, tile))
w = w.permute((0, 2, 1, 3))  # 交换K和N的块维度
w = w.reshape((self.k // tile, self.n * tile))
```

#### 2. Interleave模式（Line 57-58）
```python
interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
perm = perm.reshape((-1, 8))[:, interleave].ravel()
```

#### 3. INT4打包（Line 134-138）
```python
for i in range(8):
    q |= res[:, i::8] << 4 * i  # 8个INT4打包成一个uint32
```

### 运行时使用（CUDA端）

**文件**: `/home/qianxu/marlin/marlin/marlin_cuda_kernel.cu`

#### 1. 加载到Shared Memory（Line 428）
```cpp
cp_async4_stream(&sh_b_stage[b_sh_wr_delta * i + b_sh_wr], B_ptr[i]);
```

#### 2. 加载到寄存器（Line 465）
```cpp
frag_b_quant[k % 2] = *reinterpret_cast<I4*>(&sh_b_stage[b_sh_rd_delta * (k % b_sh_wr_iters) + b_sh_rd]);
```

#### 3. 解量化和MMA执行（Line 473-485）
```cpp
int b_quant = frag_b_quant[k % 2][j];
FragB frag_b0 = dequant(b_quant);
FragB frag_b1 = dequant(b_quant_shift);
mma_m16n16k16(frag_a[k % 2][i], frag_b0, frag_b1, frag_c[i][j][0], frag_c[i][j][1]);
```

## Threadblock计算规模

### 单个Threadblock处理

- **输出tile**: 16×128 (M×N)
- **单次迭代**: [16×128] × [128×128]
- **总迭代数**: 43次
- **总计算量**: ~1.13M MACs

### Global Reduce机制

用于Split-K归约：
- TB0: 计算 C[0:15, 0:127] 使用 K[0:5503]
- TB1: 计算 C[0:15, 0:127] 使用 K[5504:11007]
- 通过global_reduce将两个部分和累加

## 代码链接分析

### 离线预处理路径
1. 原始权重 → `/marlin/__init__.py:118` (量化)
2. → `Line 129-131` (维度重排，实现interleaving)
3. → `Line 133` (应用permutation)
4. → `Line 134-138` (INT4打包)
5. → `Line 139` (存储到self.B)

### 运行时使用路径
1. B矩阵 → `/marlin/marlin_cuda_kernel.cu:428` (cp_async加载)
2. → `Line 465` (加载到寄存器)
3. → `Line 473` (取出int32)
4. → `Line 475, 479` (dequant解量化)
5. → `Line 485` (mma_m16n16k16执行)

## 关键优化技术

### 1. INT4量化
- 减少75%内存带宽
- 离线预处理，运行时零开销

### 2. Interleaving布局
- 针对Tensor Core优化
- 避免运行时转置

### 3. Split-K并行
- 32个TB完美负载均衡
- 最少的归约开销

### 4. 4-stage Pipeline
- 隐藏内存延迟
- 计算与访存重叠

## 性能特点

- **计算密度**: ~4.25 FLOPs/Byte
- **内存带宽优化**: INT4压缩 + 异步拷贝
- **SM利用率**: 32个SM全部使用
- **归约开销**: 仅2-way reduction

---

*基于Marlin v0.1.1版本分析*
*配置：m=1, k=11008, n=2048, groupsize=128*