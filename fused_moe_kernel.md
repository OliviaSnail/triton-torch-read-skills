# 一、代码概览

`fused_moe_kernel` 是一个使用 `@triton.jit` 装饰的 **Triton kernel**，实现了 Mixture-of-Experts（MoE）中：

- 按 expert 排序后的 token 激活矩阵 A；
- 多个 expert 权重矩阵 B；
- 可选 bias；
- 多种量化方案（fp8 / int8，tensor-wise / per-channel / block-wise）；
- MoE gate 路由权重；

的 **融合矩阵乘法计算**，并把结果写回输出矩阵 C。

从数学上可以抽象为，对于每个路由后的 token 行 $m$ 与对应 expert $e$，计算：

$$
C[m, n]
  = \left( \sum_{k=0}^{K-1} A[m, k] \cdot W_e[n, k] \right)
    \cdot \mathrm{scale}_{A,B}(m, n)
    \cdot w_{\mathrm{gate}}(m)
    + \mathrm{bias}_e[n]
$$

其中：

- $A \in \mathbb{R}^{EM \times K}$：路由 + 排序 + padding 后的 token 激活；
- $W_e \in \mathbb{R}^{N \times K}$：第 $e$ 个 expert 的权重矩阵；
- $C \in \mathbb{R}^{EM \times N}$：输出；
- $\mathrm{scale}_{A,B}$ 来自量化方案（fp8/int8）；
- $w_{\mathrm{gate}}(m)$ 是该 token 对应的 MoE gate 权重；
- $\mathrm{bias}_e$ 是 expert bias（可选）。

kernel 通过合理的 tile 划分（`BLOCK_SIZE_M/N/K`）和 group 调度（`GROUP_SIZE_M`），在单个 Triton kernel 中完成上述所有操作，减少内存访存和 kernel launch 数量。


# 二、函数输入：含义与 shape

下面以逻辑张量 / 标量的视角整理 `fused_moe_kernel` 的重要参数。

## 1. 主要张量参数

| 参数名                     | 含义                                                                 | 典型 shape / 类型                              |
|----------------------------|----------------------------------------------------------------------|-----------------------------------------------|
| `a_ptr`                    | token 激活矩阵 A 的数据指针                                         | 逻辑形状约为 `[EM, K]`                        |
| `a_desc`                   | A 的 Tensor Descriptor（TMA 描述符，可选）                          | `TensorDescriptor` 或 `None`                  |
| `b_ptr`                    | expert 权重矩阵 B 的数据指针                                       | 逻辑形状 `[E, N, K]`                          |
| `b_desc`                   | B 的 Tensor Descriptor（TMA 描述符，可选）                          | `TensorDescriptor` 或 `None`                  |
| `bias_ptr`                 | expert bias 的数据指针（可选）                                      | 逻辑形状 `[E, N]`                             |
| `c_ptr`                    | 输出矩阵 C 的数据指针                                              | 逻辑形状 `[EM, N]`                            |
| `a_scale_ptr`              | A 的量化 scale 数据指针（fp8/int8 时使用）                          | tensor-wise / channel-wise / block-wise       |
| `b_scale_ptr`              | B 的量化 scale 数据指针                                             | 同上                                          |
| `topk_weights_ptr`         | 路由后每个 token 的 gate 权重                                       | 一维数组（长度 ≥ `num_valid_tokens`）         |
| `sorted_token_ids_ptr`     | 排序后的 token 索引序列                                             | `[EM]`                                        |
| `expert_ids_ptr`           | 每个 M-block 对应的 expert id                                       | `[ceil(EM / BLOCK_SIZE_M)]`                   |
| `num_tokens_post_padded_ptr` | 路由 + padding 后 token 总数 EM                                  | 标量                                          |

## 2. 维度与 stride 参数

| 参数名              | 含义                                             | 类型  |
|---------------------|--------------------------------------------------|-------|
| `N`                 | 输出维度（等于 B 的第二维）                      | int   |
| `K`                 | 输入维度（等于 B 的第三维，可能含 padding）      | int   |
| `EM`                | 路由 + padding 后的 token 总数                   | int   |
| `num_valid_tokens`  | 真实有效 token 数（不含 padding）                | int   |
| `stride_am`, `stride_ak` | A 在 M/K 维上的 stride                      | int   |
| `stride_be`, `stride_bk`, `stride_bn` | B 在 expert/K/N 维上的 stride  | int   |
| `stride_bias_e`, `stride_bias_n` | bias 在 expert/N 维上的 stride      | int   |
| `stride_cm`, `stride_cn` | C 在 M/N 维上的 stride                      | int   |
| `stride_asm`, `stride_ask` | `a_scale` 在 token/K 维上的 stride       | int   |
| `stride_bse`, `stride_bsk`, `stride_bsn` | `b_scale` 在 expert/K/N 维上的 stride | int |

## 3. 量化与调度相关 meta 参数

| 参数名             | 含义                                                        | 类型              |
|--------------------|-------------------------------------------------------------|-------------------|
| `group_n`, `group_k` | block-wise 量化的 N/K 向 block 大小                      | `tl.constexpr int`|
| `BLOCK_SIZE_M/N/K` | Triton tile 在 M/N/K 维度的大小                             | `tl.constexpr int`|
| `GROUP_SIZE_M`     | 一组中 M 向的 tile 个数，用于 grouped 调度                 | `tl.constexpr int`|
| `MUL_ROUTED_WEIGHT`| 是否乘上 MoE gate 权重                                     | `tl.constexpr bool`|
| `top_k`            | 每个 token 选择的 expert 数                                | `tl.constexpr int`|
| `compute_type`     | 计算/输出 dtype（如 `tl.float16`）                         | `tl.constexpr`    |
| `use_fp8_w8a8`     | 是否使用 fp8 权重 + fp8 激活                               | `tl.constexpr bool`|
| `use_int8_w8a8`    | 是否使用 int8 权重 + int8 激活                            | `tl.constexpr bool`|
| `use_int8_w8a16`   | 是否使用 int8 权重 + fp16 激活                            | `tl.constexpr bool`|
| `per_channel_quant`| 是否为 per-channel quant（否则 tensor-wise 或 block-wise） | `tl.constexpr bool`|
| `even_Ks`          | K 是否是 `BLOCK_SIZE_K` 的整数倍                           | `tl.constexpr bool`|
| `c_sorted`         | 输出是否保持 sorted token 顺序                             | `tl.constexpr bool`|
| `filter_expert`    | 是否过滤掉不在当前 expert parallel rank 的 expert（id=-1） | `tl.constexpr bool`|
| `swap_ab`          | 是否在某些 fp8 场景下交换 A/B 以优化 SM90 性能             | `tl.constexpr bool`|


# 三、中间变量的 shape 流动（含 BLOCK_SIZE / stride / program_id）

## 1. program id 到 C 中 tile 的映射

```python
pid = tl.program_id(axis=0)
num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

- 每个 Triton program（由 `pid` 标识）负责 C 的一个 tile：
  - 行范围：

    $$m \in [pid_m \cdot BLOCK\_SIZE\_M,\ (pid_m + 1)\cdot BLOCK\_SIZE\_M)$$

  - 列范围：

    $$n \in [pid_n \cdot BLOCK\_SIZE\_N,\ (pid_n + 1)\cdot BLOCK\_SIZE\_N)$$

- `GROUP_SIZE_M` 将若干连续的 M tiles 编为一组，在组内沿 N 方向遍历，有利于复用同一 expert 权重的 L2 cache。

## 2. token 索引与有效 mask

```python
num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
    return
offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
offs_token = offs_token.to(tl.int64)
token_mask = offs_token < num_valid_tokens
```

- `offs_token_id`：当前 tile 的 EM 维下标（在排序 + padding 后的序列中的 index）；
- `offs_token`：真正映射到「路由后的 token 索引」，部分位置可能是 padding；
- `token_mask`：标记哪些行是有效 token（`offs_token < num_valid_tokens`）。

## 3. expert 选择与过滤

```python
off_experts_i32 = tl.load(expert_ids_ptr + pid_m)
off_experts = off_experts_i32.to(tl.int64)

if filter_expert and off_experts == -1:
    write_zeros_to_output(
        c_ptr, stride_cm, stride_cn,
        pid_n, N, offs_token,
        token_mask, BLOCK_SIZE_M, BLOCK_SIZE_N, compute_type,
    )
    return
```

- 每个 `pid_m`（即一个 `BLOCK_SIZE_M` 行块）对应一个 expert id；
- 当该 expert 不在当前 expert parallel rank 上时（id 为 -1），直接调用 `write_zeros_to_output` 把对应输出块清零并返回，避免无效计算。

## 4. A/B 的 BLOCK 与 stride 地址计算

### A：token 激活块

```python
offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
offs_k = tl.arange(0, BLOCK_SIZE_K)
if a_desc is not None:
    start_offs_m = pid_m * BLOCK_SIZE_M
else:
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )
```

- `offs_token[:, None] // top_k`：大小为 `[BLOCK_SIZE_M, 1]`，将「路由后重复了 `top_k` 次的 token 行」还原到原始 token 行号；
- `offs_k[None, :]`：大小为 `[1, BLOCK_SIZE_K]`，对 K 向量做偏移；
- 广播后，`a_ptrs` 是一个 `[BLOCK_SIZE_M, BLOCK_SIZE_K]` 的地址网格，对应：

$$
A_{\mathrm{block}}(i, j)
  = A\Big( \big\lfloor \tfrac{\mathrm{offsToken}_i}{\mathrm{topK}} \big\rfloor,
           k_{\mathrm{base}} + j \Big)
$$

### B：expert 权重块

```python
if b_desc is not None:
    start_offs_n = pid_n * BLOCK_SIZE_N
else:
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )
```

- `offs_k[:, None]`：大小为 `[BLOCK_SIZE_K, 1]`，对应输入维度 K；
- `offs_bn[None, :]`：大小为 `[1, BLOCK_SIZE_N]`，对应输出维度 N；
- 广播后，`b_ptrs` 是一个 `[BLOCK_SIZE_K, BLOCK_SIZE_N]` 的地址网格，对应：

$$
B_{\mathrm{block}}(k, n)
  = B_e\big( n_{\mathrm{base}} + n,\ k_{\mathrm{base}} + k \big)
$$

使用 TMA（`a_desc` / `b_desc`）时，上述逻辑由 descriptor 内部封装。

## 5. 沿 K 维循环与 accumulator 的形状

```python
if swap_ab:
    accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
else:
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

for k_start in range(0, K, BLOCK_SIZE_K):
    # 加载 A block
    if a_desc is not None:
        a = a_desc.load([start_offs_m, k_start])
    elif even_Ks:
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
    else:
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k_start),
            other=0.0,
        )

    # 加载 B block
    if b_desc is not None:
        b = (
            b_desc.load([off_experts_i32, start_offs_n, k_start])
            .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)
            .T
        )
    elif even_Ks:
        b = tl.load(b_ptrs)
    else:
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)

    # 之后结合量化模式做 tl.dot + scale
    ...
```

不考虑 `swap_ab` 时，可以把当前 tile 的形状抽象写成：

- $a \in \mathbb{R}^{M \times K}$；
- $b \in \mathbb{R}^{K \times N}$；
- `tl.dot(a, b)` 的结果 $\in \mathbb{R}^{M \times N}$，累加进 `accumulator`。

考虑 `swap_ab = \mathrm{True}` 时，会通过 `tl.trans` 在适当时机交换维度，保证最后写回 C 时仍为 `[BLOCK_SIZE_M, BLOCK_SIZE_N]` 的布局。


# 四、关键算子：含义与用法

本节只列出对理解 kernel 至关重要的 Triton 原语及其用法。

## 1. `tl.program_id`

- **语义**：返回当前 kernel 实例在给定 axis 上的 program id（类似 CUDA 的 blockIdx）。
- **形状**：标量整型。
- **示例**：

```python
@triton.jit
def example(x_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
```

在 `fused_moe_kernel` 中，`pid` 用来映射到 C 中的 tile `(pid_m, pid_n)`。

## 2. `tl.arange` + stride 地址计算

- **语义**：在 `[start, end)` 范围内生成一维等差整数向量，常用来构造二维网格索引并结合 stride 做矩阵访问。
- **在本 kernel 中**，典型用法：

```python
offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
offs_k = tl.arange(0, BLOCK_SIZE_K)
```

与 stride 结合时，遵循典型的地址公式：

$$
\mathrm{addr}(m, n)
  = \mathrm{base}
  + m \cdot \mathrm{stride}_m
  + n \cdot \mathrm{stride}_n
$$

在代码中体现为：

```python
c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
```

## 3. `tl.load` / `tl.store`（带 mask）

- `tl.load(ptrs, mask, other)`：
  - 在 `mask=True` 的位置从内存加载；
  - `mask=False` 的位置返回 `other`。
- `tl.store(ptrs, value, mask)`：
  - 在 `mask=True` 的位置写入，`False` 的位置忽略。

示例：

```python
a = tl.load(
    a_ptrs,
    mask=token_mask[:, None] & (offs_k[None, :] < K - k_start),
    other=0.0,
)
tl.store(c_ptrs, accumulator, mask=c_mask)
```

通过 mask 可以安全处理：

- token 方向的 padding（`token_mask`）；
- K 方向不足一整 block 的尾部（`offs_k < K - k_start`）。

## 4. `tl.dot`

- **语义**：二维矩阵乘（缩并最后一维 / 第一维）。
- **形状规则**：

如果：

$$
a \in \mathbb{R}^{M \times K},\quad
b \in \mathbb{R}^{K \times N}
$$

则：

$$
\mathrm{dot}(a, b) \in \mathbb{R}^{M \times N}
$$

- **在本 kernel 中**：

```python
accumulator = tl.dot(a, b, acc=accumulator)
```

或在量化路径下：

```python
accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
```

## 5. 量化相关：scale 与 cast

- 典型的「解量化 + dot」流程：

```python
# 假设 b 最初是 int8
b = tl.load(b_ptrs)
b = b.to(tl.float32)
b = (b * b_scale).to(compute_type)
accumulator = tl.dot(a, b, acc=accumulator)
```

- block-wise / channel-wise / tensor-wise 的区别，体现在 `a_scale` / `b_scale` 的索引方式以及是「在循环中按 block 更新」还是「在循环结束后整体乘」。


# 五、函数输出：含义与 shape

kernel 通过 `tl.store` 将 `accumulator` 写回 `c_ptr` 所指向的输出矩阵 C：

```python
offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
if c_sorted:
    c_ptrs = (
        c_ptr + stride_cm * offs_token_id[:, None] + stride_cn * offs_cn[None, :]
    )
else:
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
tl.store(c_ptrs, accumulator, mask=c_mask)
```

- 逻辑形状：`[EM, N]`：
  - $EM$：等于 `sorted_token_ids.shape[0]`，即路由后 token × topk 再 padding 后的总行数；
  - $N = B.\mathrm{shape}[1]$：输出 hidden size。
- 对每个有效 token 行 $m$，对应一行：

$$
C[m, :] \approx \big(A[m, :] \cdot W_{e(m)}^\top\big) \cdot \mathrm{scale}_{A,B}(m, :)
                \cdot w_{\mathrm{gate}}(m) + \mathrm{bias}_{e(m)}
$$

- `c_sorted` 决定写回时是按：
  - `offs_token_id`（sorted 序列下标）；\n  - 还是 `offs_token`（路由后 token 索引）。


# 六、整体功能总结

从 MoE 系统视角看，`fused_moe_kernel` 是一个 **高度融合的「路由 + 量化解码 + GEMM + bias + gate 权重」kernel**：

1. 上游已经完成：
   - 对每个 token 计算 gate，选出 `top_k` 个 experts 和对应权重；
   - 根据 expert id 对 `(token, expert)` pair 做排序与 padding，生成 `sorted_token_ids` 和 `expert_ids`，保证高度结构化且便于按 block 处理。

2. `fused_moe_kernel` 针对每个 M-block：
   - 通过 `expert_ids` 选出对应 expert 的权重切片；
   - 以 `BLOCK_SIZE_M/N/K` 为粒度，从 token 激活 A 与 expert 权重 B 中取出子矩阵；
   - 按 fp8/int8 的多种量化配置，应用正确的 scale（tensor-wise / channel-wise / block-wise）；
   - 可选地加上 expert bias；
   - 若启用 `MUL_ROUTED_WEIGHT`，再乘上每个 token 的 gate 权重；
   - 将结果写入 C 中对应的 tile，可选保持 sorted 布局。

通过将这些操作整合到同一个 Triton kernel 中，`fused_moe_kernel` 减少了：

- 中间张量在全局内存中的往返（例如单独的解量化 kernel、GEMM kernel、bias add kernel 等）；
- kernel launch 次数和调度开销。

这对于大规模 MoE 推理（尤其是多 expert、量化权重、需要高吞吐的场景）而言，可以显著减轻 memory bandwidth 压力，并提升端到端性能。

****
