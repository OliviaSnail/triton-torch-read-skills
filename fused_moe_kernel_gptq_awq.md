# 一、代码概览

`fused_moe_kernel_gptq_awq` 是一个使用 `@triton.jit` 装饰的 **Triton kernel**，专门用于 **GPTQ / AWQ 风格权重量化的 MoE 融合计算**。它在保持与 `fused_moe_kernel` 相同的「按 expert 排序 + 分块 GEMM + 路由权重」主流程的同时，针对：

- **权重量化**：支持 **INT4（W4A16）** 或 **INT8（W8A16）**，即权重为 int4/int8、激活为 fp16；
- **分组 scale**：`b_scale` 按 `group_size` 在 K 维分组（GPTQ/AWQ 典型配置）；
- **可选 zero-point**：`b_zp_ptr` 可选，用于非对称量化时的零点；
- **无激活量化**：激活 A 不在此 kernel 内量化，仅权重量化。

从数学上可抽象为：对每个路由后的 token 行 $m$ 与对应 expert $e$，计算：

$$
C[m, n]
  = \left( \sum_{k=0}^{K-1} A[m, k] \cdot \big( (W_e^{\text{int}}[n,k] - \text{zp}) \cdot \text{scale}(k \bmod \text{group\_size}) \big) \right)
    \cdot w_{\text{gate}}(m)
$$

其中 $W_e^{\text{int}}$ 为量化后的权重（int4 或 int8），scale 与 zp 按 K 维分组。kernel 通过 `BLOCK_SIZE_M/N/K` 与 `GROUP_SIZE_M` 做分块与调度，在单 kernel 内完成「权重的解量化 + GEMM + 路由权重」，适配 GPTQ/AWQ 等权重量化 MoE 推理。


# 二、函数输入：含义与 shape

## 1. 主要张量参数

| 参数名                     | 含义                                                                 | 典型 shape / 类型                              |
|----------------------------|----------------------------------------------------------------------|-----------------------------------------------|
| `a_ptr`                    | token 激活矩阵 A 的数据指针（fp16，未量化）                          | 逻辑形状约为 `[M, K]`，M 为原始 token 数      |
| `b_ptr`                    | expert 权重量化后的数据指针（int4 打包或 int8）                      | 逻辑形状 `[E, N, K]`（int4 时 K 向为打包后长度） |
| `c_ptr`                    | 输出矩阵 C 的数据指针                                                | 逻辑形状 `[EM, N]`                            |
| `b_scale_ptr`              | 权重的分组 scale，按 `group_size` 在 K 维分组                       | 与 B 的 expert/N/K 分组对应，通常 3 维       |
| `b_zp_ptr`                 | 权重的分组 zero-point（可选，非对称量化时使用）                      | 与 scale 同布局或兼容 layout                  |
| `topk_weights_ptr`         | 路由后每个 token 的 gate 权重                                        | 一维，长度 ≥ `num_valid_tokens`               |
| `sorted_token_ids_ptr`     | 排序后的 token 索引序列                                              | `[EM]`                                        |
| `expert_ids_ptr`           | 每个 M-block 对应的 expert id                                        | `[ceil(EM / BLOCK_SIZE_M)]`                   |
| `num_tokens_post_padded_ptr` | 路由 + padding 后 token 总数 EM                                  | 标量                                          |

## 2. 维度与 stride 参数

| 参数名              | 含义                                             | 类型  |
|---------------------|--------------------------------------------------|-------|
| `N`                 | 输出维度（B 的第二维）                           | `tl.constexpr` |
| `K`                 | 输入维度（B 的第三维，未打包的 K）               | `tl.constexpr` |
| `EM`                | 路由 + padding 后的 token 总数                  | 标量  |
| `num_valid_tokens`   | 真实有效 token 数（不含 padding）                | 标量  |
| `stride_am`, `stride_ak` | A 在 M/K 维上的 stride                       | int   |
| `stride_be`, `stride_bk`, `stride_bn` | B 在 expert/K/N 维上的 stride       | int   |
| `stride_cm`, `stride_cn` | C 在 M/N 维上的 stride                       | int   |
| `stride_bse`, `stride_bsk`, `stride_bsn` | B_scale 在 expert/K_group/N 维上的 stride | int |
| `stride_bze`, `stride_bzk`, `stride_bzn` | B zero-point 在 expert/K_group/N 维上的 stride | int |

## 3. 量化与调度相关 meta 参数

| 参数名             | 含义                                                        | 类型              |
|--------------------|-------------------------------------------------------------|-------------------|
| `group_size`       | 权重量化在 K 维的分组大小（GPTQ/AWQ 的 group）              | `tl.constexpr int`|
| `BLOCK_SIZE_M/N/K` | Triton tile 在 M/N/K 维度的大小                             | `tl.constexpr int`|
| `GROUP_SIZE_M`     | 一组中 M 向 tile 个数，用于 grouped 调度、L2 复用            | `tl.constexpr int`|
| `MUL_ROUTED_WEIGHT`| 是否乘上 MoE gate 权重                                     | `tl.constexpr bool`|
| `top_k`            | 每个 token 选择的 expert 数                                | `tl.constexpr int`|
| `compute_type`     | 累加/输出 dtype（如 `tl.float16`）                         | `tl.constexpr`    |
| `has_zp`           | 是否使用 zero-point（非对称量化）                          | `tl.constexpr bool`|
| `use_int4_w4a16`   | 是否使用 int4 权重 + fp16 激活（权重按 2 个 int4 打包存）   | `tl.constexpr bool`|
| `use_int8_w8a16`   | 是否使用 int8 权重 + fp16 激活                            | `tl.constexpr bool`|
| `even_Ks`          | K 是否为 `BLOCK_SIZE_K` 的整数倍，用于省略 K 维 mask       | `tl.constexpr bool`|
| `filter_expert`    | 是否过滤掉不在当前 expert parallel rank 的 expert（id=-1）  | `tl.constexpr bool`|


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

- 每个 program 负责 C 的一个 tile：
  - 行范围：$m \in [pid_m \cdot BLOCK\_SIZE\_M,\ (pid_m + 1)\cdot BLOCK\_SIZE\_M)$；
  - 列范围：$n \in [pid_n \cdot BLOCK\_SIZE\_N,\ (pid_n + 1)\cdot BLOCK\_SIZE\_N)$。
- 地址计算满足：
  $$\text{addr}(m, n) = \text{base} + m \cdot \text{stride}_m + n \cdot \text{stride}_n$$

## 2. token 索引与有效 mask

```python
num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
    return
offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
token_mask = offs_token < num_valid_tokens
```

- `offs_token_id`：当前 tile 在排序后序列中的行下标，形状与 `tl.arange(0, BLOCK_SIZE_M)` 一致。
- `offs_token`：路由后的 token 索引（可能含 padding），用于索引 A 的行（`offs_token // top_k`）和 `topk_weights_ptr`。
- `token_mask`：标记当前 tile 中哪些行是有效 token，后续 load/store 的 mask 会用到。

## 3. expert 选择与过滤

```python
off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
if filter_expert and off_experts == -1:
    write_zeros_to_output(...)
    return
```

- 每个 M-block 对应一个 expert id；若该 expert 不在当前 rank（id = -1），则写零并 return。

## 4. A 的指针与 B 的指针（含 int4 打包）

**A：**

```python
a_ptrs = a_ptr + (
    offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
)
```

- `offs_token[:, None] // top_k` 形状 `[BLOCK_SIZE_M, 1]`，还原到原始 token 行号；`offs_k[None, :]` 形状 `[1, BLOCK_SIZE_K]`。因此 `a_ptrs` 对应一块 $A_{\text{block}} \in \mathbb{R}^{BLOCK\_SIZE\_M \times BLOCK\_SIZE\_K}$。

**B（int4 与 int8 分支）：**

- **INT4（W4A16）**：权重按 2 个 int4 打包在一个元素里，K 向步进以「半 K」计：

```python
b_ptrs = (
    b_ptr
    + off_experts * stride_be
    + (offs_k[:, None] // 2) * stride_bk
    + offs_bn[None, :] * stride_bn
)
b_shifter = (offs_k[:, None] % 2) * 4
```

  - 加载后用 `(b >> b_shifter) & 0xF` 取出对应 int4；K 循环中指针步进为 `(BLOCK_SIZE_K // 2) * stride_bk`。
- **INT8（W8A16）**：常规布局，`b_ptrs` 为 `[BLOCK_SIZE_K, BLOCK_SIZE_N]` 的地址块，步进 `BLOCK_SIZE_K * stride_bk`。

## 5. 分组 scale / zero-point 与解量化

- scale 按 K 维分组，组号：
  $$\text{group\_idx} = \big\lfloor (k_{\text{base}} + k) / \text{group\_size} \big\rfloor$$
  对应代码中 `(offs_k[:, None] + BLOCK_SIZE_K * k) // group_size`。

- `b_scale_ptrs` 由 expert、当前 N 块、以及上述 group 索引构造；`b_zp_ptrs` 同理（int4 时 N 向可能按 2 打包，故有 `offs_bn // 2` 等）。

- 解量化公式（与 GPTQ/AWQ 一致）：
  - 有 zero-point：$b_{\text{fp}} = (b_{\text{int}} - \text{zp}) \cdot \text{scale}$；
  - 无 zero-point：$b_{\text{fp}} = (b_{\text{int}} - \text{zp\_num}) \cdot \text{scale}$，其中 int4 用 `zp_num = 8`，int8 用 `zp_num = 128`（对称量化时的中点）。

- 累加器形状为 `[BLOCK_SIZE_M, BLOCK_SIZE_N]`，在 K 维循环中反复 `tl.dot(a, b, acc=accumulator)`，最后转为 `compute_type` 并可选乘上 `moe_weight[:, None]`。


# 四、关键算子：含义与用法

## 1. `tl.program_id`

- **语义**：当前 kernel 实例在一维 grid 上的 id，用于映射到 (pid_m, pid_n)。
- **用法**：`pid = tl.program_id(axis=0)`，再结合 `num_pid_m`、`num_pid_n`、`GROUP_SIZE_M` 得到 tile 坐标。

## 2. `tl.arange` 与 stride 地址计算

- **语义**：生成一维整数序列，与 stride 组合得到二维索引。地址满足 $\text{addr}(m,n) = \text{base} + m \cdot \text{stride}_m + n \cdot \text{stride}_n$。
- **本 kernel 中**：`offs_token_id`、`offs_k`、`offs_bn`、`offs_cn` 均由此生成，并用于构造 `a_ptrs`、`b_ptrs`、`c_ptrs`。

## 3. `tl.load` / `tl.store`（带 mask）

- **语义**：`tl.load(ptrs, mask=..., other=...)` 在 mask 为真的位置加载，否则填 `other`；`tl.store(ptrs, value, mask=...)` 仅在 mask 为真处写入。
- **本 kernel 中**：用 `token_mask` 屏蔽 padding token，用 `offs_k < K - k*BLOCK_SIZE_K` 处理 K 维尾部；写 C 时用 `c_mask = token_mask[:, None] & (offs_cn[None, :] < N)`。

## 4. `tl.dot`

- **语义**：矩阵乘，$\text{dot}(a,b) \in \mathbb{R}^{M \times N}$，其中 $a \in \mathbb{R}^{M \times K}$，$b \in \mathbb{R}^{K \times N}$。
- **本 kernel 中**：`accumulator = tl.dot(a, b, acc=accumulator)`，a 为 fp16 block，b 为解量化后的权重 block。

## 5. int4 解包与 zero-point

- **int4**：`b = (b >> b_shifter) & 0xF`，从打包存储中取出 4-bit；若有 `b_zp`，同样按位取出后参与 $(b - \text{zp}) * \text{scale}$。
- **标量 zero-point 常量**：无 `b_zp_ptr` 时，int4 用 8、int8 用 128 作为对称量化的中点减数。


# 五、函数输出：含义与 shape

kernel 将 `accumulator` 写回 `c_ptr`：

```python
offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
tl.store(c_ptrs, accumulator, mask=c_mask)
```

- **逻辑形状**：C 为 `[EM, N]`，其中 EM 为 `sorted_token_ids.shape[0]`，N 为 B 的输出维。
- **含义**：每一行对应「某个路由后 token 在对应 expert 上的线性变换结果 × 可选 gate 权重」，且权重已按 GPTQ/AWQ 分组 scale（及可选 zp）解量化后再参与 GEMM。
- 本 kernel **始终按 `offs_token` 写回**（与 `fused_moe_kernel` 的 `c_sorted=False` 行为一致），不提供 sorted 布局选项。


# 六、整体功能总结

`fused_moe_kernel_gptq_awq` 在 MoE 推理链路中的角色是：**在权重量化为 GPTQ/AWQ 风格（int4 或 int8 + 分组 scale + 可选 zero-point）时，对「按 expert 排序并 padding 后的 token」做融合的「解量化 + GEMM + 路由权重」计算**。

- **上游**：已完成 gate 与路由，得到 `sorted_token_ids`、`expert_ids`、`topk_weights`，且 B 与 `b_scale`/`b_zp` 已按 GPTQ/AWQ 格式准备好。
- **本 kernel**：按 (pid_m, pid_n) 分块，从 A 取 token block、从 B 取对应 expert 的权重 block；对 int4 做解包与按组解量化，对 int8 做按组解量化；在 fp32 累加器中完成 GEMM，再转回 `compute_type`，可选乘 gate 权重，写回 C。
- **效果**：将「权重的分组解量化 + 矩阵乘 + 路由权重」合并为单 kernel，减少中间读写与 launch 次数，适合部署 GPTQ/AWQ 量化 MoE 模型时的推理加速。
