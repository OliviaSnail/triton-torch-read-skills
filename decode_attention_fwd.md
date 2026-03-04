# 一、代码概览

`decode_attention_fwd` 是 **decode 阶段**的 memory-efficient attention 前向入口：上层传入当前步的 query `q`、以及已写入 paged KV cache 的 `k_buffer` / `v_buffer`，通过 **两阶段 Triton 流程** 得到注意力输出 `o`。

- **Stage1**：按 (batch, head, kv_split) 并行，对每个 split 内的 KV 做「分块 Q·K^T → 缩放 → 可选 logit cap / xai temperature → 在线 softmax」，得到该 split 的加权和与 log-sum-exp，写入中间张量 `attn_logits`（Att_Out）和 `attn_lse`（Att_Lse）。
- **Stage2**：按 (batch, head) 并行，把各 split 的 Att_Out / Att_Lse 按 softmax 的数值稳定形式合并，可选加上 sink token，最后写回 `o`。

入口根据 `kv_group_num = q.shape[1] // v_buffer.shape[1]` 分支：**MHA（kv_group_num == 1）** 走 `decode_attention_fwd_normal`，**GQA/MQA/MLA** 走 `decode_attention_fwd_grouped`。两者都是「Stage1 kernel + Stage2 kernel」，仅 Stage1 的实现不同（单 head 用标量/向量运算，多 head 用 `tl.dot` 等）。

数学上可抽象为对每个 batch、每个 head：用当前步的 $q$ 与整段 context 的 $K,V$ 做缩放点积注意力，得到 $o = \mathrm{softmax}(q K^\top / \sqrt{d}) V$，其中 $K,V$ 通过 `kv_indices` 从 paged buffer 中按逻辑位置取出。


# 二、函数输入：含义与 shape

## 1. 入口函数 `decode_attention_fwd` 的参数

| 参数名 | 含义 | 典型 shape / 类型 |
|--------|------|-------------------|
| `q` | 当前 decode 步的 query，每 batch 一行、每 head 一维 | `[batch, num_heads, Lk]`，dtype 多为 fp16/bf16 |
| `k_buffer` | paged KV 中的 K 缓存，按 page 存 | `[num_pages, num_kv_heads, Lk]` 或等价 layout |
| `v_buffer` | paged KV 中的 V 缓存 | `[num_pages, num_kv_heads, Lv]` |
| `o` | 输出，与 `q` 的 batch/head 对应，最后一维为 head 维 | `[batch, num_heads, Lv]` |
| `kv_indptr` | 每个 batch 在 KV 序列上的起始下标（CSR 行指针） | `[batch + 1]`，整型 |
| `kv_indices` | 该 batch 内每个逻辑位置的 page 内偏移或全局 page index，用于从 buffer 取 K/V | 一维，长度 = 该 batch 的 KV 长度之和（或由 indptr 界定） |
| `attn_logits` | Stage1 写出的中间结果：每个 (batch, head, split) 的加权 V 和 | `[batch, num_heads, max_kv_splits, Lv]` |
| `attn_lse` | Stage1 写出的 log-sum-exp，用于 Stage2 合并 | `[batch, num_heads, max_kv_splits]`（或按 Lv 打包的等价 shape） |
| `num_kv_splits` | 每个 batch 实际使用的 KV split 数 | `[batch]` |
| `max_kv_splits` | 预分配的最大 split 数，需与 `attn_logits.shape[2]` 一致 | 标量 |
| `sm_scale` | 注意力缩放因子，通常 $1/\sqrt{Lk}$ | 标量 float |
| `logit_cap` | 对 logits 的 cap（>0 时用 tanh 压成有界），0 表示不启用 | 标量 float |
| `sinks` | 可选的 sink token logits，Stage2 合并时加在分母 | `[num_heads]` 或 None |
| `xai_temperature_len` | xAI temperature 调度用长度，>0 时启用 | 标量 int |

## 2. Stage1 / Stage2 内部 kernel 的 shape 约定

- **Stage1**（`_fwd_kernel_stage1` / `_fwd_grouped_kernel_stage1`）：  
  - grid 为 `(batch, head_num, MAX_KV_SPLITS)` 或 grouped 下的 `(batch, cdiv(head_num, BLOCK_H), MAX_KV_SPLITS)`。  
  - 每个 program 读当前 batch、当前 head（或一组 head）的 `q`，以及该 split 内由 `kv_indices` 指向的 K/V 块；写回 `Att_Out` 的一段（形状与 BLOCK_DV 相关）和 `Att_Lse` 的一个标量。
- **Stage2**（`_fwd_kernel_stage2`）：  
  - grid 为 `(batch, head_num)`。  
  - 每个 program 读该 (batch, head) 下所有 split 的 `Mid_O`（即 attn_logits）和 `Mid_O_1`（即 attn_lse），做数值稳定的 merge 后写 `O` 的一行。  
- **BLOCK 常量**：`BLOCK_DMODEL` / `BLOCK_DV` / `BLOCK_N` / `MIN_BLOCK_KV` 等控制 K/V 维与序列维的分块大小，影响 load/store 的 shape 与 mask。


# 三、中间变量的 shape 流动（含 BLOCK / stride / program_id）

## 1. program_id 与 grid 划分

Stage1 的 grid 为三维：

- `cur_batch = tl.program_id(0)`：batch 维。
- `cur_head = tl.program_id(1)`（或 grouped 下的 `cur_head_id` + 组内 head 索引）：head 维。
- `split_kv_id = tl.program_id(2)`：KV 序列被划分成的 split 维。

每个 program 负责一个 (batch, head, split) 的「局部注意力」：只在该 split 对应的 KV 区间上做 Q·K^T、softmax、加权和，得到局部 Att_Out 和 Att_Lse。

Stage2 的 grid 为二维：

- `cur_batch = tl.program_id(0)`，`cur_head = tl.program_id(1)`。  
- 每个 program 负责一个 (batch, head)，读所有 split 的 Att_Out / Att_Lse，合并后写 `O[cur_batch, cur_head, :]`。

## 2. KV 区间与 stride 寻址

- `cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)`，`cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx`：得到当前 batch 的 KV 逻辑长度。
- `kv_len_per_split` 由 `cur_batch_seq_len`、`kv_splits`、`MIN_BLOCK_KV` 向上对齐得到；`split_kv_start` / `split_kv_end` 为该 split 在逻辑序列上的区间。
- K/V 从 buffer 中按「page 内位置」取出：  
  - `kv_loc = tl.load(kv_indices + cur_batch_kv_start_idx + offs_n, ...)` 得到当前 block 内各位置对应的 buffer 索引。  
  - K 的地址形如：`kv_loc[:, None] * stride_buf_kbs + cur_kv_head * stride_buf_kh + offs_d[None, :]`，即按 (page_idx, kv_head, d) 三维索引。  
- 地址计算满足：  
  - $\mathrm{addr}(b, h, d) = \mathrm{base} + b \cdot \mathrm{stride}_b + h \cdot \mathrm{stride}_h + d \cdot \mathrm{stride}_d$。  
  - 其中 stride 由 `stride_buf_kbs` / `stride_buf_kh` 等传入 kernel。

## 3. 在线 softmax 与局部 Att_Out / Att_Lse

- 在 Stage1 的 KV 循环中维护标量（或 per-head 向量）`e_max`、`e_sum` 和向量 `acc`（长度 BLOCK_DV）。  
- 每加载一块 K、V 后：  
  - 计算 `qk = q·k`（或 `tl.dot(q, k)`），乘 `sm_scale`，可选 `logit_cap * tanh(qk / logit_cap)` 和 xAI temperature。  
  - 用 `n_e_max = max(max(qk), e_max)`，`re_scale = exp(e_max - n_e_max)` 做数值稳定的更新：  
    - `acc = acc * re_scale + sum(exp(qk - n_e_max) * v)`，  
    - `e_sum = e_sum * re_scale + sum(exp(qk - n_e_max))`，  
    - `e_max = n_e_max`。  
- 循环结束后该 split 的「加权和」为 `acc / e_sum`，log-sum-exp 为 `e_max + log(e_sum)`；写入 `Att_Out` 和 `Att_Lse` 的对应位置。

## 4. Stage2 的合并与写 O

- 对 `split_kv_id in range(MAX_KV_SPLITS)` 循环加载 `Mid_O[..., split_kv_id, :]` 和 `Mid_O_1[..., split_kv_id]`。  
- 用同样的 online 合并公式（`e_max`、`e_sum`、`acc`）把各 split 的 (value_part, lse) 合并成一个 softmax 分母与分子。  
- 若 `HAS_SINK`，把 `sink_ptr` 对应的 logit 用 `exp(sink - e_max)` 加到 `e_sum`。  
- 最终 `O = acc / e_sum`，按 `stride_obs` / `stride_oh` 写回。


# 四、关键算子：含义与用法

## 1. `tl.program_id`

- **语义**：当前 kernel 实例在某一维上的 id。  
- **本模块**：Stage1 用 `program_id(0/1/2)` 对应 (batch, head, split)；Stage2 用 `program_id(0/1)` 对应 (batch, head)。

## 2. `tl.load` / `tl.store` 与 mask

- **语义**：按指针 + 偏移加载/写回，`mask` 为 True 的位置有效，否则用 `other` 或忽略。  
- **本模块**：用于从 `Q`、`K_Buffer`、`V_Buffer` 按 `kv_indices` 取数，以及写 `Att_Out`、`Att_Lse`、`O`；mask 由 `offs_n < split_kv_end`、`mask_d`、`mask_dv` 等组合得到。

## 3. `tl.dot` 与标量 q·k

- **语义**：矩阵乘或向量内积。  
- **本模块**：Grouped 路径中 `qk = tl.dot(q, k)`（q 为 [BLOCK_H, Lk]，k 为 [Lk, BLOCK_N]）；Normal 路径用 `tl.sum(q[None, :] * k, 1)` 得到每列的 q·k。

## 4. 在线 softmax（e_max, re_scale, e_sum, acc）

- **语义**：不先存全部 logits，而是逐块用「当前块最大值」与上一轮的 e_max 做 rescale，累加分子（acc）和分母（e_sum），避免溢出。  
- **本模块**：Stage1 和 Stage2 都用 `n_e_max = max(..., e_max)`，`re_scale = exp(e_max - n_e_max)`，然后更新 `acc`、`e_sum`、`e_max`。

## 5. `tanh`（logit cap）

- **语义**：对 logits 做有界变换，$y = c \cdot \tanh(x/c)$，$c = \mathrm{logit\_cap}$。  
- **本模块**：`logit_cap > 0` 时在 Stage1 对 `qk` 做该变换后再做 softmax。

## 6. `tl.cdiv`

- **语义**：向上取整除法。  
- **本模块**：用于计算 `kv_len_per_split`、grid 大小等。


# 五、函数输出：含义与 shape

- **`o`**：形状 `[batch, num_heads, Lv]`。  
  - 含义：当前 decode 步、每个 head 的注意力输出向量；即对「当前步 query 与整段 KV 做 scaled dot-product attention」的结果。  
- **`attn_logits` / `attn_lse`**：为中间结果，由 Stage1 写入、Stage2 只读；调用方通常不再单独使用，仅用于多 split 的数值稳定合并。  
- 无额外返回值；`decode_attention_fwd` 为 in-place 或通过传入的 `o` 写入结果。


# 六、整体功能总结

`decode_attention_fwd` 在 **decode 阶段** 完成单步、多 batch、多 head 的 memory-efficient attention：  
- 输入为当前步的 `q` 和已按 page 组织的 `k_buffer` / `v_buffer`，以及描述「每个 batch 的 KV 逻辑长度与在 buffer 中位置」的 `kv_indptr`、`kv_indices`。  
- 通过 **Stage1** 把 KV 序列按 split 分块，在每个 (batch, head, split) 上做局部 Q·K^T、可选 logit cap / xAI temperature、在线 softmax，得到局部加权和与 log-sum-exp 并写入中间 buffer。  
- 通过 **Stage2** 按 (batch, head) 合并各 split 的中间结果，得到数值稳定的全局 softmax 分母与分子，可选加入 sink token，最后写出 `o`。  
- 支持 **MHA**（单 head 标量/向量实现）与 **GQA/MQA/MLA**（多 head 用 `tl.dot` 的 grouped 实现），在保证数值稳定与 paged KV 兼容的前提下，减少 decode 时单步注意力的显存与带宽占用。
