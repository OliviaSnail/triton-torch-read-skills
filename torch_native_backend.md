# 一、代码概览

`TorchNativeAttnBackend` 是一个基于 **PyTorch 原生 `scaled_dot_product_attention` (SDPA)** 的注意力后端实现，继承自 `AttentionBackend`。它主要负责在 **prefill/extend 阶段** 和 **decode 阶段**：

- 从全局 KV cache 中按请求序列（`req_to_token`, `req_pool_indices`, `seq_lens`）抽取出每个序列的 K/V；
- 将多请求打平后的 `q` / `o` 重新按「请求粒度」切分，调用 SDPA 完成注意力计算；
- 支持 GQA（`enable_gqa`）、可选 causal mask、以及 cross-attention / encoder-only 两种非因果场景；
- 将结果按「打平前」的 token 顺序写回输出张量。

与 Triton 后端不同，这个类完全依赖 PyTorch 的实现（`torch.nn.functional.scaled_dot_product_attention`），不支持 Triton 内核加速，因此 `support_triton()` 返回 `False`。


# 二、函数输入：含义与 shape

## 1. 类构造与元数据

```python
class TorchNativeAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device
```

- `model_runner`：上层执行器，提供当前设备信息等；这里记录 `device`，便于后续张量创建。
- `forward_metadata`：预留的前向元数据缓存，目前未使用（`init_forward_metadata` 是空实现）。

## 2. extend 阶段内部 kernel：`_run_sdpa_forward_extend`

```python
def _run_sdpa_forward_extend(
    self,
    query: torch.Tensor,          # [num_tokens, num_heads, head_size]
    output: torch.Tensor,         # [num_tokens, num_heads, head_size]
    k_cache: torch.Tensor,        # [max_total_num_tokens, num_heads, head_size]
    v_cache: torch.Tensor,        # [max_total_num_tokens, num_heads, head_size]
    req_to_token: torch.Tensor,   # [max_num_reqs, max_context_len]
    req_pool_indices: torch.Tensor,  # [num_seqs]
    seq_lens: torch.Tensor,          # [num_seqs]
    extend_prefix_lens: torch.Tensor, # [num_seqs]
    extend_seq_lens: torch.Tensor,   # [num_seqs]
    scaling=None,
    enable_gqa=False,
    causal=False,
):
    ...
```

- `query` / `output`：
  - flatten 后所有 token 的 Q / O，形状都是 `[num_tokens, num_heads, head_size]`；
  - 内部会 `movedim` 成 `[num_heads, num_tokens, head_size]` 以满足 SDPA 要求。
- `k_cache` / `v_cache`：
  - 全局 KV cache 缓冲区，第一维是「token 在全局 KV 池中的 index」，后两维是 `[num_heads, head_size]`。
- `req_to_token` / `req_pool_indices` / `seq_lens`：
  - `req_to_token[req_pool_idx, :seq_len_kv]` 给出该请求在 KV cache 中的 token 索引序列；
  - `seq_lens[seq_idx]` 是该请求当前 context 长度（包括 prefix + 本次 extend）。
- `extend_prefix_lens` / `extend_seq_lens`：
  - `extend_prefix_lens[seq_idx]`：本次 extend 前，该请求已有多少 prefix（prefill 部分）；
  - `extend_seq_lens[seq_idx]`：本次 extend 新增 token 长度（即这次要算注意力的 Q 长度）。
- `scaling`：注意力缩放因子，传给 `scale` 参数；
- `enable_gqa`：是否启用 GQA（传给 SDPA 的 `enable_gqa`）；
- `causal`：是否使用因果 mask（传给 SDPA 的 `is_causal`）。

## 3. decode 阶段内部 kernel：`_run_sdpa_forward_decode`

```python
def _run_sdpa_forward_decode(
    self,
    query: torch.Tensor,          # [num_tokens, num_heads, head_size]
    output: torch.Tensor,         # [num_tokens, num_heads, head_size]
    k_cache: torch.Tensor,        # [max_total_num_tokens, num_heads, head_size]
    v_cache: torch.Tensor,        # [max_total_num_tokens, num_heads, head_size]
    req_to_token: torch.Tensor,   # [max_num_reqs, max_context_len]
    req_pool_indices: torch.Tensor, # [num_seqs]
    seq_lens: torch.Tensor,       # [num_seqs]
    scaling=None,
    enable_gqa=False,
    causal=False,
):
    ...
```

- 与 extend 版基本相同，区别在于 decode 阶段 **每个序列本次只解一个 token**：
  - `seq_len_q` 固定为 1；
  - `seq_len_kv = seq_lens[seq_idx]` 仍是完整 context 长度。

## 4. 面向上层接口：`forward_extend` / `forward_decode`

```python
def forward_extend(
    self,
    q, k, v,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    save_kv_cache=True,
):
    ...

def forward_decode(
    self,
    q, k, v,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    save_kv_cache=True,
):
    ...
```

- `q, k, v`：来自上层 RadixAttention 的 Q/K/V，shape 通常是 `[num_tokens, tp_q_head_num * head_dim]` 或与之兼容；
- `layer`：`RadixAttention` 层实例，提供：
  - `tp_q_head_num`, `tp_k_head_num`, `qk_head_dim`, `v_head_dim`；
  - `is_cross_attention`, `attn_type`, `layer_id`, `scaling` 等；
- `forward_batch`：本次 batch 的调度元信息与 buffer 句柄：
  - `token_to_kv_pool`：用来读写 KV cache；
  - `req_to_token_pool.req_to_token` / `req_pool_indices` / `seq_lens` 等；
  - `extend_prefix_lens`, `extend_seq_lens`, `out_cache_loc`, `encoder_out_cache_loc` 等 cache 位置索引。
- `save_kv_cache`：是否将本次的 K/V 写入 KV cache（通常 prefill/extend/decode 都需要）。


# 三、中间变量的 shape 流动

虽然这里不是 Triton kernel，但依然有清晰的「批次打平 → 按请求拆分」的维度流动过程。

## 1. 打平的 Q / O reshape 成 head 维度

在 `forward_extend` 中：

```python
q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)  # [num_tokens, num_heads, head_dim]
o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)   # [num_tokens, num_heads, head_dim]
```

随后在内部 kernel 中统一：

```python
# [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
query = query.movedim(0, query.dim() - 2)
```

因此内部 SDPA 看到的形状是：

- `query`：`[num_heads, num_tokens, head_dim]`；
- 每个请求的子块：`[num_heads, seq_len_q, head_dim]`。

## 2. 按序列循环：每次处理一个 request

两种场景共通的循环骨架：

```python
start_q, start_kv = 0, 0
for seq_idx in range(seq_lens.shape[0]):
    # seq_len_q: extend 用 extend_seq_lens / prefill + extend；decode 固定 1
    # seq_len_kv: 该序列在 KV cache 中的总长度
    end_q = start_q + seq_len_q
    end_kv = start_kv + seq_len_kv

    per_req_query = query[:, start_q:end_q, :]  # [num_heads, seq_len_q, head_dim]
    ...
    start_q, start_kv = end_q, end_kv
```

- `start_q` / `end_q`：在「打平的所有 token」维度上，为当前请求分配的 Q 子区间；
- `start_kv` / `end_kv`：用于跟踪 KV 侧的逻辑 offset（当前实现中 KV 实际通过 `req_to_token` 直接索引）。

## 3. 从 KV cache 按请求抽取 K / V

```python
req_pool_idx = req_pool_indices[seq_idx]
per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]      # [seq_len_kv]
per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)
```

- `per_req_tokens`：当前请求的 token 在全局 KV pool 中的 index 序列；
- `k_cache[per_req_tokens]`：形状 `[seq_len_kv, num_heads, head_dim]`；
- `.movedim(0, query.dim() - 2)` 后变为 `[num_heads, seq_len_kv, head_dim]`，与 `per_req_query` 的 head 维对齐。

## 4. extend 阶段的「冗余 query」处理

extend 中的核心逻辑是：

- 本轮 extend 的 Q 长度为 `extend_seq_len_q = extend_seq_lens[seq_idx]`；
- 但为了让 SDPA 的 `query/key/value` 都是 `[num_heads, seq_len_kv, head_dim]` 的形式，需要构造一个长度为 `seq_len_kv` 的「冗余 query」张量。

代码片段如下：

```python
per_req_query = query[:, start_q:end_q, :]  # [num_heads, extend_seq_len_q, head_dim]
per_req_query_redudant = torch.empty(
    (per_req_query.shape[0], seq_len_kv, per_req_query.shape[2]),
    dtype=per_req_query.dtype,
    device=per_req_query.device,
)

prefill_seq_len_q = extend_prefix_lens[seq_idx]
per_req_query_redudant[:, prefill_seq_len_q:, :] = per_req_query
```

- `prefill_seq_len_q = extend_prefix_lens[seq_idx]`：该请求在本次 extend 之前已有的 prefix 长度；
- `seq_len_kv = seq_lens[seq_idx] = prefill_seq_len_q + extend_seq_len_q`；
- `per_req_query_redudant` 形状为 `[num_heads, seq_len_kv, head_dim]`，前 `prefill_seq_len_q` 个位置仅作占位，后半段是真正此次 extend 的 query。

随后调用 SDPA 并回填输出：

```python
per_req_out_redudant = (
    scaled_dot_product_attention(
        per_req_query_redudant.unsqueeze(0),  # [1, num_heads, seq_len_kv, head_dim]
        per_req_key.unsqueeze(0),             # [1, num_heads, seq_len_kv, head_dim]
        per_req_value.unsqueeze(0),           # [1, num_heads, seq_len_kv, head_dim]
        enable_gqa=enable_gqa,
        scale=scaling,
        is_causal=causal,
    )
    .squeeze(0)
    .movedim(query.dim() - 2, 0)
)

output[start_q:end_q, :, :] = per_req_out_redudant[prefill_seq_len_q:, :, :]
```

- `per_req_out_redudant` 的形状与 `per_req_query_redudant` 相同（忽略 batch 维）为 `[num_heads, seq_len_kv, head_dim]`；
- 取后半段 `[prefill_seq_len_q:, :, :]` 对应本轮 extend 的新 tokens 输出，并写回打平后的 `output[start_q:end_q]`。

## 5. decode 阶段的更简单形状流

decode 阶段每个请求本轮只解一个 token，逻辑更简单：

```python
seq_len_q = 1
seq_len_kv = seq_lens[seq_idx]
per_req_query = query[:, start_q:end_q, :]  # [num_heads, 1, head_dim]

per_req_out = (
    scaled_dot_product_attention(
        per_req_query.unsqueeze(0),  # [1, num_heads, 1, head_dim]
        per_req_key.unsqueeze(0),    # [1, num_heads, seq_len_kv, head_dim]
        per_req_value.unsqueeze(0),  # [1, num_heads, seq_len_kv, head_dim]
        enable_gqa=enable_gqa,
        scale=scaling,
        is_causal=causal,
    )
    .squeeze(0)
    .movedim(query.dim() - 2, 0)
)

output[start_q:end_q, :, :] = per_req_out
```

- Q 长度固定为 1，因此不需要构造冗余 query，这是标准的「单 token 对整个上下文做注意力」。


# 四、关键算子：含义与用法

## 1. `torch.nn.functional.scaled_dot_product_attention`

- **语义**：
  - 给定 `query`, `key`, `value`，做缩放点积注意力：
    $$\text{Attn}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V$$
  - 在这里额外支持：
    - `scale`：手动传入缩放因子，替代默认的 $1/\sqrt{d_k}$；
    - `is_causal`：自动生成因果 mask，防止「看未来」；
    - `enable_gqa`：支持 GQA 的 head 结构。
- **形状要求**（忽略 batch 维）：
  - `query`：`[num_heads, seq_len_q, head_dim]`；
  - `key` / `value`：`[num_heads, seq_len_kv, head_dim]`。
- **在本类中的调用**：
  - extend：`query/key/value` 都是 `[num_heads, seq_len_kv, head_dim]`，通过冗余 query 实现「prefill + extend」；
  - decode：`query` 为 `[num_heads, 1, head_dim]`，`key/value` 为 `[num_heads, seq_len_kv, head_dim]`。

## 2. `movedim` / `view` / `reshape`

- `movedim(0, query.dim() - 2)`：将原先 `[num_tokens, num_heads, head_dim]` 的 token 维移动到中间位置，得到 `[num_heads, num_tokens, head_dim]`，便于按「先 head 再 seq」维度切分；
- `view(-1, num_heads, head_dim)`：把 `[batch, seq, heads, dim]` 等更高维 flatten 成 `[num_tokens, num_heads, head_dim]`，统一进入 backend。

## 3. dtype 对齐

```python
if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
    per_req_key = per_req_key.to(per_req_query.dtype)
    per_req_value = per_req_value.to(per_req_query.dtype)
```

- 确保 Q/K/V 的 dtype 一致，这是 `scaled_dot_product_attention` 的前提条件；
- 避免由于 cache / Q 侧的混合精度（如 bf16 vs fp16）导致运行时报错。


# 五、函数输出：含义与 shape

- **内部 kernel**：
  - `_run_sdpa_forward_extend` / `_run_sdpa_forward_decode` 接收一个预分配好的 `output`（`[num_tokens, num_heads, head_dim]`），在循环中按请求写入子区间，最后返回同一个张量；
- **外部接口**：
  - `forward_extend` / `forward_decode` 最终返回的 `o` 形状通常是：
    - 若 `qk_head_dim == v_head_dim`：与输入 `q` 相同；
    - 否则：`[num_tokens, tp_q_head_num * v_head_dim]`，用于后续线性层或合并。

输出的每一行都对应 **一个打平后的 token**（包括 batch 与 seq 维展平后），其含义是：

- 在给定请求的 KV context 下，对该 token 做完注意力（包括 GQA、可选 causal mask 以及 scaling）后的 head 维输出，再按 head 合并成 `tp_q_head_num * v_head_dim` 的向量。


# 六、整体功能总结

从系统角度看，`TorchNativeAttnBackend` 提供了一个基于 **PyTorch 原生 SDPA** 的注意力后端实现，用于：

1. 在 **extend/prefill 阶段**，对一批请求的「新 tokens」逐序列调用 SDPA，利用冗余 query 对齐 prefix + extend 的长度，并只回填 extend 部分的输出；
2. 在 **decode 阶段**，对每个请求当前的「最后一个 token」调用单步 SDPA，使用全 context 作为 K/V；
3. 通过 `ForwardBatch` 提供的 `req_to_token` 与 `token_to_kv_pool`，将「逻辑上的序列」与「全局 KV cache」做索引级别的桥接；
4. 在无需 Triton / CUDA 自定义 kernel 的环境下，复用 PyTorch 的高性能实现，同时保持与 RadixAttention 其他 backend 一致的接口（`forward_extend` / `forward_decode`）。

在实际部署中，可以将 `TorchNativeAttnBackend` 视为 **功能正确、实现简单但性能相对保守的 fallback 后端**，适合作为调试、开发阶段或不支持 Triton 的环境下的注意力实现。

