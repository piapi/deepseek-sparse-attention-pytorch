import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn.functional as F
from models.configuration_qwen3_dsa import Qwen3DSAConfig
from models.modeling_qwen3_dsa import Qwen3DSAForCausalLM
from transformers.models.qwen3.modeling_qwen3 import repeat_kv

B, S_q, N_heads, N_kv, D, topk = 2, 8, 4, 2, 32, 16

config = Qwen3DSAConfig(
    vocab_size=1024,
    hidden_size=N_heads * D,
    intermediate_size=512,
    num_hidden_layers=1,
    num_attention_heads=N_heads,
    num_key_value_heads=N_kv,
    head_dim=D,
    max_position_embeddings=64,
    index_n_heads=N_heads,
    index_head_dim=D,
    index_topk=topk,
)

model = Qwen3DSAForCausalLM(config)
model.eval()

input_ids = torch.randint(0, 1024, (B, S_q))

# 手动跑 attention forward
with torch.no_grad():
    hidden = model.model.embed_tokens(input_ids)
    print(f"hidden shape: {hidden.shape}")

    layer = model.model.layers[0]
    attn = layer.self_attn

    # 模拟 attention forward
    bsz = hidden.shape[0]
    input_shape = hidden.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)

    query_states = attn.q_norm(attn.q_proj(hidden).view(hidden_shape)).transpose(1, 2)
    key_states = attn.k_norm(attn.k_proj(hidden).view(hidden_shape)).transpose(1, 2)
    value_states = attn.v_proj(hidden).view(hidden_shape).transpose(1, 2)

    print(f"Q: {query_states.shape}, K: {key_states.shape}, V: {value_states.shape}")

    seqlen_q = input_shape[-1]
    seqlen_k = key_states.shape[-2]
    n_kv = key_states.shape[1]
    n_heads = query_states.shape[1]
    head_dim = key_states.shape[-1]

    # 模拟 indexer 输出
    actual_topk = min(topk, seqlen_k)
    topk_indices = torch.randint(0, seqlen_k, (bsz, seqlen_q, actual_topk))
    print(f"topk_indices: {topk_indices.shape}, actual_topk={actual_topk}")

    # gather
    sparse_indices_kv = topk_indices.unsqueeze(1).expand(-1, n_kv, -1, -1)
    sparse_indices_kv_flat = sparse_indices_kv.reshape(bsz, n_kv, -1)

    sparse_key = torch.gather(
        key_states, 2, sparse_indices_kv_flat.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    )
    sparse_value = torch.gather(
        value_states, 2, sparse_indices_kv_flat.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    )

    print(f"sparse_key after gather: {sparse_key.shape}")

    sparse_key = repeat_kv(sparse_key, attn.num_key_value_groups)
    sparse_value = repeat_kv(sparse_value, attn.num_key_value_groups)

    print(f"sparse_key after repeat_kv: {sparse_key.shape}")

    sparse_query = query_states.unsqueeze(3).expand(-1, -1, -1, actual_topk, -1).reshape(
        bsz, n_heads, -1, head_dim
    )
    print(f"sparse_query: {sparse_query.shape}")

    sparse_query_grouped = sparse_query.reshape(bsz, n_heads, seqlen_q, actual_topk, head_dim)
    sparse_key_grouped = sparse_key.reshape(bsz, n_heads, seqlen_q, actual_topk, head_dim)
    sparse_value_grouped = sparse_value.reshape(bsz, n_heads, seqlen_q, actual_topk, head_dim)

    print(f"sparse_query_grouped: {sparse_query_grouped.shape}")
    print(f"sparse_key_grouped: {sparse_key_grouped.shape}")

    q = sparse_query_grouped.unsqueeze(3)
    k = sparse_key_grouped.transpose(-2, -1)
    print(f"q for matmul: {q.shape}")
    print(f"k for matmul: {k.shape}")

    attn_weights_sparse = torch.matmul(q, k) * attn.scaling
    print(f"attn_weights_sparse: {attn_weights_sparse.shape}")

    attn_weights_sparse = F.softmax(attn_weights_sparse, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights_sparse, sparse_value_grouped)
    print(f"attn_output: {attn_output.shape}")

    attn_output = attn_output.squeeze(3)
    print(f"attn_output after squeeze: {attn_output.shape}")
    print("SUCCESS!")
