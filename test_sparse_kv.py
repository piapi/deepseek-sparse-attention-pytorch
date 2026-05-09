import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import repeat_kv

def test_correctness():
    torch.manual_seed(42)
    B, S_q, S_k, N_heads, N_kv, D, topk = 2, 4, 32, 2, 1, 16, 8

    query = torch.randn(B, N_heads, S_q, D)
    key = torch.randn(B, N_kv, S_k, D)
    value = torch.randn(B, N_kv, S_k, D)
    scaling = D ** -0.5

    topk_indices = torch.zeros(B, S_q, topk, dtype=torch.long)
    for b in range(B):
        for s in range(S_q):
            topk_indices[b, s] = torch.randperm(S_k)[:topk]

    # 方法1: eager mask (ground truth)
    mask = torch.full((B, 1, S_q, S_k), float("-inf"))
    scatter_src = torch.zeros(topk_indices.shape)
    mask.scatter_(-1, topk_indices.unsqueeze(1), scatter_src.unsqueeze(1))

    key_full = repeat_kv(key, N_heads // N_kv)
    value_full = repeat_kv(value, N_heads // N_kv)

    attn_weights_full = torch.matmul(query, key_full.transpose(-2, -1)) * scaling
    attn_weights_full = attn_weights_full + mask
    attn_weights_full = F.softmax(attn_weights_full, dim=-1, dtype=torch.float32).to(query.dtype)
    output_full = torch.matmul(attn_weights_full, value_full)

    # 方法2: 稀疏 gather + SDPA (模拟 FA2 方案)
    sparse_indices_kv = topk_indices.unsqueeze(1).expand(-1, N_kv, -1, -1)
    sparse_indices_kv_flat = sparse_indices_kv.reshape(B, N_kv, -1)

    sparse_key = torch.gather(
        key, 2, sparse_indices_kv_flat.unsqueeze(-1).expand(-1, -1, -1, D)
    )
    sparse_value = torch.gather(
        value, 2, sparse_indices_kv_flat.unsqueeze(-1).expand(-1, -1, -1, D)
    )

    sparse_key = repeat_kv(sparse_key, N_heads // N_kv)
    sparse_value = repeat_kv(sparse_value, N_heads // N_kv)

    sparse_key = sparse_key.reshape(B, N_heads, S_q, topk, D)
    sparse_value = sparse_value.reshape(B, N_heads, S_q, topk, D)

    # 拆成独立 batch
    query_expanded = query.transpose(1, 2).reshape(B * S_q, N_heads, 1, D)
    key_expanded = sparse_key.permute(0, 2, 1, 3, 4).reshape(B * S_q, N_heads, topk, D)
    value_expanded = sparse_value.permute(0, 2, 1, 3, 4).reshape(B * S_q, N_heads, topk, D)

    # SDPA
    attn_output = F.scaled_dot_product_attention(
        query_expanded, key_expanded, value_expanded,
        attn_mask=None, dropout_p=0.0, scale=scaling,
    )

    # [B*S_q, N_heads, 1, D] -> [B, N_heads, S_q, D]
    attn_output = attn_output.reshape(B, S_q, N_heads, D).transpose(1, 2)

    diff = (output_full - attn_output).abs().max().item()
    print(f"Max difference (eager mask vs sparse+SDPA): {diff}")
    if diff < 1e-4:
        print("Correctness test PASSED!")
    else:
        print("Correctness test FAILED!")

def test_full_model():
    B, S_q, N_heads, N_kv, D, topk = 2, 8, 4, 2, 32, 16

    from models.configuration_qwen3_dsa import Qwen3DSAConfig
    from models.modeling_qwen3_dsa import Qwen3DSAForCausalLM

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

    with torch.no_grad():
        output = model(input_ids)

    print(f"Output shape: {output.logits.shape}")
    print("Full model test PASSED!")

if __name__ == "__main__":
    print("=== Correctness Test ===")
    test_correctness()
    print()
    print("=== Full Model Test ===")
    test_full_model()
