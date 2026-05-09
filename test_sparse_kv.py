import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import repeat_kv

def test_correctness():
    torch.manual_seed(42)
    B, S_q, S_k, N_heads, N_kv, D, topk = 1, 4, 32, 2, 1, 16, 8

    query = torch.randn(B, N_heads, S_q, D)
    key = torch.randn(B, N_kv, S_k, D)
    value = torch.randn(B, N_kv, S_k, D)
    scaling = D ** -0.5

    topk_indices = torch.zeros(B, S_q, topk, dtype=torch.long)
    for b in range(B):
        for s in range(S_q):
            topk_indices[b, s] = torch.randperm(S_k)[:topk]

    # 方法1: eager mask
    mask = torch.full((B, 1, S_q, S_k), float("-inf"))
    scatter_src = torch.zeros(topk_indices.shape)
    mask.scatter_(-1, topk_indices.unsqueeze(1), scatter_src.unsqueeze(1))

    key_full = repeat_kv(key, N_heads // N_kv)
    value_full = repeat_kv(value, N_heads // N_kv)

    attn_weights_full = torch.matmul(query, key_full.transpose(-2, -1)) * scaling
    attn_weights_full = attn_weights_full + mask
    attn_weights_full = F.softmax(attn_weights_full, dim=-1, dtype=torch.float32).to(query.dtype)
    output_full = torch.matmul(attn_weights_full, value_full)

    # 方法2: 稀疏 KV gather
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

    sparse_key_grouped = sparse_key.reshape(B, N_heads, S_q, topk, D)
    sparse_value_grouped = sparse_value.reshape(B, N_heads, S_q, topk, D)

    attn_weights_sparse = torch.matmul(
        query.unsqueeze(3), sparse_key_grouped.transpose(-2, -1)
    ) * scaling
    attn_weights_sparse = F.softmax(attn_weights_sparse, dim=-1, dtype=torch.float32).to(query.dtype)

    output_sparse = torch.matmul(attn_weights_sparse, sparse_value_grouped).squeeze(3)

    diff = (output_full - output_sparse).abs().max().item()
    print(f"Max difference: {diff}")
    if diff < 1e-5:
        print("Correctness test PASSED!")
    else:
        print("Correctness test FAILED!")

def test_full_model():
    B, S_q, N_heads, N_kv, D, topk = 2, 8, 4, 2, 32, 16

    config_dict = {
        "vocab_size": 1024,
        "hidden_size": N_heads * D,
        "intermediate_size": 512,
        "num_hidden_layers": 1,
        "num_attention_heads": N_heads,
        "num_key_value_heads": N_kv,
        "head_dim": D,
        "max_position_embeddings": 64,
        "use_sparse_indexer": False,
        "index_n_heads": N_heads,
        "index_head_dim": D,
        "index_topk": topk,
    }

    from models.configuration_qwen3_dsa import Qwen3DSAConfig
    from models.modeling_qwen3_dsa import Qwen3DSAForCausalLM

    config = Qwen3DSAConfig(**config_dict)
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
