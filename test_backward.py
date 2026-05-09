import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import repeat_kv

def test_backward():
    torch.manual_seed(42)
    B, S_q, S_k, N_heads, N_kv, D, topk = 2, 4, 32, 2, 1, 16, 8

    query = torch.randn(B, N_heads, S_q, D, requires_grad=True)
    key = torch.randn(B, N_kv, S_k, D, requires_grad=True)
    value = torch.randn(B, N_kv, S_k, D, requires_grad=True)
    scaling = D ** -0.5

    topk_indices = torch.zeros(B, S_q, topk, dtype=torch.long)
    for b in range(B):
        for s in range(S_q):
            topk_indices[b, s] = torch.randperm(S_k)[:topk]

    # 稀疏 gather + SDPA
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

    query_expanded = query.transpose(1, 2).reshape(B * S_q, N_heads, 1, D)
    key_expanded = sparse_key.permute(0, 2, 1, 3, 4).reshape(B * S_q, N_heads, topk, D)
    value_expanded = sparse_value.permute(0, 2, 1, 3, 4).reshape(B * S_q, N_heads, topk, D)

    attn_output = F.scaled_dot_product_attention(
        query_expanded, key_expanded, value_expanded,
        attn_mask=None, dropout_p=0.0, scale=scaling,
    )

    attn_output = attn_output.reshape(B, S_q, N_heads, D).transpose(1, 2)

    # 模拟 loss
    loss = attn_output.sum()
    loss.backward()

    print(f"query.grad: {query.grad.shape}, non-zero: {(query.grad.abs() > 0).sum().item()}/{query.grad.numel()}")
    print(f"key.grad:   {key.grad.shape}, non-zero: {(key.grad.abs() > 0).sum().item()}/{key.grad.numel()}")
    print(f"value.grad: {value.grad.shape}, non-zero: {(value.grad.abs() > 0).sum().item()}/{value.grad.numel()}")
    print(f"key.grad max: {key.grad.abs().max().item():.6f}")
    print(f"value.grad max: {value.grad.abs().max().item():.6f}")

    if query.grad is not None and key.grad is not None and value.grad is not None:
        print("\nBackward test PASSED! All gradients propagated.")
    else:
        print("\nBackward test FAILED!")

if __name__ == "__main__":
    test_backward()
