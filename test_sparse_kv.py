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

    # 方法2: 分块 gather + SDPA
    chunk_size = 2
    attn_output_chunks = []

    for chunk_start in range(0, S_q, chunk_size):
        chunk_end = min(chunk_start + chunk_size, S_q)
        chunk_len = chunk_end - chunk_start

        q_chunk = query[:, :, chunk_start:chunk_end]
        topk_chunk = topk_indices[:, chunk_start:chunk_end]

        sparse_idx = topk_chunk.unsqueeze(1).expand(-1, N_kv, -1, -1)
        sparse_idx_flat = sparse_idx.reshape(B, N_kv, -1)

        sparse_k = torch.gather(
            key, 2, sparse_idx_flat.unsqueeze(-1).expand(-1, -1, -1, D)
        )
        sparse_v = torch.gather(
            value, 2, sparse_idx_flat.unsqueeze(-1).expand(-1, -1, -1, D)
        )

        sparse_k = repeat_kv(sparse_k, N_heads // N_kv)
        sparse_v = repeat_kv(sparse_v, N_heads // N_kv)

        sparse_k = sparse_k.reshape(B, N_heads, chunk_len, topk, D)
        sparse_v = sparse_v.reshape(B, N_heads, chunk_len, topk, D)

        q_exp = q_chunk.transpose(1, 2).reshape(B * chunk_len, N_heads, 1, D)
        k_exp = sparse_k.permute(0, 2, 1, 3, 4).reshape(B * chunk_len, N_heads, topk, D)
        v_exp = sparse_v.permute(0, 2, 1, 3, 4).reshape(B * chunk_len, N_heads, topk, D)

        out = F.scaled_dot_product_attention(
            q_exp, k_exp, v_exp,
            attn_mask=None, dropout_p=0.0, scale=scaling,
        )

        out = out.reshape(B, chunk_len, N_heads, D).transpose(1, 2)
        attn_output_chunks.append(out)

    output_chunked = torch.cat(attn_output_chunks, dim=2)

    diff = (output_full - output_chunked).abs().max().item()
    print(f"Max difference (eager mask vs chunked sparse+SDPA): {diff}")
    if diff < 1e-4:
        print("Correctness test PASSED!")
    else:
        print("Correctness test FAILED!")

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

    chunk_size = 2
    attn_output_chunks = []

    for chunk_start in range(0, S_q, chunk_size):
        chunk_end = min(chunk_start + chunk_size, S_q)
        chunk_len = chunk_end - chunk_start

        q_chunk = query[:, :, chunk_start:chunk_end]
        topk_chunk = topk_indices[:, chunk_start:chunk_end]

        sparse_idx = topk_chunk.unsqueeze(1).expand(-1, N_kv, -1, -1)
        sparse_idx_flat = sparse_idx.reshape(B, N_kv, -1)

        sparse_k = torch.gather(
            key, 2, sparse_idx_flat.unsqueeze(-1).expand(-1, -1, -1, D)
        )
        sparse_v = torch.gather(
            value, 2, sparse_idx_flat.unsqueeze(-1).expand(-1, -1, -1, D)
        )

        sparse_k = repeat_kv(sparse_k, N_heads // N_kv)
        sparse_v = repeat_kv(sparse_v, N_heads // N_kv)

        sparse_k = sparse_k.reshape(B, N_heads, chunk_len, topk, D)
        sparse_v = sparse_v.reshape(B, N_heads, chunk_len, topk, D)

        q_exp = q_chunk.transpose(1, 2).reshape(B * chunk_len, N_heads, 1, D)
        k_exp = sparse_k.permute(0, 2, 1, 3, 4).reshape(B * chunk_len, N_heads, topk, D)
        v_exp = sparse_v.permute(0, 2, 1, 3, 4).reshape(B * chunk_len, N_heads, topk, D)

        out = F.scaled_dot_product_attention(
            q_exp, k_exp, v_exp,
            attn_mask=None, dropout_p=0.0, scale=scaling,
        )

        out = out.reshape(B, chunk_len, N_heads, D).transpose(1, 2)
        attn_output_chunks.append(out)

    output = torch.cat(attn_output_chunks, dim=2)
    loss = output.sum()
    loss.backward()

    print(f"query.grad: {query.grad.shape}, non-zero: {(query.grad.abs() > 0).sum().item()}/{query.grad.numel()}")
    print(f"key.grad:   {key.grad.shape}, non-zero: {(key.grad.abs() > 0).sum().item()}/{key.grad.numel()}")
    print(f"value.grad: {value.grad.shape}, non-zero: {(value.grad.abs() > 0).sum().item()}/{value.grad.numel()}")

    if query.grad is not None and key.grad is not None and value.grad is not None:
        print("Backward test PASSED!")
    else:
        print("Backward test FAILED!")

if __name__ == "__main__":
    print("=== Correctness Test (Chunked) ===")
    test_correctness()
    print()
    print("=== Backward Test (Chunked) ===")
    test_backward()
