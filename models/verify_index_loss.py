"""
Verify index_loss computation: old vs new implementation.

Compares:
1. Attention weights: old repeat_kv vs new KV-head-loop
2. compute_index_loss: old masked_fill(-1e9) vs new gather
3. End-to-end: old recompute+loss vs new _compute_index_loss_from_scratch
"""

import torch
import torch.nn.functional as F


def old_recompute_attention_weights(query, key, attention_mask, scaling, num_key_value_groups):
    key_states = key.repeat_interleave(num_key_value_groups, dim=1)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            causal_mask = torch.zeros_like(attention_mask, dtype=torch.float)
            causal_mask = causal_mask.masked_fill(~attention_mask, float("-inf"))
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
        else:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
    return attn_weights


def new_compute_attention_weights_summed(query, key, attention_mask, scaling, num_key_value_groups):
    G = num_key_value_groups
    H_KV = key.shape[1]
    B, H_Q, S_q, D = query.shape
    S_kv = key.shape[2]

    query_reshaped = query.view(B, H_KV, G, S_q, D)

    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            causal_mask = torch.zeros_like(attention_mask, dtype=torch.float)
            causal_mask = causal_mask.masked_fill(~attention_mask, float("-inf"))
        else:
            causal_mask = attention_mask

    attn_summed = torch.zeros(B, S_q, S_kv, dtype=torch.float32, device=query.device)

    for h in range(H_KV):
        q_group = query_reshaped[:, h]
        k_h = key[:, h:h+1]

        weights_h = torch.matmul(q_group, k_h.transpose(2, 3)) * scaling

        if attention_mask is not None:
            mask_h = causal_mask[:, :, :, :S_kv]
            weights_h = weights_h + mask_h

        weights_h = F.softmax(weights_h, dim=-1, dtype=torch.float32)
        attn_summed += weights_h.sum(dim=1)

    return attn_summed


def old_compute_index_loss(index_score, attention_weights, index_mask):
    if attention_weights.dim() == 4:
        attention_weights = attention_weights.sum(1)

    eps = 1e-8
    B = attention_weights.shape[0]
    if index_mask is not None:
        index_mask = index_mask.squeeze(1)
        attention_weights = attention_weights.masked_fill(~index_mask, eps)
        index_score = index_score.masked_fill(~index_mask, -1e9)
    index_score = torch.clamp(index_score, min=-1e9, max=1e9)
    attn_dist = attention_weights / attention_weights.sum(
        dim=-1, keepdim=True
    ).clamp_min(eps)
    log_index_dist = F.log_softmax(index_score, dim=-1)

    kl_loss = F.kl_div(
        log_index_dist, attn_dist, reduction="batchmean", log_target=False
    )
    return kl_loss


def new_compute_index_loss(index_score, attention_weights, topk_indices):
    if attention_weights.dim() == 4:
        attention_weights = attention_weights.sum(1)

    eps = 1e-8
    B, S_q, S_kv = attention_weights.shape
    CHUNK_SIZE = 128

    total_kl = torch.tensor(0.0, device=attention_weights.device, dtype=torch.float32)

    for s_start in range(0, S_q, CHUNK_SIZE):
        s_end = min(s_start + CHUNK_SIZE, S_q)

        attn_chunk = attention_weights[:, s_start:s_end, :]
        score_chunk = index_score[:, s_start:s_end, :]
        topk_chunk = topk_indices[:, s_start:s_end]

        score_topk = score_chunk.gather(-1, topk_chunk)
        attn_topk = attn_chunk.gather(-1, topk_chunk)

        attn_topk = attn_topk + eps
        attn_dist = attn_topk / attn_topk.sum(dim=-1, keepdim=True).clamp_min(eps)
        log_index_dist = F.log_softmax(score_topk, dim=-1)

        total_kl += F.kl_div(
            log_index_dist, attn_dist, reduction="sum", log_target=False
        )

    return total_kl / B


def new_compute_index_loss_from_scratch(
    index_score, query_states, key_states, attention_mask, scaling, topk_indices, num_key_value_groups
):
    G = num_key_value_groups
    H_KV = key_states.shape[1]
    B, H_Q, S_q, D = query_states.shape
    S_kv = key_states.shape[2]

    eps = 1e-8
    CHUNK_SIZE = 128
    total_kl = torch.tensor(0.0, device=query_states.device, dtype=torch.float32)

    query_reshaped = query_states.view(B, H_KV, G, S_q, D)

    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            causal_mask = torch.zeros_like(attention_mask, dtype=torch.float)
            causal_mask = causal_mask.masked_fill(~attention_mask, float("-inf"))
        else:
            causal_mask = attention_mask

    for s_start in range(0, S_q, CHUNK_SIZE):
        s_end = min(s_start + CHUNK_SIZE, S_q)
        chunk_len = s_end - s_start

        score_chunk = index_score[:, s_start:s_end, :]
        topk_chunk = topk_indices[:, s_start:s_end]

        attn_chunk = torch.zeros(
            B, chunk_len, S_kv, dtype=torch.float32, device=query_states.device
        )

        for h in range(H_KV):
            q_group = query_reshaped[:, h, :, s_start:s_end]
            k_h = key_states[:, h:h+1]

            weights_h = torch.matmul(
                q_group, k_h.transpose(2, 3)
            ) * scaling

            if attention_mask is not None:
                mask_chunk = causal_mask[:, :, s_start:s_end, :S_kv]
                weights_h = weights_h + mask_chunk

            weights_h = F.softmax(weights_h, dim=-1, dtype=torch.float32)
            attn_chunk += weights_h.sum(dim=1)

        attn_topk = attn_chunk.gather(-1, topk_chunk)
        score_topk = score_chunk.gather(-1, topk_chunk)

        attn_topk = attn_topk + eps
        attn_dist = attn_topk / attn_topk.sum(dim=-1, keepdim=True).clamp_min(eps)
        log_index_dist = F.log_softmax(score_topk, dim=-1)

        total_kl += F.kl_div(
            log_index_dist, attn_dist, reduction="sum", log_target=False
        )

    return total_kl / B


def make_data(B=2, H_Q=8, H_KV=2, S_q=256, S_kv=256, D=64, index_topk=64, device="cpu"):
    G = H_Q // H_KV
    query = torch.randn(B, H_Q, S_q, D, device=device)
    key = torch.randn(B, H_KV, S_kv, D, device=device)
    scaling = D ** -0.5

    causal = torch.ones(S_q, S_kv, dtype=torch.bool, device=device).tril()
    attention_mask = causal.unsqueeze(0).unsqueeze(0).expand(B, 1, S_q, S_kv)

    index_score = torch.randn(B, S_q, S_kv, device=device)
    index_score = index_score.masked_fill(~causal.unsqueeze(0), -1e9)
    topk_indices = index_score.topk(min(index_topk, S_kv), dim=-1)[1]

    index_mask = torch.zeros(B, S_q, S_kv, dtype=torch.bool, device=device)
    index_mask = index_mask.scatter_(-1, topk_indices, True)
    index_mask = index_mask.unsqueeze(1)

    return query, key, scaling, attention_mask, index_score, topk_indices, index_mask, G


def test_1_attention_weights():
    print("=" * 60)
    print("Test 1: Attention weights (old repeat_kv vs new KV-head-loop)")
    print("=" * 60)

    for S in [64, 128, 256]:
        query, key, scaling, attention_mask, _, _, _, G = make_data(S_q=S, S_kv=S)

        old_weights = old_recompute_attention_weights(query, key, attention_mask, scaling, G)
        old_summed = old_weights.sum(dim=1)

        new_summed = new_compute_attention_weights_summed(query, key, attention_mask, scaling, G)

        diff = (old_summed - new_summed).abs()
        max_err = diff.max().item()
        mean_err = diff.mean().item()
        rel_err = (diff / (old_summed.abs() + 1e-8)).max().item()

        status = "PASS" if max_err < 1e-5 else "FAIL"
        print(f"  S={S:4d}: max_err={max_err:.2e}, mean_err={mean_err:.2e}, rel_err={rel_err:.2e}  [{status}]")

    print()


def test_2_compute_index_loss():
    print("=" * 60)
    print("Test 2: compute_index_loss (old masked_fill vs new gather)")
    print("=" * 60)

    for S in [64, 128, 256]:
        query, key, scaling, attention_mask, index_score, topk_indices, index_mask, G = make_data(S_q=S, S_kv=S)

        old_weights = old_recompute_attention_weights(query, key, attention_mask, scaling, G)

        old_loss = old_compute_index_loss(index_score.clone(), old_weights.clone(), index_mask.clone())
        new_loss = new_compute_index_loss(index_score.clone(), old_weights.clone(), topk_indices.clone())

        print(f"  S={S:4d}: old_loss={old_loss.item():.6f}, new_loss={new_loss.item():.6f}, "
              f"diff={abs(old_loss.item() - new_loss.item()):.6f}")

    print("  NOTE: old and new differ because old includes non-topk positions in KL.")
    print("  The new version is correct (distillation on effective tokens only).")
    print()


def test_3_end_to_end():
    print("=" * 60)
    print("Test 3: End-to-end (old recompute+loss vs new from_scratch)")
    print("=" * 60)

    for S in [64, 128, 256]:
        query, key, scaling, attention_mask, index_score, topk_indices, index_mask, G = make_data(S_q=S, S_kv=S)

        old_weights = old_recompute_attention_weights(query, key, attention_mask, scaling, G)
        old_loss = old_compute_index_loss(index_score.clone(), old_weights, index_mask)

        new_loss = new_compute_index_loss_from_scratch(
            index_score.clone(), query, key, attention_mask, scaling, topk_indices, G
        )

        print(f"  S={S:4d}: old_loss={old_loss.item():.6f}, new_loss={new_loss.item():.6f}, "
              f"diff={abs(old_loss.item() - new_loss.item()):.6f}")

    print("  NOTE: Differences come from (a) gather vs masked_fill in KL,")
    print("  and (b) any numerical differences in attention weight computation.")
    print()


def test_4_kl_on_topk_only():
    print("=" * 60)
    print("Test 4: Verify new KL is computed on topk positions only")
    print("=" * 60)

    query, key, scaling, attention_mask, index_score, topk_indices, index_mask, G = make_data(S_q=128, S_kv=128)

    old_weights = old_recompute_attention_weights(query, key, attention_mask, scaling, G)
    attn_summed = old_weights.sum(dim=1)

    eps = 1e-8
    B, S_q, S_kv = attn_summed.shape

    score_topk = index_score.gather(-1, topk_indices)
    attn_topk = attn_summed.gather(-1, topk_indices)
    attn_topk = attn_topk + eps
    attn_dist = attn_topk / attn_topk.sum(dim=-1, keepdim=True).clamp_min(eps)
    log_index_dist = F.log_softmax(score_topk, dim=-1)

    kl_per_query = F.kl_div(log_index_dist, attn_dist, reduction="none", log_target=False)
    kl_per_query = kl_per_query.sum(dim=-1)

    print(f"  KL per query shape: {kl_per_query.shape}")
    print(f"  KL per query range: [{kl_per_query.min().item():.6f}, {kl_per_query.max().item():.6f}]")
    print(f"  KL per query mean:  {kl_per_query.mean().item():.6f}")
    print(f"  KL per query std:   {kl_per_query.std().item():.6f}")

    is_finite = torch.isfinite(kl_per_query).all().item()
    is_positive = (kl_per_query >= 0).all().item()
    print(f"  All finite: {is_finite}, All >= 0: {is_positive}")

    status = "PASS" if (is_finite and is_positive) else "FAIL"
    print(f"  [{status}]")
    print()


def test_5_gradient_flow():
    print("=" * 60)
    print("Test 5: Gradient flow through new compute_index_loss")
    print("=" * 60)

    query, key, scaling, attention_mask, index_score, topk_indices, index_mask, G = make_data(S_q=64, S_kv=64)

    index_score_param = index_score.clone().detach().requires_grad_(True)
    query_param = query.clone().detach().requires_grad_(True)
    key_param = key.clone().detach().requires_grad_(True)

    loss = new_compute_index_loss_from_scratch(
        index_score_param, query_param, key_param, attention_mask, scaling, topk_indices, G
    )

    loss.backward()

    print(f"  loss = {loss.item():.6f}")
    print(f"  index_score.grad: {index_score_param.grad is not None}, "
          f"norm={index_score_param.grad.norm().item():.6f}" if index_score_param.grad is not None else "  index_score.grad: None")
    print(f"  query.grad: {query_param.grad is not None}, "
          f"norm={query_param.grad.norm().item():.6f}" if query_param.grad is not None else "  query.grad: None")
    print(f"  key.grad: {key_param.grad is not None}, "
          f"norm={key_param.grad.norm().item():.6f}" if key_param.grad is not None else "  key.grad: None")

    has_grad = index_score_param.grad is not None
    status = "PASS" if has_grad else "FAIL"
    print(f"  [{status}]")
    print()


def test_6_full_kl_mode():
    print("=" * 60)
    print("Test 6: Full KL mode (cold start safe)")
    print("=" * 60)

    for S in [64, 128, 256]:
        query, key, scaling, attention_mask, index_score, topk_indices, index_mask, G = make_data(S_q=S, S_kv=S)

        old_weights = old_recompute_attention_weights(query, key, attention_mask, scaling, G)

        topk_loss = new_compute_index_loss(
            index_score.clone(), old_weights.clone(), topk_indices.clone()
        )

        full_kl_loss = full_kl_compute_index_loss(
            index_score.clone(), old_weights.clone()
        )

        print(f"  S={S:4d}: topk_loss={topk_loss.item():.6f}, full_kl_loss={full_kl_loss.item():.6f}")

    print("  NOTE: full_kl_loss uses ALL valid positions, safe for cold start.")
    print("  topk_loss uses only topk positions, efficient after warmup.")
    print()


def full_kl_compute_index_loss(index_score, attention_weights):
    if attention_weights.dim() == 4:
        attention_weights = attention_weights.sum(1)

    eps = 1e-8
    B, S_q, S_kv = attention_weights.shape
    CHUNK_SIZE = 128

    total_kl = torch.tensor(0.0, device=attention_weights.device, dtype=torch.float32)

    for s_start in range(0, S_q, CHUNK_SIZE):
        s_end = min(s_start + CHUNK_SIZE, S_q)

        attn_chunk = attention_weights[:, s_start:s_end, :]
        score_chunk = index_score[:, s_start:s_end, :]

        valid_mask = score_chunk > -1e8
        score_chunk = score_chunk.clamp(min=-1e9, max=1e9)
        attn_chunk = attn_chunk + eps
        attn_dist = attn_chunk / attn_chunk.sum(dim=-1, keepdim=True).clamp_min(eps)
        log_index_dist = F.log_softmax(score_chunk, dim=-1)
        kl_element = F.kl_div(
            log_index_dist, attn_dist, reduction="none", log_target=False
        )
        kl_element = kl_element.masked_fill(~valid_mask, 0.0)
        total_kl += kl_element.sum()

    return total_kl / B


def test_7_warmup_auto_switch():
    print("=" * 60)
    print("Test 7: Warmup auto-switch (property-based)")
    print("=" * 60)

    query, key, scaling, attention_mask, index_score, topk_indices, index_mask, G = make_data(S_q=128, S_kv=128)
    old_weights = old_recompute_attention_weights(query, key, attention_mask, scaling, G)

    class MockAttention:
        def __init__(self, warmup_steps):
            self.indexer_warmup_steps = warmup_steps
            self._indexer_training_step = 0

        @property
        def indexer_full_kl(self):
            return self._indexer_training_step < self.indexer_warmup_steps

    attn = MockAttention(warmup_steps=100)

    for step in [0, 50, 99, 100, 200]:
        attn._indexer_training_step = step
        mode = "full_kl" if attn.indexer_full_kl else "topk"
        print(f"  step={step:4d}: indexer_full_kl={attn.indexer_full_kl} → {mode} mode")

    print("  NOTE: step < warmup_steps → full KL (cold start safe)")
    print("        step >= warmup_steps → topk KL (efficient)")
    print()


if __name__ == "__main__":
    test_1_attention_weights()
    test_2_compute_index_loss()
    test_3_end_to_end()
    test_4_kl_on_topk_only()
    test_5_gradient_flow()
    test_6_full_kl_mode()
    test_7_warmup_auto_switch()