import torch
import torch.nn.functional as F


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def old_recompute(query, key, attention_mask, scaling, num_key_value_groups):
    key_states = repeat_kv(key, num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            causal_mask = torch.zeros_like(attention_mask, dtype=torch.float)
            causal_mask = causal_mask.masked_fill(~attention_mask, float("-inf"))
            causal_mask = causal_mask[:, :, :, :key_states.shape[-2]]
        else:
            causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    return attn_weights


def new_recompute(query, key, attention_mask, scaling, num_key_value_groups):
    G = num_key_value_groups
    H_KV = key.shape[1]
    B, H_Q, S_q, D = query.shape
    S_kv = key.shape[2]

    CHUNK_SIZE = 128

    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            causal_mask = torch.zeros_like(attention_mask, dtype=torch.float)
            causal_mask = causal_mask.masked_fill(~attention_mask, float("-inf"))
            causal_mask = causal_mask[:, :, :, :S_kv]
        else:
            causal_mask = attention_mask[:, :, :, :S_kv]
    else:
        causal_mask = None

    query_reshaped = query.view(B, H_KV, G, S_q, D)
    attn_weights_sum = torch.zeros(B, S_q, S_kv, dtype=torch.float32, device=query.device)

    for s_start in range(0, S_q, CHUNK_SIZE):
        s_end = min(s_start + CHUNK_SIZE, S_q)
        chunk_len = s_end - s_start
        chunk_weights = torch.zeros(B, chunk_len, S_kv, dtype=torch.float32, device=query.device)

        for h in range(H_KV):
            q_group = query_reshaped[:, h, :, s_start:s_end]
            k_h = key[:, h:h+1]

            weights_h = torch.matmul(q_group, k_h.transpose(2, 3)) * scaling

            if causal_mask is not None:
                weights_h = weights_h + causal_mask[:, :, s_start:s_end]

            weights_h = F.softmax(weights_h, dim=-1, dtype=torch.float32)
            chunk_weights += weights_h.sum(dim=1)

        attn_weights_sum[:, s_start:s_end] = chunk_weights

    return attn_weights_sum.unsqueeze(1).to(query.dtype)


def old_compute_index_loss(index_score, attention_weights, index_mask):
    if attention_weights.dim() == 4:
        attention_weights = attention_weights.sum(1)

    eps = 1e-8
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


def new_compute_index_loss(index_score, attention_weights, index_mask):
    if attention_weights.dim() == 4:
        attention_weights = attention_weights.sum(1)

    eps = 1e-8
    B, S_q, S_kv = attention_weights.shape
    CHUNK_SIZE = 128

    if index_mask is not None:
        index_mask = index_mask.squeeze(1)

    total_kl = torch.tensor(0.0, device=attention_weights.device, dtype=torch.float32)

    for s_start in range(0, S_q, CHUNK_SIZE):
        s_end = min(s_start + CHUNK_SIZE, S_q)

        attn_chunk = attention_weights[:, s_start:s_end, :]
        score_chunk = index_score[:, s_start:s_end, :]

        if index_mask is not None:
            mask_chunk = index_mask[:, s_start:s_end, :]
            attn_chunk = attn_chunk.masked_fill(~mask_chunk, eps)
            score_chunk = score_chunk.masked_fill(~mask_chunk, -1e9)

        score_chunk = torch.clamp(score_chunk, min=-1e9, max=1e9)
        attn_dist = attn_chunk / attn_chunk.sum(
            dim=-1, keepdim=True
        ).clamp_min(eps)
        log_index_dist = F.log_softmax(score_chunk, dim=-1)

        total_kl += F.kl_div(
            log_index_dist, attn_dist, reduction="sum", log_target=False
        )

    return total_kl / B


def test_case(name, B, H_Q, H_KV, S_q, S_kv, D, with_mask=True):
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"  shape: B={B}, H_Q={H_Q}, H_KV={H_KV}, S_q={S_q}, S_kv={S_kv}, D={D}")
    G = H_Q // H_KV
    print(f"  num_key_value_groups={G}")

    dtype = torch.float16
    device = "cpu"

    torch.manual_seed(42)
    query = torch.randn(B, H_Q, S_q, D, dtype=dtype, device=device)
    key = torch.randn(B, H_KV, S_kv, D, dtype=dtype, device=device)
    scaling = D ** -0.5

    if with_mask:
        attention_mask = torch.ones(B, 1, S_q, S_kv, dtype=torch.bool, device=device).tril_()
    else:
        attention_mask = None

    old_out = old_recompute(query, key, attention_mask, scaling, G)
    new_out = new_recompute(query, key, attention_mask, scaling, G)

    old_summed = old_out.sum(dim=1)
    new_summed = new_out.squeeze(1)

    abs_diff = (old_summed.float() - new_summed.float()).abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()

    old_max = old_summed.float().abs().max().item()
    rel_err = max_abs / (old_max + 1e-8)

    print(f"  old shape: {old_out.shape} -> summed: {old_summed.shape}")
    print(f"  new shape: {new_out.shape} -> squeezed: {new_summed.shape}")
    print(f"  max abs diff:  {max_abs:.10f}")
    print(f"  mean abs diff: {mean_abs:.10f}")
    print(f"  relative err:  {rel_err:.10f}")

    if rel_err < 1e-5:
        print(f"  [PASS] Results are CONSISTENT!")
        return True
    elif rel_err < 1e-3:
        print(f"  [PASS] Results are consistent within float16 tolerance")
        return True
    else:
        print(f"  [FAIL] Results DIFFER!")
        return False


def test_index_loss(name, B, S_q, S_kv, with_mask=True):
    print(f"\n{'='*60}")
    print(f"Test index_loss: {name}")
    print(f"  shape: B={B}, S_q={S_q}, S_kv={S_kv}")

    device = "cpu"
    torch.manual_seed(42)

    index_score = torch.randn(B, S_q, S_kv, dtype=torch.float32, device=device)
    attention_weights = F.softmax(torch.randn(B, S_q, S_kv, dtype=torch.float32, device=device), dim=-1)
    attention_weights_4d = attention_weights.unsqueeze(1)

    if with_mask:
        index_mask = torch.ones(B, 1, S_q, S_kv, dtype=torch.bool, device=device).tril_()
    else:
        index_mask = None

    old_loss = old_compute_index_loss(index_score.clone(), attention_weights_4d.clone(),
                                       index_mask.clone() if index_mask is not None else None)
    new_loss = new_compute_index_loss(index_score.clone(), attention_weights_4d.clone(),
                                       index_mask.clone() if index_mask is not None else None)

    diff = abs(old_loss.item() - new_loss.item())
    rel_err = diff / (abs(old_loss.item()) + 1e-8)

    print(f"  old loss: {old_loss.item():.10f}")
    print(f"  new loss: {new_loss.item():.10f}")
    print(f"  abs diff:  {diff:.12f}")
    print(f"  rel diff:  {rel_err:.12f}")

    if rel_err < 1e-7:
        print(f"  [PASS] Results are CONSISTENT!")
        return True
    elif rel_err < 1e-5:
        print(f"  [PASS] Results are consistent within float32 tolerance")
        return True
    else:
        print(f"  [FAIL] Results DIFFER!")
        return False


def main():
    print("=" * 60)
    print("Verification: old vs new recompute_attention_weights")
    print("=" * 60)

    all_pass = True

    all_pass &= test_case("small_no_chunk", B=2, H_Q=8, H_KV=2, S_q=64, S_kv=64, D=64, with_mask=True)
    all_pass &= test_case("small_no_mask", B=2, H_Q=8, H_KV=2, S_q=64, S_kv=64, D=64, with_mask=False)
    all_pass &= test_case("medium_cross_chunk", B=1, H_Q=16, H_KV=4, S_q=256, S_kv=256, D=128, with_mask=True)
    all_pass &= test_case("large_seq", B=1, H_Q=32, H_KV=8, S_q=512, S_kv=512, D=128, with_mask=True)
    all_pass &= test_case("prefill_asymmetric", B=1, H_Q=32, H_KV=8, S_q=256, S_kv=1024, D=128, with_mask=True)

    print("\n" + "=" * 60)
    print("Verification: old vs new compute_index_loss")
    print("=" * 60)

    all_pass &= test_index_loss("small_no_chunk", B=2, S_q=64, S_kv=64, with_mask=True)
    all_pass &= test_index_loss("small_no_mask", B=2, S_q=64, S_kv=64, with_mask=False)
    all_pass &= test_index_loss("medium_cross_chunk", B=1, S_q=256, S_kv=256, with_mask=True)
    all_pass &= test_index_loss("large_seq", B=1, S_q=512, S_kv=512, with_mask=True)
    all_pass &= test_index_loss("prefill_asymmetric", B=1, S_q=256, S_kv=1024, with_mask=True)

    print("\n" + "=" * 60)
    if all_pass:
        print("[PASS] All tests passed! Old and new versions are CONSISTENT.")
    else:
        print("[FAIL] Some tests failed!")


if __name__ == "__main__":
    main()