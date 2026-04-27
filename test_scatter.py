import torch

bsz, seqlen_q, seqlen_k, topk = 1, 8, 8, 3

topk_indices = torch.tensor([[[2, 5, 7],
                               [0, 3, 6],
                               [1, 4, 7],
                               [0, 2, 5],
                               [3, 6, 7],
                               [1, 4, 5],
                               [0, 2, 6],
                               [1, 3, 7]]])

scatter_src = torch.zeros(topk_indices.shape, dtype=torch.float32)

row_pos = torch.arange(seqlen_q).unsqueeze(1)
future_mask = topk_indices > row_pos
scatter_src[future_mask] = float('-inf')

print("=== scatter_src ===")
print(scatter_src)

mask_for_eager = torch.full(
    (bsz, 1, seqlen_q, seqlen_k),
    float('-inf'),
    dtype=torch.float32,
)
mask_for_eager.scatter_(-1, topk_indices.unsqueeze(1), scatter_src.unsqueeze(1))

print("\n=== mask_for_eager (new way) ===")
print(mask_for_eager.squeeze())

index_mask = torch.zeros((bsz, seqlen_q, seqlen_k), dtype=torch.bool)
index_mask = index_mask.scatter_(-1, topk_indices, True)
causal_mask = torch.ones((bsz, seqlen_q, seqlen_k), dtype=torch.bool).tril_(0)
index_mask = index_mask & causal_mask

mask_ref = torch.where(
    index_mask,
    torch.tensor(0.0, dtype=torch.float32),
    torch.tensor(float('-inf'), dtype=torch.float32),
)

print("\n=== mask_ref (old way) ===")
print(mask_ref.squeeze())

match = torch.allclose(mask_for_eager.squeeze(), mask_ref.squeeze(), atol=1e-6)
print(f"\n=== MATCH: {match} ===")

if not match:
    diff = (mask_for_eager.squeeze() - mask_ref.squeeze())
    print("Diff:")
    print(diff)
