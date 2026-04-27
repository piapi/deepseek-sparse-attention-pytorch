import torch
import traceback

try:
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

    mask_new = torch.full((bsz, 1, seqlen_q, seqlen_k), float('-inf'), dtype=torch.float32)
    mask_new.scatter_(-1, topk_indices.unsqueeze(1), scatter_src.unsqueeze(1))

    index_mask = torch.zeros((bsz, seqlen_q, seqlen_k), dtype=torch.bool)
    index_mask = index_mask.scatter_(-1, topk_indices, True)
    causal_mask = torch.ones((bsz, seqlen_q, seqlen_k), dtype=torch.bool).tril_(0)
    index_mask = index_mask & causal_mask
    mask_ref = torch.where(index_mask, torch.tensor(0.0), torch.tensor(float('-inf')))

    match = torch.allclose(mask_new.squeeze(), mask_ref.squeeze(), atol=1e-6)

    with open("e:/PyCharmDoc/deepseek-sparse-attention-pytorch/test_result.txt", "w", encoding="utf-8") as f:
        f.write(f"PyTorch version: {torch.__version__}\n")
        f.write(f"MATCH: {match}\n\n")
        f.write("=== topk_indices ===\n")
        f.write(str(topk_indices.squeeze()) + "\n\n")
        f.write("=== scatter_src ===\n")
        f.write(str(scatter_src.squeeze()) + "\n\n")
        f.write("=== mask_new (scatter way) ===\n")
        for row in mask_new.squeeze():
            vals = [f"{v:7.1f}" for v in row.tolist()]
            f.write("[" + ", ".join(vals) + "]\n")
        f.write("\n")
        f.write("=== mask_ref (old way) ===\n")
        for row in mask_ref.squeeze():
            vals = [f"{v:7.1f}" for v in row.tolist()]
            f.write("[" + ", ".join(vals) + "]\n")
        f.write("\n")
        if not match:
            diff = (mask_new.squeeze() - mask_ref.squeeze())
            f.write("=== DIFF ===\n")
            f.write(str(diff) + "\n")
except Exception as e:
    with open("e:/PyCharmDoc/deepseek-sparse-attention-pytorch/test_result.txt", "w", encoding="utf-8") as f:
        f.write(f"ERROR: {e}\n\n")
        traceback.print_exc(file=f)
