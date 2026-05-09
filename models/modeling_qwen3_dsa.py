from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen3Attention,
    Qwen3ForCausalLM,
    Qwen3Model,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.deprecation import deprecate_kwarg

from .configuration_qwen3_dsa import Qwen3DSAConfig

def fp16_index(q, weights, k):
    # q: (bsz, seqlen, n_heads, head_dim)
    # weights: (bsz, seq_len, n_heads, 1)
    # k: (bsz, seqlen_k, head_dim)
    index_score = torch.einsum(
        "bsnd,btd->bsnt", q, k
    )  # (bsz, seqlen, n_heads, seqlen_k)
    index_score = F.relu(index_score)
    weighted = index_score * weights  # (bsz, seqlen, n_heads, seqlen_k)
    index_score = weighted.sum(dim=2)  # (bsz, seqlen, seqlen_k)
    return index_score


class Indexer(nn.Module):
    def __init__(self, config: Qwen3DSAConfig):
        super().__init__()
        self.d_model = config.hidden_size
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk

        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.head_dim, bias=False)
        self.w_proj = nn.Linear(self.d_model, self.n_heads, bias=False)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.softmax_scale = self.head_dim**-0.5
        self.k_cache = None
        self.max_seq_len = config.max_position_embeddings

    def update(self, k, start_pos, end_pos):
        bsz, seqlen, _ = k.shape
        assert seqlen == end_pos - start_pos, "k length must match [start_pos, end_pos)"
        if self.k_cache is None:
            self.k_cache = torch.zeros(
                bsz, self.max_seq_len, self.head_dim, dtype=k.dtype, device=k.device
            )
        self.k_cache[:, start_pos:end_pos] = k

    def forward(
        self, x: torch.Tensor, start_pos: int, end_pos: int, attention_mask: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        hidden_shape = (bsz, seqlen, -1, self.head_dim)
        q = self.q_norm(self.q_proj(x).view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(hidden_shape)).transpose(1, 2)

        position_embeddings = kwargs["position_embeddings"]
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)  # (bsz, n_heads, seqlen, head_dim)
        q = q.transpose(1, 2)  # (bsz, seqlen, n_heads, head_dim)
        k = k.transpose(1, 2).squeeze(2)  # (bsz, seqlen, head_dim)

        if kwargs["use_cache"] and start_pos >= 0 and end_pos >= 0:
            self.update(k, start_pos, end_pos)
            k = self.k_cache[:, :end_pos]

        weights = self.w_proj(x) * self.n_heads**-0.5  # (bsz,  n_heads)
        weights = (
            weights.unsqueeze(-1) * self.softmax_scale
        )  # (bsz, seqlen, n_heads, 1)

        index_score = fp16_index(q, weights, k)  # (bsz, seqlen, seqlen_k)

        if attention_mask is not None:
            if attention_mask.dim() == 4:
                attention_mask = attention_mask.squeeze(1)
            index_score = index_score.masked_fill(~attention_mask, float("-inf"))
        elif seqlen > 1:
            seqlen_k = index_score.shape[-1]
            mask = torch.full(
                (seqlen, seqlen_k), float("-inf"), device=x.device
            ).triu_(1)
            index_score += mask

        topk_indices = index_score.topk(min(self.index_topk, end_pos), dim=-1)[1]
        # topk_indices_ = topk_indices.clone()
        # dist.broadcast(topk_indices_, src=0)
        # assert torch.all(topk_indices == topk_indices_), f"{topk_indices=} {topk_indices_=}"
        return topk_indices, index_score


class Qwen3DSAAttention(Qwen3Attention):
    def __init__(self, config: Qwen3DSAConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.indexer = Indexer(config)
        self.index_loss = 0

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(
            self.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )  # bsz, n_heads, seqlen, head_dim

        seqlen = key_states.shape[-2]
        start_pos, end_pos = 0, seqlen
        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
            start_pos, end_pos = key_states.shape[-2] - seqlen, key_states.shape[-2]

        topk_indices, index_score = self.indexer(
            hidden_states, start_pos, end_pos, attention_mask=attention_mask, position_embeddings=position_embeddings, **kwargs
        )

        seqlen_q = input_shape[-1]
        seqlen_k = key_states.shape[-2]
        bsz = query_states.shape[0]

        if self.config._attn_implementation != "eager":
            # 稀疏 KV 选择：用 topk_indices 从 K/V 中 gather 出子集
            # topk_indices: [B, S_q, topk]，每个 query 位置选 topk 个 key 位置
            # 策略：沿 query 维度分块，每块 gather + FA2/SDPA，避免一次性物化大张量
            n_kv = key_states.shape[1]
            n_heads = query_states.shape[1]
            head_dim = key_states.shape[-1]
            topk = topk_indices.shape[-1]
            use_fa2 = self.config._attn_implementation == "flash_attention_2"

            # 分块处理：每次处理 chunk_size 个 query 位置
            chunk_size = min(512, seqlen_q)
            attn_output_chunks = []

            for chunk_start in range(0, seqlen_q, chunk_size):
                chunk_end = min(chunk_start + chunk_size, seqlen_q)
                chunk_len = chunk_end - chunk_start

                # Q chunk: [B, N_heads, chunk_len, D]
                q_chunk = query_states[:, :, chunk_start:chunk_end]

                # topk_indices chunk: [B, chunk_len, topk]
                topk_chunk = topk_indices[:, chunk_start:chunk_end]

                # 扩展到 KV 头维度: [B, chunk_len, topk] -> [B, N_kv, chunk_len, topk]
                sparse_idx = topk_chunk.unsqueeze(1).expand(-1, n_kv, -1, -1)
                sparse_idx_flat = sparse_idx.reshape(bsz, n_kv, -1)

                # gather 稀疏 K/V: [B, N_kv, S_k, D] -> [B, N_kv, chunk_len*topk, D]
                sparse_k = torch.gather(
                    key_states, 2, sparse_idx_flat.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                )
                sparse_v = torch.gather(
                    value_states, 2, sparse_idx_flat.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                )

                # GQA: repeat KV heads
                if n_kv < n_heads:
                    sparse_k = repeat_kv(sparse_k, self.num_key_value_groups)
                    sparse_v = repeat_kv(sparse_v, self.num_key_value_groups)

                # reshape: [B, N_heads, chunk_len*topk, D] -> [B, N_heads, chunk_len, topk, D]
                sparse_k = sparse_k.reshape(bsz, n_heads, chunk_len, topk, head_dim)
                sparse_v = sparse_v.reshape(bsz, n_heads, chunk_len, topk, head_dim)

                # 拆成独立 batch: 每个 (batch, query_pos) 独立做 [1 × topk] attention
                # Q: [B, N_heads, chunk_len, D] -> [B*chunk_len, N_heads, 1, D]
                q_exp = q_chunk.transpose(1, 2).reshape(bsz * chunk_len, n_heads, 1, head_dim)
                # K: [B, N_heads, chunk_len, topk, D] -> [B*chunk_len, N_heads, topk, D]
                k_exp = sparse_k.permute(0, 2, 1, 3, 4).reshape(bsz * chunk_len, n_heads, topk, head_dim)
                v_exp = sparse_v.permute(0, 2, 1, 3, 4).reshape(bsz * chunk_len, n_heads, topk, head_dim)

                if use_fa2:
                    from flash_attn import flash_attn_func
                    # FA2 格式: [batch, seqlen, num_heads, head_dim]
                    out = flash_attn_func(
                        q_exp.transpose(1, 2), k_exp.transpose(1, 2), v_exp.transpose(1, 2),
                        dropout_p=0.0 if not self.training else self.attention_dropout,
                        softmax_scale=self.scaling,
                        causal=False,
                    ).transpose(1, 2)
                else:
                    out = F.scaled_dot_product_attention(
                        q_exp, k_exp, v_exp,
                        attn_mask=None,
                        dropout_p=0.0 if not self.training else self.attention_dropout,
                        scale=self.scaling,
                    )

                # [B*chunk_len, N_heads, 1, D] -> [B, chunk_len, N_heads, D] -> [B, N_heads, chunk_len, D]
                out = out.reshape(bsz, chunk_len, n_heads, head_dim).transpose(1, 2)
                attn_output_chunks.append(out)

            # 拼接所有 chunk: [B, N_heads, S_q, D]
            attn_output = torch.cat(attn_output_chunks, dim=2)
            attn_weights = None

            if self.training:
                attention_weights = self.recompute_attention_weights(
                    query_states, key_states, attention_mask, self.scaling
                )
                scatter_src = torch.zeros(
                    topk_indices.shape, dtype=query_states.dtype, device=query_states.device
                )
                mask_for_eager = torch.full(
                    (input_shape[0], 1, seqlen_q, seqlen_k),
                    torch.finfo(query_states.dtype).min,
                    dtype=query_states.dtype,
                    device=query_states.device,
                )
                mask_for_eager.scatter_(-1, topk_indices.unsqueeze(1), scatter_src.unsqueeze(1))
                index_mask = mask_for_eager.squeeze(1) == 0.0
                self.index_loss = self.compute_index_loss(
                    index_score, attention_weights, index_mask
                )
        else:
            # eager 模式：使用 mask 方式（原有逻辑）
            scatter_src = torch.zeros(
                topk_indices.shape, dtype=query_states.dtype, device=query_states.device
            )
            mask_for_eager = torch.full(
                (input_shape[0], 1, seqlen_q, seqlen_k),
                torch.finfo(query_states.dtype).min,
                dtype=query_states.dtype,
                device=query_states.device,
            )
            mask_for_eager.scatter_(-1, topk_indices.unsqueeze(1), scatter_src.unsqueeze(1))

            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                mask_for_eager,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )

            if self.training:
                attention_weights = attn_weights
                if attention_weights is None:
                    attention_weights = self.recompute_attention_weights(
                        query_states, key_states, attention_mask, self.scaling
                    )
                index_mask = mask_for_eager.squeeze(1) == 0.0
                self.index_loss = self.compute_index_loss(
                    index_score, attention_weights, index_mask
                )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def compute_index_loss(
        self,
        index_score: torch.Tensor,
        attention_weights: torch.Tensor,
        index_mask: torch.Tensor | None,
    ):
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

    def recompute_attention_weights(self, query, key, attention_mask, scaling):
        key_states = repeat_kv(key, self.num_key_value_groups)
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        seqlen_q, seqlen_k = attn_weights.shape[-2], attn_weights.shape[-1]
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                causal_mask = torch.zeros_like(attention_mask, dtype=torch.float)
                causal_mask = causal_mask.masked_fill(~attention_mask, float("-inf"))
                causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
            else:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        else:
            causal_mask = torch.zeros(
                (seqlen_q, seqlen_k), dtype=attn_weights.dtype, device=attn_weights.device
            )
            if seqlen_q > 1:
                causal_mask = causal_mask.masked_fill(
                    torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=attn_weights.device).triu_(1),
                    float("-inf"),
                )
        attn_weights = attn_weights + causal_mask
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query.dtype)

        return attn_weights


class Qwen3DSAModel(Qwen3Model):
    def __init__(self, config: Qwen3DSAConfig):
        super().__init__(config)

        for layer in self.layers:
            old_attn = layer.self_attn
            new_attn = Qwen3DSAAttention(config, layer_idx=old_attn.layer_idx)
            layer.self_attn = new_attn

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping[
                    "sliding_attention"
                ] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Qwen3DSAForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    config_class = Qwen3DSAConfig
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: Qwen3DSAConfig):
        super().__init__(config)
        self.model = Qwen3DSAModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        if not hasattr(config, "sparse_lambda"):
            config.sparse_lambda = 0.01

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if labels is not None:
            sparse_loss = sum(layer.self_attn.index_loss for layer in self.model.layers)

            loss = loss + self.config.sparse_lambda * sparse_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


