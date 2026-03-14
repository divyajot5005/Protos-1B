from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import nn

from model.attention import AttentionConfig, CausalSelfAttention
from model.ffn_blocks import FFNConfig, ParallelFFN
from model.lora import LoRAConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


@dataclass
class ModelConfig:
    vocab_size: int = 151936
    hidden_size: int = 1536
    intermediate_size: int = 6144
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 96
    max_position_embeddings: int = 2048
    attention_dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    lora_enabled: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    ffn_num_blocks: int = 8
    train_base_weights: bool = False
    dense_baseline: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass
class ForwardContext:
    active_layers: set[int] | None = None
    full_update: bool = False
    active_ffn_blocks: dict[int, list[int]] = field(default_factory=dict)
    scale_ffn_outputs: bool = False


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        lora_config = LoRAConfig(
            enabled=config.lora_enabled and not config.dense_baseline,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        )
        train_base = config.train_base_weights or config.dense_baseline
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = CausalSelfAttention(
            AttentionConfig(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                attention_dropout=config.attention_dropout,
                lora=lora_config,
                train_base=train_base,
            )
        )
        self.ffn = ParallelFFN(
            FFNConfig(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_blocks=config.ffn_num_blocks,
                lora=lora_config,
                train_base=train_base,
            )
        )

    def set_lora_trainable(self, enabled: bool) -> None:
        self.attention.set_lora_trainable(enabled)
        self.ffn.set_lora_trainable(enabled)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, block_index: int, ctx: ForwardContext | None = None) -> tuple[torch.Tensor, list[int]]:
        ctx = ctx or ForwardContext()
        layer_active = ctx.full_update or ctx.active_layers is None or block_index in ctx.active_layers
        active_blocks = ctx.active_ffn_blocks.get(block_index)

        if self.training and not layer_active:
            with torch.no_grad():
                attn_out = self.attention(self.attn_norm(x), position_ids)
                x = x + attn_out
                ffn_out, used_blocks = self.ffn(self.ffn_norm(x), active_blocks, ctx.scale_ffn_outputs)
                x = x + ffn_out
            return x, used_blocks

        attn_out = self.attention(self.attn_norm(x), position_ids)
        x = x + attn_out
        ffn_out, used_blocks = self.ffn(self.ffn_norm(x), active_blocks, ctx.scale_ffn_outputs)
        x = x + ffn_out
        return x, used_blocks


class CausalLMModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

        if config.lora_enabled and not config.dense_baseline:
            self.embed_tokens.weight.requires_grad_(False)

    def set_active_lora_layers(self, active_layers: set[int] | None, full_update: bool) -> None:
        if self.config.dense_baseline or not self.config.lora_enabled:
            return
        for idx, layer in enumerate(self.layers):
            layer.set_lora_trainable(full_update or active_layers is None or idx in active_layers)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        forward_context: ForwardContext | None = None,
    ) -> dict[str, Any]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        hidden_states = self.embed_tokens(input_ids)

        block_usage: dict[int, list[int]] = {}
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, used_blocks = layer(hidden_states, position_ids, layer_idx, forward_context)
            block_usage[layer_idx] = used_blocks

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return {"loss": loss, "logits": logits, "block_usage": block_usage}
