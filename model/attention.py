from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from model.lora import LoRAConfig, LoRALinear


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1)
    return rotated.flatten(start_dim=-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings

    def forward(self, positions: torch.Tensor, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(device=device, dtype=dtype)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


def repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return x
    batch, heads, seq_len, dim = x.shape
    x = x[:, :, None, :, :].expand(batch, heads, repeats, seq_len, dim)
    return x.reshape(batch, heads * repeats, seq_len, dim)


@dataclass
class AttentionConfig:
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_position_embeddings: int
    attention_dropout: float = 0.0
    lora: LoRAConfig | None = None
    train_base: bool = False


class CausalSelfAttention(nn.Module):
    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.dropout_p = config.attention_dropout
        self.kv_repeat = self.num_heads // self.num_kv_heads

        proj_out = self.num_heads * self.head_dim
        kv_out = self.num_kv_heads * self.head_dim
        self.q_proj = LoRALinear(config.hidden_size, proj_out, lora_config=config.lora, train_base=config.train_base)
        self.k_proj = LoRALinear(config.hidden_size, kv_out, lora_config=config.lora, train_base=config.train_base)
        self.v_proj = LoRALinear(config.hidden_size, kv_out, lora_config=config.lora, train_base=config.train_base)
        self.o_proj = LoRALinear(proj_out, config.hidden_size, lora_config=config.lora, train_base=config.train_base)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.max_position_embeddings)

    def set_lora_trainable(self, enabled: bool) -> None:
        for module in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            module.set_lora_trainable(enabled)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(position_ids[0], device=x.device, dtype=q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        k = repeat_kv(k, self.kv_repeat)
        v = repeat_kv(v, self.kv_repeat)

        attn = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(attn)
