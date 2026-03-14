from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from model.lora import LoRAConfig, LoRALinear


@dataclass
class FFNConfig:
    hidden_size: int
    intermediate_size: int
    num_blocks: int = 8
    lora: LoRAConfig | None = None
    train_base: bool = False


class SwiGLUBlock(nn.Module):
    def __init__(self, config: FFNConfig) -> None:
        super().__init__()
        block_size = config.intermediate_size // config.num_blocks
        self.gate_proj = LoRALinear(config.hidden_size, block_size, lora_config=config.lora, train_base=config.train_base)
        self.up_proj = LoRALinear(config.hidden_size, block_size, lora_config=config.lora, train_base=config.train_base)
        self.down_proj = LoRALinear(block_size, config.hidden_size, lora_config=config.lora, train_base=config.train_base)

    def set_lora_trainable(self, enabled: bool) -> None:
        for module in (self.gate_proj, self.up_proj, self.down_proj):
            module.set_lora_trainable(enabled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ParallelFFN(nn.Module):
    def __init__(self, config: FFNConfig) -> None:
        super().__init__()
        self.num_blocks = config.num_blocks
        self.blocks = nn.ModuleList([SwiGLUBlock(config) for _ in range(config.num_blocks)])

    def set_lora_trainable(self, enabled: bool) -> None:
        for block in self.blocks:
            block.set_lora_trainable(enabled)

    def forward(self, x: torch.Tensor, active_block_indices: list[int] | None = None, scale_outputs: bool = False) -> tuple[torch.Tensor, list[int]]:
        block_indices = active_block_indices or list(range(self.num_blocks))
        outputs = [self.blocks[idx](x) for idx in block_indices]
        combined = torch.stack(outputs, dim=0).sum(dim=0)
        if scale_outputs and len(block_indices) > 0:
            combined = combined * (self.num_blocks / len(block_indices))
        return combined, block_indices
