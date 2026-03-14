from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from model.lora import LoRAConfig


@dataclass
class FFNConfig:
    hidden_size: int
    intermediate_size: int
    num_blocks: int = 8
    lora: LoRAConfig | None = None
    train_base: bool = False


class TensorizedParallelFFN(nn.Module):
    def __init__(self, config: FFNConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_blocks = config.num_blocks
        self.block_size = config.intermediate_size // config.num_blocks
        self.lora_config = config.lora or LoRAConfig(enabled=False)
        self.train_base = config.train_base or not self.lora_config.enabled

        self.gate_weight = nn.Parameter(torch.empty(self.num_blocks, self.block_size, self.hidden_size))
        self.up_weight = nn.Parameter(torch.empty(self.num_blocks, self.block_size, self.hidden_size))
        self.down_weight = nn.Parameter(torch.empty(self.num_blocks, self.hidden_size, self.block_size))

        if self.lora_config.enabled and self.lora_config.rank > 0:
            rank = self.lora_config.rank
            self.gate_lora_a = nn.Parameter(torch.empty(self.num_blocks, rank, self.hidden_size))
            self.gate_lora_b = nn.Parameter(torch.zeros(self.num_blocks, self.block_size, rank))
            self.up_lora_a = nn.Parameter(torch.empty(self.num_blocks, rank, self.hidden_size))
            self.up_lora_b = nn.Parameter(torch.zeros(self.num_blocks, self.block_size, rank))
            self.down_lora_a = nn.Parameter(torch.empty(self.num_blocks, rank, self.block_size))
            self.down_lora_b = nn.Parameter(torch.zeros(self.num_blocks, self.hidden_size, rank))
            self.lora_dropout = nn.Dropout(self.lora_config.dropout)
        else:
            self.gate_lora_a = None
            self.gate_lora_b = None
            self.up_lora_a = None
            self.up_lora_b = None
            self.down_lora_a = None
            self.down_lora_b = None
            self.lora_dropout = nn.Identity()

        self.reset_parameters()
        self._set_base_trainable(self.train_base)

    @property
    def has_lora(self) -> bool:
        return self.gate_lora_a is not None

    def _set_base_trainable(self, enabled: bool) -> None:
        self.gate_weight.requires_grad_(enabled)
        self.up_weight.requires_grad_(enabled)
        self.down_weight.requires_grad_(enabled)

    def set_lora_trainable(self, enabled: bool) -> None:
        if self.has_lora:
            for param in (
                self.gate_lora_a,
                self.gate_lora_b,
                self.up_lora_a,
                self.up_lora_b,
                self.down_lora_a,
                self.down_lora_b,
            ):
                param.requires_grad_(enabled)

    def reset_parameters(self) -> None:
        for weight in (self.gate_weight, self.up_weight, self.down_weight):
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        if self.has_lora:
            for param in (self.gate_lora_a, self.up_lora_a, self.down_lora_a):
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            for param in (self.gate_lora_b, self.up_lora_b, self.down_lora_b):
                nn.init.zeros_(param)

    def _select_blocks(self, tensor: torch.Tensor, block_indices: list[int]) -> torch.Tensor:
        if len(block_indices) == self.num_blocks:
            return tensor
        index = torch.tensor(block_indices, device=tensor.device, dtype=torch.long)
        return tensor.index_select(0, index)

    def _project_input(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        lora_a: torch.Tensor | None,
        lora_b: torch.Tensor | None,
    ) -> torch.Tensor:
        output = torch.einsum("bsh,koh->bsko", x, weight)
        if lora_a is not None and lora_b is not None:
            lora_x = self.lora_dropout(x)
            low_rank = torch.einsum("bsh,krh->bskr", lora_x, lora_a)
            delta = torch.einsum("bskr,kor->bsko", low_rank, lora_b)
            output = output + delta * self.lora_config.scaling
        return output

    def _project_output(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        lora_a: torch.Tensor | None,
        lora_b: torch.Tensor | None,
    ) -> torch.Tensor:
        output = torch.einsum("bski,koi->bsko", x, weight)
        if lora_a is not None and lora_b is not None:
            low_rank = torch.einsum("bski,kri->bskr", x, lora_a)
            delta = torch.einsum("bskr,kor->bsko", low_rank, lora_b)
            output = output + delta * self.lora_config.scaling
        return output

    def forward(self, x: torch.Tensor, active_block_indices: list[int] | None = None, scale_outputs: bool = False) -> tuple[torch.Tensor, list[int]]:
        block_indices = active_block_indices or list(range(self.num_blocks))

        gate_weight = self._select_blocks(self.gate_weight, block_indices)
        up_weight = self._select_blocks(self.up_weight, block_indices)
        down_weight = self._select_blocks(self.down_weight, block_indices)

        gate_lora_a = self._select_blocks(self.gate_lora_a, block_indices) if self.gate_lora_a is not None else None
        gate_lora_b = self._select_blocks(self.gate_lora_b, block_indices) if self.gate_lora_b is not None else None
        up_lora_a = self._select_blocks(self.up_lora_a, block_indices) if self.up_lora_a is not None else None
        up_lora_b = self._select_blocks(self.up_lora_b, block_indices) if self.up_lora_b is not None else None
        down_lora_a = self._select_blocks(self.down_lora_a, block_indices) if self.down_lora_a is not None else None
        down_lora_b = self._select_blocks(self.down_lora_b, block_indices) if self.down_lora_b is not None else None

        gated = F.silu(self._project_input(x, gate_weight, gate_lora_a, gate_lora_b))
        up = self._project_input(x, up_weight, up_lora_a, up_lora_b)
        hidden = gated * up
        down = self._project_output(hidden, down_weight, down_lora_a, down_lora_b)
        combined = down.sum(dim=2)
        if scale_outputs and len(block_indices) > 0:
            combined = combined * (self.num_blocks / len(block_indices))
        return combined, block_indices


ParallelFFN = TensorizedParallelFFN
