from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class LoRAConfig:
    enabled: bool = True
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0

    @property
    def scaling(self) -> float:
        if self.rank <= 0:
            return 1.0
        return self.alpha / self.rank


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        lora_config: LoRAConfig | None = None,
        train_base: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_config = lora_config or LoRAConfig(enabled=False)
        self.train_base = train_base or not self.lora_config.enabled

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        if self.lora_config.enabled and self.lora_config.rank > 0:
            self.lora_a = nn.Parameter(torch.empty(self.lora_config.rank, in_features))
            self.lora_b = nn.Parameter(torch.zeros(out_features, self.lora_config.rank))
            self.lora_dropout = nn.Dropout(self.lora_config.dropout)
        else:
            self.lora_a = None
            self.lora_b = None
            self.lora_dropout = nn.Identity()

        self.reset_parameters()
        self.weight.requires_grad_(self.train_base)
        if self.bias is not None:
            self.bias.requires_grad_(self.train_base)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        if self.lora_a is not None:
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        if self.lora_b is not None:
            nn.init.zeros_(self.lora_b)

    @property
    def has_lora(self) -> bool:
        return self.lora_a is not None and self.lora_b is not None

    def set_lora_trainable(self, enabled: bool) -> None:
        if self.has_lora:
            self.lora_a.requires_grad_(enabled)
            self.lora_b.requires_grad_(enabled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = F.linear(x, self.weight, self.bias)
        if self.has_lora:
            lora_x = self.lora_dropout(x)
            delta = F.linear(F.linear(lora_x, self.lora_a), self.lora_b)
            output = output + delta * self.lora_config.scaling
        return output
