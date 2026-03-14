from __future__ import annotations

import random
from dataclasses import dataclass

from model.transformer import ForwardContext


@dataclass
class LayerSamplingConfig:
    enabled: bool = True
    active_layers: int = 8
    full_update_interval: int = 200
    ffn_block_subsampling: bool = True
    active_ffn_blocks_min: int = 1
    active_ffn_blocks_max: int = 2
    total_ffn_blocks: int = 8


class LayerSampler:
    def __init__(self, config: LayerSamplingConfig, total_layers: int) -> None:
        self.config = config
        self.total_layers = total_layers

    def build_context(self, global_step: int) -> ForwardContext:
        if not self.config.enabled:
            return ForwardContext(active_layers=None, full_update=True, scale_ffn_outputs=False)

        full_update = global_step > 0 and global_step % self.config.full_update_interval == 0
        active_layers = set(range(self.total_layers)) if full_update else set(
            random.sample(range(self.total_layers), k=min(self.config.active_layers, self.total_layers))
        )

        active_ffn_blocks: dict[int, list[int]] = {}
        if self.config.ffn_block_subsampling and not full_update:
            for layer_idx in range(self.total_layers):
                num_blocks = random.randint(self.config.active_ffn_blocks_min, self.config.active_ffn_blocks_max)
                active_ffn_blocks[layer_idx] = sorted(
                    random.sample(range(self.config.total_ffn_blocks), k=min(num_blocks, self.config.total_ffn_blocks))
                )

        return ForwardContext(
            active_layers=active_layers,
            full_update=full_update,
            active_ffn_blocks=active_ffn_blocks,
            scale_ffn_outputs=self.config.ffn_block_subsampling and not full_update,
        )
