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
            return ForwardContext(active_layers=None, active_layer_flags=None, full_update=True, scale_ffn_outputs=False)

        full_update = global_step > 0 and global_step % self.config.full_update_interval == 0
        active_layers = set(range(self.total_layers)) if full_update else set(
            random.sample(range(self.total_layers), k=min(self.config.active_layers, self.total_layers))
        )

        max_active_blocks = max(1, min(self.config.active_ffn_blocks_max, self.config.total_ffn_blocks))
        default_indices = tuple(range(max_active_blocks))
        default_mask = tuple(True for _ in range(max_active_blocks))
        active_ffn_block_indices: list[tuple[int, ...]] = [default_indices for _ in range(self.total_layers)]
        active_ffn_block_mask: list[tuple[bool, ...]] = [default_mask for _ in range(self.total_layers)]
        if self.config.ffn_block_subsampling and not full_update:
            for layer_idx in range(self.total_layers):
                num_blocks = random.randint(self.config.active_ffn_blocks_min, self.config.active_ffn_blocks_max)
                selected_blocks = sorted(
                    random.sample(range(self.config.total_ffn_blocks), k=min(num_blocks, self.config.total_ffn_blocks))
                )
                padded_indices = list(selected_blocks)
                padded_mask = [True] * len(selected_blocks)
                pad_value = selected_blocks[0] if selected_blocks else 0
                while len(padded_indices) < max_active_blocks:
                    padded_indices.append(pad_value)
                    padded_mask.append(False)
                active_ffn_block_indices[layer_idx] = tuple(padded_indices)
                active_ffn_block_mask[layer_idx] = tuple(padded_mask)

        active_layer_flags = tuple(layer_idx in active_layers for layer_idx in range(self.total_layers))
        return ForwardContext(
            active_layers=active_layers,
            active_layer_flags=active_layer_flags,
            full_update=full_update,
            active_ffn_block_indices=tuple(active_ffn_block_indices),
            active_ffn_block_mask=tuple(active_ffn_block_mask),
            scale_ffn_outputs=self.config.ffn_block_subsampling and not full_update,
        )
