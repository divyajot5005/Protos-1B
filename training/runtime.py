from __future__ import annotations

import inspect

import torch
from torch.optim import AdamW


def configure_torch_runtime(device: torch.device) -> None:
    torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def build_adamw(parameters, lr: float, betas: tuple[float, float], weight_decay: float, device: torch.device) -> AdamW:
    kwargs = {
        "lr": lr,
        "betas": betas,
        "weight_decay": weight_decay,
    }
    if device.type == "cuda" and "fused" in inspect.signature(AdamW).parameters:
        kwargs["fused"] = True
    return AdamW(parameters, **kwargs)
