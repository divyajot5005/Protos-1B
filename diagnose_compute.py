from __future__ import annotations

import argparse
import json
import time
from statistics import mean

import torch

from data.streaming_dataset import build_streaming_dataset
from data.tokenizer_pipeline import load_qwen_tokenizer
from model.transformer import CausalLMModel
from training.runtime import build_adamw, configure_torch_runtime
from training.trainer import TrainingConfig, load_training_config


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose training-step bottlenecks")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--dense-baseline", action="store_true")
    parser.add_argument("--disable-layer-sampling", action="store_true")
    parser.add_argument("--disable-ffn-subsampling", action="store_true")
    parser.add_argument("--disable-lora", action="store_true")
    return parser.parse_args()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def now() -> float:
    return time.perf_counter()


def timed_fetch(dataset, batch_size: int, device: torch.device):
    start = now()
    batch = dataset.next_batch(batch_size, device)
    sync(device)
    return batch, now() - start


def timed_step(model, optimizer, batch, device: torch.device, autocast_dtype: torch.dtype | None):
    stats = {}

    start = now()
    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=device.type == "cuda" and autocast_dtype is not None):
        output = model(batch["input_ids"], labels=batch["labels"])
    sync(device)
    stats["forward_seconds"] = now() - start

    start = now()
    output["loss"].backward()
    sync(device)
    stats["backward_seconds"] = now() - start

    start = now()
    optimizer.step()
    sync(device)
    stats["optimizer_seconds"] = now() - start

    start = now()
    optimizer.zero_grad(set_to_none=True)
    sync(device)
    stats["zero_grad_seconds"] = now() - start

    stats["loss"] = float(output["loss"].item())
    return stats


def configure(config: TrainingConfig, args) -> TrainingConfig:
    config.batch_size = args.batch_size
    if args.test_mode:
        config.test_mode = True
        config.data.test_mode = True
        config.data.max_samples = max(config.data.max_samples or 0, args.batch_size * (args.num_steps + args.warmup_steps + 4))
    if args.dense_baseline:
        config.model.dense_baseline = True
        config.model.lora_enabled = False
        config.layer_sampling.enabled = False
        config.layer_sampling.ffn_block_subsampling = False
    if args.disable_lora:
        config.model.lora_enabled = False
    if args.disable_layer_sampling:
        config.layer_sampling.enabled = False
    if args.disable_ffn_subsampling:
        config.layer_sampling.ffn_block_subsampling = False
    return config


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "avg": round(mean(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def main():
    args = parse_args()
    config: TrainingConfig = configure(load_training_config(args.config), args)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    configure_torch_runtime(device)
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else None

    tokenizer = load_qwen_tokenizer(config.tokenizer_name)
    config.model.vocab_size = len(tokenizer)
    dataset = build_streaming_dataset(config.data, tokenizer, validation=False)

    model = CausalLMModel(config.model).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    model.train()

    optimizer = build_adamw([p for p in model.parameters() if p.requires_grad], lr=config.optimizer.lr, betas=config.optimizer.betas, weight_decay=config.optimizer.weight_decay, device=device)

    fetch_times = []
    forward_times = []
    backward_times = []
    optimizer_times = []
    zero_grad_times = []
    losses = []

    total_steps = args.warmup_steps + args.num_steps
    for step_idx in range(total_steps):
        batch, fetch_seconds = timed_fetch(dataset, args.batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        step_stats = timed_step(model, optimizer, batch, device, autocast_dtype)

        if step_idx >= args.warmup_steps:
            fetch_times.append(fetch_seconds)
            forward_times.append(step_stats["forward_seconds"])
            backward_times.append(step_stats["backward_seconds"])
            optimizer_times.append(step_stats["optimizer_seconds"])
            zero_grad_times.append(step_stats["zero_grad_seconds"])
            losses.append(step_stats["loss"])

    tokens_per_step = args.batch_size * config.data.sequence_length
    total_step_seconds = [f + fw + bw + op + zg for f, fw, bw, op, zg in zip(fetch_times, forward_times, backward_times, optimizer_times, zero_grad_times)]
    avg_total_seconds = mean(total_step_seconds)
    tokens_per_sec = tokens_per_step / avg_total_seconds

    result = {
        "device": str(device),
        "config": args.config,
        "tokens_per_step": tokens_per_step,
        "steps_profiled": args.num_steps,
        "compile": args.compile,
        "dense_baseline": config.model.dense_baseline,
        "lora_enabled": config.model.lora_enabled,
        "layer_sampling_enabled": config.layer_sampling.enabled,
        "ffn_block_subsampling": config.layer_sampling.ffn_block_subsampling,
        "fetch_seconds": summarize(fetch_times),
        "forward_seconds": summarize(forward_times),
        "backward_seconds": summarize(backward_times),
        "optimizer_seconds": summarize(optimizer_times),
        "zero_grad_seconds": summarize(zero_grad_times),
        "loss": summarize(losses),
        "avg_total_step_seconds": round(avg_total_seconds, 4),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "phase_share": {
            "fetch": round(mean(fetch_times) / avg_total_seconds, 3),
            "forward": round(mean(forward_times) / avg_total_seconds, 3),
            "backward": round(mean(backward_times) / avg_total_seconds, 3),
            "optimizer": round(mean(optimizer_times) / avg_total_seconds, 3),
            "zero_grad": round(mean(zero_grad_times) / avg_total_seconds, 3),
        },
    }

    hints = []
    if result["phase_share"]["fetch"] > 0.2:
        hints.append("data_fetch_is_material")
    if result["phase_share"]["forward"] > 0.35:
        hints.append("forward_path_is_expensive")
    if result["phase_share"]["backward"] > 0.35:
        hints.append("backward_path_is_expensive")
    if result["phase_share"]["optimizer"] > 0.15:
        hints.append("optimizer_overhead_is_high")
    if result["tokens_per_sec"] < 10000 and device.type == "cuda":
        hints.append("overall_compute_throughput_is_low_for_h100")
    result["hints"] = hints

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
