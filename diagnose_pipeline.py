from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from data.streaming_dataset import build_streaming_dataset
from data.tokenizer_pipeline import load_qwen_tokenizer
from model.transformer import CausalLMModel
from training.runtime import build_adamw, configure_torch_runtime
from training.trainer import TrainingConfig, load_training_config


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose data and training throughput bottlenecks")
    parser.add_argument("--config", type=str, required=True, help="Experiment config path")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test-mode", action="store_true")
    return parser.parse_args()


def measure_batch_fetch(dataset, batch_size: int, device: torch.device, num_batches: int) -> list[float]:
    durations = []
    for _ in range(num_batches):
        start = time.perf_counter()
        batch = dataset.next_batch(batch_size, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - start)
        _ = batch["input_ids"].shape
    return durations


def measure_train_step(model, batch, num_steps: int = 3) -> list[float]:
    optimizer = build_adamw([p for p in model.parameters() if p.requires_grad], lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1, device=batch["input_ids"].device)
    durations = []
    for _ in range(num_steps):
        optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        with torch.autocast(device_type=batch["input_ids"].device.type, dtype=torch.bfloat16, enabled=batch["input_ids"].device.type == "cuda"):
            output = model(batch["input_ids"], labels=batch["labels"])
        output["loss"].backward()
        optimizer.step()
        if batch["input_ids"].device.type == "cuda":
            torch.cuda.synchronize(batch["input_ids"].device)
        durations.append(time.perf_counter() - start)
    return durations


def main():
    args = parse_args()
    config: TrainingConfig = load_training_config(args.config)
    if args.test_mode:
        config.test_mode = True
        config.data.test_mode = True
        config.data.max_samples = max(config.data.max_samples or 0, args.batch_size * args.num_batches * 2)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    configure_torch_runtime(device)

    overall_start = time.perf_counter()

    tokenizer_start = time.perf_counter()
    tokenizer = load_qwen_tokenizer(config.tokenizer_name)
    tokenizer_seconds = time.perf_counter() - tokenizer_start

    dataset = build_streaming_dataset(config.data, tokenizer, validation=False)
    first_batch_times = measure_batch_fetch(dataset, args.batch_size, device, args.num_batches)

    config.model.vocab_size = len(tokenizer)
    model_start = time.perf_counter()
    model = CausalLMModel(config.model).to(device)
    model.train()
    model_build_seconds = time.perf_counter() - model_start

    train_batch = dataset.next_batch(args.batch_size, device)
    step_times = measure_train_step(model, train_batch)

    tokens_per_batch = train_batch["input_ids"].numel()
    avg_fetch = sum(first_batch_times) / len(first_batch_times)
    avg_step = sum(step_times) / len(step_times)
    fetch_tokens_per_sec = tokens_per_batch / avg_fetch if avg_fetch > 0 else float("inf")
    step_tokens_per_sec = tokens_per_batch / avg_step if avg_step > 0 else float("inf")

    diagnosis = {
        "config": str(Path(args.config)),
        "device": str(device),
        "tokenizer_seconds": round(tokenizer_seconds, 4),
        "model_build_seconds": round(model_build_seconds, 4),
        "batch_fetch_seconds": [round(x, 4) for x in first_batch_times],
        "avg_batch_fetch_seconds": round(avg_fetch, 4),
        "batch_fetch_tokens_per_sec": round(fetch_tokens_per_sec, 2),
        "train_step_seconds": [round(x, 4) for x in step_times],
        "avg_train_step_seconds": round(avg_step, 4),
        "train_step_tokens_per_sec": round(step_tokens_per_sec, 2),
        "tokens_per_batch": tokens_per_batch,
        "total_wall_seconds": round(time.perf_counter() - overall_start, 4),
    }

    hints = []
    if avg_fetch > avg_step * 1.5:
        hints.append("data_fetch_slower_than_train_step")
    if avg_fetch > 1.0:
        hints.append("stream_reinitialization_or_tokenization_overhead_likely")
    if avg_step > 1.0 and device.type == "cuda":
        hints.append("model_step_itself_is_nontrivial")
    diagnosis["hints"] = hints

    print(json.dumps(diagnosis, indent=2))


if __name__ == "__main__":
    main()
