from __future__ import annotations

import json
import math
import os
import random
import shutil
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW

from data.streaming_dataset import DataSource, DatasetConfig, build_streaming_dataset
from data.tokenizer_pipeline import load_qwen_tokenizer
from model.transformer import CausalLMModel, ModelConfig
from training.layer_sampler import LayerSampler, LayerSamplingConfig
from training.schedulers import build_cosine_scheduler


@dataclass
class OptimizerConfig:
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip_norm: float = 1.0


@dataclass
class CheckpointConfig:
    save_optimizer_state: bool = True
    save_rng_state: bool = True
    keep_last_k: int = 3
    save_on_exit: bool = True


@dataclass
class TrainingConfig:
    output_dir: str = "outputs/default"
    tokenizer_name: str = "Qwen/Qwen2.5-1.5B"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DatasetConfig = field(default_factory=DatasetConfig)
    layer_sampling: LayerSamplingConfig = field(default_factory=LayerSamplingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    checkpointing: CheckpointConfig = field(default_factory=CheckpointConfig)
    batch_size: int = 1
    grad_accum_steps: int = 16
    max_steps: int = 1000
    warmup_steps: int = 100
    val_interval: int = 100
    val_steps: int = 20
    log_interval: int = 10
    checkpoint_interval: int = 200
    bf16: bool = True
    compile_model: bool = False
    seed: int = 7
    test_mode: bool = False
    projection_target_tokens: int = 200_000_000_000


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


DEFAULT_PROJECT_ROOT = Path("/root/Protos-1B")
PROJECT_ROOT = Path(os.environ.get("PROTOS_PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))).resolve()


def _resolve_project_path(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_training_config(config_path: str | None) -> TrainingConfig:
    config = asdict(TrainingConfig())
    if config_path:
        resolved_config_path = _resolve_project_path(config_path)
        with open(resolved_config_path, "r", encoding="utf-8") as handle:
            file_config = json.load(handle)
        config = _deep_update(config, file_config)

    config["model"] = ModelConfig(**config["model"])
    config["data"]["sources"] = [DataSource(**source) for source in config["data"].get("sources", [])]
    config["data"]["validation_sources"] = [DataSource(**source) for source in config["data"].get("validation_sources", [])]
    config["data"] = DatasetConfig(**config["data"])
    config["layer_sampling"] = LayerSamplingConfig(**config["layer_sampling"])
    config["optimizer"] = OptimizerConfig(**config["optimizer"])
    config["checkpointing"] = CheckpointConfig(**config["checkpointing"])
    return TrainingConfig(**config)


def _format_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "inf"
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{days}d {hours}h {minutes}m"


class Trainer:
    def __init__(self, config: TrainingConfig, resume_path: str | None = None, auto_resume: bool = False) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(config.seed)
        random.seed(config.seed)

        if self.config.model.dense_baseline:
            self.config.model.lora_enabled = False
            self.config.layer_sampling.enabled = False
            self.config.layer_sampling.ffn_block_subsampling = False
        self.config.layer_sampling.total_ffn_blocks = self.config.model.ffn_num_blocks

        self.output_dir = _resolve_project_path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.latest_checkpoint_path = self.output_dir / "latest.pt"

        self.tokenizer = load_qwen_tokenizer(config.tokenizer_name)
        self.config.model.vocab_size = len(self.tokenizer)
        self.config.data.test_mode = self.config.test_mode or self.config.data.test_mode

        self.model = CausalLMModel(self.config.model).to(self.device)
        if self.config.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        self.train_dataset = build_streaming_dataset(self.config.data, self.tokenizer, validation=False)
        self.val_dataset = build_streaming_dataset(self.config.data, self.tokenizer, validation=True)
        self.layer_sampler = LayerSampler(self.config.layer_sampling, self.config.model.num_hidden_layers)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=self.config.optimizer.lr,
            betas=self.config.optimizer.betas,
            weight_decay=self.config.optimizer.weight_decay,
        )
        self.scheduler = build_cosine_scheduler(self.optimizer, self.config.warmup_steps, self.config.max_steps)
        self.autocast_dtype = torch.bfloat16 if self.config.bf16 and self.device.type == "cuda" else torch.float32

        self.global_step = 0
        self.tokens_processed = 0
        self.best_val_perplexity: float | None = None
        self.last_log_time = time.perf_counter()
        self.train_start_time = time.perf_counter()
        self.loaded_resume_path: str | None = None
        self._exit_checkpoint_written = False

        with open(self.output_dir / "resolved_config.json", "w", encoding="utf-8") as handle:
            json.dump(asdict(self.config), handle, indent=2)

        candidate_resume = _resolve_project_path(resume_path)
        if candidate_resume is None and auto_resume and self.latest_checkpoint_path.exists():
            candidate_resume = self.latest_checkpoint_path
        if candidate_resume is not None:
            self.resume(str(candidate_resume))

        self._register_signal_handlers()

    def _register_signal_handlers(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._handle_exit_signal)
            except (ValueError, AttributeError):
                continue

    def _handle_exit_signal(self, signum, _frame) -> None:
        signal_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        if self.config.checkpointing.save_on_exit and self.global_step > 0 and not self._exit_checkpoint_written:
            self.save_checkpoint(self.global_step, tag=signal_name.lower())
            self._exit_checkpoint_written = True
        print(json.dumps({"signal": signal_name, "step": self.global_step, "checkpoint_saved": self._exit_checkpoint_written}))
        raise KeyboardInterrupt

    def _capture_rng_state(self) -> dict[str, Any]:
        state = {
            "python": random.getstate(),
            "torch": torch.get_rng_state().cpu(),
        }
        if torch.cuda.is_available():
            state["cuda"] = [rng_state.cpu() for rng_state in torch.cuda.get_rng_state_all()]
        return state

    def _coerce_rng_state_tensor(self, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(dtype=torch.uint8, device="cpu")
        return torch.tensor(value, dtype=torch.uint8)

    def _restore_rng_state(self, state: dict[str, Any] | None) -> None:
        if not state:
            return
        if "python" in state:
            random.setstate(state["python"])
        if "torch" in state:
            torch.set_rng_state(self._coerce_rng_state_tensor(state["torch"]))
        if torch.cuda.is_available() and "cuda" in state:
            cuda_states = [self._coerce_rng_state_tensor(rng_state) for rng_state in state["cuda"]]
            torch.cuda.set_rng_state_all(cuda_states)

    def _checkpoint_payload(self, step: int, last_val_perplexity: float | None = None) -> dict[str, Any]:
        payload = {
            "step": step,
            "tokens_processed": self.tokens_processed,
            "best_val_perplexity": self.best_val_perplexity,
            "last_val_perplexity": last_val_perplexity,
            "config": asdict(self.config),
            "model": self.model.state_dict(),
            "train_dataset": self.train_dataset.state_dict(),
            "val_dataset": self.val_dataset.state_dict(),
        }
        if self.config.checkpointing.save_optimizer_state:
            payload["optimizer"] = self.optimizer.state_dict()
            payload["scheduler"] = self.scheduler.state_dict()
        if self.config.checkpointing.save_rng_state:
            payload["rng_state"] = self._capture_rng_state()
        return payload

    def _cleanup_old_checkpoints(self) -> None:
        keep_last_k = self.config.checkpointing.keep_last_k
        if keep_last_k <= 0:
            return
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_step_*.pt"), key=lambda path: path.stat().st_mtime)
        stale = checkpoints[:-keep_last_k]
        for path in stale:
            path.unlink(missing_ok=True)

    def _save_latest_alias(self, checkpoint_path: Path) -> None:
        shutil.copy2(checkpoint_path, self.latest_checkpoint_path)

    def save_checkpoint(self, step: int, last_val_perplexity: float | None = None, tag: str | None = None) -> Path:
        suffix = f"_{tag}" if tag else ""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}{suffix}.pt"
        torch.save(self._checkpoint_payload(step, last_val_perplexity), checkpoint_path)
        self._save_latest_alias(checkpoint_path)
        self._cleanup_old_checkpoints()
        meta = {
            "latest_checkpoint": str(checkpoint_path),
            "step": step,
            "tokens_processed": self.tokens_processed,
            "updated_at": time.time(),
        }
        with open(self.output_dir / "resume_state.json", "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
        return checkpoint_path

    def resume(self, checkpoint_path: str) -> None:
        payload = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(payload["model"])
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        if "scheduler" in payload:
            self.scheduler.load_state_dict(payload["scheduler"])
        self.train_dataset.load_state_dict(payload.get("train_dataset"))
        self.val_dataset.load_state_dict(payload.get("val_dataset"))
        self._restore_rng_state(payload.get("rng_state"))
        self.global_step = int(payload.get("step", 0))
        self.tokens_processed = int(payload.get("tokens_processed", 0))
        self.best_val_perplexity = payload.get("best_val_perplexity")
        self.train_start_time = time.perf_counter()
        self.loaded_resume_path = checkpoint_path
        print(json.dumps({"resume_checkpoint": checkpoint_path, "step": self.global_step, "tokens_processed": self.tokens_processed}))

    def _next_batch(self, split: str) -> dict[str, torch.Tensor]:
        dataset = self.train_dataset if split == "train" else self.val_dataset
        return dataset.next_batch(self.config.batch_size, self.device)

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float]:
        self.model.eval()
        losses = []
        saved_val_state = self.val_dataset.state_dict()
        for _ in range(self.config.val_steps):
            try:
                batch = self._next_batch("val")
            except StopIteration:
                break
            with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.device.type == "cuda"):
                output = self.model(batch["input_ids"], labels=batch["labels"])
            losses.append(output["loss"].item())
        self.val_dataset.load_state_dict(saved_val_state)
        self.model.train()
        if not losses:
            return float("inf"), float("inf")

        avg_loss = sum(losses) / len(losses)
        if not math.isfinite(avg_loss):
            return avg_loss, float("inf")
        if avg_loss >= 80.0:
            return avg_loss, float("inf")
        return avg_loss, math.exp(avg_loss)

    def train(self) -> None:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        self.last_log_time = time.perf_counter()

        try:
            for step in range(self.global_step + 1, self.config.max_steps + 1):
                step_loss = 0.0
                ctx = self.layer_sampler.build_context(step)
                self.model.set_active_lora_layers(ctx.active_layers, ctx.full_update)
                last_block_usage: dict[int, list[int]] = {}
                last_val_perplexity = None

                for _ in range(self.config.grad_accum_steps):
                    batch = self._next_batch("train")
                    with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.device.type == "cuda"):
                        output = self.model(batch["input_ids"], labels=batch["labels"], forward_context=ctx)
                        loss = output["loss"] / self.config.grad_accum_steps
                    loss.backward()
                    step_loss += loss.item()
                    self.tokens_processed += batch["input_ids"].numel()
                    last_block_usage = output["block_usage"]

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.optimizer.grad_clip_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step = step

                if step % self.config.log_interval == 0:
                    now = time.perf_counter()
                    elapsed = max(now - self.last_log_time, 1e-6)
                    tokens_per_sec = (self.config.log_interval * self.config.grad_accum_steps * self.config.batch_size * self.config.data.sequence_length) / elapsed
                    active_layers = sorted(ctx.active_layers) if ctx.active_layers is not None else list(range(self.config.model.num_hidden_layers))
                    usage_summary = {layer: blocks for layer, blocks in list(last_block_usage.items())[:4]}
                    print(
                        json.dumps(
                            {
                                "step": step,
                                "loss": step_loss,
                                "lr": self.scheduler.get_last_lr()[0],
                                "tokens_processed": self.tokens_processed,
                                "tokens_per_sec": round(tokens_per_sec, 2),
                                "active_layers": active_layers,
                                "full_update": ctx.full_update,
                                "ffn_block_usage_sample": usage_summary,
                                "resumed_from": self.loaded_resume_path,
                            }
                        )
                    )
                    self.last_log_time = now

                if step % self.config.val_interval == 0:
                    val_loss, last_val_perplexity = self.evaluate()
                    if math.isfinite(last_val_perplexity):
                        self.best_val_perplexity = last_val_perplexity if self.best_val_perplexity is None else min(self.best_val_perplexity, last_val_perplexity)
                    print(json.dumps({"step": step, "validation_loss": val_loss, "validation_perplexity": last_val_perplexity, "best_validation_perplexity": self.best_val_perplexity}))

                if step % self.config.checkpoint_interval == 0:
                    self.save_checkpoint(step, last_val_perplexity)

            self.save_checkpoint(self.global_step, tag="final")
            if self.config.test_mode and self.tokens_processed > 0:
                total_elapsed = max(time.perf_counter() - self.train_start_time, 1e-6)
                avg_tokens_per_sec = self.tokens_processed / total_elapsed
                projected_seconds = self.config.projection_target_tokens / avg_tokens_per_sec
                print(
                    json.dumps(
                        {
                            "test_mode_benchmark": True,
                            "measured_tokens": self.tokens_processed,
                            "avg_tokens_per_sec": round(avg_tokens_per_sec, 2),
                            "projection_target_tokens": self.config.projection_target_tokens,
                            "projected_hours": round(projected_seconds / 3600.0, 2),
                            "projected_days": round(projected_seconds / 86400.0, 2),
                            "projected_duration": _format_duration(projected_seconds),
                        }
                    )
                )
        except KeyboardInterrupt:
            if self.config.checkpointing.save_on_exit and self.global_step > 0 and not self._exit_checkpoint_written:
                self.save_checkpoint(self.global_step, tag="interrupt")
                self._exit_checkpoint_written = True
            raise


def main() -> int:
    return 0


if __name__ == "__main__":
    sys.exit(main())


