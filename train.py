from __future__ import annotations

import argparse

from training.trainer import Trainer, load_training_config


def parse_args():
    parser = argparse.ArgumentParser(description="Sparse-update causal LM pretraining framework")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON experiment config")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--test-mode", action="store_true", help="Run against a tiny in-memory dataset")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap packed sequences for smoke tests")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume from")
    parser.add_argument("--auto-resume", action="store_true", help="Resume from outputs/.../latest.pt if it exists")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_training_config(args.config)

    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.grad_accum_steps is not None:
        config.grad_accum_steps = args.grad_accum_steps
    if args.max_samples is not None:
        config.data.max_samples = args.max_samples
    if args.test_mode:
        config.test_mode = True
        config.data.test_mode = True
        config.val_steps = min(config.val_steps, 2)
        config.val_interval = max(1, min(config.val_interval, 2))
        config.checkpoint_interval = max(config.max_steps + 1, config.checkpoint_interval)
    if args.compile:
        config.compile_model = True

    trainer = Trainer(config, resume_path=args.resume, auto_resume=args.auto_resume)
    trainer.train()


if __name__ == "__main__":
    main()
