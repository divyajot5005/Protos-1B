from __future__ import annotations

import argparse
import json
from copy import deepcopy

from tqdm.auto import tqdm

from training.trainer import Trainer, load_training_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train dense until a validation-loss threshold, then switch to LoRA+sparse training")
    parser.add_argument("--config", type=str, required=True, help="Base sparse experiment config")
    parser.add_argument("--switch-val-loss", type=float, required=True, help="Validation-loss threshold for switching from dense to sparse")
    parser.add_argument("--dense-max-steps", type=int, default=None, help="Safety cap for dense bootstrap phase")
    parser.add_argument("--max-steps", type=int, default=None, help="Final total max steps for the sparse phase")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dense-batch-size", type=int, default=None, help="Override microbatch size for the dense bootstrap phase only")
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--dense-grad-accum-steps", type=int, default=None, help="Override gradient accumulation for the dense bootstrap phase only")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--dense-output-dir", type=str, default=None)
    parser.add_argument("--sparse-output-dir", type=str, default=None)
    parser.add_argument("--dense-resume", type=str, default=None, help="Resume the dense phase from a checkpoint")
    parser.add_argument("--dense-auto-resume", action="store_true", help="Auto-resume the dense phase from its latest checkpoint")
    return parser.parse_args()


def apply_common_overrides(config, args):
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.grad_accum_steps is not None:
        config.grad_accum_steps = args.grad_accum_steps
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.compile:
        config.compile_model = True
    return config


def main():
    args = parse_args()
    sparse_config = apply_common_overrides(load_training_config(args.config), args)

    dense_config = deepcopy(sparse_config)
    dense_config.output_dir = args.dense_output_dir or f"{sparse_config.output_dir}_dense_bootstrap"
    dense_config.max_steps = args.dense_max_steps or sparse_config.max_steps
    if args.dense_batch_size is not None:
        dense_config.batch_size = args.dense_batch_size
    if args.dense_grad_accum_steps is not None:
        dense_config.grad_accum_steps = args.dense_grad_accum_steps
    dense_config.stop_at_val_loss = args.switch_val_loss
    dense_config.model.dense_baseline = True
    dense_config.model.train_base_weights = True
    dense_config.model.lora_enabled = False
    dense_config.layer_sampling.enabled = False
    dense_config.layer_sampling.ffn_block_subsampling = False

    dense_trainer = Trainer(dense_config, resume_path=args.dense_resume, auto_resume=args.dense_auto_resume)
    dense_result = dense_trainer.train() or {}
    dense_checkpoint = dense_result.get("checkpoint") or dense_trainer.last_checkpoint_path or str(dense_trainer.latest_checkpoint_path)

    if dense_trainer.last_validation_loss is None or dense_trainer.last_validation_loss > args.switch_val_loss:
        tqdm.write(json.dumps({
            "stage": "dense",
            "status": "threshold_not_reached",
            "validation_loss": dense_trainer.last_validation_loss,
            "target_val_loss": args.switch_val_loss,
            "checkpoint": dense_checkpoint,
        }))
        return

    sparse_config.output_dir = args.sparse_output_dir or sparse_config.output_dir
    sparse_config.stop_at_val_loss = None
    sparse_trainer = Trainer(sparse_config)
    sparse_trainer.load_stage_checkpoint(dense_checkpoint)
    sparse_trainer.train()


if __name__ == "__main__":
    main()
