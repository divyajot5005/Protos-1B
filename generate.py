from __future__ import annotations

import argparse
import json

import torch

from data.tokenizer_pipeline import load_qwen_tokenizer
from model.transformer import CausalLMModel
from training.runtime import configure_torch_runtime
from training.trainer import load_training_config


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from a Protos trainer checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to the training config JSON")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trainer checkpoint")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt to generate from")
    parser.add_argument("--interactive", action="store_true", help="Start an interactive prompt loop")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    return parser.parse_args()


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    probs = torch.softmax(logits / temperature, dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep_mask = cumulative <= top_p
        keep_mask[..., 0] = True
        filtered_probs = torch.where(keep_mask, sorted_probs, torch.zeros_like(sorted_probs))
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
        sampled = torch.multinomial(filtered_probs, num_samples=1)
        return sorted_indices.gather(-1, sampled)

    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_text(
    model: CausalLMModel,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    eos_token_id = tokenizer.eos_token_id

    model.eval()
    for _ in range(max_new_tokens):
        context = input_ids[:, -model.config.max_position_embeddings :]
        logits = model(context)["logits"][:, -1, :]
        next_token = sample_next_token(logits, temperature=temperature, top_p=top_p)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def load_model(config_path: str, checkpoint_path: str, device: torch.device) -> tuple[CausalLMModel, object]:
    config = load_training_config(config_path)
    tokenizer = load_qwen_tokenizer(config.tokenizer_name)
    config.model.vocab_size = len(tokenizer)

    model = CausalLMModel(config.model).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model"])
    return model, tokenizer


def main():
    args = parse_args()
    if not args.interactive and args.prompt is None:
        raise SystemExit("--prompt is required unless --interactive is used")

    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    configure_torch_runtime(device)
    model, tokenizer = load_model(args.config, args.checkpoint, device)

    if args.interactive:
        while True:
            try:
                prompt = input("prompt> ").strip()
            except EOFError:
                break
            if not prompt or prompt.lower() in {"quit", "exit"}:
                break
            completion = generate_text(
                model,
                tokenizer,
                prompt,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print(json.dumps({"prompt": prompt, "completion": completion}, ensure_ascii=False))
        return

    print(
        generate_text(
            model,
            tokenizer,
            args.prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    )


if __name__ == "__main__":
    main()
