# Sparse-Update CLM Pretraining Framework

This project provides a modular PyTorch training scaffold for running dense and sparse-update pretraining experiments on a roughly 1B-parameter decoder-only language model.

## Highlights

- Qwen tokenizer loaded directly from Hugging Face, with vocabulary-driven embeddings and tied LM head weights
- LLaMA/Qwen-style decoder with RMSNorm, RoPE, GQA, and SwiGLU FFNs
- Parallel FFN blocks for block subsampling experiments
- LoRA-only training mode with frozen base weights
- Layer sampling and periodic full-update steps
- Hugging Face streaming dataset pipeline with on-the-fly tokenization and sequence packing
- Checkpointing with resume support for multi-day training sessions
- Test-mode flags for quick smoke checks

## Quick start

```bash
python train.py --config experiments/lora_layer_sampling_ffn.json --test-mode --max-steps 2 --max-samples 4 --batch-size 1 --grad-accum-steps 1
```

## Resume training

```bash
python train.py --config experiments/lora_layer_sampling_ffn.json --auto-resume
python train.py --config experiments/lora_layer_sampling_ffn.json --resume /root/Protos-1B/outputs/lora_layer_sampling_ffn/latest.pt
```

Checkpoints now store model weights, optimizer state, scheduler state, RNG state, token counts, and the packed streaming-dataset cursor so restarted jobs continue from the same training position. Relative config, output, and resume paths are resolved under `/root/Protos-1B` by default.

## Notes

- `dense_baseline=true` disables LoRA, layer sampling, and FFN block subsampling for fair comparison.
- Streaming datasets keep local storage low, but the first real run will require network access to Hugging Face.
- `torch.nn.functional.scaled_dot_product_attention` is used so PyTorch can route to FlashAttention kernels on H100 when available.
"# Protos-1B" 

