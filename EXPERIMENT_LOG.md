# Experiment Log

## 2026-03-15

### Goal
Improve training throughput for the sparse-update 1B pretraining framework on 1x H100 and track bottlenecks systematically.

### Baseline observations
- Real streaming run showed extremely low throughput around 277 tokens/sec.
- Test-mode diagnosis showed data fetch was fast, but compute throughput was still low for H100.
- The real-data path repeatedly printed `Resolving data files`, indicating dataset stream reinitialization overhead.

### Changes made

#### Checkpointing and resume
- Added full checkpoint save/load for model, optimizer, scheduler, RNG state, token counters, and dataset cursor.
- Added `latest.pt` alias and `resume_state.json` metadata.
- Added `--resume` and `--auto-resume` CLI support.
- Added signal-based save-on-exit for interrupted runs.

Status: Success

#### Path handling
- Made relative config, output, and resume paths resolve under `/root/Protos-1B` by default.
- Added `PROTOS_PROJECT_ROOT` override support.

Status: Success

#### Validation overflow fix
- Validation perplexity could overflow on very large early losses.
- Changed evaluation to log both validation loss and perplexity, returning `inf` perplexity safely when loss is too large or non-finite.

Status: Success

#### Test-mode projection
- Added test-mode projection for total runtime at 200B tokens based on measured average tokens/sec.

Status: Success

#### Diagnosis scripts
- Added `diagnose_pipeline.py` to isolate tokenizer, batch fetch, and training-step timing.
- Added `diagnose_compute.py` to split fetch, forward, backward, optimizer, and zero-grad timings.

Status: Success

#### Runtime optimizations
- Added `training/runtime.py`.
- Enabled:
  - `torch.set_float32_matmul_precision("high")`
  - TF32 matmul
  - SDPA backend toggles
  - cuDNN benchmark
  - fused AdamW when supported
- Updated trainer and diagnosis scripts to use bf16 autocast consistently.

Status: Success

### Results

#### Pipeline diagnosis in test mode
- Batch fetch reached about 28k tok/s at batch size 1 and about 153k-172k tok/s at batch size 4.
- Conclusion: test-mode data path is not the bottleneck.

Status: Success

#### Compute diagnosis before runtime tuning
Sparse mode, batch size 4, test mode:
- Around 5.0k tok/s without compile.
- Around 5.6k tok/s with compile.
- Forward about 41-42% of step time.
- Backward about 55-56% of step time.
- Disabling FFN subsampling had almost no effect.
- Disabling layer sampling had almost no effect.
- Dense baseline was slower than sparse LoRA mode at the same batch size.

Conclusion:
- Sparse control logic is not the main compute bottleneck.
- Forward/backward math path is the main bottleneck.

Status: Success

#### Real-data diagnosis
- Real-data `diagnose_pipeline.py` showed repeated `Resolving data files` lines.
- Batch fetch was extremely slow on early batches, with average fetch time around 10.5s and only about 194 tok/s.
- Conclusion: real streaming path is reinitializing dataset/file resolution too often.

Status: Failure in current implementation

### FlashAttention notes
- `pip install flash-attn` corresponds to the main `flash-attn` package line, not Hopper FA3.
- FA3 requires installing from the upstream repo `hopper/` directory.
- Installing `flash-attn` initially failed because `psutil` was missing.

Status: Not completed yet

### Current conclusions
- Data fetch in real streaming mode is broken/inefficient and must be fixed.
- Compute path is still too slow for H100 even when data fetch is not the bottleneck.
- Compile helps modestly.
- The biggest remaining compute opportunities are likely:
  - better attention kernels
  - tensorized FFN implementation
  - fewer small-kernel launches in the model hot path

### Open next steps
1. Re-run `diagnose_compute.py` after the new runtime tuning and compare results.
2. Fix the real-data streaming iterator so it stays open instead of re-resolving data repeatedly.
3. Tensorize `model/ffn_blocks.py` to reduce Python/module overhead and tiny matmuls.
4. Finish FlashAttention/FA3 installation and compare again.
5. Try activation checkpointing if it helps raise microbatch above 4.


#### Tensorized FFN path
- Replaced the Python-looped FFN block stack with a packed tensorized implementation using batched einsums.
- Preserved per-block selection, scaling, and LoRA updates in the FFN path.
- Goal: reduce kernel launch overhead and make forward/backward more compiler- and GPU-friendly.

Status: Implemented, benchmark pending


#### Persistent streaming iterator fix
- Reworked the streaming dataset to keep one live iterator open across sequences and batches.
- Removed the per-sequence dataset rebuild that was causing repeated `Resolving data files` work.
- Reset the live iterator cleanly on checkpoint resume.
- Disabled datasets progress bars to avoid noisy shard-resolution output.

Status: Implemented, benchmark pending


#### Runtime-tuned compute diagnosis
- Re-ran compute diagnosis after enabling bf16 autocast, TF32/high matmul precision, SDPA backend toggles, fused AdamW, and `torch.compile`.
- Sparse mode, batch size 4, test mode, compile on:
  - `forward_seconds`: about 0.106s before FFN tensorization
  - `backward_seconds`: about 0.163s before FFN tensorization
  - `tokens_per_sec`: about 26.9k
- Conclusion: runtime/kernel tuning alone produced a major jump and made the compute path viable on H100.

Status: Success

#### Tensorized FFN benchmark result
- Re-ran compute diagnosis after tensorizing the FFN block path.
- Sparse mode, batch size 4, test mode, compile on:
  - `forward_seconds`: about 0.078s
  - `backward_seconds`: about 0.116s
  - `avg_total_step_seconds`: about 0.220s
  - `tokens_per_sec`: about 37.3k
- Compared with the pre-tensorized compiled result, throughput improved by roughly 39%.
- Conclusion: FFN Python/module overhead was a major remaining compute bottleneck and tensorization helped substantially.

Status: Success

#### Real-data training after persistent iterator fix
- Re-ran real training with compile on, batch size 4, grad accumulation 1.
- The repeated `Resolving data files` spam disappeared after switching to a persistent streaming iterator.
- Throughput observations:
  - early warmup/compile step at step 10: about 361 tok/s
  - stabilized throughput by step 20: about 44.4k tok/s
- Conclusion: the real-data bottleneck was caused by per-sequence stream rebuilds, and fixing the iterator unlocked end-to-end throughput close to the compute-only path.

Status: Success

#### Remaining issues after throughput recovery
- `torch.compile` still hits the recompile limit because `forward_context.active_layers` is a Python `set` used in the model forward path.
- The run ended with `terminate called without an active exception` after finishing the short benchmark run, likely during shutdown/teardown rather than the actual training step.

Status: Open

#### Updated takeaways
- End-to-end real training is now in the ~44k tok/s range on 1x H100 for the short run tested.
- Major wins came from:
  - persistent streaming iterator
  - bf16/TF32/runtime tuning
  - fused AdamW
  - tensorized FFN path
- The next likely optimization target is reducing compile recompilations from dynamic layer selection.


#### Background RAM prefetch buffer
- Added a background training-batch prefetcher with a default 1000-step RAM buffer.
- Training now consumes prefetched CPU batches while a daemon worker streams, tokenizes, and packs future batches in the background.
- Prefetched batches are pinned in memory for faster GPU transfer.
- Checkpoints now snapshot unread prefetched batches so resume does not silently skip buffered data.

Status: Implemented, benchmark pending


#### Flash attention confirmation and latest throughput
- Verified with `torch.profiler` on H100 that PyTorch SDPA is dispatching to flash attention kernels:
  - `aten::_scaled_dot_product_flash_attention`
  - `aten::_flash_attention_forward`
  - `pytorch_flash::flash_fwd_kernel`
- Short real-data training run with compile on reached about `49.0k tok/s` by step 20.
- Flash attention is active in practice, but `torch.compile` still recompiles because FFN block routing is represented as variable-length Python lists.

Status: Success

#### Fixed-width FFN routing for compile stability
- Replaced per-layer variable-length FFN block lists in the forward context with fixed-width padded block index tuples plus a matching boolean mask.
- The FFN module now consumes a stable-shape routing representation and masks padded slots instead of relying on variable-length selections.
- Goal: stop Dynamo recompiles caused by changing `len(active_ffn_blocks[layer])` while preserving sparse FFN execution.

Status: Implemented, benchmark pending


#### Reverted fixed-width FFN routing
- Reverted the padded fixed-width FFN routing representation.
- The change moved the Dynamo recompile from FFN block-count specialization to dynamic layer flags and reduced end-to-end throughput from roughly `49-50k tok/s` to roughly `47k tok/s`.
- Restored the previous variable-length FFN routing path because it performed better in practice.

Status: Reverted after regression


#### Layer-activity branch removal
- Replaced the per-layer Python `if self.training and not layer_active` branch with a tensor-gated detach path.
- Trainer now prepares a per-step `active_layer_mask` tensor on the target device and passes it through the forward context.
- Goal: keep one compiled forward path while still blocking gradients for inactive layers.

Status: Implemented, benchmark pending
