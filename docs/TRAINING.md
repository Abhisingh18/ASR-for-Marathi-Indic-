# Training configuration

All six training runs share this recipe (see `scripts/training/`); only the encoder path/name and,
for the Sarvam-1 run, the LLM, differ between scripts.

| Hyperparameter | Value |
|---|---|
| LLM | Gemma-3-4B-IT (5 of 6 runs) or Sarvam-1 (2B, 1 run) |
| LoRA | r=8, alpha=32, dropout=0.05, target modules: q/k/v/o/gate/up/down proj |
| Encoder | frozen |
| LLM base weights | frozen (LoRA-only) |
| Optimizer | AdamW, lr=1e-4, warmup=1000 steps (DeepSpeed WarmupLR) |
| Precision | bf16, ZeRO stage 2 |
| Batch | micro-batch 4 x n_gpus x grad-accum (grad-accum = 8/n_gpus) = effective batch **32** |
| Epochs | 1 (most comparison runs) or 5 (later runs, still evaluated at epoch-1/step-32000 here for a fair comparison) |
| Checkpoint/eval interval | every 1000 steps |
| GPUs | 4x NVIDIA RTX 6000 Ada (49GB), `CUDA_DEVICE_ORDER=PCI_BUS_ID` |

## Resume support

`src_patches/finetune_deepspeed_new.py` includes a fix so that resuming from a checkpoint restores
the **full DeepSpeed engine state** -- optimizer moments, LR-scheduler position, global step count
-- not just the model weights. Without this, a resumed run silently re-warms the LR from 0 and
restarts Adam from scratch, which would make resumed runs incomparable to runs that never crashed.
See `docs/RESUME.md` for the mechanism.
