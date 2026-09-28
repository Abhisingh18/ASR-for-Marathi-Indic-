# Checkpoint resume mechanism

Training on shared, occasionally-unstable GPU hardware means runs get killed (reboots, OOM,
driver hangs) and need to resume mid-epoch without restarting the LR warmup or losing optimizer
state. Two things are restored on resume, both necessary:

1. **Weights** -- `ckpt_path=<ckpt_dir>/pytorch_model.bin`, loaded by the model factory
   (LoRA + projector only; the frozen encoder/LLM base weights are not in this file).
2. **DeepSpeed engine state** -- `model_engine.load_checkpoint(ckpt_dir, load_optimizer_states=True,
   load_lr_scheduler_states=True)`, added in `src_patches/finetune_deepspeed_new.py`. This restores
   Adam's moment estimates and the WarmupLR scheduler's position, so a resumed run's loss curve
   continues smoothly instead of spiking (from a re-warmed LR) or drifting (from a reset optimizer).

The dataloader is also fast-forwarded past already-completed batches (`resume_step`/`resume_epoch`,
parsed from the checkpoint directory name `asr_epoch_<E>_step_<S>`), so the same batch order
resumes from where it left off rather than restarting the epoch.