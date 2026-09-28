# XEUS (CMU/ESPnet, pretrained-only) + Gemma-3-4B-IT

| | |
|---|---|
| Encoder | XEUS (CMU/ESPnet, 577M params, ~1M-hour multilingual pretraining, pretrained-only), frozen, dim 1024 |
| LLM | Gemma-3-4B-IT, frozen text decoder + LoRA (r=8, alpha=32) |
| Training script | [`finetune_xeus_gemma3_marathi_cs.sh`](../../scripts/training/finetune_xeus_gemma3_marathi_cs.sh) |
| Decode script | [`inference_xeus_gemma3_marathi_7testsets.sh`](../../scripts/decoding/inference_xeus_gemma3_marathi_7testsets.sh) |
| Checkpoint evaluated | epoch 1, step 32,000 |
| Effective batch | 32 (micro-batch 4 x 4 GPUs x grad-accum 2) |