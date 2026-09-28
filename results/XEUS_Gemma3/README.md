# XEUS (CMU/ESPnet, pretrained-only) + Gemma-3-4B-IT

| | |
|---|---|
| Encoder | XEUS (CMU/ESPnet, 577M params, ~1M-hour multilingual pretraining, pretrained-only), frozen, dim 1024 |
| LLM | Gemma-3-4B-IT, frozen text decoder + LoRA (r=8, alpha=32) |
| Training script | [`finetune_xeus_gemma3_marathi_cs.sh`](../../scripts/training/finetune_xeus_gemma3_marathi_cs.sh) |
| Decode script | [`inference_xeus_gemma3_marathi_7testsets.sh`](../../scripts/decoding/inference_xeus_gemma3_marathi_7testsets.sh) |
| Checkpoint evaluated | epoch 1, step 32,000 |
| Effective batch | 32 (micro-batch 4 x 4 GPUs x grad-accum 2) |

## WER%

| Test set | WER% |
|---|---|
| commonvoice | 35.80 |
| fleurs | 30.43 |
| indictts | 21.90 |
| kathbath | 31.96 |
| kathbath_noisy | 33.38 |
| mucs | 35.28 |
| R12 (preliminary) | see [`../SUMMARY.md`](../SUMMARY.md) |