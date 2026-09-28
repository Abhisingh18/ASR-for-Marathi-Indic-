# data2vec-AQC (multilingual SSL, pretrained-only) + Gemma-3-4B-IT

| | |
|---|---|
| Encoder | SPRING-INX data2vec-AQC, **multilingual SSL pretrained-only** (no Marathi CTC fine-tuning), frozen, dim 1024 |
| LLM | Gemma-3-4B-IT, frozen text decoder + LoRA (r=8, alpha=32) |
| Training script | [`finetune_data2vec_gemma3_marathi_ssl_cs.sh`](../../scripts/training/finetune_data2vec_gemma3_marathi_ssl_cs.sh) |
| Decode script | [`inference_data2vec_ssl_gemma3_marathi_7testsets.sh`](../../scripts/decoding/inference_data2vec_ssl_gemma3_marathi_7testsets.sh) |
| Checkpoint evaluated | epoch 1, step 32,000 |
| Effective batch | 32 (micro-batch 4 x 4 GPUs x grad-accum 2) |

## WER%

| Test set | WER% |
|---|---|
| commonvoice | 25.23 |
| fleurs | 26.57 |
| indictts | 19.27 |
| kathbath | 25.04 |
| kathbath_noisy | 27.25 |
| mucs | 34.82 |
| R12 (preliminary) | see [`../SUMMARY.md`](../SUMMARY.md) |

Same architecture as the FT row above, but the encoder never saw Marathi-labeled CTC data —
consistently a few WER points behind, isolating the value of Marathi-specific encoder fine-tuning.
