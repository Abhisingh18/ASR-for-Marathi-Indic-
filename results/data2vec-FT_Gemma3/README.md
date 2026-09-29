# data2vec-AQC (Marathi fine-tuned) + Gemma-3-4B-IT

| | |
|---|---|
| Encoder | SPRING-INX data2vec-AQC, **Marathi CTC fine-tuned**, frozen, dim 1024 |
| LLM | Gemma-3-4B-IT, frozen text decoder + LoRA (r=8, alpha=32) |
| Training script | [`finetune_data2vec_gemma3_marathi_cs.sh`](../../scripts/training/finetune_data2vec_gemma3_marathi_cs.sh) |
| Decode script | [`inference_data2vec_gemma3_marathi_7testsets.sh`](../../scripts/decoding/inference_data2vec_gemma3_marathi_7testsets.sh) |
| Checkpoint evaluated | epoch 1, step 32,000 |
| Effective batch | 32 (micro-batch 4 x 4 GPUs x grad-accum 2) |

## WER%

| Test set | WER% |
|---|---|
| commonvoice | **22.49** |
| fleurs | **23.88** |
| indictts | **16.15** |
| kathbath | **21.29** |
| kathbath_noisy | **22.14** |
| mucs | 29.24 |
| R12 | 58.20 |

Best overall of the six encoders compared here — the encoder was fine-tuned on Marathi CTC data
before this SLAM-ASR stage, giving it an in-domain head start over the pretrained-only encoders.
