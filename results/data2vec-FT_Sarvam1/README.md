# data2vec-AQC (Marathi fine-tuned) + Sarvam-1

| | |
|---|---|
| Encoder | SPRING-INX data2vec-AQC, **Marathi CTC fine-tuned**, frozen, dim 1024 (same checkpoint as the Gemma-3 row) |
| LLM | Sarvam-1 (2B, llm_dim 2048), frozen + LoRA (r=8, alpha=32), vicuna-style prompt |
| Training script | [`finetune_data2vec_sarvam1_marathi_cs.sh`](../../scripts/training/finetune_data2vec_sarvam1_marathi_cs.sh) |
| Decode script | [`inference_data2vec_sarvam1_marathi_7testsets.sh`](../../scripts/decoding/inference_data2vec_sarvam1_marathi_7testsets.sh) |
| Checkpoint evaluated | epoch 1, step 32,000 |
| Effective batch | 32 (micro-batch 4 x 4 GPUs x grad-accum 2) |

## WER%

| Test set | WER% |
|---|---|
| commonvoice | 23.53 |
| fleurs | 26.77 |
| indictts | 17.26 |
| kathbath | 22.73 |
| kathbath_noisy | 23.22 |
| mucs | 43.62 |
| R12 | 57.89 |

Same encoder as the top-performing Gemma-3 row, swapping only the LLM — isolates how much the
downstream LLM choice matters versus the encoder. Competitive on most sets, but falls sharply
behind on `mucs`.
