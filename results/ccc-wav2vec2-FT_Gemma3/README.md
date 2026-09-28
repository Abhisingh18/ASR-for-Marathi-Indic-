# ccc-wav2vec2.0 (Marathi CTC fine-tuned) + Gemma-3-4B-IT

| | |
|---|---|
| Encoder | SPRING-INX ccc-wav2vec2.0 (fairseq wav2vec2), **Marathi CTC fine-tuned**, frozen, dim 1024 |
| LLM | Gemma-3-4B-IT, frozen text decoder + LoRA (r=8, alpha=32) |
| Training script | not included here (trained on a separate host; see the CTC-checkpoint source at asr.iitm.ac.in/models) |
| Decode script | [`inference_ccc_wav2vec2_gemma3_marathi_7testsets.sh`](../../scripts/decoding/inference_ccc_wav2vec2_gemma3_marathi_7testsets.sh) |
| Checkpoint evaluated | epoch 1, step 32,000 |
| Effective batch | 32 (micro-batch 4 x 4 GPUs x grad-accum 2) |

## WER%

| Test set | WER% |
|---|---|
| commonvoice | 24.16 |
| fleurs | 26.57 |
| indictts | 18.26 |
| kathbath | 23.74 |
| kathbath_noisy | 23.97 |
| mucs | 27.96 |
| R12 (preliminary) | see [`../SUMMARY.md`](../SUMMARY.md) |

A second Marathi-fine-tuned encoder family (fairseq wav2vec2 architecture, contrastive+CTC
pretraining) — close runner-up to the data2vec-AQC encoder above.
