# data2vec-AQC (Marathi fine-tuned) + Gemma-3-4B-IT

| | |
|---|---|
| Encoder | SPRING-INX data2vec-AQC, **Marathi CTC fine-tuned**, frozen, dim 1024 |
| LLM | Gemma-3-4B-IT, frozen text decoder + LoRA (r=8, alpha=32) |
| Training script | [`finetune_data2vec_gemma3_marathi_cs.sh`](../../scripts/training/finetune_data2vec_gemma3_marathi_cs.sh) |
| Decode script | [`inference_data2vec_gemma3_marathi_7testsets.sh`](../../scripts/decoding/inference_data2vec_gemma3_marathi_7testsets.sh) |
| Checkpoint evaluated | epoch 1, step 32,000 |
| Effective batch | 32 (micro-batch 4 x 4 GPUs x grad-accum 2) |