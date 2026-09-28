# data2vec-AQC (Marathi fine-tuned) + Sarvam-1

| | |
|---|---|
| Encoder | SPRING-INX data2vec-AQC, **Marathi CTC fine-tuned**, frozen, dim 1024 (same checkpoint as the Gemma-3 row) |
| LLM | Sarvam-1 (2B, llm_dim 2048), frozen + LoRA (r=8, alpha=32), vicuna-style prompt |
| Training script | [`finetune_data2vec_sarvam1_marathi_cs.sh`](../../scripts/training/finetune_data2vec_sarvam1_marathi_cs.sh) |
| Decode script | [`inference_data2vec_sarvam1_marathi_7testsets.sh`](../../scripts/decoding/inference_data2vec_sarvam1_marathi_7testsets.sh) |
| Checkpoint evaluated | epoch 1, step 32,000 |
| Effective batch | 32 (micro-batch 4 x 4 GPUs x grad-accum 2) |