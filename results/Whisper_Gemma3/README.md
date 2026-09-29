# Whisper large-v3 (multilingual) + Gemma-3-4B-IT

| | |
|---|---|
| Encoder | OpenAI Whisper large-v3, frozen, dim 1280, mel input (128 mel bins) |
| LLM | Gemma-3-4B-IT, frozen text decoder + LoRA (r=8, alpha=32) |
| Training script | [`finetune_whisper_gemma3_marathi_cs.sh`](../../scripts/training/finetune_whisper_gemma3_marathi_cs.sh) |
| Decode script | [`inference_whisper_gemma3_marathi_7testsets.sh`](../../scripts/decoding/inference_whisper_gemma3_marathi_7testsets.sh) |
| Checkpoint evaluated | epoch 1, step 32,000 |
| Effective batch | 32 (micro-batch 4 x 4 GPUs x grad-accum 2) |

## WER%

| Test set | WER% |
|---|---|
| commonvoice | 29.27 |
| fleurs | 29.57 |
| indictts | 18.97 |
| kathbath | 26.99 |
| kathbath_noisy | 28.94 |
| mucs | **26.99** |
| R12 | 50.44 |

Only mel-spectrogram-input encoder in this comparison (the rest are raw-waveform). Wins on
`mucs`, otherwise mid-pack.
