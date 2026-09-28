# Encoders compared

| Encoder | Params | Pretraining | Marathi fine-tuned? | Input | Dim |
|---|---|---|---|---|---|
| data2vec-AQC (FT) | ~313M | SPRING-INX multilingual SSL, then Marathi CTC FT | Yes | raw wav | 1024 |
| data2vec-AQC (SSL) | ~313M | SPRING-INX multilingual SSL only | No | raw wav | 1024 |
| ccc-wav2vec2.0 (FT) | ~315M | fairseq wav2vec2, contrastive+CTC, Marathi FT | Yes | raw wav | 1024 |
| Whisper large-v3 | ~635M (encoder) | OpenAI, 680k+ hours multilingual | No | log-mel (128 bins) | 1280 |
| XEUS | 577M | CMU/ESPnet, ~1M-hour multilingual SSL | No | raw wav | 1024 |

All encoders are **frozen** during SLAM-ASR training -- only the linear projector and the LLM's
LoRA adapters receive gradients. This isolates each encoder's out-of-the-box representation
quality for Marathi+English speech.

## Why this comparison matters

Marathi fine-tuning gives a real, measurable edge (see `results/SUMMARY.md`): both
Marathi-fine-tuned encoders (data2vec-AQC FT, ccc-wav2vec2 FT) outperform every pretrained-only
encoder on every complete test set. Among the pretrained-only encoders, Whisper (680k+ hours,
many of them likely Indian-language) edges out XEUS and the SSL-only data2vec-AQC variant on
most sets.
