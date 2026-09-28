# Encoders compared

| Encoder | Params | Pretraining | Marathi fine-tuned? | Input | Dim |
|---|---|---|---|---|---|
| data2vec-AQC (FT) | ~313M | SPRING-INX multilingual SSL, then Marathi CTC FT | Yes | raw wav | 1024 |
| data2vec-AQC (SSL) | ~313M | SPRING-INX multilingual SSL only | No | raw wav | 1024 |
| ccc-wav2vec2.0 (FT) | ~315M | fairseq wav2vec2, contrastive+CTC, Marathi FT | Yes | raw wav | 1024 |
| Whisper large-v3 | ~635M (encoder) | OpenAI, 680k+ hours multilingual | No | log-mel (128 bins) | 1280 |
| XEUS | 577M | CMU/ESPnet, ~1M-hour multilingual SSL | No | raw wav | 1024 |