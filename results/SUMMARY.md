# Results Summary

**Checkpoint:** all 6 models are evaluated at the end of **epoch 1** (step 32,000) of a bilingual
Marathi+English code-switch fine-tune, so the comparison is controlled — same data, same steps,
same LoRA recipe, only the frozen speech encoder (and, for one row, the LLM) differs.

**Metric:** WER%, computed with `akshaya_wer.py` (NFC-normalized, punctuation removed).

## WER% — six complete testsets

| Test set | data2vec-FT+Gemma3 | ccc-wav2vec2+Gemma3 | Sarvam-1+FT | data2vec-SSL+Gemma3 | Whisper+Gemma3 | XEUS+Gemma3 |
|---|---|---|---|---|---|---|
| commonvoice      | **22.49** | 24.16 | 23.53 | 25.23 | 29.27 | 35.80 |
| fleurs           | **23.88** | 26.57 | 26.77 | 26.57 | 29.57 | 30.43 |
| indictts         | **16.15** | 18.26 | 17.26 | 19.27 | 18.97 | 21.90 |
| kathbath         | **21.29** | 23.74 | 22.73 | 25.04 | 26.99 | 31.96 |
| kathbath_noisy   | **22.14** | 23.97 | 23.22 | 27.25 | 28.94 | 33.38 |
| mucs             | 29.24 | 27.96 | 43.62 | 34.82 | **26.99** | 35.28 |

**Bold** = best (lowest WER) for that test set.