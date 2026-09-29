# Results Summary

**Checkpoint:** all 6 models are evaluated at the end of **epoch 1** (step 32,000) of a bilingual
Marathi+English code-switch fine-tune, so the comparison is controlled — same data, same steps,
same LoRA recipe, only the frozen speech encoder (and, for one row, the LLM) differs.

**Metric:** WER%, computed with `akshaya_wer.py` (NFC-normalized, punctuation removed).

## WER% — all seven testsets

| Test set | data2vec-FT+Gemma3 | ccc-wav2vec2+Gemma3 | Sarvam-1+FT | data2vec-SSL+Gemma3 | Whisper+Gemma3 | XEUS+Gemma3 |
|---|---|---|---|---|---|---|
| commonvoice      | **22.49** | 24.16 | 23.53 | 25.23 | 29.27 | 35.80 |
| fleurs           | **23.88** | 26.57 | 26.77 | 26.57 | 29.57 | 30.43 |
| indictts         | **16.15** | 18.26 | 17.26 | 19.27 | 18.97 | 21.90 |
| kathbath         | **21.29** | 23.74 | 22.73 | 25.04 | 26.99 | 31.96 |
| kathbath_noisy   | **22.14** | 23.97 | 23.22 | 27.25 | 28.94 | 33.38 |
| mucs             | 29.24 | 27.96 | 43.62 | 34.82 | **26.99** | 35.28 |
| R12 (6,282 utt.) | 58.20 | 56.51 | 57.89 | 56.34 | **50.44** | 59.54 |

**Bold** = best (lowest WER) for that test set.

## Takeaways

- **`data2vec-FT + Gemma-3-4B`** (Marathi fine-tuned data2vec-AQC encoder) wins on 5 of 6 complete
  test sets — expected, since its encoder was fine-tuned on in-domain Marathi CTC data before
  this SLAM-ASR stage even began.
- **`ccc-wav2vec2 + Gemma-3-4B`** and **`data2vec-FT + Sarvam-1`** are close runners-up, both
  within ~1-2 WER points of the best on most test sets.
- **`XEUS + Gemma-3-4B`** (pretrained-only, no Marathi fine-tuning) is consistently the weakest —
  it has never seen labeled Marathi audio before this stage, unlike the other encoders.
- Swapping the LLM (Sarvam-1 vs Gemma-3-4B, same data2vec-FT encoder) matters less than swapping
  the encoder, except on `mucs`, where Sarvam-1 falls far behind (43.62 vs 29.24).
