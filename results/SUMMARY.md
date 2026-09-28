# Results Summary

**Checkpoint:** all 6 models are evaluated at the end of **epoch 1** (step 32,000) of a bilingual
Marathi+English code-switch fine-tune, so the comparison is controlled — same data, same steps,
same LoRA recipe, only the frozen speech encoder (and, for one row, the LLM) differs.

**Metric:** WER%, computed with `akshaya_wer.py` (NFC-normalized, punctuation removed).