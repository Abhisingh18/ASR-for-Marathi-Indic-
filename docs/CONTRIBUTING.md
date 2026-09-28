# Adding a new encoder or LLM to this comparison

To keep results comparable, a new run should change **exactly one thing** from an existing script
in `scripts/training/` — copy the closest match and diff it against this checklist:

1. `model_config.encoder_name` / `encoder_path` / `encoder_dim` (or `llm_name` / `llm_path` /
   `llm_dim` for an LLM swap).
2. `dataset_config.input_type` (`raw` for waveform encoders, `mel` for Whisper) and `mel_size` if
   applicable.
3. `dataset_config.normalize` — match whatever the encoder's own preprocessing expects.
4. Everything else (LoRA config, batch size, LR, warmup, data, prompt template) should be
   **left untouched**, so any WER difference is attributable to the one thing you changed.
5. Build the matching decode script the same way: copy the closest `scripts/decoding/inference_*`
   script and swap only the encoder/LLM block (see `docs/DECODING.md` for the `model_config.file`
   gotcha — it must match whatever model factory the training script used).
6. Run all 7 test sets and add a `results/<YourModel>/` folder with the same layout as the
   existing ones (`config.json`, `README.md`, `wer_*` per test set), then update
   `results/SUMMARY.md` and the main `README.md` results table.
