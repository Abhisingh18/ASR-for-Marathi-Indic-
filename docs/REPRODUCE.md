# Reproducing a run end-to-end

1. Prepare `train_mr_en_merged.jsonl` / `dev_mr_en_merged.jsonl` (schema in `docs/DATASET.md`).
2. Pick a training script from `scripts/training/`, set `GPU_INCLUDE`, and fire it:
   ```bash
   GPU_INCLUDE=0,1,2,3 bash scripts/training/finetune_data2vec_gemma3_marathi_cs.sh
   ```
   Add `DRYRUN=1` first to sanity-check paths/config without touching a GPU.
3. Once trained, rewrite the test-set paths (`docs/DECODING.md`) and run the matching decode
   script from `scripts/decoding/`, pointing `CKPT_DIR` at your checkpoint if not the default.
4. Run `compute_wer_marathi_7testsets.sh` (set `DECODE_DIR` to match the decode script's output).
5. Compare against `results/SUMMARY.md`.
