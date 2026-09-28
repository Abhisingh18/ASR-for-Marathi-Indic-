# Decoding pipeline

Each of the 6 decode scripts (`scripts/decoding/inference_*.sh`) runs one model's checkpoint over
all 7 test sets on a single GPU, with a 5-attempt retry loop per test set (the shared cluster
occasionally throws a transient CUDA fault mid-`generate()`).

Steps to reproduce:

1. **Rewrite test-set audio paths.** The test-set JSONLs were originally prepared on a different
   host, so their `source` fields point to that host's filesystem. `rewrite_marathi_testset_paths.py`
   rewrites them to wherever you've placed the audio locally, and reports any files it can't find.
2. **Run inference**, e.g.:
   ```bash
   GPU=6 bash scripts/decoding/inference_data2vec_gemma3_marathi_7testsets.sh
   ```
   Override `CKPT_DIR=...` to decode a different checkpoint than the epoch-1/step-32000 default.
3. **Score.** `compute_wer_marathi_7testsets.sh` runs `akshaya_wer.py` (NFC-normalized WER/CER)
   over each test set's `_gt`/`_pred` files and writes one `wer_<testset>` file per test set.

## Cross-host data transfer

The test-set audio and transcripts lived on a separate CDAC HPC cluster whose login node has no
direct route to the training host (outbound SSH blocked both ways). The working path was a
two-hop bridge through a third machine that had a route to both: pull from CDAC to that machine's
local disk, then push from there to the training host. Ordinary single-command `rsync` between
two remote hosts doesn't work here -- rsync requires at least one side to be local.
