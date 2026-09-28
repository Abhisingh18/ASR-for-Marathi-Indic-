#!/bin/bash
# Compute WER for the Marathi 7-testset decode (run after
# inference_data2vec_gemma3_marathi_7testsets.sh completes).
# Uses akshaya_wer.py (same one used in the bilingual/Kannada decode scripts).
# Override DECODE_DIR=... to match whatever CKPT_DIR the inference script used.

output_dir=/speech/abhishek/output/marathi-cs-data2vec-ft-gemma3-4b-finetuned
decode_dir=${DECODE_DIR:-$output_dir/decode_results_asr_epoch_1_step_32000_7testsets}
wer_dir=$decode_dir/wer
mkdir -p "$wer_dir"

py=/speech/abhishek/miniconda3/envs/slam_llm/bin/python3
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
script=$script_dir/akshaya_wer.py

testsets=(
    commonvoice_marathi_test
    fleurs_marathi_test
    indictts_marathi_test
    kathbath_marathi_test
    kathbath_noisy_marathi_test
    mucs_marathi_test
    R12_marathi_eval_filtered
)