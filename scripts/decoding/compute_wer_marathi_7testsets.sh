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

fail=0
for split in "${testsets[@]}"; do
    gt=$decode_dir/decode_${split}_gt
    pred=$decode_dir/decode_${split}_pred
    out=$wer_dir/wer_${split}

    if [ ! -s "$gt" ] || [ ! -s "$pred" ]; then
        echo "$split: NO RESULT (gt/pred missing or empty)" >&2
        fail=1
        continue
    fi

    $py "$script" "$gt" "$pred" "$out"
done

echo "=== Summary ($wer_dir) ==="
for split in "${testsets[@]}"; do
    out=$wer_dir/wer_${split}
    if [ -f "$out" ]; then
        echo "$split: $(head -1 "$out")"
    else
        echo "$split: MISSING"
    fi
done
if [ $fail -ne 0 ]; then
    echo "[WARN] some testsets had no result -- check the inference log" >&2
fi
