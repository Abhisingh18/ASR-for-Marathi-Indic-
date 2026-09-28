#!/bin/bash
# Inference: ccc-wav2vec2.0 (SPRING-INX, Marathi CTC fine-tuned, frozen) + linear projector +
# Gemma-3-4B-IT + LoRA, on 7 benchmark Marathi(+English CS) testsets pulled
# from CDAC (nithyar's dump_slam30, "data_ENMR"): commonvoice, fleurs,
# indictts, kathbath, kathbath_noisy, mucs, R12_eval_filtered.
#
# Template: inference_data2vec_gemma3_kannada_dualencoder_4testsets.sh, minus
# the dual-encoder fusion + this run's dataset_config.file override (Marathi
# needs speech_dataset_marathi.py for the mr/en/mr-en language note, same as
# training: finetune_data2vec_gemma3_marathi_cs.sh).
#
# BEFORE RUNNING: the testset jsonls' "source" fields are CDAC absolute paths
# (/nlsasfs/home/dibd/...) -- rewrite them to local audio paths first with
# rewrite_marathi_testset_paths.py (see that script's header).
#
# Override CKPT_DIR=... to decode a different checkpoint (default: the
# completed FT+Gemma-3 run's final checkpoint, step 32000/epoch 1).
# Override GPU=... (default 6) -- re-check nvidia-smi first.

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTHONPATH=/speech/abhishek/fairseq:$PYTHONPATH
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:/speech/abhishek/miniconda3/envs/slam_llm/bin:$PATH
export NVIDIA_TF32_OVERRIDE=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export HYDRA_FULL_ERROR=1

run_dir=/speech/abhishek/SLAM-LLM
cd $run_dir
code_dir=examples/asr_librispeech

speech_encoder_path=/speech/abhishek/marathi_data/encoders/ccc_wav2vec2/SPRING_INX_ccc_wav2vec2_Marathi.pt
llm_path=/speech/abhishek/SLAM_Hindi/models/google/gemma-3-4b-it

output_dir=/speech/abhishek/output/marathi-cs-ccc-wav2vec2-ft-gemma3-4b-finetuned
ckpt_dir=${CKPT_DIR:-$output_dir/asr_epoch_1_step_32000}

decode_dir=$output_dir/decode_results_$(basename $ckpt_dir)_7testsets
mkdir -p $decode_dir

test_dir=/speech/abhishek/marathi_data/decode_test/data_ENMR

lora_targets=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]

testsets=(
    commonvoice_marathi_test
    fleurs_marathi_test
    indictts_marathi_test
    kathbath_marathi_test
    kathbath_noisy_marathi_test
    mucs_marathi_test
    R12_marathi_eval_filtered
)

for split in "${testsets[@]}"; do
    echo "========================================================"
    echo "Decoding: $split"
    echo "========================================================"

    val_data_path=$test_dir/${split}_output_prev.local.jsonl
    decode_log=$decode_dir/decode_${split}

    if [ ! -f "$val_data_path" ]; then
        echo "[FATAL] $val_data_path not found -- run rewrite_marathi_testset_paths.py first" >&2
        continue
    fi