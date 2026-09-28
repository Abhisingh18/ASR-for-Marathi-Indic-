#!/bin/bash
# Inference: XEUS (CMU/ESPnet, 577M, pretrained-only, frozen) + linear projector +
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
export CUDA_VISIBLE_DEVICES=${GPU:-8}
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

speech_encoder_path=/speech/abhishek/kannada_data/encoders/XEUS_repo/model/xeus_checkpoint_new.pth
llm_path=/speech/abhishek/SLAM_Hindi/models/google/gemma-3-4b-it

output_dir=/speech/abhishek/output/marathi-cs-xeus-normed-gemma3-4b-finetuned
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

    attempt=1
    max_attempts=5
    until /speech/abhishek/miniconda3/envs/slam_llm/bin/python $code_dir/inference_asr_batch_new.py \
        --config-path "conf" \
        --config-name "prompt_marathi_ctx.yaml" \
        hydra.run.dir=$ckpt_dir \
        ++model_config.file=$code_dir/model/slam_model_asr_new.py:model_factory \
        ++model_config.llm_name=gemma-3-4b-it \
        ++model_config.llm_path=$llm_path \
        ++model_config.llm_dim=2560 \
        ++model_config.encoder_name=xeus \
        ++model_config.normalize=false \
        ++dataset_config.normalize=false \
        ++model_config.encoder_projector_ds_rate=5 \
        ++model_config.encoder_path=$speech_encoder_path \
        ++model_config.encoder_dim=1024 \
        ++model_config.encoder_projector=linear \
        ++dataset_config.dataset=speech_dataset \
        ++dataset_config.file=src/slam_llm/datasets/speech_dataset_marathi.py:get_speech_dataset \
        ++dataset_config.val_data_path=$val_data_path \
        ++dataset_config.input_type=raw \
        ++dataset_config.prompt_style=gemma2 \
        ++dataset_config.use_history_context=true \
        ++dataset_config.inference_mode=true \
        ++train_config.model_name=asr \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.use_peft=true \
        ++train_config.peft_config.peft_method=lora \
        ++train_config.peft_config.r=8 \
        ++train_config.peft_config.lora_alpha=32 \
        ++train_config.peft_config.target_modules=$lora_targets \
        ++train_config.peft_config.lora_dropout=0.05 \
        ++train_config.peft_config.bias=none \
        ++train_config.peft_config.task_type=CAUSAL_LM \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=2 \
        ++train_config.num_workers_dataloader=2 \
        ++train_config.output_dir=$output_dir \
        ++decode_log=$decode_log \
        ++ckpt_path=$ckpt_dir/pytorch_model.bin \
        ++log_config.log_file=${decode_log}.log \
        ++log_config.use_wandb=false
    do
        echo "[retry] $split attempt $attempt failed (exit $?)"
        attempt=$((attempt+1))
        if [ $attempt -gt $max_attempts ]; then
            echo "[FATAL] $split failed after $max_attempts attempts, skipping"
            break
        fi
        sleep 10
    done

    echo "Done: $split"
    echo "GT  : ${decode_log}_gt"
    echo "PRED: ${decode_log}_pred"
    echo ""
done

echo "All 7 Marathi testsets done! Results in: $decode_dir/"
