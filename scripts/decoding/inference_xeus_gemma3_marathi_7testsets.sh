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