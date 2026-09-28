#!/bin/bash
# Marathi+English bilingual SLAM-ASR: data2vec-AQC (Marathi fine-tuned, frozen) + linear projector
# + Gemma-3-4B-IT (LoRA r=8/alpha=32). Same recipe/hyper-parameters as the Kannada baseline
# (finetune_data2vec_gemma3_kannada_cs.sh) so results are comparable; only the data/encoder are Marathi.
# Effective batch 32 = micro-batch 4 x n_gpus x grad-accum (grad-accum = 8 / n_gpus is set automatically).
#
# PORTABLE: written to be fired from gpu18 (or gpu17) -- every machine-specific thing is an env var.
# Nothing here modifies an existing file; it uses the *_new pipeline (full wandb curves) and the
# speech_dataset_marathi.py loader copy (adds the `mr-en` language note).
#
# Minimal usage (on the machine whose GPUs you want to use; CUDA indices, PCI_BUS_ID order):
#   GPU_INCLUDE=0,1,2,3 bash finetune_data2vec_gemma3_marathi_cs.sh
# See what it WOULD run, without touching any GPU:
#   DRYRUN=1 GPU_INCLUDE=0,1,2,3 bash finetune_data2vec_gemma3_marathi_cs.sh
#
# Overridable env vars (defaults assume the gpu17 tree was copied to the SAME paths on the target machine):
#   GPU_INCLUDE   (required) e.g. 0,1,2,3 -- 1,2,4 or 8 GPUs (must divide 8 to keep batch 32)
#   SLAM_DIR      /speech/abhishek/SLAM-LLM          FAIRSEQ_DIR /speech/abhishek/fairseq
#   ENV_BIN       /speech/abhishek/miniconda3/envs/slam_llm/bin   (folder with python + deepspeed)
#   DATA_DIR      /speech/abhishek/marathi_data      (jsonl/ + audio/ + encoders/)
#   ENCODER_PATH  $DATA_DIR/encoders/SPRING_INX_data2vec_aqc_Marathi.pt
#   LLM_PATH      /speech/abhishek/SLAM_Hindi/models/google/gemma-3-4b-it
#   TRAIN_JSONL   $DATA_DIR/jsonl/train_mr_en_merged.jsonl   DEV_JSONL $DATA_DIR/jsonl/dev_mr_en_merged.jsonl
#   OUTPUT_DIR    /speech/abhishek/output/marathi-cs-data2vec-ft-gemma3-4b-finetuned
#   MASTER_PORT   29527      NUM_EPOCHS 1 (run stops by itself; last checkpoint = last multiple of 1000 steps)
#   USE_WANDB     auto|true|false (auto = on only if `wandb login` was done / WANDB_API_KEY set)
#   RESUME_CKPT   unset = auto-resume from the latest complete checkpoint in OUTPUT_DIR; none = fresh run; or a path
#   FORCE=1       skip the "GPUs must be free and >=40 GB" safety check
#   CUDA_HOME     /usr/local/cuda-12.4 (nvcc is needed by DeepSpeed's fused AdamW JIT build)

set -e

GPU_INCLUDE=${GPU_INCLUDE:?set GPU_INCLUDE to the CUDA indices to use, e.g. GPU_INCLUDE=0,1,2,3 (check nvidia-smi first; never use GPUs other people are using)}
SLAM_DIR=${SLAM_DIR:-/speech/abhishek/SLAM-LLM}
FAIRSEQ_DIR=${FAIRSEQ_DIR:-/speech/abhishek/fairseq}
ENV_BIN=${ENV_BIN:-/speech/abhishek/miniconda3/envs/slam_llm/bin}
DATA_DIR=${DATA_DIR:-/speech/abhishek/marathi_data}
ENCODER_PATH=${ENCODER_PATH:-$DATA_DIR/encoders/SPRING_INX_data2vec_aqc_Marathi.pt}
LLM_PATH=${LLM_PATH:-/speech/abhishek/SLAM_Hindi/models/google/gemma-3-4b-it}
TRAIN_JSONL=${TRAIN_JSONL:-$DATA_DIR/jsonl/train_mr_en_merged.jsonl}
DEV_JSONL=${DEV_JSONL:-$DATA_DIR/jsonl/dev_mr_en_merged.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-/speech/abhishek/output/marathi-cs-data2vec-ft-gemma3-4b-finetuned}
MASTER_PORT=${MASTER_PORT:-29527}
NUM_EPOCHS=${NUM_EPOCHS:-1}
WANDB_ENTITY=${WANDB_ENTITY:-abhisingh964800-iit-madras-foundation}
WANDB_PROJECT=${WANDB_PROJECT:-Marathi_English_Encoder_Comparison}
WANDB_EXP_NAME=${WANDB_EXP_NAME:-data2vec-aqc-marathi-finetuned-bilingual-cs-gemma3-4b-it-lora}