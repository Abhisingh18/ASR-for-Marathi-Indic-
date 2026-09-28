#!/bin/bash
# Marathi+English bilingual SLAM-ASR: data2vec-AQC (Marathi fine-tuned, frozen) + linear projector
# + Gemma-3-4B-IT (LoRA r=8/alpha=32). Same recipe/hyper-parameters as the Kannada baseline
# (finetune_data2vec_gemma3_kannada_xeus_cs.sh). Identical to finetune_data2vec_qwen3_5_marathi_cs.sh except
# Effective batch 32 = micro-batch 4 x n_gpus x grad-accum (grad-accum = 8 / n_gpus is set automatically).
#
# PORTABLE: written to be fired from gpu18 (or gpu17) -- every machine-specific thing is an env var.
# Nothing here modifies an existing file; it uses the *_new pipeline (full wandb curves) and the
# speech_dataset_marathi.py loader copy (adds the `mr-en` language note).
#
# Minimal usage (on the machine whose GPUs you want to use; CUDA indices, PCI_BUS_ID order):
#   GPU_INCLUDE=6,7,8,9 bash finetune_data2vec_qwen3_5_marathi_cs.sh
# See what it WOULD run, without touching any GPU:
#   DRYRUN=1 GPU_INCLUDE=6,7,8,9 bash finetune_data2vec_qwen3_5_marathi_cs.sh
#
# Overridable env vars (defaults assume the gpu17 tree was copied to the SAME paths on the target machine):
#   GPU_INCLUDE   (required) e.g. 0,1,2,3 -- 1,2,4 or 8 GPUs (must divide 8 to keep batch 32)
#   SLAM_DIR      /speech/abhishek/SLAM-LLM          FAIRSEQ_DIR /speech/abhishek/fairseq
#   ENV_BIN       /speech/abhishek/miniconda3/envs/slam_llm/bin   (folder with python + deepspeed)
#   DATA_DIR      /speech/abhishek/marathi_data      (jsonl/ + audio/ + encoders/)
#   ENCODER_PATH  $DATA_DIR/encoders/SPRING_INX_data2vec_aqc_Marathi.pt
#   LLM_PATH      /speech/abhishek/hf_downloads/Qwen3.5-4B
#   TRAIN_JSONL   $DATA_DIR/jsonl/train_mr_en_merged.jsonl   DEV_JSONL $DATA_DIR/jsonl/dev_mr_en_merged.jsonl
#   OUTPUT_DIR    /speech/abhishek/output/marathi-cs-data2vec-qwen3_5-finetuned
#   MASTER_PORT   29532      NUM_EPOCHS 5 (run stops by itself; last checkpoint = last multiple of 1000 steps)
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
LLM_PATH=${LLM_PATH:-/speech/abhishek/hf_downloads/Qwen3.5-4B}
TRAIN_JSONL=${TRAIN_JSONL:-$DATA_DIR/jsonl/train_mr_en_merged.jsonl}
DEV_JSONL=${DEV_JSONL:-$DATA_DIR/jsonl/dev_mr_en_merged.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-/speech/abhishek/output/marathi-cs-data2vec-qwen3_5-finetuned}
MASTER_PORT=${MASTER_PORT:-29532}
NUM_EPOCHS=${NUM_EPOCHS:-5}
WANDB_ENTITY=${WANDB_ENTITY:-abhisingh964800-iit-madras-foundation}
WANDB_PROJECT=${WANDB_PROJECT:-Marathi_English_Encoder_Comparison}
WANDB_EXP_NAME=${WANDB_EXP_NAME:-data2vec-aqc-marathi-finetuned-bilingual-cs-qwen3_5-4b-it-lora}

# ---------------- environment (same flags as the Kannada Gemma-3 runs) ----------------
export PYTHONPATH=$SLAM_DIR/src:$FAIRSEQ_DIR:$PYTHONPATH    # our tree MUST win over any editable slam_llm install
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
[ -x "$ENV_BIN/python" ] && export PATH=$ENV_BIN:$PATH
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export DS_SKIP_CUDA_CHECK=1
export HYDRA_FULL_ERROR=1
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_DEBUG=WARN
# stability mitigations used for every Gemma run on the RTX 6000 Ada boxes (no TF32, deterministic cuBLAS)
export NVIDIA_TF32_OVERRIDE=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800

# ---------------- effective batch = 2 x n_gpus x grad_accum = 32 ----------------
# micro-batch dropped 4 -> 2 (grad_accum doubled to compensate, same effective
# batch 32) after this run OOM'd at step 4 on 47.5 GiB cards: Qwen3.5-4B's own
# memory footprint differs from Gemma-3-4B/Sarvam-1's, so the batch size that
# fit them left too little headroom for this model's longer training batches.
n_gpus=$(echo "$GPU_INCLUDE" | awk -F, '{print NF}')
if [ $((16 % n_gpus)) -ne 0 ]; then
    echo "[FATAL] n_gpus=$n_gpus does not divide 16; use 1, 2, 4, 8 or 16 GPUs to keep effective batch 32 at micro-batch 2" >&2
    exit 1
fi
grad_accum=$((16 / n_gpus))

cd "$SLAM_DIR"
code_dir=examples/asr_librispeech

# ---------------- pre-flight checks (fail early, before any GPU work) ----------------
command -v deepspeed >/dev/null || { echo "[FATAL] 'deepspeed' not found on PATH (set ENV_BIN to the env's bin dir)" >&2; exit 1; }
py_slam=$(python -c 'import slam_llm; print(slam_llm.__file__)')
case "$py_slam" in
    "$SLAM_DIR"/*) echo "[check] slam_llm from: $py_slam" ;;
    *) echo "[FATAL] slam_llm is imported from $py_slam, not from \$SLAM_DIR=$SLAM_DIR -- refusing to run the wrong code" >&2; exit 1 ;;
esac
python -c "import fairseq" 2>/dev/null || { echo "[FATAL] cannot import fairseq (FAIRSEQ_DIR=$FAIRSEQ_DIR)" >&2; exit 1; }
[ -x "$CUDA_HOME/bin/nvcc" ] || echo "[WARN] no nvcc at $CUDA_HOME/bin -- DeepSpeed's fused AdamW may fail to build (set CUDA_HOME to a CUDA 12.x toolkit)"
[ -f "$ENCODER_PATH" ] || { echo "[FATAL] encoder checkpoint not found: $ENCODER_PATH" >&2; exit 1; }
[ -f "$LLM_PATH/config.json" ] || { echo "[FATAL] Gemma-3-4B-IT not found at LLM_PATH=$LLM_PATH (need config.json + weights)" >&2; exit 1; }
[ -f "$TRAIN_JSONL" ] && [ -f "$DEV_JSONL" ] || { echo "[FATAL] train/dev jsonl not found ($TRAIN_JSONL / $DEV_JSONL)" >&2; exit 1; }
missing=$( (head -2 "$TRAIN_JSONL"; shuf -n 200 "$TRAIN_JSONL"; shuf -n 100 "$DEV_JSONL") | python -c "
import sys, json, os
bad = [json.loads(l)['source'] for l in sys.stdin if not os.path.isfile(json.loads(l)['source'])]
print(len(bad), bad[0] if bad else '')")
case "$missing" in
    "0 "*) echo "[check] sampled 300 train/dev audio paths: all exist" ;;
    *) echo "[FATAL] audio missing for sampled jsonl rows (count + first): $missing -- was the audio directory copied to the same path?" >&2; exit 1 ;;
esac

# the chosen GPUs must be free and big enough (protects other people's jobs and catches a CUDA-vs-nvidia-smi index mix-up)
if [ "${DRYRUN:-0}" != "1" ] && [ "${FORCE:-0}" != "1" ]; then
    CUDA_VISIBLE_DEVICES=$GPU_INCLUDE python - <<'EOF'
import sys, torch
n = torch.cuda.device_count()
ok = True
for i in range(n):
    free, total = torch.cuda.mem_get_info(i)
    name = torch.cuda.get_device_name(i)
    print(f"[gpu-check] cuda:{i} {name}  free {free/2**30:.1f} / {total/2**30:.1f} GiB")
    if total < 40 * 2**30 or free < 40 * 2**30:
        ok = False
if not ok:
    sys.exit("[FATAL] a selected GPU is busy or smaller than 40 GB (FORCE=1 to override). Check `nvidia-smi` and GPU_INCLUDE.")
EOF
fi

# ---------------- wandb ----------------
USE_WANDB=${USE_WANDB:-auto}
if [ "$USE_WANDB" = "auto" ]; then
    if [ -n "${WANDB_API_KEY:-}" ] || grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
        USE_WANDB=true
        # logged in, but can this machine reach wandb's server? (any HTTP reply counts; a timeout would otherwise crash wandb.init)
        if curl -sS -m 8 -o /dev/null https://api.wandb.ai 2>/dev/null; then
            echo "[wandb] logged in + server reachable -> online logging to $WANDB_ENTITY/$WANDB_PROJECT"
        else
            export WANDB_MODE=offline
            echo "[wandb] logged in but api.wandb.ai is NOT reachable from here -> WANDB_MODE=offline. Curves are saved under \${WANDB_DIR:-OUTPUT_DIR}/wandb; upload later from a machine with internet: wandb sync <that dir>/wandb/offline-run-*"
        fi
    else
        USE_WANDB=false
        echo "[wandb] not logged in on this machine -> wandb OFF. Run 'wandb login' (or export WANDB_API_KEY=...) and re-run to get the curves; or USE_WANDB=true WANDB_MODE=offline to record locally and sync later."
    fi
else
    echo "[wandb] USE_WANDB=$USE_WANDB (forced)  WANDB_MODE=${WANDB_MODE:-online}"
fi

# ---------------- DeepSpeed config: the Gemma-3 bilingual one, only grad-accum changed ----------------
mkdir -p "$OUTPUT_DIR"
ds_config_path=$OUTPUT_DIR/ds_config_marathi.json
python - <<EOF
import json
cfg = json.load(open("$SLAM_DIR/$code_dir/conf/ds_config_gemma3_bilingual.json"))
cfg["gradient_accumulation_steps"] = $grad_accum
json.dump(cfg, open("$ds_config_path", "w"), indent=2)
EOF
echo "[cfg] n_gpus=$n_gpus micro-batch=2 grad_accum=$grad_accum -> effective batch $((2 * n_gpus * grad_accum))   ds_config=$ds_config_path"

# ---------------- auto-resume ----------------
# A checkpoint is complete only if it has pytorch_model.bin, `latest` and one optimizer shard per GPU.
resume_ckpt=""
if [ "${RESUME_CKPT:-}" = "none" ]; then
    resume_ckpt=""
elif [ -n "${RESUME_CKPT:-}" ]; then
    resume_ckpt="$RESUME_CKPT"
else
    for d in $(ls -dt "$OUTPUT_DIR"/asr_epoch_*_step_* 2>/dev/null); do
        if [ -f "$d/pytorch_model.bin" ] && [ -f "$d/latest" ]; then
            gstep_dir="$d/$(cat "$d/latest" 2>/dev/null)"
            n_optim=$(ls "$gstep_dir"/bf16_zero_pp_rank_*_mp_rank_00_optim_states.pt 2>/dev/null | wc -l)
            if [ "$n_optim" -eq "$n_gpus" ]; then resume_ckpt="$d"; break; fi
        fi
    done
fi
resume_arg=""
if [ -n "$resume_ckpt" ]; then
    echo "[resume] resuming from: $resume_ckpt"
    # BOTH are needed: ckpt_path restores the weights, resume_ckpt fast-forwards the dataloader.
    resume_arg="++ckpt_path=$resume_ckpt/pytorch_model.bin ++train_config.resume_ckpt=$resume_ckpt"
else
    echo "[resume] no complete checkpoint -> fresh run in $OUTPUT_DIR"
fi

lora_targets=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]

hydra_args="
hydra.run.dir=$OUTPUT_DIR \
++model_config.file=$code_dir/model/slam_model_asr_qwen.py:model_factory \
++model_config.llm_name=qwen3_5 \
++model_config.llm_path=$LLM_PATH \
++model_config.llm_dim=2560 \
++model_config.encoder_name=data2vec_aqc \
++dataset_config.normalize=true \
++model_config.encoder_projector_ds_rate=5 \
++model_config.encoder_path=$ENCODER_PATH \
++model_config.encoder_dim=1024 \
++model_config.encoder_projector=linear \
++dataset_config.dataset=speech_dataset \
++dataset_config.file=src/slam_llm/datasets/speech_dataset_marathi.py:get_speech_dataset \
++dataset_config.train_data_path=$TRAIN_JSONL \
++dataset_config.val_data_path=$DEV_JSONL \
++dataset_config.input_type=raw \
++dataset_config.prompt_style=qwen3_5 \
++dataset_config.use_history_context=true \
++train_config.model_name=asr \
++train_config.num_epochs=$NUM_EPOCHS \
++train_config.enable_deepspeed=true \
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
++train_config.use_fp16=false \
++train_config.warmup_steps=1000 \
++train_config.total_steps=200000 \
++train_config.lr=1e-4 \
++train_config.validation_interval=1000 \
++train_config.checkpoint_interval=1000 \
++train_config.batch_size_training=2 \
++train_config.val_batch_size=2 \
++train_config.num_workers_dataloader=8 \
++train_config.output_dir=$OUTPUT_DIR \
$resume_arg \
++deepspeed_config=$ds_config_path \
++metric=acc \
++log_config.log_file=$OUTPUT_DIR/train.log \
++log_config.use_wandb=$USE_WANDB \
++log_config.wandb_dir=${WANDB_DIR:-$OUTPUT_DIR} \
++log_config.wandb_entity_name=$WANDB_ENTITY \
++log_config.wandb_project_name=$WANDB_PROJECT \
++log_config.wandb_exp_name=$WANDB_EXP_NAME \
++log_config.log_interval=5 \
"

cmd="deepspeed --include=localhost:$GPU_INCLUDE --master_port=$MASTER_PORT $code_dir/deepspeed_finetune_asr_new.py --config-path conf --config-name prompt_marathi_ctx.yaml $hydra_args"
if [ "${DRYRUN:-0}" = "1" ]; then
    echo "[DRYRUN] nothing launched. Command that would run:"; echo "$cmd" | tr -s ' ' | sed 's/ \(--\|++\|hydra\.\)/\n  \1/g'
    exit 0
fi

deepspeed \
    --include=localhost:$GPU_INCLUDE \
    --master_port=$MASTER_PORT \
    $code_dir/deepspeed_finetune_asr_new.py \
    --config-path "conf" \
    --config-name "prompt_marathi_ctx.yaml" \
    $hydra_args
