#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/gui-grounding-agent
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

mkdir -p logs/finqa outputs/finqa/predictions outputs/finqa/metrics saves/finqa data/finqa/processed/grpo

export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MAIN_LOG=logs/finqa/grpo_r2_full_with_fallback.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$MAIN_LOG"; }

log "STEP 1 prepare full GRPO dataset"
python scripts/finqa/prepare_finqa_grpo.py \
  --input_jsonl external/LLaMA-Factory/data/finqa_train_full.jsonl \
  --output_jsonl data/finqa/processed/grpo/finqa_grpo_train_full.jsonl \
  --limit 0 \
  --seed 42 2>&1 | tee logs/finqa/prepare_grpo_r2_full.log

run_train() {
  local name="$1"
  local gens="$2"
  local grad_acc="$3"
  local out_dir="$4"
  local train_log="$5"
  log "STEP 2 train ${name}: num_generations=${gens}, grad_acc=${grad_acc}, prompt=1792, completion=96, max_steps=782"
  python scripts/finqa/train_finqa_grpo.py \
    --model_path models/Qwen2.5-7B-Instruct \
    --sft_adapter_path saves/finqa/qwen25-7b-finqa-full-sft \
    --train_jsonl data/finqa/processed/grpo/finqa_grpo_train_full.jsonl \
    --output_dir "$out_dir" \
    --max_steps 782 \
    --num_generations "$gens" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps "$grad_acc" \
    --max_prompt_length 1792 \
    --max_completion_length 96 \
    --learning_rate 5e-7 2>&1 | tee "$train_log"
}

ACTIVE_NAME="r2_full_g8"
ACTIVE_OUT="saves/finqa/qwen25-7b-finqa-sft-grpo-r2-full-g8"
if ! run_train "r2_full_g8" 8 8 "$ACTIVE_OUT" logs/finqa/train_grpo_r2_full_g8.log; then
  log "G8 training failed; fallback to num_generations=6, grad_acc=4"
  ACTIVE_NAME="r2_full_g6_fallback"
  ACTIVE_OUT="saves/finqa/qwen25-7b-finqa-sft-grpo-r2-full-g6-fallback"
  run_train "r2_full_g6_fallback" 6 4 "$ACTIVE_OUT" logs/finqa/train_grpo_r2_full_g6_fallback.log
fi

log "STEP 3 infer ${ACTIVE_NAME} full dev"
PYTHONPATH=. python eval/infer_finqa_qwen.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path "$ACTIVE_OUT" \
  --input_jsonl external/LLaMA-Factory/data/finqa_dev.jsonl \
  --output_jsonl "outputs/finqa/predictions/${ACTIVE_NAME}_dev.jsonl" \
  --max_new_tokens 96 2>&1 | tee "logs/finqa/infer_${ACTIVE_NAME}_dev.log"

log "STEP 4 eval ${ACTIVE_NAME} full dev"
python eval/eval_finqa_answer.py \
  --pred_jsonl "outputs/finqa/predictions/${ACTIVE_NAME}_dev.jsonl" \
  --out_metrics "outputs/finqa/metrics/${ACTIVE_NAME}_dev_metrics.json" 2>&1 | tee "logs/finqa/eval_${ACTIVE_NAME}_dev.log"

log "GRPO r2 full with fallback completed: ${ACTIVE_NAME}"
