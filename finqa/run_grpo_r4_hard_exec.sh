#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/gui-grounding-agent
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

mkdir -p logs/finqa outputs/finqa/predictions outputs/finqa/metrics saves/finqa data/finqa/processed/grpo

export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG=logs/finqa/grpo_r4_hard_exec_pipeline.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

MERGED_MODEL=models/Qwen2.5-7B-Instruct-FinQA-SFT-Merged
TRAIN_JSONL=data/finqa/processed/grpo/finqa_grpo_r4_hard2500.jsonl
OUT_DIR=saves/finqa/qwen25-7b-finqa-grpo-r4-hard-exec-r16-g8
PRED_JSONL=outputs/finqa/predictions/grpo_r4_hard_exec_dev.jsonl
METRICS_JSON=outputs/finqa/metrics/grpo_r4_hard_exec_dev_metrics.json
EXEC_METRICS_JSON=outputs/finqa/metrics/grpo_r4_hard_exec_dev_exec_metrics.json

log "STEP 1 build hard GRPO set from train SFT candidates"
python scripts/finqa/build_grpo_hardset_r4.py \
  --out_jsonl "$TRAIN_JSONL" \
  --target_size 2500 \
  --wrong_size 1500 \
  --category_size 650 \
  --anchor_size 350 \
  --seed 43 2>&1 | tee logs/finqa/build_grpo_r4_hardset.log

log "STEP 2 train GRPO r4 hard exec: rank16, g8, temp=0.9, top_p=0.95, steps=500"
python scripts/finqa/train_finqa_grpo.py \
  --model_path "$MERGED_MODEL" \
  --sft_adapter_path "" \
  --init_lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target q_proj,k_proj,v_proj,o_proj \
  --train_jsonl "$TRAIN_JSONL" \
  --output_dir "$OUT_DIR" \
  --max_steps 500 \
  --num_generations 8 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_prompt_length 1792 \
  --max_completion_length 96 \
  --learning_rate 5e-7 \
  --temperature 0.9 \
  --top_p 0.95 2>&1 | tee logs/finqa/train_grpo_r4_hard_exec.log

log "STEP 3 infer full dev"
PYTHONPATH=. python eval/infer_finqa_qwen.py \
  --model_path "$MERGED_MODEL" \
  --adapter_path "$OUT_DIR" \
  --input_jsonl external/LLaMA-Factory/data/finqa_dev.jsonl \
  --output_jsonl "$PRED_JSONL" \
  --max_new_tokens 96 2>&1 | tee logs/finqa/infer_grpo_r4_hard_exec_dev.log

log "STEP 4 eval full dev answer metrics"
python eval/eval_finqa_answer.py \
  --pred_jsonl "$PRED_JSONL" \
  --out_metrics "$METRICS_JSON" 2>&1 | tee logs/finqa/eval_grpo_r4_hard_exec_dev.log

log "STEP 5 eval full dev execution metrics"
python eval/eval_finqa_answer_exec.py \
  --pred_jsonl "$PRED_JSONL" \
  --out_metrics "$EXEC_METRICS_JSON" 2>&1 | tee logs/finqa/eval_grpo_r4_hard_exec_dev_exec.log

log "GRPO r4 hard exec pipeline completed"
