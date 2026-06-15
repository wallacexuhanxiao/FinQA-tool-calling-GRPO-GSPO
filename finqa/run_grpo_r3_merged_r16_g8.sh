#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/gui-grounding-agent
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

mkdir -p logs/finqa outputs/finqa/predictions outputs/finqa/metrics saves/finqa data/finqa/processed/grpo models

export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG=logs/finqa/grpo_r3_merged_r16_g8_pipeline.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

MERGED_MODEL=models/Qwen2.5-7B-Instruct-FinQA-SFT-Merged
OUT_DIR=saves/finqa/qwen25-7b-finqa-sftmerged-grpo-r16-g8
TRAIN_JSONL=data/finqa/processed/grpo/finqa_grpo_train_full.jsonl
PRED_JSONL=outputs/finqa/predictions/grpo_r3_merged_r16_g8_dev.jsonl
METRICS_JSON=outputs/finqa/metrics/grpo_r3_merged_r16_g8_dev_metrics.json

log "STEP 1 merge SFT LoRA into base model if needed"
python scripts/finqa/merge_sft_lora.py \
  --base_model models/Qwen2.5-7B-Instruct \
  --adapter saves/finqa/qwen25-7b-finqa-full-sft \
  --output_dir "$MERGED_MODEL" 2>&1 | tee logs/finqa/merge_sft_lora_r3.log

log "STEP 2 prepare full GRPO dataset"
python scripts/finqa/prepare_finqa_grpo.py \
  --input_jsonl external/LLaMA-Factory/data/finqa_train_full.jsonl \
  --output_jsonl "$TRAIN_JSONL" \
  --limit 0 \
  --seed 42 2>&1 | tee logs/finqa/prepare_grpo_r3_full.log

log "STEP 3 train GRPO r3: merged SFT base + new LoRA rank16, num_generations=8, grad_acc=8"
python scripts/finqa/train_finqa_grpo.py \
  --model_path "$MERGED_MODEL" \
  --sft_adapter_path "" \
  --init_lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target q_proj,k_proj,v_proj,o_proj \
  --train_jsonl "$TRAIN_JSONL" \
  --output_dir "$OUT_DIR" \
  --max_steps 782 \
  --num_generations 8 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_prompt_length 1792 \
  --max_completion_length 96 \
  --learning_rate 5e-7 2>&1 | tee logs/finqa/train_grpo_r3_merged_r16_g8.log

log "STEP 4 infer full dev"
PYTHONPATH=. python eval/infer_finqa_qwen.py \
  --model_path "$MERGED_MODEL" \
  --adapter_path "$OUT_DIR" \
  --input_jsonl external/LLaMA-Factory/data/finqa_dev.jsonl \
  --output_jsonl "$PRED_JSONL" \
  --max_new_tokens 96 2>&1 | tee logs/finqa/infer_grpo_r3_merged_r16_g8_dev.log

log "STEP 5 eval full dev"
python eval/eval_finqa_answer.py \
  --pred_jsonl "$PRED_JSONL" \
  --out_metrics "$METRICS_JSON" 2>&1 | tee logs/finqa/eval_grpo_r3_merged_r16_g8_dev.log

log "GRPO r3 merged rank16 g8 pipeline completed"
