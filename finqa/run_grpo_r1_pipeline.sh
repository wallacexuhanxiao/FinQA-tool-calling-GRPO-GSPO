#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/gui-grounding-agent
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

mkdir -p logs/finqa outputs/finqa/predictions outputs/finqa/metrics saves/finqa data/finqa/processed/grpo

export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo "[$(date '+%F %T')] $*" | tee -a logs/finqa/grpo_r1_pipeline.log; }

log "STEP 1 prepare GRPO train subset"
python scripts/finqa/prepare_finqa_grpo.py \
  --input_jsonl external/LLaMA-Factory/data/finqa_train_full.jsonl \
  --output_jsonl data/finqa/processed/grpo/finqa_grpo_train1500.jsonl \
  --limit 1500 \
  --seed 42 2>&1 | tee logs/finqa/prepare_grpo_r1.log

log "STEP 2 train GRPO r1: 7B, num_generations=4, max_prompt_length=1792, max_completion_length=96"
python scripts/finqa/train_finqa_grpo.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --sft_adapter_path saves/finqa/qwen25-7b-finqa-full-sft \
  --train_jsonl data/finqa/processed/grpo/finqa_grpo_train1500.jsonl \
  --output_dir saves/finqa/qwen25-7b-finqa-sft-grpo-r1 \
  --max_steps 100 \
  --num_generations 4 \
  --max_prompt_length 1792 \
  --max_completion_length 96 \
  --learning_rate 5e-7 2>&1 | tee logs/finqa/train_grpo_r1.log

log "STEP 3 infer GRPO r1 dev200"
PYTHONPATH=. python eval/infer_finqa_qwen.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/finqa/qwen25-7b-finqa-sft-grpo-r1 \
  --input_jsonl external/LLaMA-Factory/data/finqa_dev.jsonl \
  --output_jsonl outputs/finqa/predictions/grpo_r1_dev200.jsonl \
  --limit 200 \
  --max_new_tokens 96 2>&1 | tee logs/finqa/infer_grpo_r1_dev200.log

log "STEP 4 eval GRPO r1 dev200"
python eval/eval_finqa_answer.py \
  --pred_jsonl outputs/finqa/predictions/grpo_r1_dev200.jsonl \
  --out_metrics outputs/finqa/metrics/grpo_r1_dev200_metrics.json 2>&1 | tee logs/finqa/eval_grpo_r1_dev200.log

log "GRPO r1 pipeline completed"
