#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/gui-grounding-agent

SESSION_NAME="manual"
LOG="logs/finqa/grpo_r5_exec_consistency_pipeline.log"
OUT="saves/finqa/qwen25-7b-sftmerged-grpo-r5-exec-consistency-r32-g10"
PRED="outputs/finqa/predictions/grpo_r5_exec_consistency_dev.jsonl"
MET="outputs/finqa/metrics/grpo_r5_exec_consistency_dev_metrics.json"
EXMET="outputs/finqa/metrics/grpo_r5_exec_consistency_dev_exec_metrics.json"
TRAINLOG="logs/finqa/train_grpo_r5_exec_consistency.log"
INFERLOG="logs/finqa/infer_grpo_r5_exec_consistency_dev.log"

mkdir -p logs/finqa outputs/finqa/predictions outputs/finqa/metrics saves/finqa

echo "[$(date '+%F %T')] STEP 1 train GRPO r5 exec-consistency: SFT-merged base, full train, rank32, g10, grad_acc=10, steps=1200" | tee "$LOG"
PYTHONPATH=. python scripts/finqa/train_finqa_grpo.py \
  --model_path models/Qwen2.5-7B-Instruct-FinQA-SFT-Merged \
  --sft_adapter_path "" \
  --train_jsonl data/finqa/processed/grpo/finqa_grpo_train_full.jsonl \
  --output_dir "$OUT" \
  --init_lora_rank 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --num_generations 10 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 10 \
  --max_prompt_length 1792 \
  --max_completion_length 96 \
  --learning_rate 5e-7 \
  --temperature 0.9 \
  --top_p 0.95 \
  --max_steps 1200 \
  2>&1 | tee "$TRAINLOG"

echo "[$(date '+%F %T')] STEP 2 infer full dev" | tee -a "$LOG"
PYTHONPATH=. python eval/infer_finqa_qwen.py \
  --model_path models/Qwen2.5-7B-Instruct-FinQA-SFT-Merged \
  --adapter_path "$OUT" \
  --input_jsonl external/LLaMA-Factory/data/finqa_dev.jsonl \
  --output_jsonl "$PRED" \
  --max_new_tokens 96 \
  2>&1 | tee "$INFERLOG"

echo "[$(date '+%F %T')] STEP 3 eval answer metrics" | tee -a "$LOG"
python eval/eval_finqa_answer.py --pred_jsonl "$PRED" --out_metrics "$MET" \
  2>&1 | tee logs/finqa/eval_grpo_r5_exec_consistency_dev.log

echo "[$(date '+%F %T')] STEP 4 eval execution metrics" | tee -a "$LOG"
PYTHONPATH=. python eval/eval_finqa_answer_exec.py --pred_jsonl "$PRED" --out_metrics "$EXMET" \
  2>&1 | tee logs/finqa/eval_grpo_r5_exec_consistency_dev_exec.log

echo "[$(date '+%F %T')] GRPO r5 exec-consistency pipeline completed" | tee -a "$LOG"
