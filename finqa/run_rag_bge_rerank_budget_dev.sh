#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/gui-grounding-agent
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
mkdir -p logs/finqa outputs/finqa/rag/predictions outputs/finqa/rag/metrics

PYTHONPATH=. python eval/infer_finqa_qwen.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/finqa/qwen25-7b-finqa-dpo-clean-balanced \
  --input_jsonl data/finqa/processed/rag/generation/dense_bge_small_dev_top15_bge_rerank_budget900.jsonl \
  --output_jsonl outputs/finqa/rag/predictions/dpo_clean_balanced_dense_bge_small_dev_top15_bge_rerank_budget900.jsonl \
  --max_new_tokens 256 \
  2>&1 | tee logs/finqa/infer_dpo_clean_balanced_dense_top15_bge_rerank_budget900_dev.log

python eval/eval_finqa_answer_exec.py \
  --pred_jsonl outputs/finqa/rag/predictions/dpo_clean_balanced_dense_bge_small_dev_top15_bge_rerank_budget900.jsonl \
  --out_metrics outputs/finqa/rag/metrics/dpo_clean_balanced_dense_bge_small_dev_top15_bge_rerank_budget900_answer_exec_metrics.json \
  2>&1 | tee logs/finqa/eval_dpo_clean_balanced_dense_top15_bge_rerank_budget900_dev.log
