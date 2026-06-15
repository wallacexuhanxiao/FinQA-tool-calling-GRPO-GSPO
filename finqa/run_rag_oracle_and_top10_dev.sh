#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/gui-grounding-agent
mkdir -p outputs/finqa/rag/predictions outputs/finqa/rag/metrics logs/finqa
log(){ echo "[$(date '+%F %T')] $*"; }
MODEL=models/Qwen2.5-7B-Instruct
ADAPTER=saves/finqa/qwen25-7b-finqa-dpo-clean-balanced

log "infer DPO clean-balanced on oracle gold expanded dev"
PYTHONPATH=. python eval/infer_finqa_qwen.py \
  --model_path "$MODEL" \
  --adapter_path "$ADAPTER" \
  --input_jsonl data/finqa/processed/rag/generation/oracle_gold_dev_expanded.jsonl \
  --output_jsonl outputs/finqa/rag/predictions/dpo_clean_balanced_oracle_gold_dev_expanded.jsonl \
  --max_new_tokens 256
log "eval oracle gold expanded dev"
python eval/eval_finqa_answer_exec.py \
  --pred_jsonl outputs/finqa/rag/predictions/dpo_clean_balanced_oracle_gold_dev_expanded.jsonl \
  --out_metrics outputs/finqa/rag/metrics/dpo_clean_balanced_oracle_gold_dev_expanded_answer_exec_metrics.json

log "infer DPO clean-balanced on dense top10 expanded dev"
PYTHONPATH=. python eval/infer_finqa_qwen.py \
  --model_path "$MODEL" \
  --adapter_path "$ADAPTER" \
  --input_jsonl data/finqa/processed/rag/generation/dense_bge_small_dev_top10_expanded.jsonl \
  --output_jsonl outputs/finqa/rag/predictions/dpo_clean_balanced_dense_bge_small_dev_top10_expanded.jsonl \
  --max_new_tokens 256
log "eval dense top10 expanded dev"
python eval/eval_finqa_answer_exec.py \
  --pred_jsonl outputs/finqa/rag/predictions/dpo_clean_balanced_dense_bge_small_dev_top10_expanded.jsonl \
  --out_metrics outputs/finqa/rag/metrics/dpo_clean_balanced_dense_bge_small_dev_top10_expanded_answer_exec_metrics.json
log "done"
