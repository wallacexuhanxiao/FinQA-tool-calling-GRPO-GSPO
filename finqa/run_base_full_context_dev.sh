#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/gui-grounding-agent
mkdir -p outputs/finqa/predictions outputs/finqa/metrics logs/finqa
log(){ echo "[$(date '+%F %T')] $*"; }
log "infer BASE full-context dev with Qwen2.5-7B-Instruct, no adapter"
PYTHONPATH=. python eval/infer_finqa_qwen.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --input_jsonl external/LLaMA-Factory/data/finqa_dev.jsonl \
  --output_jsonl outputs/finqa/predictions/base_qwen25_7b_full_context_dev.jsonl \
  --max_new_tokens 256
log "eval BASE full-context dev"
python eval/eval_finqa_answer_exec.py \
  --pred_jsonl outputs/finqa/predictions/base_qwen25_7b_full_context_dev.jsonl \
  --out_metrics outputs/finqa/metrics/base_qwen25_7b_full_context_dev_answer_exec_metrics.json
log "done"
