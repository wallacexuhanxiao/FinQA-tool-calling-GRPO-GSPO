#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/gui-grounding-agent
mkdir -p data/finqa/processed/rag/generation outputs/finqa/rag/predictions outputs/finqa/rag/metrics logs/finqa
log(){ echo "[$(date '+%F %T')] $*"; }
log "build RAG generation input: dense top5 dev"
python scripts/finqa/build_finqa_rag_generation_input.py \
  --retrieval_jsonl outputs/finqa/rag/retrieval/dense_bge_small_dev.jsonl \
  --sft_jsonl external/LLaMA-Factory/data/finqa_dev.jsonl \
  --output_jsonl data/finqa/processed/rag/generation/dense_bge_small_dev_top5.jsonl \
  --top_k 5
log "infer RAG dense top5 dev with DPO clean-balanced"
PYTHONPATH=. python eval/infer_finqa_qwen.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/finqa/qwen25-7b-finqa-dpo-clean-balanced \
  --input_jsonl data/finqa/processed/rag/generation/dense_bge_small_dev_top5.jsonl \
  --output_jsonl outputs/finqa/rag/predictions/dpo_clean_balanced_dense_bge_small_dev_top5.jsonl \
  --max_new_tokens 256
log "eval RAG dense top5 dev"
python eval/eval_finqa_answer_exec.py \
  --pred_jsonl outputs/finqa/rag/predictions/dpo_clean_balanced_dense_bge_small_dev_top5.jsonl \
  --out_metrics outputs/finqa/rag/metrics/dpo_clean_balanced_dense_bge_small_dev_top5_answer_exec_metrics.json
log "done"
