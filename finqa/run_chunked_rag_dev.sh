#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/gui-grounding-agent
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONPATH=.

mkdir -p outputs/finqa/rag outputs/finqa/metrics external/LLaMA-Factory/data logs/finqa

RERANKER_MODEL="${RERANKER_MODEL:-models/bge-reranker-base}"
DENSE_MODEL="${DENSE_MODEL:-}"

if [ ! -d "${RERANKER_MODEL}" ]; then
  echo "Reranker not found at ${RERANKER_MODEL}; falling back to lexical rerank."
  RERANKER_MODEL=""
fi

if [ -n "${DENSE_MODEL}" ] && [ ! -d "${DENSE_MODEL}" ]; then
  echo "Dense model not found at ${DENSE_MODEL}; disabling dense retrieval."
  DENSE_MODEL=""
fi

python scripts/finqa/run_finqa_chunked_rag.py \
  --split validation \
  --input_jsonl data/finqa/processed/sft/dev.jsonl \
  --out_retrieval_jsonl outputs/finqa/rag/chunked_rag_dev_retrieval.jsonl \
  --out_generation_jsonl external/LLaMA-Factory/data/finqa_chunked_rag_dev.jsonl \
  --out_metrics outputs/finqa/metrics/chunked_rag_dev_retrieval_metrics.json \
  --bm25_top_n 50 \
  --dense_model "${DENSE_MODEL}" \
  --dense_top_n 50 \
  --reranker_model "${RERANKER_MODEL}" \
  --rerank_top_n 20 \
  --final_top_k 8 \
  --max_context_words 700 \
  --sentence_window 2 \
  2>&1 | tee logs/finqa/run_chunked_rag_dev.log

echo "Wrote:"
echo "- outputs/finqa/rag/chunked_rag_dev_retrieval.jsonl"
echo "- external/LLaMA-Factory/data/finqa_chunked_rag_dev.jsonl"
echo "- outputs/finqa/metrics/chunked_rag_dev_retrieval_metrics.json"
