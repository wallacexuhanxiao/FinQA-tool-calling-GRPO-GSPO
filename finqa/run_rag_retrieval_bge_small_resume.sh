#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/gui-grounding-agent
mkdir -p outputs/finqa/rag/retrieval outputs/finqa/rag/metrics logs/finqa
MODEL=models/BAAI/bge-small-en-v1.5-ms
log(){ echo "[$(date '+%F %T')] $*"; }
for split in dev test train; do
  log "bm25 top50 ${split}"
  python scripts/finqa/retrieve_finqa_bm25.py \
    --corpus_jsonl data/finqa/processed/rag/corpus_${split}.jsonl \
    --queries_jsonl data/finqa/processed/rag/queries_${split}.jsonl \
    --output_jsonl outputs/finqa/rag/retrieval/bm25_${split}_top50.jsonl \
    --top_k 50

  log "eval bm25 top50 ${split}"
  python eval/eval_finqa_retrieval.py \
    --retrieval_jsonl outputs/finqa/rag/retrieval/bm25_${split}_top50.jsonl \
    --out_metrics outputs/finqa/rag/metrics/bm25_${split}_top50_retrieval_metrics.json

  if [ ! -s outputs/finqa/rag/retrieval/dense_bge_small_${split}.jsonl ]; then
    log "dense retrieval ${split} with bge-small-en-v1.5"
    python scripts/finqa/retrieve_finqa_dense.py \
      --model_path "$MODEL" \
      --corpus_jsonl data/finqa/processed/rag/corpus_${split}.jsonl \
      --queries_jsonl data/finqa/processed/rag/queries_${split}.jsonl \
      --output_jsonl outputs/finqa/rag/retrieval/dense_bge_small_${split}.jsonl \
      --top_k 50 \
      --batch_size 64
  else
    log "dense exists ${split}, skip"
  fi

  log "eval dense ${split}"
  python eval/eval_finqa_retrieval.py \
    --retrieval_jsonl outputs/finqa/rag/retrieval/dense_bge_small_${split}.jsonl \
    --out_metrics outputs/finqa/rag/metrics/dense_bge_small_${split}_retrieval_metrics.json

  log "hybrid rrf ${split}"
  python scripts/finqa/retrieve_finqa_hybrid_rrf.py \
    --bm25_jsonl outputs/finqa/rag/retrieval/bm25_${split}_top50.jsonl \
    --dense_jsonl outputs/finqa/rag/retrieval/dense_bge_small_${split}.jsonl \
    --output_jsonl outputs/finqa/rag/retrieval/hybrid_bm25_bge_small_${split}.jsonl \
    --top_k 50 \
    --rrf_k 60

  log "eval hybrid ${split}"
  python eval/eval_finqa_retrieval.py \
    --retrieval_jsonl outputs/finqa/rag/retrieval/hybrid_bm25_bge_small_${split}.jsonl \
    --out_metrics outputs/finqa/rag/metrics/hybrid_bm25_bge_small_${split}_retrieval_metrics.json
done
log "done"
