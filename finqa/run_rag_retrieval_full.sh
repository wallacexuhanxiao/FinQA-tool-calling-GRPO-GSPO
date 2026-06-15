#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/gui-grounding-agent
mkdir -p logs/finqa models/BAAI outputs/finqa/rag/retrieval outputs/finqa/rag/metrics
PIPE=logs/finqa/rag_retrieval_full_pipeline.log

echo "[$(date '+%F %T')] STEP 1 download BGE models via hf-mirror" | tee "$PIPE"
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 python scripts/finqa/download_rag_models.py 2>&1 | tee logs/finqa/download_rag_models.log

for split in dev test; do
  echo "[$(date '+%F %T')] STEP 2 dense ${split}" | tee -a "$PIPE"
  python scripts/finqa/retrieve_finqa_dense.py \
    --model_path models/BAAI/bge-m3 \
    --corpus_jsonl data/finqa/processed/rag/corpus_${split}.jsonl \
    --queries_jsonl data/finqa/processed/rag/queries_${split}.jsonl \
    --output_jsonl outputs/finqa/rag/retrieval/dense_bge_m3_${split}_top20.jsonl \
    --top_k 20 --batch_size 64 2>&1 | tee logs/finqa/retrieve_dense_bge_m3_${split}.log

  echo "[$(date '+%F %T')] STEP 3 eval dense ${split}" | tee -a "$PIPE"
  python eval/eval_finqa_retrieval.py \
    --retrieval_jsonl outputs/finqa/rag/retrieval/dense_bge_m3_${split}_top20.jsonl \
    --out_metrics outputs/finqa/rag/metrics/dense_bge_m3_${split}_retrieval_metrics.json \
    2>&1 | tee logs/finqa/eval_dense_bge_m3_${split}.log

  echo "[$(date '+%F %T')] STEP 4 hybrid RRF ${split}" | tee -a "$PIPE"
  python scripts/finqa/retrieve_finqa_hybrid_rrf.py \
    --bm25_jsonl outputs/finqa/rag/retrieval/bm25_${split}_top20.jsonl \
    --dense_jsonl outputs/finqa/rag/retrieval/dense_bge_m3_${split}_top20.jsonl \
    --output_jsonl outputs/finqa/rag/retrieval/hybrid_rrf_${split}_top50.jsonl \
    --top_k 50 2>&1 | tee logs/finqa/retrieve_hybrid_rrf_${split}.log

  echo "[$(date '+%F %T')] STEP 5 eval hybrid ${split}" | tee -a "$PIPE"
  python eval/eval_finqa_retrieval.py \
    --retrieval_jsonl outputs/finqa/rag/retrieval/hybrid_rrf_${split}_top50.jsonl \
    --out_metrics outputs/finqa/rag/metrics/hybrid_rrf_${split}_retrieval_metrics.json \
    2>&1 | tee logs/finqa/eval_hybrid_rrf_${split}.log

  echo "[$(date '+%F %T')] STEP 6 rerank ${split}" | tee -a "$PIPE"
  python scripts/finqa/rerank_finqa_bge.py \
    --reranker_path models/BAAI/bge-reranker-v2-m3 \
    --input_jsonl outputs/finqa/rag/retrieval/hybrid_rrf_${split}_top50.jsonl \
    --output_jsonl outputs/finqa/rag/retrieval/hybrid_rerank_bge_${split}_top20.jsonl \
    --candidate_k 50 --top_k 20 --batch_size 32 2>&1 | tee logs/finqa/rerank_bge_${split}.log

  echo "[$(date '+%F %T')] STEP 7 eval rerank ${split}" | tee -a "$PIPE"
  python eval/eval_finqa_retrieval.py \
    --retrieval_jsonl outputs/finqa/rag/retrieval/hybrid_rerank_bge_${split}_top20.jsonl \
    --out_metrics outputs/finqa/rag/metrics/hybrid_rerank_bge_${split}_retrieval_metrics.json \
    2>&1 | tee logs/finqa/eval_hybrid_rerank_bge_${split}.log
done

echo "[$(date '+%F %T')] STEP 8 summarize retrieval metrics" | tee -a "$PIPE"
python - <<'PY' 2>&1 | tee logs/finqa/summarize_rag_retrieval_full.log
import json
from pathlib import Path
methods=['bm25','dense_bge_m3','hybrid_rrf','hybrid_rerank_bge']
for split in ['dev','test']:
    print('\nSPLIT', split)
    for method in methods:
        p=Path(f'outputs/finqa/rag/metrics/{method}_{split}_retrieval_metrics.json')
        if not p.exists():
            print(method, 'missing'); continue
        m=json.loads(p.read_text())
        keep=['mrr','recall_at_1','recall_at_3','recall_at_5','recall_at_10','evidence_coverage_at_5','all_recall_at_5','ndcg_at_5']
        print(method, {k:round(m.get(k,0)*100,2) for k in keep})
PY

echo "[$(date '+%F %T')] RAG retrieval full pipeline completed" | tee -a "$PIPE"
