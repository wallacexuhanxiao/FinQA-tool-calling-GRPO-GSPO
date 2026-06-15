#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/gui-grounding-agent
mkdir -p logs/finqa models/sentence-transformers outputs/finqa/rag/retrieval outputs/finqa/rag/metrics
PIPE=logs/finqa/rag_dense_hybrid_small_pipeline.log

echo "[$(date '+%F %T')] STEP 1 download all-MiniLM via hf-mirror" | tee "$PIPE"
HF_ENDPOINT=https://hf-mirror.com python scripts/finqa/download_small_embed.py 2>&1 | tee logs/finqa/download_small_embed.log

for split in dev test; do
  echo "[$(date '+%F %T')] STEP 2 dense small ${split}" | tee -a "$PIPE"
  python scripts/finqa/retrieve_finqa_dense.py \
    --model_path models/sentence-transformers/all-MiniLM-L6-v2 \
    --corpus_jsonl data/finqa/processed/rag/corpus_${split}.jsonl \
    --queries_jsonl data/finqa/processed/rag/queries_${split}.jsonl \
    --output_jsonl outputs/finqa/rag/retrieval/dense_minilm_${split}_top20.jsonl \
    --top_k 20 --batch_size 128 2>&1 | tee logs/finqa/retrieve_dense_minilm_${split}.log

  echo "[$(date '+%F %T')] STEP 3 eval dense small ${split}" | tee -a "$PIPE"
  python eval/eval_finqa_retrieval.py \
    --retrieval_jsonl outputs/finqa/rag/retrieval/dense_minilm_${split}_top20.jsonl \
    --out_metrics outputs/finqa/rag/metrics/dense_minilm_${split}_retrieval_metrics.json \
    2>&1 | tee logs/finqa/eval_dense_minilm_${split}.log

  echo "[$(date '+%F %T')] STEP 4 hybrid small ${split}" | tee -a "$PIPE"
  python scripts/finqa/retrieve_finqa_hybrid_rrf.py \
    --bm25_jsonl outputs/finqa/rag/retrieval/bm25_${split}_top20.jsonl \
    --dense_jsonl outputs/finqa/rag/retrieval/dense_minilm_${split}_top20.jsonl \
    --output_jsonl outputs/finqa/rag/retrieval/hybrid_minilm_rrf_${split}_top50.jsonl \
    --top_k 50 2>&1 | tee logs/finqa/retrieve_hybrid_minilm_rrf_${split}.log

  echo "[$(date '+%F %T')] STEP 5 eval hybrid small ${split}" | tee -a "$PIPE"
  python eval/eval_finqa_retrieval.py \
    --retrieval_jsonl outputs/finqa/rag/retrieval/hybrid_minilm_rrf_${split}_top50.jsonl \
    --out_metrics outputs/finqa/rag/metrics/hybrid_minilm_rrf_${split}_retrieval_metrics.json \
    2>&1 | tee logs/finqa/eval_hybrid_minilm_rrf_${split}.log
done

echo "[$(date '+%F %T')] STEP 6 summarize" | tee -a "$PIPE"
python - <<'PY' 2>&1 | tee logs/finqa/summarize_rag_dense_hybrid_small.log
import json
from pathlib import Path
methods=['bm25','dense_minilm','hybrid_minilm_rrf']
for split in ['dev','test']:
    print('\nSPLIT', split)
    for method in methods:
        p=Path(f'outputs/finqa/rag/metrics/{method}_{split}_retrieval_metrics.json')
        if not p.exists(): print(method,'missing'); continue
        m=json.loads(p.read_text())
        keep=['mrr','recall_at_1','recall_at_3','recall_at_5','recall_at_10','evidence_coverage_at_5','all_recall_at_5','ndcg_at_5']
        print(method, {k:round(m.get(k,0)*100,2) for k in keep})
PY

echo "[$(date '+%F %T')] RAG dense/hybrid small pipeline completed" | tee -a "$PIPE"
