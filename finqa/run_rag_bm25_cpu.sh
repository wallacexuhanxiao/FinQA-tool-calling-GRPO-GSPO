#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/gui-grounding-agent
mkdir -p logs/finqa data/finqa/processed/rag outputs/finqa/rag/retrieval outputs/finqa/rag/metrics
PIPE=logs/finqa/rag_bm25_cpu_pipeline.log

echo "[$(date '+%F %T')] STEP 1 build FinQA RAG corpus" | tee "$PIPE"
python scripts/finqa/build_finqa_rag_corpus.py 2>&1 | tee logs/finqa/build_rag_corpus.log

for split in dev test train; do
  echo "[$(date '+%F %T')] STEP 2 retrieve BM25 ${split}" | tee -a "$PIPE"
  python scripts/finqa/retrieve_finqa_bm25.py \
    --corpus_jsonl data/finqa/processed/rag/corpus_${split}.jsonl \
    --queries_jsonl data/finqa/processed/rag/queries_${split}.jsonl \
    --output_jsonl outputs/finqa/rag/retrieval/bm25_${split}_top20.jsonl \
    --top_k 20 2>&1 | tee logs/finqa/retrieve_bm25_${split}.log

  echo "[$(date '+%F %T')] STEP 3 eval BM25 ${split}" | tee -a "$PIPE"
  python eval/eval_finqa_retrieval.py \
    --retrieval_jsonl outputs/finqa/rag/retrieval/bm25_${split}_top20.jsonl \
    --out_metrics outputs/finqa/rag/metrics/bm25_${split}_retrieval_metrics.json \
    2>&1 | tee logs/finqa/eval_bm25_${split}.log
done

echo "[$(date '+%F %T')] STEP 4 summarize BM25 metrics" | tee -a "$PIPE"
python - <<'PY' 2>&1 | tee logs/finqa/summarize_bm25_retrieval.log
import json
from pathlib import Path
for split in ['dev','test','train']:
    p=Path(f'outputs/finqa/rag/metrics/bm25_{split}_retrieval_metrics.json')
    if not p.exists():
        print(split, 'missing')
        continue
    m=json.loads(p.read_text())
    keep=['num_queries','num_queries_with_gold_chunks','num_queries_without_gold_chunks','mrr','recall_at_1','recall_at_3','recall_at_5','recall_at_10','evidence_coverage_at_5','all_recall_at_5','ndcg_at_5']
    print(split, {k:m.get(k) for k in keep})
PY

echo "[$(date '+%F %T')] RAG BM25 CPU pipeline completed" | tee -a "$PIPE"
