import json
from pathlib import Path
cov=json.load(open('outputs/finqa/rag/analysis/dense_bge_small_dev_expansion_coverage.json'))['settings']
err=json.load(open('outputs/finqa/rag/analysis/dpo_clean_balanced_dense_top5_expanded_error_attribution.json'))
keys=['top5_text1_table1','top10_text1_table1','top15_text1_table1','top20_text1_table1','top10_text1_table2','top15_text1_table2','top20_text2_table2']
print('Coverage table')
print('| setting | Any | All | Coverage | Avg chunks | P95 chunks | missing table_row |')
print('|---|---:|---:|---:|---:|---:|---:|')
for k in keys:
    m=cov[k]
    print(f"| {k} | {m['any_recall']} | {m['all_recall']} | {m['evidence_coverage']} | {m['avg_evidence_chunks']} | {m['p95_evidence_chunks']} | {m['missing_type_counts'].get('table_row',0)} |")
print('\nError attribution')
print(json.dumps(err['counts'],ensure_ascii=False,indent=2))
print(json.dumps(err['rates'],ensure_ascii=False,indent=2))
print('\nPrepared files')
for p in [
'data/finqa/processed/rag/generation/dense_bge_small_dev_top10_expanded.jsonl',
'data/finqa/processed/rag/generation/dense_bge_small_dev_top15_expanded.jsonl',
'data/finqa/processed/rag/generation/dense_bge_small_dev_top20_expanded.jsonl',
'data/finqa/processed/rag/generation/oracle_gold_dev_expanded.jsonl']:
    print(p, sum(1 for _ in open(p,encoding='utf-8')))
