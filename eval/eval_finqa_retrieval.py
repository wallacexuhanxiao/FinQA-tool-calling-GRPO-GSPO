import argparse
import json
import math
from pathlib import Path


def dcg(rels):
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--retrieval_jsonl', required=True)
    ap.add_argument('--out_metrics', required=True)
    ap.add_argument('--ks', default='1,3,5,10,20')
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(',') if x]

    totals = {k: {'hit': 0, 'all': 0, 'coverage': 0.0, 'ndcg': 0.0, 'table_hit': 0, 'text_hit': 0} for k in ks}
    n = 0
    mrr = 0.0
    no_gold = 0
    gold_count_sum = 0
    examples = []

    with open(args.retrieval_jsonl, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            n += 1
            gold = set(ex.get('gold_chunk_ids') or [])
            gold_count_sum += len(gold)
            if not gold:
                no_gold += 1
                continue
            retrieved = ex.get('retrieved') or []
            ranks = []
            type_by_gold = {}
            for r in retrieved:
                if r['chunk_id'] in gold:
                    ranks.append(r['rank'])
                    type_by_gold[r['chunk_id']] = r.get('chunk_type', '')
            if ranks:
                mrr += 1.0 / min(ranks)
            for k in ks:
                top = retrieved[:k]
                hit_ids = {r['chunk_id'] for r in top if r['chunk_id'] in gold}
                totals[k]['hit'] += bool(hit_ids)
                totals[k]['all'] += hit_ids == gold
                totals[k]['coverage'] += len(hit_ids) / len(gold)
                rels = [1 if r['chunk_id'] in gold else 0 for r in top]
                ideal = [1] * min(len(gold), k) + [0] * max(0, k - len(gold))
                denom = dcg(ideal)
                totals[k]['ndcg'] += dcg(rels) / denom if denom else 0.0
                totals[k]['table_hit'] += any('table' in r.get('chunk_type', '') for r in top if r['chunk_id'] in gold)
                totals[k]['text_hit'] += any(r.get('chunk_type', '') in {'pre_text', 'post_text'} for r in top if r['chunk_id'] in gold)
            if len(examples) < 20 and not any(r['chunk_id'] in gold for r in retrieved[:5]):
                examples.append({
                    'sample_id': ex.get('sample_id'),
                    'question': ex.get('question'),
                    'gold_inds': ex.get('gold_inds'),
                    'gold_chunk_ids': list(gold),
                    'top5': retrieved[:5],
                })

    denom = max(n - no_gold, 1)
    metrics = {
        'num_queries': n,
        'num_queries_with_gold_chunks': n - no_gold,
        'num_queries_without_gold_chunks': no_gold,
        'avg_gold_chunks': gold_count_sum / n if n else 0,
        'mrr': mrr / denom,
        'bad_top5_preview': examples,
    }
    for k in ks:
        metrics[f'recall_at_{k}'] = totals[k]['hit'] / denom
        metrics[f'all_recall_at_{k}'] = totals[k]['all'] / denom
        metrics[f'evidence_coverage_at_{k}'] = totals[k]['coverage'] / denom
        metrics[f'ndcg_at_{k}'] = totals[k]['ndcg'] / denom
        metrics[f'table_hit_at_{k}'] = totals[k]['table_hit'] / denom
        metrics[f'text_hit_at_{k}'] = totals[k]['text_hit'] / denom

    out = Path(args.out_metrics)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
