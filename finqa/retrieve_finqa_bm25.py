import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from rank_bm25 import BM25Okapi


def tokenize(s):
    return re.findall(r'[a-z]+|\d+(?:\.\d+)?|[%$]', str(s).lower())


def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus_jsonl', required=True)
    ap.add_argument('--queries_jsonl', required=True)
    ap.add_argument('--output_jsonl', required=True)
    ap.add_argument('--top_k', type=int, default=20)
    args = ap.parse_args()

    chunks = load_jsonl(args.corpus_jsonl)
    queries = load_jsonl(args.queries_jsonl)
    by_sample = defaultdict(list)
    for c in chunks:
        by_sample[c['sample_id']].append(c)

    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for q in queries:
            local_chunks = by_sample[q['sample_id']]
            tokenized = [tokenize(c['text']) for c in local_chunks]
            bm25 = BM25Okapi(tokenized)
            scores = bm25.get_scores(tokenize(q['question']))
            order = sorted(range(len(local_chunks)), key=lambda i: float(scores[i]), reverse=True)[:args.top_k]
            retrieved = []
            for rank, i in enumerate(order, start=1):
                c = local_chunks[i]
                score = float(scores[i])
                if math.isnan(score):
                    score = 0.0
                retrieved.append({
                    'rank': rank,
                    'chunk_id': c['chunk_id'],
                    'score': score,
                    'chunk_type': c['chunk_type'],
                    'text': c['text'],
                    'is_gold': c.get('is_gold', False),
                })
            rec = dict(q)
            rec['retriever'] = 'bm25_sample_local'
            rec['retrieved'] = retrieved
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(json.dumps({'queries': len(queries), 'output': str(out), 'top_k': args.top_k}, indent=2))


if __name__ == '__main__':
    main()
