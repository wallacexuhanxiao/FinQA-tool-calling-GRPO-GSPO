import argparse
import json
from pathlib import Path


def load_by_id(path):
    d = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                ex = json.loads(line)
                d[ex['sample_id']] = ex
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bm25_jsonl', required=True)
    ap.add_argument('--dense_jsonl', required=True)
    ap.add_argument('--output_jsonl', required=True)
    ap.add_argument('--top_k', type=int, default=50)
    ap.add_argument('--rrf_k', type=int, default=60)
    args = ap.parse_args()
    bm25 = load_by_id(args.bm25_jsonl)
    dense = load_by_id(args.dense_jsonl)
    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for sid, b in bm25.items():
            d = dense[sid]
            chunks = {}
            scores = {}
            sources = {}
            for source_name, ex in [('bm25', b), ('dense', d)]:
                for r in ex['retrieved']:
                    cid = r['chunk_id']
                    chunks[cid] = r
                    scores[cid] = scores.get(cid, 0.0) + 1.0 / (args.rrf_k + int(r['rank']))
                    sources.setdefault(cid, {})[source_name] = int(r['rank'])
            order = sorted(scores, key=scores.get, reverse=True)[:args.top_k]
            retrieved = []
            for rank, cid in enumerate(order, start=1):
                r = dict(chunks[cid])
                r['rank'] = rank
                r['score'] = scores[cid]
                r['source_ranks'] = sources.get(cid, {})
                retrieved.append(r)
            rec = dict(b)
            rec['retriever'] = 'bm25_dense_rrf_sample_local'
            rec['retrieved'] = retrieved
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(json.dumps({'queries': len(bm25), 'output': str(out), 'top_k': args.top_k}, indent=2))


if __name__ == '__main__':
    main()
