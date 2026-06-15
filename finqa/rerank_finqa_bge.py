import argparse
import json
from pathlib import Path
from FlagEmbedding import FlagReranker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reranker_path', default='models/BAAI/bge-reranker-v2-m3')
    ap.add_argument('--input_jsonl', required=True)
    ap.add_argument('--output_jsonl', required=True)
    ap.add_argument('--candidate_k', type=int, default=50)
    ap.add_argument('--top_k', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=32)
    args = ap.parse_args()
    reranker = FlagReranker(args.reranker_path, use_fp16=True)
    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.input_jsonl, encoding='utf-8') as f, out.open('w', encoding='utf-8') as g:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            cand = (ex.get('retrieved') or [])[:args.candidate_k]
            pairs = [[ex['question'], r['text']] for r in cand]
            scores = reranker.compute_score(pairs, batch_size=args.batch_size) if pairs else []
            if isinstance(scores, float):
                scores = [scores]
            for r, s in zip(cand, scores):
                r['rerank_score'] = float(s)
            cand = sorted(cand, key=lambda r: r.get('rerank_score', -1e9), reverse=True)[:args.top_k]
            for rank, r in enumerate(cand, start=1):
                r['rank'] = rank
                r['score'] = r.get('rerank_score', r.get('score', 0.0))
            rec = dict(ex)
            rec['retriever'] = 'hybrid_rrf_bge_reranker_sample_local'
            rec['retrieved'] = cand
            g.write(json.dumps(rec, ensure_ascii=False) + '\n')
            n += 1
            if n % 100 == 0:
                print({'processed': n}, flush=True)
    print(json.dumps({'queries': n, 'output': str(out), 'top_k': args.top_k}, indent=2))


if __name__ == '__main__':
    main()
