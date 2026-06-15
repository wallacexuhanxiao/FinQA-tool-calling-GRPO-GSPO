import argparse
import json
from collections import defaultdict
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', default='models/BAAI/bge-m3')
    ap.add_argument('--corpus_jsonl', required=True)
    ap.add_argument('--queries_jsonl', required=True)
    ap.add_argument('--output_jsonl', required=True)
    ap.add_argument('--top_k', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=64)
    args = ap.parse_args()

    chunks = load_jsonl(args.corpus_jsonl)
    queries = load_jsonl(args.queries_jsonl)
    by_sample = defaultdict(list)
    for c in chunks:
        by_sample[c['sample_id']].append(c)

    model = SentenceTransformer(args.model_path, trust_remote_code=True, device='cuda')
    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for qi, q in enumerate(queries, start=1):
            local_chunks = by_sample[q['sample_id']]
            texts = [c['text'] for c in local_chunks]
            doc_emb = model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            q_emb = model.encode([q['question']], batch_size=1, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            doc_emb = np.asarray(doc_emb, dtype='float32')
            q_emb = np.asarray(q_emb, dtype='float32')
            index = faiss.IndexFlatIP(doc_emb.shape[1])
            index.add(doc_emb)
            scores, idxs = index.search(q_emb, min(args.top_k, len(local_chunks)))
            retrieved = []
            for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), start=1):
                c = local_chunks[int(idx)]
                retrieved.append({
                    'rank': rank,
                    'chunk_id': c['chunk_id'],
                    'score': float(score),
                    'chunk_type': c['chunk_type'],
                    'text': c['text'],
                    'is_gold': c.get('is_gold', False),
                })
            rec = dict(q)
            rec['retriever'] = 'bge_m3_dense_sample_local'
            rec['retrieved'] = retrieved
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            if qi % 100 == 0:
                print({'processed': qi, 'total': len(queries)}, flush=True)
    print(json.dumps({'queries': len(queries), 'output': str(out), 'top_k': args.top_k}, indent=2))


if __name__ == '__main__':
    main()
