import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi


SYSTEM = (
    "You are a financial reasoning assistant. "
    "Solve numerical reasoning questions over retrieved financial evidence. "
    "Return JSON only with keys: program and answer."
)


def norm(s):
    return re.sub(r"\s+", " ", str(s or "").lower()).strip()


def tokenize(s):
    return re.findall(r"[a-zA-Z]+|[-+]?\d+(?:\.\d+)?%?", norm(s))


def sent_split(text):
    text = str(text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def table_row_text(row, header=None):
    row = [str(x) for x in (row or [])]
    header = [str(x) for x in (header or [])]
    if header and len(header) == len(row):
        pairs = [f"{h}: {v}" for h, v in zip(header, row)]
        return " | ".join(pairs)
    return " | ".join(row)


@dataclass
class Chunk:
    sample_id: str
    chunk_id: str
    chunk_type: str
    ordinal: int
    text: str
    is_gold: bool = False
    bm25_score: float = 0.0
    dense_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0


def gold_strings(ex):
    gold = ex.get("gold_inds") or []
    if isinstance(gold, dict):
        vals = list(gold.values()) + list(gold.keys())
    else:
        vals = gold
    out = []
    for item in vals:
        if isinstance(item, (list, tuple)):
            out.extend(str(x) for x in item)
        else:
            out.append(str(item))
    return [norm(x) for x in out if str(x).strip()]


def is_gold_chunk(text, ordinal, golds):
    t = norm(text)
    for g in golds:
        if not g:
            continue
        if g.isdigit() and int(g) == ordinal:
            return True
        if len(g) >= 12 and (g in t or t in g):
            return True
    return False


def build_chunks(ex, sentence_window=2):
    sid = str(ex.get("id") or "")
    golds = gold_strings(ex)
    chunks = []

    ordinal = 0
    for block_name in ["pre_text", "post_text"]:
        paras = ex.get(block_name) or []
        for para_i, para in enumerate(paras):
            sents = sent_split(para)
            if not sents:
                continue
            for start in range(0, len(sents), sentence_window):
                text = " ".join(sents[start:start + sentence_window])
                chunks.append(Chunk(
                    sample_id=sid,
                    chunk_id=f"{sid}::{block_name}:{para_i}:{start}",
                    chunk_type=block_name,
                    ordinal=ordinal,
                    text=text,
                    is_gold=is_gold_chunk(text, ordinal, golds),
                ))
                ordinal += 1

    table = ex.get("table") or []
    header = table[0] if table else []
    for row_i, row in enumerate(table):
        text = table_row_text(row, header if row_i else None)
        if not text.strip():
            continue
        chunks.append(Chunk(
            sample_id=sid,
            chunk_id=f"{sid}::table:{row_i}",
            chunk_type="table_row" if row_i else "table_header",
            ordinal=ordinal,
            text=text,
            is_gold=is_gold_chunk(text, ordinal, golds),
        ))
        ordinal += 1
    return chunks


def parse_sft_context(user_text):
    marker = "Financial context:"
    q_marker = "\n\nQuestion:"
    if marker in user_text:
        ctx = user_text.split(marker, 1)[1]
    else:
        ctx = user_text
    if q_marker in ctx:
        ctx = ctx.split(q_marker, 1)[0]

    sections = {"pre_text": [], "post_text": [], "table": []}
    current = None
    for line in ctx.splitlines():
        stripped = line.strip()
        low = stripped.lower()
        if low.startswith("pre-text:"):
            current = "pre_text"
            continue
        if low.startswith("table:"):
            current = "table"
            continue
        if low.startswith("post-text:"):
            current = "post_text"
            continue
        if current == "table":
            if stripped and not re.match(r"^\|\s*-", stripped):
                sections["table"].append(stripped.strip("| "))
        elif current in ("pre_text", "post_text") and stripped:
            sections[current].append(stripped)
    return sections


def sft_to_raw(row):
    messages = row.get("messages") or []
    meta = row.get("meta") or {}
    user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
    ctx = parse_sft_context(user)
    table = []
    for line in ctx["table"]:
        if "|" in line:
            table.append([x.strip() for x in line.split("|")])
        elif line.strip():
            table.append([line.strip()])
    return {
        "id": meta.get("id") or row.get("id"),
        "pre_text": ctx["pre_text"],
        "post_text": ctx["post_text"],
        "table": table,
        "question": meta.get("question") or "",
        "answer": meta.get("answer") or "",
        "final_result": meta.get("answer") or "",
        "program_re": meta.get("program_re") or "",
        "gold_inds": meta.get("gold_inds") or [],
    }


def load_examples(args):
    if args.input_jsonl:
        rows = []
        with open(args.input_jsonl, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(sft_to_raw(json.loads(line)))
                    if args.limit and len(rows) >= args.limit:
                        break
        return rows
    ds = load_dataset("ibm-research/finqa", split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    return list(ds)


def zscore(values):
    arr = np.asarray(values, dtype=np.float32)
    if len(arr) == 0:
        return arr
    std = arr.std()
    if std < 1e-8:
        return arr * 0
    return (arr - arr.mean()) / std


def bm25_candidates(question, chunks, top_n):
    tokenized = [tokenize(c.text) for c in chunks]
    if not tokenized:
        return []
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(tokenize(question))
    order = np.argsort(scores)[::-1][:top_n]
    out = []
    for idx in order:
        c = chunks[int(idx)]
        c.bm25_score = float(scores[int(idx)])
        out.append(c)
    return out


def try_load_dense(model_path):
    if not model_path:
        return None
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_path)


def try_load_reranker(model_path):
    if not model_path:
        return None
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_path)


def dense_candidates(question, chunks, model, top_n):
    if model is None or not chunks:
        return []
    texts = [c.text for c in chunks]
    q_emb = model.encode([question], normalize_embeddings=True, show_progress_bar=False)
    c_emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    scores = np.matmul(c_emb, q_emb[0])
    order = np.argsort(scores)[::-1][:top_n]
    out = []
    for idx in order:
        c = chunks[int(idx)]
        c.dense_score = float(scores[int(idx)])
        out.append(c)
    return out


def fuse_candidates(bm25, dense, top_n):
    by_id = {}
    for c in bm25 + dense:
        by_id[c.chunk_id] = c
    candidates = list(by_id.values())
    bm = {c.chunk_id: c.bm25_score for c in candidates}
    de = {c.chunk_id: c.dense_score for c in candidates}
    bm_z = dict(zip(bm.keys(), zscore(list(bm.values()))))
    de_z = dict(zip(de.keys(), zscore(list(de.values()))))
    for c in candidates:
        c.fused_score = float(bm_z.get(c.chunk_id, 0.0) + de_z.get(c.chunk_id, 0.0))
    return sorted(candidates, key=lambda c: c.fused_score, reverse=True)[:top_n]


def rerank(question, candidates, reranker, top_k):
    if not candidates:
        return []
    if reranker is not None:
        pairs = [(question, c.text) for c in candidates]
        scores = reranker.predict(pairs)
        for c, score in zip(candidates, scores):
            c.rerank_score = float(score)
        return sorted(candidates, key=lambda c: c.rerank_score, reverse=True)[:top_k]
    # Fallback lexical rerank: coverage of query tokens plus fused score.
    q = set(tokenize(question))
    for c in candidates:
        toks = set(tokenize(c.text))
        overlap = len(q & toks) / max(len(q), 1)
        c.rerank_score = overlap + 0.05 * c.fused_score
    return sorted(candidates, key=lambda c: c.rerank_score, reverse=True)[:top_k]


def budget_select(chunks, max_words):
    selected = []
    used = 0
    for c in chunks:
        words = c.text.split()
        if selected and used + len(words) > max_words:
            continue
        selected.append(c)
        used += len(words)
        if used >= max_words:
            break
    return selected


def build_user_prompt(question, chunks):
    evidence = []
    for i, c in enumerate(chunks, 1):
        evidence.append(f"[{i}] ({c.chunk_type}) {c.text}")
    evidence_text = "\n".join(evidence)
    return (
        "Given the retrieved financial evidence and question, generate a short executable reasoning program "
        "and the final answer.\n\n"
        "Return JSON only:\n"
        "{\"program\": \"...\", \"answer\": \"...\"}\n\n"
        f"Retrieved evidence:\n{evidence_text}\n\n"
        f"Question:\n{question}"
    )


def gold_answer(ex):
    return str(ex.get("final_result") or ex.get("answer") or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--input_jsonl", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out_retrieval_jsonl", required=True)
    ap.add_argument("--out_generation_jsonl", required=True)
    ap.add_argument("--out_metrics", required=True)
    ap.add_argument("--bm25_top_n", type=int, default=40)
    ap.add_argument("--dense_model", default="")
    ap.add_argument("--dense_top_n", type=int, default=40)
    ap.add_argument("--reranker_model", default="")
    ap.add_argument("--rerank_top_n", type=int, default=15)
    ap.add_argument("--final_top_k", type=int, default=8)
    ap.add_argument("--max_context_words", type=int, default=700)
    ap.add_argument("--sentence_window", type=int, default=2)
    args = ap.parse_args()

    ds = load_examples(args)

    dense_model = try_load_dense(args.dense_model)
    reranker = try_load_reranker(args.reranker_model)

    Path(args.out_retrieval_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_generation_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_metrics).parent.mkdir(parents=True, exist_ok=True)

    total = 0
    gold_available = 0
    recall_at = {1: 0, 3: 0, 5: 0, 8: 0, 15: 0}
    context_words = []
    chunk_counts = []

    with open(args.out_retrieval_jsonl, "w", encoding="utf-8") as ret_f, open(args.out_generation_jsonl, "w", encoding="utf-8") as gen_f:
        for ex in ds:
            total += 1
            question = ex["question"]
            chunks = build_chunks(ex, sentence_window=args.sentence_window)
            chunk_counts.append(len(chunks))
            bm = bm25_candidates(question, chunks, args.bm25_top_n)
            de = dense_candidates(question, chunks, dense_model, args.dense_top_n)
            fused = fuse_candidates(bm, de, max(args.bm25_top_n, args.dense_top_n))
            ranked = rerank(question, fused, reranker, args.rerank_top_n)
            selected = budget_select(ranked[:args.final_top_k], args.max_context_words)

            has_gold = any(c.is_gold for c in chunks)
            if has_gold:
                gold_available += 1
                for k in recall_at:
                    if any(c.is_gold for c in ranked[:k]):
                        recall_at[k] += 1

            context_words.append(sum(len(c.text.split()) for c in selected))
            ret_record = {
                "id": ex.get("id"),
                "question": question,
                "answer": gold_answer(ex),
                "gold_available": has_gold,
                "chunks_total": len(chunks),
                "selected_context_words": context_words[-1],
                "retrieved": [asdict(c) for c in ranked],
                "selected": [asdict(c) for c in selected],
            }
            ret_f.write(json.dumps(ret_record, ensure_ascii=False) + "\n")

            gen_record = {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": build_user_prompt(question, selected)},
                ],
                "meta": {
                    "id": ex.get("id"),
                    "split": args.split,
                    "question": question,
                    "answer": gold_answer(ex),
                    "program_re": ex.get("program_re", ""),
                    "gold_inds": ex.get("gold_inds", []),
                    "rag_selected_chunk_ids": [c.chunk_id for c in selected],
                    "rag_context_words": context_words[-1],
                },
            }
            gen_f.write(json.dumps(gen_record, ensure_ascii=False) + "\n")

    denom = max(gold_available, 1)
    metrics = {
        "num_samples": total,
        "gold_available": gold_available,
        "avg_chunks_per_sample": float(np.mean(chunk_counts)) if chunk_counts else 0.0,
        "avg_selected_context_words": float(np.mean(context_words)) if context_words else 0.0,
        "bm25_top_n": args.bm25_top_n,
        "dense_model": args.dense_model,
        "dense_top_n": args.dense_top_n,
        "reranker_model": args.reranker_model,
        "rerank_top_n": args.rerank_top_n,
        "final_top_k": args.final_top_k,
        "max_context_words": args.max_context_words,
    }
    for k, v in recall_at.items():
        metrics[f"recall_at_{k}"] = v / denom if gold_available else None
    Path(args.out_metrics).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
