import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SYSTEM = (
    "You are a financial reasoning assistant. "
    "Solve numerical reasoning questions over retrieved financial evidence. "
    "Return JSON only with keys: program and answer."
)


def load_sft_meta(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        ex = json.loads(line)
        meta = ex.get("meta") or {}
        sid = meta.get("id")
        if sid:
            out[sid] = meta
    return out


def load_corpus(path):
    by_chunk = {}
    by_sample_type = defaultdict(dict)
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        c = json.loads(line)
        by_chunk[c["chunk_id"]] = c
        by_sample_type[(c["sample_id"], c["chunk_type"])][int(c.get("chunk_index", 0))] = c
    return by_chunk, by_sample_type


def normalize_text(s):
    return re.sub(r"\s+", " ", s or "").strip()


def expand(c, by_sample_type, text_window=1, table_window=1):
    sid = c.get("sample_id")
    typ = c.get("chunk_type")
    idx = int(c.get("chunk_index", 0))
    out = []
    if typ == "table_row":
        h = by_sample_type.get((sid, "table_header"), {}).get(0)
        if h:
            out.append((h, "table_header"))
        rows = by_sample_type.get((sid, "table_row"), {})
        for j in range(idx - table_window, idx + table_window + 1):
            if j in rows:
                out.append((rows[j], "neighbor_table_row" if j != idx else "primary_table_row"))
    elif typ == "table_header":
        rows = by_sample_type.get((sid, "table_row"), {})
        for j in range(1, table_window + 2):
            if j in rows:
                out.append((rows[j], "header_following_row"))
    elif typ in {"pre_text", "post_text"}:
        rows = by_sample_type.get((sid, typ), {})
        for j in range(idx - text_window, idx + text_window + 1):
            if j in rows:
                out.append((rows[j], "neighbor_text" if j != idx else "primary_text"))
    return out


def add(out, seen, c, reason, score=None, rank=None):
    cid = c.get("chunk_id")
    if not cid or cid in seen:
        return
    x = dict(c)
    x["pack_reason"] = reason
    x["rerank_score"] = score
    x["rerank_rank"] = rank
    out.append(x)
    seen.add(cid)


def fmt(c, i):
    score = c.get("rerank_score")
    score_s = f"score={score:.4f}" if isinstance(score, (int, float)) else "expanded"
    text = normalize_text(c.get("text", ""))
    return f"[{i}] ({score_s}; {c.get('pack_reason')}; {c.get('chunk_type')}) {text}"


def build_messages(question, evidence):
    user = (
        "Given the reranked financial evidence and question, generate a short executable reasoning program "
        "and the final answer. Prefer higher-scored primary evidence, but use expanded neighboring rows/sentences "
        "for table headers, units, years, and context. Do not use facts outside the evidence.\n\n"
        "Return JSON only:\n{\"program\": \"...\", \"answer\": \"...\"}\n\n"
        f"Reranked and expanded financial evidence:\n{evidence}\n\nQuestion:\n{question}"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


class Reranker:
    def __init__(self, model_path, device="auto", batch_size=32, max_length=512):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score(self, question, chunks):
        pairs = [(question, normalize_text(c.get("text", ""))) for c in chunks]
        scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits.view(-1).float().detach().cpu().tolist()
            scores.extend(float(x) for x in logits)
        return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval_jsonl", required=True)
    ap.add_argument("--sft_jsonl", required=True)
    ap.add_argument("--corpus_jsonl", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--reranker_model", default="models/BAAI/bge-reranker-base")
    ap.add_argument("--prompt_model_path", default="models/Qwen2.5-7B-Instruct")
    ap.add_argument("--candidate_top_k", type=int, default=15)
    ap.add_argument("--token_budget", type=int, default=900)
    ap.add_argument("--text_window", type=int, default=1)
    ap.add_argument("--table_window", type=int, default=1)
    ap.add_argument("--max_evidence_chunks", type=int, default=40)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--rerank_max_length", type=int, default=512)
    args = ap.parse_args()

    prompt_tok = AutoTokenizer.from_pretrained(args.prompt_model_path, trust_remote_code=True)
    reranker = Reranker(args.reranker_model, args.device, args.batch_size, args.rerank_max_length)
    meta_by_id = load_sft_meta(args.sft_jsonl)
    by_chunk, by_sample_type = load_corpus(args.corpus_jsonl)
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)

    stats = {"written": 0, "avg_tokens": 0, "avg_chunks": 0, "all_gold": 0, "any_gold": 0, "coverage_sum": 0}
    rows = [json.loads(x) for x in open(args.retrieval_jsonl, encoding="utf-8") if x.strip()]

    with open(args.output_jsonl, "w", encoding="utf-8") as out:
        for r in tqdm(rows, desc="bge rerank+pack"):
            sid = r.get("sample_id") or r.get("id")
            meta = dict(meta_by_id.get(sid, {}))
            q = r.get("question") or meta.get("question")
            candidates = []
            for rc in (r.get("retrieved") or [])[: args.candidate_top_k]:
                c = dict(by_chunk.get(rc.get("chunk_id"), rc))
                for k in ("rank", "score", "is_gold"):
                    if k in rc:
                        c[k] = rc[k]
                candidates.append(c)

            scores = reranker.score(q, candidates) if candidates else []
            ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

            packed, seen, final_msgs, final_tokens = [], set(), None, 0
            for rank, (score, c) in enumerate(ranked, 1):
                trial, trial_seen = list(packed), set(seen)
                add(trial, trial_seen, c, "primary_bge_reranked", score, rank)
                for ec, reason in expand(c, by_sample_type, args.text_window, args.table_window):
                    add(trial, trial_seen, ec, reason, None, None)
                trial = trial[: args.max_evidence_chunks]
                evidence = "\n".join(fmt(x, i + 1) for i, x in enumerate(trial))
                msgs = build_messages(q, evidence)
                prompt = prompt_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                ntok = len(prompt_tok(prompt, add_special_tokens=False).input_ids)
                if ntok <= args.token_budget or not packed:
                    packed, seen, final_msgs, final_tokens = trial, trial_seen, msgs, ntok

            if final_msgs is None:
                final_msgs = build_messages(q, "")
                final_tokens = 0
            gold = set(r.get("gold_chunk_ids") or [])
            ev = {x.get("chunk_id") for x in packed}
            hit = gold & ev
            stats["written"] += 1
            stats["avg_tokens"] += final_tokens
            stats["avg_chunks"] += len(packed)
            stats["any_gold"] += 1 if hit else 0
            stats["all_gold"] += 1 if gold and gold <= ev else 0
            stats["coverage_sum"] += len(hit) / len(gold) if gold else 0
            row = {
                "messages": final_msgs,
                "meta": {
                    **meta,
                    "id": sid,
                    "question": q,
                    "answer": meta.get("answer") or r.get("answer"),
                    "program_re": meta.get("program_re") or r.get("program_re"),
                    "bge_rerank_budget": True,
                    "candidate_top_k": args.candidate_top_k,
                    "token_budget": args.token_budget,
                    "gold_chunk_ids": list(gold),
                    "evidence_chunk_ids": [x.get("chunk_id") for x in packed],
                    "prompt_tokens": final_tokens,
                },
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = stats["written"] or 1
    summary = {
        "written": stats["written"],
        "avg_tokens": round(stats["avg_tokens"] / n, 1),
        "avg_chunks": round(stats["avg_chunks"] / n, 2),
        "any_recall": round(stats["any_gold"] / n * 100, 2),
        "all_recall": round(stats["all_gold"] / n * 100, 2),
        "coverage": round(stats["coverage_sum"] / n * 100, 2),
        "output": args.output_jsonl,
        "reranker_device": str(reranker.device),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
