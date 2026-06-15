import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

SYSTEM = (
    "You are a financial reasoning assistant. "
    "Solve numerical reasoning questions over retrieved financial evidence. "
    "Return JSON only with keys: program and answer."
)

def load_sft_meta(path):
    by_id = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            meta = ex.get("meta") or {}
            sid = meta.get("id")
            if sid:
                by_id[sid] = meta
    return by_id

def load_corpus(path):
    by_chunk = {}
    by_sample_type = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            by_chunk[c["chunk_id"]] = c
            by_sample_type[(c["sample_id"], c["chunk_type"])][int(c.get("chunk_index", 0))] = c
    return by_chunk, by_sample_type

def add_chunk(out, seen, c, reason, primary_rank=None):
    if not c:
        return
    cid = c.get("chunk_id")
    if cid in seen:
        if primary_rank is not None and seen[cid].get("primary_rank") is None:
            seen[cid]["primary_rank"] = primary_rank
        return
    item = dict(c)
    item["expand_reason"] = reason
    item["primary_rank"] = primary_rank
    seen[cid] = item
    out.append(item)

def expand_for_chunk(c, by_sample_type, text_window=1, table_window=1):
    sid = c.get("sample_id")
    ctype = c.get("chunk_type")
    idx = int(c.get("chunk_index", 0))
    expanded = []
    if ctype == "table_row":
        header = by_sample_type.get((sid, "table_header"), {}).get(0)
        if header:
            expanded.append((header, "table_header"))
        rows = by_sample_type.get((sid, "table_row"), {})
        for j in range(idx - table_window, idx + table_window + 1):
            if j in rows:
                expanded.append((rows[j], "neighbor_table_row" if j != idx else "primary_table_row"))
    elif ctype == "table_header":
        rows = by_sample_type.get((sid, "table_row"), {})
        for j in range(1, table_window + 2):
            if j in rows:
                expanded.append((rows[j], "header_following_row"))
    elif ctype in {"pre_text", "post_text"}:
        rows = by_sample_type.get((sid, ctype), {})
        for j in range(idx - text_window, idx + text_window + 1):
            if j in rows:
                expanded.append((rows[j], "neighbor_text" if j != idx else "primary_text"))
    return expanded

def compact_evidence_text(c, n):
    rank = c.get("primary_rank")
    rank_s = f"rank={rank}" if rank is not None else "expanded"
    reason = c.get("expand_reason", "retrieved")
    ctype = c.get("chunk_type", "chunk")
    text = re.sub(r"\s+", " ", c.get("text", "")).strip()
    return f"[{n}] ({rank_s}; {reason}; {ctype}) {text}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval_jsonl", required=True)
    ap.add_argument("--sft_jsonl", required=True)
    ap.add_argument("--corpus_jsonl", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--text_window", type=int, default=1)
    ap.add_argument("--table_window", type=int, default=1)
    ap.add_argument("--max_evidence_chunks", type=int, default=18)
    args = ap.parse_args()

    meta_by_id = load_sft_meta(args.sft_jsonl)
    by_chunk, by_sample_type = load_corpus(args.corpus_jsonl)
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    stats = {"raw_chunks": 0, "expanded_chunks": 0}
    with open(args.retrieval_jsonl, encoding="utf-8") as f, open(args.output_jsonl, "w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            sid = r.get("sample_id") or r.get("id")
            meta = dict(meta_by_id.get(sid, {}))
            question = r.get("question") or meta.get("question")
            retrieved = (r.get("retrieved") or r.get("top_chunks") or [])[: args.top_k]

            evidence = []
            seen = {}
            for rc in retrieved:
                cid = rc.get("chunk_id")
                c = by_chunk.get(cid, rc)
                c = {**c, **{k: rc[k] for k in rc if k in {"rank", "score", "is_gold"}}}
                add_chunk(evidence, seen, c, "retrieved", rc.get("rank"))
                for ec, reason in expand_for_chunk(c, by_sample_type, args.text_window, args.table_window):
                    add_chunk(evidence, seen, ec, reason, None)

            evidence = evidence[: args.max_evidence_chunks]
            stats["raw_chunks"] += len(retrieved)
            stats["expanded_chunks"] += len(evidence)
            evidence_lines = [compact_evidence_text(c, i + 1) for i, c in enumerate(evidence)]
            evidence_text = "\n".join(evidence_lines)
            user = (
                "Given the retrieved financial evidence and question, generate a short executable reasoning program "
                "and the final answer. Prefer higher-rank retrieved evidence, but use expanded neighboring rows/sentences "
                "for table headers, units, years, and context. Do not use facts outside the evidence.\n\n"
                "Return JSON only:\n"
                "{\"program\": \"...\", \"answer\": \"...\"}\n\n"
                f"Retrieved and expanded financial evidence:\n{evidence_text}\n\n"
                f"Question:\n{question}"
            )
            row = {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                ],
                "meta": {
                    **meta,
                    "id": sid,
                    "question": question,
                    "answer": meta.get("answer") or r.get("answer") or r.get("gold_answer"),
                    "program_re": meta.get("program_re") or r.get("gold_program"),
                    "rag_top_k": args.top_k,
                    "rag_expanded": True,
                    "retrieval_source": args.retrieval_jsonl,
                    "gold_chunk_ids": r.get("gold_chunk_ids", []),
                    "retrieved_chunk_ids": [c.get("chunk_id") for c in retrieved],
                    "evidence_chunk_ids": [c.get("chunk_id") for c in evidence],
                },
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    stats["written"] = n
    stats["avg_raw_chunks"] = stats["raw_chunks"] / n if n else 0
    stats["avg_expanded_chunks"] = stats["expanded_chunks"] / n if n else 0
    stats["output"] = args.output_jsonl
    print(json.dumps(stats, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
