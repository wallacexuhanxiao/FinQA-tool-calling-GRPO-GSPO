import argparse
import json
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval_jsonl", required=True)
    ap.add_argument("--sft_jsonl", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()

    meta_by_id = load_sft_meta(args.sft_jsonl)
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.retrieval_jsonl, encoding="utf-8") as f, open(args.output_jsonl, "w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            sid = r.get("sample_id") or r.get("id")
            meta = dict(meta_by_id.get(sid, {}))
            question = r.get("question") or meta.get("question")
            chunks = r.get("retrieved") or r.get("top_chunks") or []
            selected = chunks[: args.top_k]
            evidence_lines = []
            for c in selected:
                rank = c.get("rank", len(evidence_lines) + 1)
                ctype = c.get("chunk_type", "chunk")
                text = c.get("text", "")
                evidence_lines.append(f"[{rank}] ({ctype}) {text}")
            evidence = "\n".join(evidence_lines)
            user = (
                "Given the retrieved financial evidence and question, generate a short executable reasoning program "
                "and the final answer. Use only the retrieved evidence.\n\n"
                "Return JSON only:\n"
                "{\"program\": \"...\", \"answer\": \"...\"}\n\n"
                f"Retrieved financial evidence:\n{evidence}\n\n"
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
                    "retrieval_source": args.retrieval_jsonl,
                    "gold_chunk_ids": r.get("gold_chunk_ids", []),
                    "retrieved_chunk_ids": [c.get("chunk_id") for c in selected],
                },
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(json.dumps({"written": n, "output": args.output_jsonl, "top_k": args.top_k}, ensure_ascii=False))

if __name__ == "__main__":
    main()
