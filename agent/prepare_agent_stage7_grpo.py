import argparse
import json
import random
from pathlib import Path


HARD_KEYWORDS = (
    "percent", "percentage", "rate", "ratio", "margin", "average", "change",
    "increase", "decrease", "growth", "cagr", "return", "divided", "per",
)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def first_prompt(messages):
    prompt = []
    for msg in messages:
        if msg.get("role") == "assistant":
            break
        prompt.append(msg)
    return prompt


def first_assistant(messages):
    for msg in messages:
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def source_gold_tool(ex):
    meta = ex.get("meta") or {}
    if meta.get("gold_tool"):
        return meta.get("gold_tool")
    raw = first_assistant(ex.get("messages") or [])
    try:
        obj = json.loads(raw)
        call = obj.get("tool_call") or {}
        return call.get("name") or ""
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage6_train", default="data/agent/processed/stage6/agent_stage6_finqa_train.jsonl")
    ap.add_argument("--stage6_candidates", default="outputs/finqa/predictions/stage6_agent_finqa_train_for_stage7_dpo.jsonl")
    ap.add_argument("--out_jsonl", default="data/agent/processed/stage7/agent_stage7_finqa_grpo_balanced.jsonl")
    ap.add_argument("--seed", type=int, default=27)
    ap.add_argument("--max_wrong", type=int, default=1800)
    ap.add_argument("--max_anchor", type=int, default=1200)
    ap.add_argument("--max_keyword", type=int, default=800)
    ap.add_argument("--max_replay", type=int, default=800)
    args = ap.parse_args()

    random.seed(args.seed)
    rows = load_jsonl(args.stage6_train)
    preds = load_jsonl(args.stage6_candidates)
    by_id = {(ex.get("meta") or {}).get("id"): ex for ex in rows if (ex.get("meta") or {}).get("source") == "finqa_agent"}

    wrong_ids, correct_ids = [], []
    for pr in preds:
        sid = pr.get("id")
        if sid not in by_id:
            continue
        if pr.get("exec_scaled_at_5"):
            correct_ids.append(sid)
        else:
            wrong_ids.append(sid)

    keyword_ids = []
    for sid, ex in by_id.items():
        q = str((ex.get("meta") or {}).get("question") or "").lower()
        prog = str((ex.get("meta") or {}).get("program_re") or "").lower()
        if any(k in q or k in prog for k in HARD_KEYWORDS):
            keyword_ids.append(sid)

    random.shuffle(wrong_ids)
    random.shuffle(correct_ids)
    random.shuffle(keyword_ids)

    selected = []
    seen = set()

    def add_finqa(ids, tag, limit):
        added = 0
        for sid in ids:
            if added >= limit:
                break
            if (sid, tag) in seen:
                continue
            ex = by_id[sid]
            meta = ex.get("meta") or {}
            selected.append({
                "prompt": first_prompt(ex.get("messages") or []),
                "answer": str(meta.get("answer") or ""),
                "gold_program": str(meta.get("program_re") or ""),
                "task_type": "finqa",
                "tag": tag,
                "id": sid,
                "question": meta.get("question"),
            })
            seen.add((sid, tag))
            added += 1

    add_finqa(wrong_ids, "stage6_wrong", args.max_wrong)
    add_finqa(correct_ids, "stage6_correct_anchor", args.max_anchor)
    add_finqa(keyword_ids, "keyword_hard", args.max_keyword)

    replay = []
    for ex in rows:
        meta = ex.get("meta") or {}
        if meta.get("source") == "finqa_agent":
            continue
        gold_tool = source_gold_tool(ex)
        if not gold_tool:
            continue
        replay.append({
            "prompt": first_prompt(ex.get("messages") or []),
            "gold_tool": gold_tool,
            "task_type": "route_replay",
            "tag": "agent_replay",
            "id": meta.get("id") or meta.get("bfcl_id") or meta.get("source"),
        })
    random.shuffle(replay)
    selected.extend(replay[: args.max_replay])
    random.shuffle(selected)

    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "num_samples": len(selected),
        "wrong_available": len(wrong_ids),
        "correct_available": len(correct_ids),
        "keyword_available": len(keyword_ids),
        "requested": {
            "wrong": args.max_wrong,
            "anchor": args.max_anchor,
            "keyword": args.max_keyword,
            "replay": args.max_replay,
        },
        "actual_counts": {},
        "out_jsonl": str(out),
    }
    for row in selected:
        summary["actual_counts"][row["tag"]] = summary["actual_counts"].get(row["tag"], 0) + 1
    out.with_name("stage7_grpo_data_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
