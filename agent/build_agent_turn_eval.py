import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.input_jsonl, encoding="utf-8") as f, open(args.output_jsonl, "w", encoding="utf-8") as out:
        for row_idx, line in enumerate(f):
            if args.limit and row_idx >= args.limit:
                break
            if not line.strip():
                continue
            ex = json.loads(line)
            messages = ex["messages"]
            assistant_seen = 0
            for i, msg in enumerate(messages):
                if msg.get("role") != "assistant":
                    continue
                rec = {
                    "prompt_messages": messages[:i],
                    "gold_output": msg.get("content", ""),
                    "meta": {
                        **(ex.get("meta") or {}),
                        "row_idx": row_idx,
                        "turn_idx": assistant_seen,
                    },
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                assistant_seen += 1
                n += 1
    print("wrote", n, args.output_jsonl)


if __name__ == "__main__":
    main()
