import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", default="data/agent/processed/stage6/agent_stage6_finqa_train.jsonl")
    ap.add_argument("--output_jsonl", default="data/agent/processed/stage7/agent_stage7_finqa_train_only.jsonl")
    args = ap.parse_args()

    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.input_jsonl, encoding="utf-8") as f, out.open("w", encoding="utf-8") as g:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            meta = ex.get("meta") or {}
            if meta.get("source") == "finqa_agent" and meta.get("answer") and meta.get("program_re"):
                g.write(json.dumps(ex, ensure_ascii=False) + "\n")
                n += 1
    print(json.dumps({"saved": str(out), "num_finqa": n}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
