import json
from pathlib import Path


def main():
    src = Path("external/LLaMA-Factory/data/finqa_toolcall_train_full.jsonl")
    out = Path("data/finqa/processed/grpo/finqa_toolcall_grpo_full.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with src.open(encoding="utf-8") as f, out.open("w", encoding="utf-8") as g:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            meta = ex.get("meta") or {}
            prompt = [m for m in ex["messages"] if m["role"] != "assistant"]
            row = {
                "prompt": prompt,
                "answer": str(meta.get("answer") or ""),
                "gold_program": str(meta.get("program_re") or ""),
                "id": meta.get("id"),
                "question": meta.get("question"),
            }
            g.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(json.dumps({"saved": str(out), "num_rows": n}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
