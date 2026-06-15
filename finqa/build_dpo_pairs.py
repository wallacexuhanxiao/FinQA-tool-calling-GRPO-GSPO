import argparse
import json
from pathlib import Path

from eval.eval_finqa_answer import extract_json, norm_text, numeric_match


def role_to_sharegpt(message):
    role = message["role"]
    if role == "system":
        return {"from": "system", "value": message["content"]}
    if role == "user":
        return {"from": "human", "value": message["content"]}
    if role == "assistant":
        return {"from": "gpt", "value": message["content"]}
    raise ValueError(f"unsupported role: {role}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_jsonl", required=True)
    ap.add_argument("--pred_jsonl", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--dataset_info", default="external/LLaMA-Factory/data/dataset_info.json")
    ap.add_argument("--dataset_name", default="finqa_dpo_full")
    ap.add_argument("--copy_to_dataset_dir", default="external/LLaMA-Factory/data/finqa_dpo_full.json")
    args = ap.parse_args()

    src_rows = [json.loads(x) for x in open(args.src_jsonl, encoding="utf-8") if x.strip()]
    pred_rows = [json.loads(x) for x in open(args.pred_jsonl, encoding="utf-8") if x.strip()]

    pairs = []
    skipped_correct = 0
    skipped_empty = 0

    for ex, pr in zip(src_rows, pred_rows):
        gold_msg = ex["messages"][-1]["content"]
        gold_obj = json.loads(gold_msg)
        gold_answer = str(gold_obj.get("answer", ""))

        raw = str(pr.get("raw_output", "")).strip()
        obj = extract_json(raw)
        if isinstance(obj, dict):
            rejected_answer = str(obj.get("answer", ""))
            rejected = json.dumps(
                {
                    "program": str(obj.get("program", "")),
                    "answer": rejected_answer,
                },
                ensure_ascii=False,
            )
        else:
            rejected_answer = raw.split("\n")[0].strip()
            rejected = raw

        if not rejected.strip():
            skipped_empty += 1
            continue

        if norm_text(rejected_answer) == norm_text(gold_answer) or numeric_match(
            rejected_answer, gold_answer, 0.05
        ):
            skipped_correct += 1
            continue

        prompt_messages = [
            role_to_sharegpt(m) for m in ex["messages"] if m["role"] != "assistant"
        ]
        pairs.append(
            {
                "conversations": prompt_messages,
                "chosen": {"from": "gpt", "value": gold_msg},
                "rejected": {"from": "gpt", "value": rejected},
                "meta": ex.get("meta", {}),
            }
        )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")

    copy_path = Path(args.copy_to_dataset_dir)
    copy_path.parent.mkdir(parents=True, exist_ok=True)
    copy_path.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")

    info_path = Path(args.dataset_info)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info[args.dataset_name] = {
        "file_name": copy_path.name,
        "ranking": True,
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "chosen": "chosen",
            "rejected": "rejected",
        },
    }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "src": args.src_jsonl,
        "pred": args.pred_jsonl,
        "out": str(out_path),
        "dataset_file": str(copy_path),
        "dataset_name": args.dataset_name,
        "pairs": len(pairs),
        "skipped_correct": skipped_correct,
        "skipped_empty": skipped_empty,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
