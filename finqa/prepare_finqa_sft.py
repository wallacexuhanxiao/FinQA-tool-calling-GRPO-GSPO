import json
from pathlib import Path

from datasets import load_dataset, load_from_disk


SYSTEM = (
    "You are a financial reasoning assistant. "
    "Solve numerical reasoning questions over financial reports. "
    "Return JSON only with keys: program and answer."
)


def table_to_markdown(table):
    if not table:
        return ""
    rows = [[str(x) for x in row] for row in table]
    if len(rows) == 1:
        return " | ".join(rows[0])

    header = rows[0]
    md = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows[1:]:
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        md.append("| " + " | ".join(row[: len(header)]) + " |")
    return "\n".join(md)


def build_context(ex):
    pre = "\n".join(ex.get("pre_text") or [])
    post = "\n".join(ex.get("post_text") or [])
    table = table_to_markdown(ex.get("table") or [])
    return f"Pre-text:\n{pre}\n\nTable:\n{table}\n\nPost-text:\n{post}".strip()


def gold_answer(ex):
    return str(ex.get("final_result") or ex.get("answer") or "")


def load_finqa(raw_dir):
    raw_path = Path(raw_dir)
    if raw_path.exists():
        return load_from_disk(str(raw_path))
    return load_dataset("ibm-research/finqa", trust_remote_code=True)


def convert_split(ds, split, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ex in ds[split]:
            context = build_context(ex)
            question = ex["question"]
            answer = gold_answer(ex)
            program = str(ex.get("program_re") or "")

            user = (
                "Given the financial context and question, generate a short executable reasoning program "
                "and the final answer.\n\n"
                "Return JSON only:\n"
                '{"program": "...", "answer": "..."}\n\n'
                f"Financial context:\n{context}\n\n"
                f"Question:\n{question}"
            )

            assistant = json.dumps(
                {"program": program, "answer": answer},
                ensure_ascii=False,
            )

            row = {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ],
                "meta": {
                    "id": ex.get("id"),
                    "split": split,
                    "question": question,
                    "answer": answer,
                    "program_re": program,
                    "gold_inds": ex.get("gold_inds", []),
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(split, n, out_path)


def main():
    ds = load_finqa("data/finqa/raw/hf_finqa")
    out_dir = Path("data/finqa/processed/sft")
    convert_split(ds, "train", out_dir / "train.jsonl")
    convert_split(ds, "validation", out_dir / "dev.jsonl")
    convert_split(ds, "test", out_dir / "test.jsonl")


if __name__ == "__main__":
    main()
