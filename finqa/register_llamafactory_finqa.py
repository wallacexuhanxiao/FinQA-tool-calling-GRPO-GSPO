import json
from pathlib import Path


DATASETS = [
    ("finqa_debug100", "finqa_debug100.jsonl"),
    ("finqa_train1k", "finqa_train1k.jsonl"),
    ("finqa_train_full", "finqa_train_full.jsonl"),
    ("finqa_dev", "finqa_dev.jsonl"),
    ("finqa_test", "finqa_test.jsonl"),
]


def main():
    p = Path("external/LLaMA-Factory/data/dataset_info.json")
    info = json.loads(p.read_text(encoding="utf-8"))

    base = {
        "formatting": "sharegpt",
        "columns": {"messages": "messages"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }

    for name, file_name in DATASETS:
        item = dict(base)
        item["file_name"] = file_name
        info[name] = item

    p.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print("registered FinQA datasets:", ", ".join(name for name, _ in DATASETS))


if __name__ == "__main__":
    main()
