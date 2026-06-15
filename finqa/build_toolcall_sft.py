import json
from pathlib import Path


SYSTEM = (
    "You are a financial reasoning assistant. "
    "Read the financial context and question, then call the calculator tool. "
    "Return JSON only."
)

TOOLCALL_INSTRUCTION = (
    "\n\nReturn a calculator tool call in this exact JSON format:\n"
    "{\"tool_call\":{\"name\":\"calculator\",\"arguments\":{\"program\":\"...\"}}}\n\n"
    "The program should use operations such as add, subtract, multiply, divide, average, max, min.\n"
    "Use #0, #1, ... to refer to previous step results.\n"
    "Do not output the final answer yourself."
)


def convert_row(row):
    messages = row["messages"]
    meta = row.get("meta") or {}
    program = str(meta.get("program_re") or "")
    prompt_messages = []
    for message in messages:
        if message["role"] == "assistant":
            continue
        if message["role"] == "system":
            prompt_messages.append({"role": "system", "content": SYSTEM})
        elif message["role"] == "user":
            prompt_messages.append({"role": "user", "content": message["content"] + TOOLCALL_INSTRUCTION})
    assistant = {
        "tool_call": {
            "name": "calculator",
            "arguments": {"program": program},
        }
    }
    return {
        "messages": prompt_messages + [{"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False)}],
        "meta": meta,
    }


def write_jsonl(rows, path, limit=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            if limit is not None and count >= limit:
                break
            f.write(json.dumps(convert_row(row), ensure_ascii=False) + "\n")
            count += 1
    print(path, count)


def load_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def main():
    train = load_jsonl("data/finqa/processed/sft/train.jsonl")
    dev = load_jsonl("data/finqa/processed/sft/dev.jsonl")
    test = load_jsonl("data/finqa/processed/sft/test.jsonl")

    out = Path("external/LLaMA-Factory/data")
    write_jsonl(train, out / "finqa_toolcall_debug100.jsonl", limit=100)
    write_jsonl(train, out / "finqa_toolcall_train_full.jsonl")
    write_jsonl(dev, out / "finqa_toolcall_dev.jsonl")
    write_jsonl(test, out / "finqa_toolcall_test.jsonl")

    info_path = out / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
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
    for name, file_name in [
        ("finqa_toolcall_debug100", "finqa_toolcall_debug100.jsonl"),
        ("finqa_toolcall_train_full", "finqa_toolcall_train_full.jsonl"),
        ("finqa_toolcall_dev", "finqa_toolcall_dev.jsonl"),
        ("finqa_toolcall_test", "finqa_toolcall_test.jsonl"),
    ]:
        item = dict(base)
        item["file_name"] = file_name
        info[name] = item
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print("registered toolcall datasets")


if __name__ == "__main__":
    main()
