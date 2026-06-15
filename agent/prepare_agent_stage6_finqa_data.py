import json
import random
import re
from pathlib import Path


ROOT = Path("/root/autodl-tmp/gui-grounding-agent")

SYSTEM = (
    "You are a financial tool-use agent. Read the financial context and question, "
    "call the calculator tool with an executable FinQA program, then use the tool observation "
    "to return the final answer. Return JSON only. Valid actions are tool_call and final."
)


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_context(old_user):
    text = str(old_user or "")
    m = re.search(r"Financial context:\n(.*)", text, flags=re.S)
    if m:
        return m.group(1).strip()
    return text.strip()


def make_finqa_agent_row(ex, split):
    meta = ex.get("meta") or {}
    old_user = ""
    for msg in ex.get("messages", []):
        if msg.get("role") == "user":
            old_user = msg.get("content", "")
            break
    context_and_question = extract_context(old_user)
    question = meta.get("question", "")
    answer = str(meta.get("answer", ""))
    program = str(meta.get("program_re", ""))

    user = (
        "Solve this FinQA financial numerical reasoning task by using the calculator tool.\n\n"
        "First return a calculator tool call in this JSON format:\n"
        "{\"action\":\"tool_call\",\"tool_call\":{\"name\":\"calculator\",\"arguments\":{\"program\":\"...\"}}}\n\n"
        "After the calculator observation is provided, return final JSON:\n"
        "{\"action\":\"final\",\"answer\":\"...\"}\n\n"
        "Use operations such as add, subtract, multiply, divide, average, max, min. "
        "Use #0, #1, ... to refer to previous calculator step results.\n\n"
        f"Financial context and question:\n{context_and_question}"
    )
    tool_call = json.dumps(
        {"action": "tool_call", "tool_call": {"name": "calculator", "arguments": {"program": program}}},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    obs = "Tool observation from calculator:\n" + json.dumps({"ok": True, "result": answer}, ensure_ascii=False)
    final = json.dumps({"action": "final", "answer": answer}, ensure_ascii=False, separators=(",", ":"))
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": tool_call},
            {"role": "user", "content": obs},
            {"role": "assistant", "content": final},
        ],
        "meta": {**meta, "source": "finqa_agent", "split": split},
    }


def sample_replay(path, n, rng):
    rows = read_jsonl(path)
    rng.shuffle(rows)
    return rows[:n]


def register_lf_dataset(names):
    info_p = ROOT / "external/LLaMA-Factory/data/dataset_info.json"
    info = json.loads(info_p.read_text(encoding="utf-8"))
    base = {
        "formatting": "sharegpt",
        "columns": {"messages": "messages"},
        "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"},
    }
    for name, file_name in names:
        item = dict(base)
        item["file_name"] = file_name
        info[name] = item
    info_p.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    rng = random.Random(20260606)
    out_dir = ROOT / "data/agent/processed/stage6"
    lf_dir = ROOT / "external/LLaMA-Factory/data"
    train_finqa = [make_finqa_agent_row(ex, "train") for ex in read_jsonl(ROOT / "data/finqa/processed/sft/train.jsonl")]
    dev_finqa = [make_finqa_agent_row(ex, "dev") for ex in read_jsonl(ROOT / "data/finqa/processed/sft/dev.jsonl")]

    replay = []
    replay.extend(sample_replay(ROOT / "data/agent/processed/stage4/agent_stage4_train.jsonl", 1200, rng))
    replay.extend(sample_replay(ROOT / "data/agent/processed/stage3/agent_stage3_train.jsonl", 800, rng))
    replay.extend(sample_replay(ROOT / "data/agent/processed/stage2/agent_stage2_train.jsonl", 600, rng))
    train = train_finqa + replay
    rng.shuffle(train)

    write_jsonl(out_dir / "agent_stage6_finqa_train.jsonl", train)
    write_jsonl(out_dir / "agent_stage6_finqa_dev.jsonl", dev_finqa)
    write_jsonl(lf_dir / "agent_stage6_finqa_train.jsonl", train)
    write_jsonl(lf_dir / "agent_stage6_finqa_dev.jsonl", dev_finqa)
    register_lf_dataset([
        ("agent_stage6_finqa_train", "agent_stage6_finqa_train.jsonl"),
        ("agent_stage6_finqa_dev", "agent_stage6_finqa_dev.jsonl"),
    ])
    summary = {
        "finqa_train": len(train_finqa),
        "agent_replay": len(replay),
        "train_total": len(train),
        "finqa_dev": len(dev_finqa),
    }
    (out_dir / "stage6_data_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
