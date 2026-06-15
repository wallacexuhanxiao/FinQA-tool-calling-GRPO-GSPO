import json
import random
from pathlib import Path


ROOT = Path("/root/autodl-tmp/gui-grounding-agent")

SYSTEM = (
    "You are a multi-step tool-use agent. Read the user task, call tools when needed, "
    "use tool observations to decide the next step, and return a final answer when done. "
    "Return JSON only. Valid actions are tool_call, final, and clarify."
)

BFCL_FILES = [
    ROOT / "data/agent/raw/bfcl/BFCL_v3_multi_turn_base.json",
    ROOT / "data/agent/raw/bfcl/BFCL_v3_multi_turn_composite.json",
    ROOT / "data/agent/raw/bfcl/BFCL_v3_multi_turn_long_context.json",
    ROOT / "data/agent/raw/bfcl/BFCL_v3_multi_turn_miss_func.json",
    ROOT / "data/agent/raw/bfcl/BFCL_v3_multi_turn_miss_param.json",
]


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


def questions_text(qs):
    lines = []
    for i, turn in enumerate(qs or [], 1):
        parts = []
        for msg in turn:
            role = msg.get("role", "user")
            content = str(msg.get("content", "")).strip()
            if content:
                parts.append(f"{role}: {content}")
        if parts:
            lines.append(f"Turn {i}: " + " ".join(parts))
    return "\n".join(lines)


def collect_bfcl_raw():
    raw = []
    all_tools = []
    for path in BFCL_FILES:
        for obj in read_jsonl(path):
            tools = [str(x) for x in (obj.get("path") or []) if x]
            if not tools:
                continue
            raw.append((path.stem, obj, tools))
            all_tools.extend(tools)
    return raw, sorted(set(all_tools))


def bfcl_multiturn_rows():
    rng = random.Random(20260606)
    raw, tool_pool = collect_bfcl_raw()
    rows = []
    for source, obj, path in raw:
        unique_path = list(dict.fromkeys(path))
        distractors = [x for x in tool_pool if x not in unique_path]
        rng.shuffle(distractors)
        candidates = unique_path + distractors[:10]
        rng.shuffle(candidates)
        task = questions_text(obj.get("question") or [])
        cand_text = "\n".join(f"- {x}" for x in candidates)
        involved = ", ".join(obj.get("involved_classes") or [])
        user = (
            "Complete this BFCL multi-turn task by selecting tools step by step.\n\n"
            f"Sample id: {obj.get('id', '')}\n"
            f"Involved classes: {involved}\n\n"
            f"Task conversation:\n{task}\n\n"
            f"Candidate tools:\n{cand_text}\n\n"
            "For each tool step, return JSON: "
            "{\"action\":\"tool_call\",\"tool_call\":{\"name\":\"...\",\"arguments\":{}}}. "
            "After the required tool sequence is complete, return JSON final."
        )
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        for step_idx, tool in enumerate(path):
            messages.append({
                "role": "assistant",
                "content": json.dumps({"action": "tool_call", "tool_call": {"name": tool, "arguments": {}}}, ensure_ascii=False, separators=(",", ":")),
            })
            obs = {"ok": True, "selected_tool": tool, "step": step_idx + 1, "remaining_steps": len(path) - step_idx - 1}
            messages.append({"role": "user", "content": "Tool observation from bfcl_router:\n" + json.dumps(obs, ensure_ascii=False)})
        messages.append({
            "role": "assistant",
            "content": json.dumps({"action": "final", "answer": "Completed tool plan: " + " -> ".join(path)}, ensure_ascii=False, separators=(",", ":")),
        })
        rows.append({
            "messages": messages,
            "meta": {"source": "bfcl_multiturn_trace", "bfcl_file": source, "bfcl_id": str(obj.get("id", "")), "path": path},
        })
    rng.shuffle(rows)
    return rows


def split_finance_rows():
    train = []
    dev = []
    for p, target in [
        (ROOT / "data/agent/processed/stage1/agent_stage1_train.jsonl", train),
        (ROOT / "data/agent/processed/stage1/agent_stage1_dev.jsonl", dev),
    ]:
        for ex in read_jsonl(p):
            if (ex.get("meta") or {}).get("source") == "synthetic_finance_agent":
                target.append(ex)
    return train, dev


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
    out_dir = ROOT / "data/agent/processed/stage4"
    lf_dir = ROOT / "external/LLaMA-Factory/data"

    bfcl = bfcl_multiturn_rows()
    bfcl_dev = bfcl[:600]
    bfcl_train = bfcl[600:]
    finance_train, finance_dev = split_finance_rows()

    replay = []
    replay.extend(sample_replay(ROOT / "data/agent/processed/stage3/agent_stage3_train.jsonl", 2200, rng))
    replay.extend(sample_replay(ROOT / "data/agent/processed/stage2/agent_stage2_train.jsonl", 1000, rng))
    replay.extend(sample_replay(ROOT / "data/agent/processed/stage1/agent_stage1_train.jsonl", 800, rng))

    train = bfcl_train + finance_train + replay
    rng.shuffle(train)

    write_jsonl(out_dir / "agent_stage4_train.jsonl", train)
    write_jsonl(out_dir / "agent_stage4_bfcl_multiturn_dev.jsonl", bfcl_dev)
    write_jsonl(out_dir / "agent_stage4_finance_multiturn_dev.jsonl", finance_dev)
    write_jsonl(lf_dir / "agent_stage4_train.jsonl", train)
    write_jsonl(lf_dir / "agent_stage4_bfcl_multiturn_dev.jsonl", bfcl_dev)
    write_jsonl(lf_dir / "agent_stage4_finance_multiturn_dev.jsonl", finance_dev)
    register_lf_dataset([
        ("agent_stage4_train", "agent_stage4_train.jsonl"),
        ("agent_stage4_bfcl_multiturn_dev", "agent_stage4_bfcl_multiturn_dev.jsonl"),
        ("agent_stage4_finance_multiturn_dev", "agent_stage4_finance_multiturn_dev.jsonl"),
    ])

    summary = {
        "bfcl_trace_total": len(bfcl),
        "bfcl_trace_train": len(bfcl_train),
        "bfcl_trace_dev": len(bfcl_dev),
        "finance_train": len(finance_train),
        "finance_dev": len(finance_dev),
        "replay_total": len(replay),
        "train_total": len(train),
    }
    (out_dir / "stage4_data_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
