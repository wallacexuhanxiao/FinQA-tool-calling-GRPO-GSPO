import json
import random
from pathlib import Path


ROOT = Path("/root/autodl-tmp/gui-grounding-agent")


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


def first_turn_rows(path, limit, rng):
    rows = read_jsonl(path)
    rng.shuffle(rows)
    out = []
    for ex in rows[:limit]:
        messages = ex.get("messages") or []
        prompt = []
        gold = ""
        for msg in messages:
            if msg.get("role") == "assistant":
                gold = msg.get("content", "")
                break
            prompt.append(msg)
        if not prompt or not gold:
            continue
        try:
            obj = json.loads(gold)
            gold_tool = ((obj.get("tool_call") or {}).get("name") or "")
        except Exception:
            gold_tool = ""
        if not gold_tool:
            continue
        out.append({
            "prompt": prompt,
            "gold_output": gold,
            "gold_tool": gold_tool,
            "source": (ex.get("meta") or {}).get("source", "unknown"),
            "meta": ex.get("meta", {}),
        })
    return out


def turn_rows(path, limit, rng):
    rows = read_jsonl(path)
    rng.shuffle(rows)
    out = []
    for ex in rows[:limit]:
        messages = ex.get("messages") or []
        for i, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            gold = msg.get("content", "")
            try:
                obj = json.loads(gold)
            except Exception:
                continue
            if obj.get("action") != "tool_call":
                continue
            gold_tool = ((obj.get("tool_call") or {}).get("name") or "")
            if not gold_tool:
                continue
            out.append({
                "prompt": messages[:i],
                "gold_output": gold,
                "gold_tool": gold_tool,
                "source": (ex.get("meta") or {}).get("source", "unknown"),
                "meta": {**(ex.get("meta") or {}), "turn_index": i},
            })
            if len(out) >= limit:
                break
        if len(out) >= limit:
            break
    return out


def main():
    rng = random.Random(20260606)
    out = []
    # Hard target: BFCL route/planning. These are exactly where Stage4 is around 0.80.
    out.extend(first_turn_rows(ROOT / "data/agent/processed/stage3/agent_stage3_train.jsonl", 2600, rng))
    out.extend(turn_rows(ROOT / "data/agent/processed/stage4/agent_stage4_train.jsonl", 2200, rng))
    # Replay to keep API and finance tool formats alive.
    out.extend(first_turn_rows(ROOT / "data/agent/processed/stage2/agent_stage2_train.jsonl", 800, rng))
    out.extend(first_turn_rows(ROOT / "data/agent/processed/stage1/agent_stage1_train.jsonl", 800, rng))
    rng.shuffle(out)
    path = ROOT / "data/agent/processed/stage5/agent_stage5_grpo_toolroute.jsonl"
    write_jsonl(path, out)
    summary = {"num_samples": len(out), "output": str(path)}
    (ROOT / "data/agent/processed/stage5/stage5_data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
