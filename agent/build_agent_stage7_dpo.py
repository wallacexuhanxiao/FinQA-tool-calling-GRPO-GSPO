import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.finqa_calculator import execute_program, numeric_match_scaled
from scripts.agent.run_finqa_agent_controller import extract_json, parse_tool


def to_sharegpt_prompt(messages):
    out = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            out.append({"from": "system", "value": msg.get("content", "")})
        elif role == "user":
            out.append({"from": "human", "value": msg.get("content", "")})
        elif role == "assistant":
            break
    return out


def first_assistant(messages):
    for msg in messages:
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def is_toolcall_json(text):
    obj = extract_json(text)
    return (
        isinstance(obj, dict)
        and obj.get("action") == "tool_call"
        and isinstance(obj.get("tool_call"), dict)
    )


def normalize_rejected(raw):
    raw = str(raw or "").strip()
    if raw:
        return raw
    return '{"action":"final","answer":""}'


def direct_answer_reject(gold_answer):
    return json.dumps({"action": "final", "answer": str(gold_answer)}, ensure_ascii=False)


def wrong_tool_reject():
    return json.dumps({"action": "tool_call", "tool_call": {"name": "web_search", "arguments": {"query": ""}}}, ensure_ascii=False)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def candidate_status(pred, gold):
    raw = pred.get("raw_tool") or ""
    obj, tool_name, program = parse_tool(raw)
    if not isinstance(obj, dict):
        return "invalid_json", False
    if tool_name != "calculator":
        return "wrong_tool", False
    if not str(program).strip():
        return "empty_program", False
    exe = execute_program(program)
    if not exe.get("ok"):
        return "not_executable", False
    # Use scaled correctness. Scale-only differences are treated as correct and
    # therefore ignored as negatives.
    if numeric_match_scaled(exe.get("result"), gold, 0.05, pred_is_value=True):
        return "scaled_correct", True
    return "scaled_wrong", False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finqa_train_jsonl", default="data/agent/processed/stage7/agent_stage7_finqa_train_only.jsonl")
    ap.add_argument("--pred_jsonl", default="outputs/finqa/predictions/stage6_agent_finqa_train_for_stage7_dpo.jsonl")
    ap.add_argument("--replay_jsonl", default="data/agent/processed/stage6/agent_stage6_finqa_train.jsonl")
    ap.add_argument("--out_json", default="data/agent/processed/stage7/agent_stage7_finqa_dpo_balanced.json")
    ap.add_argument("--out_external", default="external/LLaMA-Factory/data/agent_stage7_finqa_dpo_balanced.json")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_wrong", type=int, default=3200)
    ap.add_argument("--max_anchor", type=int, default=1600)
    ap.add_argument("--max_replay", type=int, default=900)
    args = ap.parse_args()

    random.seed(args.seed)
    train_rows = load_jsonl(args.finqa_train_jsonl)
    pred_rows = load_jsonl(args.pred_jsonl)
    by_id = {(row.get("meta") or {}).get("id"): row for row in train_rows}

    wrong_pairs = []
    anchor_candidates = []
    status_counts = Counter()

    for pred in pred_rows:
        sid = pred.get("id")
        ex = by_id.get(sid)
        if not ex:
            continue
        meta = ex.get("meta") or {}
        gold = str(meta.get("answer") or "")
        chosen = first_assistant(ex.get("messages") or [])
        prompt = to_sharegpt_prompt(ex.get("messages") or [])
        status, scaled_ok = candidate_status(pred, gold)
        status_counts[status] += 1
        raw_rejected = normalize_rejected(pred.get("raw_tool") or "")

        if not scaled_ok:
            # Keep real model negatives: invalid JSON, wrong tool, non-executable,
            # or executable but scaled-wrong. Scale-only errors are excluded above.
            wrong_pairs.append({
                "conversations": prompt,
                "chosen": {"from": "gpt", "value": chosen},
                "rejected": {"from": "gpt", "value": raw_rejected},
                "meta": {**meta, "pair_type": "finqa_wrong", "candidate_status": status},
            })
        else:
            anchor_candidates.append((prompt, chosen, gold, meta))

    random.shuffle(wrong_pairs)
    random.shuffle(anchor_candidates)
    wrong_pairs = wrong_pairs[: args.max_wrong]

    anchor_pairs = []
    for prompt, chosen, gold, meta in anchor_candidates[: args.max_anchor]:
        # Correct-sample anchors prevent the DPO set from becoming only an error
        # distribution. The rejected output is a direct final answer, so we do
        # not penalize alternative correct calculator programs.
        anchor_pairs.append({
            "conversations": prompt,
            "chosen": {"from": "gpt", "value": chosen},
            "rejected": {"from": "gpt", "value": direct_answer_reject(gold)},
            "meta": {**meta, "pair_type": "finqa_correct_anchor", "candidate_status": "scaled_correct"},
        })

    replay_rows = []
    for ex in load_jsonl(args.replay_jsonl):
        meta = ex.get("meta") or {}
        if meta.get("source") == "finqa_agent":
            continue
        chosen = first_assistant(ex.get("messages") or [])
        if not chosen:
            continue
        prompt = to_sharegpt_prompt(ex.get("messages") or [])
        rejected = direct_answer_reject("")
        if is_toolcall_json(chosen):
            rejected = wrong_tool_reject()
        replay_rows.append({
            "conversations": prompt,
            "chosen": {"from": "gpt", "value": chosen},
            "rejected": {"from": "gpt", "value": rejected},
            "meta": {**meta, "pair_type": "agent_replay"},
        })
    random.shuffle(replay_rows)
    replay_pairs = replay_rows[: args.max_replay]

    pairs = wrong_pairs + anchor_pairs + replay_pairs
    random.shuffle(pairs)

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    ext = Path(args.out_external)
    ext.parent.mkdir(parents=True, exist_ok=True)
    ext.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "num_pairs": len(pairs),
        "wrong_pairs": len(wrong_pairs),
        "correct_anchor_pairs": len(anchor_pairs),
        "agent_replay_pairs": len(replay_pairs),
        "candidate_status_counts": dict(status_counts),
        "out_json": str(out),
        "out_external": str(ext),
    }
    summary_path = out.with_name("stage7_dpo_data_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
