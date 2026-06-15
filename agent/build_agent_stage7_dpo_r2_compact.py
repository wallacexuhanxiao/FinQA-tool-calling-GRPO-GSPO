import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.finqa_calculator import execute_program, numeric_match_scaled
from scripts.agent.build_agent_stage7_dpo import (
    direct_answer_reject,
    first_assistant,
    load_jsonl,
    normalize_rejected,
    wrong_tool_reject,
)
from scripts.agent.run_finqa_agent_controller import parse_tool


COMPACT_SYSTEM = (
    "You are a financial tool-use agent. Use the calculator tool for FinQA numerical reasoning. "
    "Return JSON only. First return {\"action\":\"tool_call\",\"tool_call\":{\"name\":\"calculator\",\"arguments\":{\"program\":\"...\"}}}."
)


def compact_prompt(ex):
    meta = ex.get("meta") or {}
    evidence = meta.get("gold_inds") or []
    if isinstance(evidence, list):
        evidence_text = "\n".join(str(x) for x in evidence)
    else:
        evidence_text = str(evidence)
    user = (
        "Write the calculator tool call for this FinQA item.\n\n"
        f"Question:\n{meta.get('question', '')}\n\n"
        f"Gold supporting evidence:\n{evidence_text}\n\n"
        "Use operations such as add, subtract, multiply, divide, average, max, min. "
        "Use #0, #1, ... for previous results. Do not output the final answer directly."
    )
    return [{"from": "system", "value": COMPACT_SYSTEM}, {"from": "human", "value": user}]


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
    if numeric_match_scaled(exe.get("result"), gold, 0.05, pred_is_value=True):
        return "scaled_correct", True
    return "scaled_wrong", False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finqa_train_jsonl", default="data/agent/processed/stage7/agent_stage7_finqa_train_only.jsonl")
    ap.add_argument("--pred_jsonl", default="outputs/finqa/predictions/stage6_agent_finqa_train_for_stage7_dpo.jsonl")
    ap.add_argument("--replay_jsonl", default="data/agent/processed/stage6/agent_stage6_finqa_train.jsonl")
    ap.add_argument("--out_json", default="data/agent/processed/stage7/agent_stage7_finqa_dpo_r2_compact.json")
    ap.add_argument("--out_external", default="external/LLaMA-Factory/data/agent_stage7_finqa_dpo_r2_compact.json")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--max_wrong", type=int, default=900)
    ap.add_argument("--max_anchor", type=int, default=900)
    ap.add_argument("--max_replay", type=int, default=1500)
    args = ap.parse_args()

    random.seed(args.seed)
    train_rows = load_jsonl(args.finqa_train_jsonl)
    pred_rows = load_jsonl(args.pred_jsonl)
    by_id = {(row.get("meta") or {}).get("id"): row for row in train_rows}

    wrong, anchors, status_counts = [], [], Counter()
    for pred in pred_rows:
        ex = by_id.get(pred.get("id"))
        if not ex:
            continue
        meta = ex.get("meta") or {}
        gold = str(meta.get("answer") or "")
        chosen = first_assistant(ex.get("messages") or [])
        status, scaled_ok = candidate_status(pred, gold)
        status_counts[status] += 1
        if scaled_ok:
            anchors.append({
                "conversations": compact_prompt(ex),
                "chosen": {"from": "gpt", "value": chosen},
                "rejected": {"from": "gpt", "value": direct_answer_reject(gold)},
                "meta": {**meta, "pair_type": "finqa_correct_anchor_compact", "candidate_status": status},
            })
        else:
            wrong.append({
                "conversations": compact_prompt(ex),
                "chosen": {"from": "gpt", "value": chosen},
                "rejected": {"from": "gpt", "value": normalize_rejected(pred.get("raw_tool") or "")},
                "meta": {**meta, "pair_type": "finqa_wrong_compact", "candidate_status": status},
            })

    random.shuffle(wrong)
    random.shuffle(anchors)
    pairs = wrong[: args.max_wrong] + anchors[: args.max_anchor]

    replay = []
    for ex in load_jsonl(args.replay_jsonl):
        meta = ex.get("meta") or {}
        if meta.get("source") == "finqa_agent":
            continue
        chosen = first_assistant(ex.get("messages") or [])
        if not chosen:
            continue
        prompt = []
        for msg in ex.get("messages") or []:
            role = msg.get("role")
            if role == "assistant":
                break
            if role == "system":
                prompt.append({"from": "system", "value": msg.get("content", "")})
            elif role == "user":
                prompt.append({"from": "human", "value": msg.get("content", "")})
        replay.append({
            "conversations": prompt,
            "chosen": {"from": "gpt", "value": chosen},
            "rejected": {"from": "gpt", "value": wrong_tool_reject()},
            "meta": {**meta, "pair_type": "agent_replay_r2"},
        })
    random.shuffle(replay)
    pairs += replay[: args.max_replay]
    random.shuffle(pairs)

    for path in [args.out_json, args.out_external]:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "num_pairs": len(pairs),
        "wrong_pairs": min(len(wrong), args.max_wrong),
        "correct_anchor_pairs": min(len(anchors), args.max_anchor),
        "agent_replay_pairs": min(len(replay), args.max_replay),
        "candidate_status_counts": dict(status_counts),
        "prompt_style": "compact_question_gold_evidence",
        "note": "Scale-correct candidates are anchors, not negatives. Starts from Stage6 adapter, not failed Stage7.",
    }
    Path(args.out_json).with_name("stage7_dpo_r2_data_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
