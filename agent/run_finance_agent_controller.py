import argparse
import ast
import json
import math
import operator
import re
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def extract_json(text):
    s = str(text or "").strip()
    s = re.sub(r"^```json\s*", "", s)
    s = re.sub(r"^```\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        s = m.group(0)
    try:
        return json.loads(s)
    except Exception:
        return None


def normalize_args(args):
    return json.dumps(args or {}, ensure_ascii=False, sort_keys=True)


def build_tool_cache(messages):
    cache = {}
    for i, msg in enumerate(messages[:-1]):
        if msg.get("role") != "assistant":
            continue
        obj = extract_json(msg.get("content", ""))
        if not isinstance(obj, dict) or obj.get("action") != "tool_call":
            continue
        call = obj.get("tool_call") or {}
        name = call.get("name")
        args = call.get("arguments") or {}
        nxt = messages[i + 1]
        if nxt.get("role") == "user" and nxt.get("content", "").startswith("Tool observation"):
            cache[(name, normalize_args(args))] = nxt.get("content", "")
    return cache


def safe_calc(expr):
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
            return OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"unsupported expression node: {type(node).__name__}")

    tree = ast.parse(str(expr), mode="eval")
    val = ev(tree)
    if not math.isfinite(val):
        raise ValueError("non-finite result")
    return round(val, 6)


def extract_numbers(text):
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", str(text or "").replace(",", ""))]


def extract_pred_num(text):
    nums = extract_numbers(text)
    return nums[-1] if nums else None


def extract_gold_num(text):
    text = str(text or "")
    try:
        payload = text.split("\n", 1)[1] if "\n" in text else text
        obj = json.loads(payload)
        return float(obj.get("result"))
    except Exception:
        nums = extract_numbers(text)
        return nums[-1] if nums else None


def numeric_match(pred, gold, tol=0.01):
    p = extract_pred_num(pred)
    g = extract_gold_num(gold)
    if p is None or g is None:
        return False
    return abs(p - g) / max(abs(g), 1e-6) <= tol


def gold_scaled(ex):
    return (ex.get("meta") or {}).get("gold_scaled_answer", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--adapter_path", default="")
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--out_metrics", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_turns", type=int, default=6)
    ap.add_argument("--max_new_tokens", type=int, default=192)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    rows = []
    with open(args.input_jsonl, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            if line.strip():
                ex = json.loads(line)
                if (ex.get("meta") or {}).get("source") == "synthetic_finance_agent":
                    rows.append(ex)

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_metrics).parent.mkdir(parents=True, exist_ok=True)

    counters = {
        "samples": 0,
        "completed": 0,
        "json_steps": 0,
        "total_steps": 0,
        "tool_steps": 0,
        "tool_success": 0,
        "finance_api_success": 0,
        "calculator_success": 0,
        "final_acc_1pct": 0,
        "final_acc_5pct": 0,
    }
    bad = []

    with open(args.output_jsonl, "w", encoding="utf-8") as out:
        for ex in tqdm(rows):
            counters["samples"] += 1
            messages = [ex["messages"][0], ex["messages"][1]]
            cache = build_tool_cache(ex["messages"])
            trace = []
            final = ""
            for _ in range(args.max_turns):
                text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tok([text], return_tensors="pt").to(model.device)
                with torch.no_grad():
                    gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                raw = tok.decode(gen[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
                obj = extract_json(raw)
                counters["total_steps"] += 1
                if isinstance(obj, dict):
                    counters["json_steps"] += 1
                trace.append({"assistant_raw": raw, "assistant_json": obj})
                messages.append({"role": "assistant", "content": raw})

                if not isinstance(obj, dict):
                    break
                if obj.get("action") == "final":
                    final = obj.get("answer", "")
                    counters["completed"] += 1
                    break
                if obj.get("action") != "tool_call":
                    break

                counters["tool_steps"] += 1
                call = obj.get("tool_call") or {}
                name = call.get("name")
                call_args = call.get("arguments") or {}
                obs = ""
                if name == "finance_api":
                    obs = cache.get((name, normalize_args(call_args)), "")
                    if obs:
                        counters["tool_success"] += 1
                        counters["finance_api_success"] += 1
                    else:
                        obs = "Tool observation from finance_api:\n" + json.dumps({"error": "not_found", "arguments": call_args}, ensure_ascii=False)
                elif name == "calculator":
                    try:
                        result = safe_calc(call_args.get("expression", ""))
                        obs = "Tool observation from calculator:\n" + json.dumps({"ok": True, "result": result}, ensure_ascii=False)
                        counters["tool_success"] += 1
                        counters["calculator_success"] += 1
                    except Exception as e:
                        obs = "Tool observation from calculator:\n" + json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
                else:
                    obs = "Tool observation from unknown:\n" + json.dumps({"error": "unsupported_tool", "name": name}, ensure_ascii=False)
                trace[-1]["observation"] = obs
                messages.append({"role": "user", "content": obs})

            gold = gold_scaled(ex)
            acc1 = numeric_match(final, gold, 0.01)
            acc5 = numeric_match(final, gold, 0.05)
            counters["final_acc_1pct"] += int(acc1)
            counters["final_acc_5pct"] += int(acc5)
            rec = {
                "question": ex["messages"][1]["content"],
                "final_answer": final,
                "gold_scaled_answer": gold,
                "acc1": acc1,
                "acc5": acc5,
                "trace": trace,
                "meta": ex.get("meta", {}),
            }
            if len(bad) < 20 and not acc5:
                bad.append(rec)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()

    n = counters["samples"]
    metrics = {
        "num_samples": n,
        "completion_rate": counters["completed"] / n if n else 0,
        "json_valid_step_rate": counters["json_steps"] / counters["total_steps"] if counters["total_steps"] else 0,
        "tool_success_rate": counters["tool_success"] / counters["tool_steps"] if counters["tool_steps"] else 0,
        "finance_api_success": counters["finance_api_success"],
        "calculator_success": counters["calculator_success"],
        "final_numeric_acc_1pct": counters["final_acc_1pct"] / n if n else 0,
        "final_numeric_acc_5pct": counters["final_acc_5pct"] / n if n else 0,
        "bad_preview": bad,
    }
    Path(args.out_metrics).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
