import argparse
import json
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.finqa_calculator import execute_program, format_number, numeric_match, numeric_match_scaled


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


def parse_tool(raw):
    obj = extract_json(raw)
    if not isinstance(obj, dict):
        return obj, "", ""
    call = obj.get("tool_call") or {}
    if not isinstance(call, dict):
        return obj, "", ""
    args = call.get("arguments") or {}
    if not isinstance(args, dict):
        args = {}
    return obj, str(call.get("name") or ""), str(args.get("program") or "")


def parse_final(raw):
    obj = extract_json(raw)
    if isinstance(obj, dict) and obj.get("action") == "final":
        return str(obj.get("answer") or ""), True
    if isinstance(obj, dict) and "answer" in obj:
        return str(obj.get("answer") or ""), False
    return "", False


def generate(tok, model, messages, max_new_tokens):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--adapter_path", default="")
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--out_metrics", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_new_tokens_tool", type=int, default=160)
    ap.add_argument("--max_new_tokens_final", type=int, default=96)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
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
                rows.append(json.loads(line))

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_metrics).parent.mkdir(parents=True, exist_ok=True)
    stats = {"n": 0, "json": 0, "calc": 0, "prog": 0, "exec": 0, "final": 0, "exec1": 0, "exec5": 0, "f1": 0, "f5": 0}
    bad = []
    with open(args.output_jsonl, "w", encoding="utf-8") as out:
        for ex in tqdm(rows):
            stats["n"] += 1
            meta = ex.get("meta") or {}
            gold = str(meta.get("answer") or "")
            messages = []
            for msg in ex["messages"]:
                if msg.get("role") == "assistant":
                    break
                messages.append(msg)
            raw_tool = generate(tok, model, messages, args.max_new_tokens_tool)
            tool_obj, tool_name, program = parse_tool(raw_tool)
            stats["json"] += isinstance(tool_obj, dict)
            stats["calc"] += tool_name == "calculator"
            stats["prog"] += bool(program.strip())
            exec_result = execute_program(program) if program.strip() else {"ok": False, "result": None, "error": "empty program", "ops": []}
            stats["exec"] += bool(exec_result.get("ok"))
            exec_value = exec_result.get("result")
            stats["exec1"] += numeric_match_scaled(exec_value, gold, 0.01, pred_is_value=True)
            stats["exec5"] += numeric_match_scaled(exec_value, gold, 0.05, pred_is_value=True)

            obs_payload = {"ok": bool(exec_result.get("ok")), "result": format_number(exec_value) if exec_result.get("ok") else None, "error": exec_result.get("error")}
            messages2 = messages + [{"role": "assistant", "content": raw_tool}, {"role": "user", "content": "Tool observation from calculator:\n" + json.dumps(obs_payload, ensure_ascii=False)}]
            raw_final = generate(tok, model, messages2, args.max_new_tokens_final)
            final_answer, final_ok = parse_final(raw_final)
            stats["final"] += final_ok
            stats["f1"] += numeric_match_scaled(final_answer, gold, 0.01)
            stats["f5"] += numeric_match_scaled(final_answer, gold, 0.05)
            rec = {
                "id": meta.get("id"),
                "question": meta.get("question"),
                "gold_answer": gold,
                "raw_tool": raw_tool,
                "tool_json_valid": isinstance(tool_obj, dict),
                "tool_name": tool_name,
                "program": program,
                "program_executable": bool(exec_result.get("ok")),
                "execution_result": exec_value,
                "exec_error": exec_result.get("error"),
                "raw_final": raw_final,
                "final_answer": final_answer,
                "final_action_valid": final_ok,
                "exec_scaled_at_5": numeric_match_scaled(exec_value, gold, 0.05, pred_is_value=True),
                "final_scaled_at_5": numeric_match_scaled(final_answer, gold, 0.05),
                "meta": meta,
            }
            if len(bad) < 30 and not rec["final_scaled_at_5"]:
                bad.append({k: rec.get(k) for k in ["id", "question", "gold_answer", "final_answer", "execution_result", "exec_error", "raw_tool", "raw_final"]})
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
    n = stats["n"] or 1
    metrics = {
        "num_samples": stats["n"],
        "tool_json_valid_rate": stats["json"] / n,
        "calculator_call_rate": stats["calc"] / n,
        "program_nonempty_rate": stats["prog"] / n,
        "program_executable_rate": stats["exec"] / n,
        "scaled_execution_acc_at_1pct": stats["exec1"] / n,
        "scaled_execution_acc_at_5pct": stats["exec5"] / n,
        "final_action_valid_rate": stats["final"] / n,
        "scaled_final_acc_at_1pct": stats["f1"] / n,
        "scaled_final_acc_at_5pct": stats["f5"] / n,
        "bad_preview": bad,
    }
    Path(args.out_metrics).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
