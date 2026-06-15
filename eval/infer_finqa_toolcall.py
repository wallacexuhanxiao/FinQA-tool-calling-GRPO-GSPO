import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.finqa_calculator import execute_program, format_number


def extract_json(text):
    s = str(text or "").strip()
    s = re.sub(r"^```json\s*", "", s)
    s = re.sub(r"^```\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    match = re.search(r"\{.*\}", s, flags=re.S)
    if match:
        s = match.group(0)
    try:
        return json.loads(s)
    except Exception:
        return None


def parse_tool_call(raw):
    obj = extract_json(raw)
    if not isinstance(obj, dict):
        return False, None, None, "", "invalid json"
    call = obj.get("tool_call")
    if not isinstance(call, dict):
        return True, None, None, "", "missing tool_call"
    name = call.get("name")
    args = call.get("arguments") or {}
    if not isinstance(args, dict):
        args = {}
    program = str(args.get("program") or "")
    return True, call, name, program, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--adapter_path", default="")
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    rows = []
    with open(args.input_jsonl, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            if line.strip():
                rows.append(json.loads(line))

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as out:
        for ex in tqdm(rows):
            meta = ex.get("meta") or {}
            prompt_messages = [m for m in ex["messages"] if m["role"] != "assistant"]
            text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            gen = generated[0][inputs.input_ids.shape[1]:]
            raw = tokenizer.decode(gen, skip_special_tokens=True)

            json_valid, _, tool_name, program, parse_error = parse_tool_call(raw)
            exec_result = execute_program(program) if program else {
                "ok": False,
                "result": None,
                "steps": [],
                "error": parse_error or "empty program",
                "ops": [],
            }
            final_answer = format_number(exec_result["result"]) if exec_result["ok"] else ""
            record = {
                "id": meta.get("id"),
                "question": meta.get("question"),
                "gold_answer": meta.get("answer"),
                "gold_program": meta.get("program_re"),
                "raw_output": raw,
                "tool_json_valid": json_valid,
                "tool_name": tool_name,
                "program": program,
                "program_parse_ok": bool(program),
                "program_executable": exec_result["ok"],
                "execution_result": exec_result["result"],
                "final_answer": final_answer,
                "exec_error": exec_result["error"],
                "ops": exec_result["ops"],
                "meta": meta,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
