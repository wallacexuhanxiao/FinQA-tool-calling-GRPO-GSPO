import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_prompt(ex):
    return ex["messages"], ex.get("meta") or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_path",
        default="models/Qwen2.5-7B-Instruct",
        help="Pure text Qwen model path.",
    )
    ap.add_argument("--adapter_path", default="")
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=256)
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
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            rows.append(json.loads(line))

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for ex in tqdm(rows):
            messages, meta = load_prompt(ex)
            prompt_messages = [m for m in messages if m["role"] != "assistant"]
            text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer([text], return_tensors="pt").to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )

            gen = out[0][inputs.input_ids.shape[1] :]
            raw = tokenizer.decode(gen, skip_special_tokens=True)

            record = {
                "id": meta.get("id"),
                "question": meta.get("question"),
                "answer": meta.get("answer"),
                "gold_program": meta.get("program_re"),
                "gold_inds": meta.get("gold_inds"),
                "raw_output": raw,
                "meta": meta,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()


if __name__ == "__main__":
    main()
