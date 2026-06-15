import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_prompts(path, limit, tokenizer):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            messages = [m for m in ex["messages"] if m["role"] != "assistant"]
            rows.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            if limit and len(rows) >= limit:
                break
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="saves/finqa/qwen25-7b-finqa-toolcall-grpo-r32-scaled-full-merged")
    parser.add_argument("--input_jsonl", default="external/LLaMA-Factory/data/finqa_toolcall_dev.jsonl")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--out_json", default="outputs/finqa/metrics/hf_benchmark.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    prompts = load_prompts(args.input_jsonl, args.limit, tokenizer)

    latencies = []
    out_tokens = 0
    start_all = time.perf_counter()
    for i in tqdm(range(0, len(prompts), args.batch_size)):
        batch = prompts[i : i + args.batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        elapsed = time.perf_counter() - start
        latencies.extend([elapsed / len(batch)] * len(batch))
        out_tokens += int((outputs.shape[1] - inputs.input_ids.shape[1]) * len(batch))
    total = time.perf_counter() - start_all

    lats = sorted(latencies)
    def pct(q):
        if not lats:
            return None
        return lats[min(len(lats) - 1, int(round((len(lats) - 1) * q)))]

    metrics = {
        "backend": "hf_transformers",
        "model": args.model_path,
        "num_requests": len(prompts),
        "batch_size": args.batch_size,
        "elapsed_sec": total,
        "requests_per_sec": len(prompts) / total if total else None,
        "output_tokens_per_sec": out_tokens / total if total else None,
        "latency_p50_sec": pct(0.50),
        "latency_p95_sec": pct(0.95),
        "latency_mean_sec": statistics.mean(lats) if lats else None,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
