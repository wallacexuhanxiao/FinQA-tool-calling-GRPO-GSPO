import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from openai import AsyncOpenAI


def load_prompts(path, limit):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            rows.append([m for m in ex["messages"] if m["role"] != "assistant"])
            if limit and len(rows) >= limit:
                break
    return rows


async def worker(client, model, queue, latencies, max_tokens):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return
        idx, messages = item
        start = time.perf_counter()
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=max_tokens,
            )
            usage = getattr(resp, "usage", None)
            out_tokens = getattr(usage, "completion_tokens", 0) or 0
            ok = True
        except Exception:
            out_tokens = 0
            ok = False
        latencies.append((time.perf_counter() - start, out_tokens, ok))
        queue.task_done()


async def run(args):
    prompts = load_prompts(args.input_jsonl, args.limit)
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    queue = asyncio.Queue()
    latencies = []
    for item in enumerate(prompts):
        queue.put_nowait(item)
    for _ in range(args.concurrency):
        queue.put_nowait(None)

    start = time.perf_counter()
    tasks = [
        asyncio.create_task(worker(client, args.model, queue, latencies, args.max_tokens))
        for _ in range(args.concurrency)
    ]
    await queue.join()
    await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start

    ok = sum(1 for _, _, status in latencies if status)
    out_tokens = sum(tokens for _, tokens, _ in latencies)
    lats = sorted(lat for lat, _, _ in latencies)
    def pct(q):
        if not lats:
            return None
        return lats[min(len(lats) - 1, int(round((len(lats) - 1) * q)))]

    metrics = {
        "backend": "vllm",
        "model": args.model,
        "num_requests": len(prompts),
        "ok_requests": ok,
        "concurrency": args.concurrency,
        "elapsed_sec": elapsed,
        "requests_per_sec": len(prompts) / elapsed if elapsed else None,
        "output_tokens_per_sec": out_tokens / elapsed if elapsed else None,
        "latency_p50_sec": pct(0.50),
        "latency_p95_sec": pct(0.95),
        "latency_mean_sec": statistics.mean(lats) if lats else None,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--model", default="finqa-toolcall-qwen25-7b")
    parser.add_argument("--input_jsonl", default="external/LLaMA-Factory/data/finqa_toolcall_dev.jsonl")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--out_json", default="outputs/finqa/metrics/vllm_benchmark.json")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
