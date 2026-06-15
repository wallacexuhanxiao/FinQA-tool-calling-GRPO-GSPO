import argparse
import json
from pathlib import Path

from scripts.agent.langchain_tool_agent import LangChainQwenAgent


DEMO_TASKS = [
    {
        "id": "calculator_cagr",
        "task": "Use the calculator tool to compute NVIDIA revenue CAGR from 27.0 in FY2023 to 130.5 in FY2025. Return the final percentage.",
        "expected_tools": ["calculator"],
    },
    {
        "id": "finance_then_calculator",
        "task": "Use finance_api to get NVDA revenue for FY2023 and FY2025, then use calculator to compute the CAGR percentage. Return the final answer.",
        "expected_tools": ["finance_api", "calculator"],
    },
    {
        "id": "web_search",
        "task": "Use web_search with max_results 5 and top_k 3 to look up 'LangChain framework official', then summarize what LangChain is in one sentence.",
        "expected_tools": ["web_search"],
    },
]


def tools_called(trace):
    return [step.get("tool_name") for step in trace if step.get("tool_name")]


def rag_summary(trace):
    for step in trace:
        if step.get("tool_name") == "web_search":
            obs = step.get("observation") or {}
            return {
                "backend": obs.get("backend"),
                "live": obs.get("live"),
                "retrieved_count": obs.get("retrieved_count"),
                "rerank_top_k": obs.get("rerank_top_k"),
                "reranker": obs.get("reranker"),
                "top_titles": [row.get("title") for row in (obs.get("reranked_results") or [])],
            }
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter_path", default="saves/agent/qwen25-7b-agent-stage7-finqa-grpo-r32")
    ap.add_argument("--output_jsonl", default="outputs/agent/predictions/langchain_agent_demo.jsonl")
    ap.add_argument("--out_metrics", default="outputs/agent/metrics/langchain_agent_demo_metrics.json")
    ap.add_argument("--max_turns", type=int, default=6)
    ap.add_argument("--max_new_tokens", type=int, default=192)
    args = ap.parse_args()

    agent = LangChainQwenAgent(args.model_path, args.adapter_path, args.max_new_tokens)
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_metrics).parent.mkdir(parents=True, exist_ok=True)

    results = []
    with open(args.output_jsonl, "w", encoding="utf-8") as out:
        for item in DEMO_TASKS:
            result = agent.run(item["task"], max_turns=args.max_turns)
            called = tools_called(result.get("trace", []))
            expected_hit = all(tool in called for tool in item["expected_tools"])
            record = {
                "id": item["id"],
                "task": item["task"],
                "expected_tools": item["expected_tools"],
                "tools_called": called,
                "rag_summary": rag_summary(result.get("trace", [])),
                "expected_tools_hit": expected_hit,
                **result,
            }
            results.append(record)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

    metrics = {
        "num_tasks": len(results),
        "completion_rate": sum(bool(r.get("ok")) for r in results) / len(results),
        "expected_tool_hit_rate": sum(bool(r.get("expected_tools_hit")) for r in results) / len(results),
        "tool_call_rate": sum(bool(r.get("tools_called")) for r in results) / len(results),
        "tools_called_by_task": {r["id"]: r.get("tools_called", []) for r in results},
        "rag_by_task": {r["id"]: r.get("rag_summary", {}) for r in results},
        "final_answers": {r["id"]: r.get("final_answer", "") for r in results},
    }
    Path(args.out_metrics).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
