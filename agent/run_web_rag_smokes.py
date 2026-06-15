import json
import re
from pathlib import Path
from scripts.agent.web_chunked_rag import web_chunked_rag

out_dir = Path("outputs/agent/web_rag_smoke")
out_dir.mkdir(parents=True, exist_ok=True)
queries = [
    ("marketcap_tesla_apple", "Tesla market cap Apple market cap difference"),
    ("marketcap_nvidia_microsoft", "NVIDIA market cap Microsoft market cap difference"),
    ("langchain_agents", "LangChain agents tool calling documentation"),
    ("qwen_function_calling", "Qwen function calling tool use documentation"),
    ("zh_tesla_apple_marketcap", "特斯拉市值和苹果市值差多少"),
]
summary = ["# Web Chunked RAG Smoke Tests", ""]
for name, query in queries:
    print(f"RUN {name}: {query}", flush=True)
    data = web_chunked_rag(
        query=query,
        max_search_results=6,
        max_pages_to_fetch=6,
        retrieve_top_n=30,
        final_top_k=5,
    )
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    summary.append(f"## {name}")
    summary.append(f"- query: {query}")
    summary.append(f"- ok: {data.get('ok')}")
    summary.append(f"- backend: {data.get('backend')}")
    summary.append(f"- chunk_count: {data.get('chunk_count')}")
    summary.append(f"- retrieved_count: {data.get('retrieved_count')}")
    summary.append(f"- selected_count: {data.get('selected_count')}")
    summary.append(f"- reranker: {data.get('reranker')}")
    summary.append("- search_urls:")
    for r in data.get("search_results", [])[:6]:
        summary.append(f"  - {r.get('title')} | {r.get('url')}")
    summary.append("- citations:")
    for c in data.get("citations", [])[:5]:
        summary.append(f"  - {c.get('title')} | {c.get('url')} | {c.get('chunk_id')}")
    ctx = (data.get("rag_context") or "").replace("\n", " ")
    ctx = re.sub(r"\s+", " ", ctx)[:700]
    summary.append(f"- context_preview: {ctx}")
    summary.append("")
(out_dir / "WEB_RAG_SMOKE_SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")
print("DONE", out_dir / "WEB_RAG_SMOKE_SUMMARY.md")
