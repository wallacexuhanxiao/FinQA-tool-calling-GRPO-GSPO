# Web Chunked RAG Smoke Demo

## Location

Smoke outputs are stored under:

`remote_files/outputs/agent/web_rag_smoke/`

Files:

- `WEB_RAG_SMOKE_SUMMARY.md`: compact summary.
- `marketcap_tesla_apple.json`: English market-cap query.
- `marketcap_nvidia_microsoft.json`: second market-cap query.
- `zh_tesla_apple_marketcap.json`: Chinese market-cap query.
- `langchain_agents.json`: technical documentation query.
- `qwen_function_calling.json`: Qwen/tool-use documentation query.

The archive copied from the server is:

`remote_files/outputs/agent/web_rag_smoke_demo.tgz`

## What Was Tested

Each smoke test runs:

1. Web search.
2. Curated URL injection for market-cap queries.
3. Page fetch.
4. HTML cleanup.
5. Page text chunking.
6. Lexical chunk retrieval.
7. BGE reranker chunk reranking.
8. Context assembly with citations.

## Observations

The market-cap tests work well:

- Tesla vs Apple injects CompaniesMarketCap pages for both companies.
- NVIDIA vs Microsoft injects CompaniesMarketCap pages for both companies.
- The Chinese Tesla/Apple query also resolves to CompaniesMarketCap pages.

The technical documentation tests run end to end, but source quality is less controlled:

- LangChain query retrieved Chinese tutorial/community pages and LangChain-related pages.
- Qwen query retrieved Qwen/Alibaba-related pages, but not always the most useful function-calling documentation page.

## Next Improvements

- Add source priors for documentation queries:
  - LangChain: prefer `python.langchain.com`, `docs.langchain.com`, `langchain-ai.github.io`.
  - Qwen: prefer `qwen.readthedocs.io`, `qwenlm.github.io`, `github.com/QwenLM`, `help.aliyun.com`.
- Add query rewriting for documentation queries, similar to the market-cap rewrite.
- Add an optional allowlist/denylist by task type.

## Demo Command

On the remote machine:

```bash
cd /root/autodl-tmp/gui-grounding-agent
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
PYTHONPATH=. python scripts/agent/run_web_rag_smokes.py
```
