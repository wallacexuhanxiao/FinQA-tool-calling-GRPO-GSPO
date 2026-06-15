# FinQA Tool-Calling GRPO / GSPO Agent

This repository contains a FinQA financial numerical reasoning project built around Qwen2.5-7B-Instruct, tool-calling post-training, execution-based rewards, web chunked RAG, and agent-style tool orchestration.

The core idea is to move financial QA from direct answer generation to an executable tool-calling workflow:

```text
question / financial context
  -> model emits structured JSON tool call
  -> Python controller parses the call
  -> calculator / market_data / web_search tools execute
  -> controller returns observation
  -> model emits final answer
```

## Highlights

- Built SFT, DPO, GRPO, and GSPO post-training pipelines on FinQA.
- Converted direct `{program, answer}` outputs into structured calculator API calls.
- Implemented a safe FinQA calculator executor without Python `eval()`.
- Designed GRPO rewards around JSON validity, tool routing, program executability, and scaled execution accuracy.
- Added web-search chunked RAG: search pages, fetch HTML, split into chunks, retrieve, rerank, and return cited context.
- Added LangChain-style agent tools for `calculator`, `market_data`, and `web_search`.
- Included vLLM benchmark scripts for merged LoRA deployment and throughput testing.

## Repository Layout

```text
configs/finqa/                 LLaMA-Factory SFT/DPO/tool-call configs
scripts/finqa/                 FinQA data prep, SFT/DPO/GRPO, RAG, vLLM benchmarks
scripts/agent/                 Agent training, controllers, web RAG, LangChain/FastAPI demos
eval/                          FinQA answer, execution, retrieval, and tool-call evaluators
results/finqa_metrics/         Key FinQA evaluation metrics
results/agent_metrics/         Agent and forgetting-eval metrics
results/web_rag_smoke/         Web chunked RAG smoke-test outputs
logs/agent/                    Selected training/evaluation logs
docs/                          Detailed project notes and experiment writeups
```

## Core Results

FinQA dev set, 883 samples:

| Model / Method | Program Exec | Scaled Exec@1 | Scaled Exec@5 | Scaled Final@1 | Scaled Final@5 |
|---|---:|---:|---:|---:|---:|
| Stage6 Tool-call SFT-r32 | 0.9694 | 0.6104 | 0.6920 | 0.6195 | 0.7022 |
| Stage7 GRPO-r32 400 steps | 0.9819 | 0.6217 | 0.7022 | 0.6308 | 0.7123 |
| Stage7 GSPO-r32 400 steps | 0.9807 | 0.6206 | 0.7010 | 0.6285 | 0.7101 |
| Stage7 GRPO-r32 2400 steps | **0.9921** | **0.6478** | **0.7271** | **0.6546** | **0.7339** |

The longer GRPO run improves the Stage6 tool-calling SFT baseline by about 3.17 points on `scaled_final@5`.

## Web Chunked RAG Smoke Tests

The web RAG pipeline does not simply pass search snippets to the model. It performs:

```text
query -> web search -> candidate URL injection -> page fetch -> HTML cleanup
      -> text chunking -> chunk retrieval -> BGE reranking -> cited context
```

Smoke examples are in [`results/web_rag_smoke`](results/web_rag_smoke):

- Tesla vs Apple market cap
- NVIDIA vs Microsoft market cap
- Chinese Tesla/Apple market-cap query
- LangChain agent documentation query
- Qwen function-calling query

Market-cap queries inject CompaniesMarketCap URLs before reranking, so the RAG context contains high-value financial evidence rather than generic homepages.

## Key Commands

### Prepare FinQA SFT Data

```bash
python scripts/finqa/prepare_finqa_sft.py
python scripts/finqa/register_llamafactory_finqa.py
```

### Train Tool-Calling SFT

```bash
llamafactory-cli train configs/finqa/qwen25_7b_finqa_toolcall_full_sft_r32.yaml
```

### Train Stage7 FinQA GRPO Long Run

```bash
bash scripts/agent/run_stage7_finqa_grpo_long_pipeline.sh
```

Important settings:

```text
max_steps=2400
num_generations=8
max_prompt_length=2048
max_completion_length=128
learning_rate=2e-7
save_total_limit=1
```

### Run Web Chunked RAG Smokes

```bash
PYTHONPATH=. python scripts/agent/run_web_rag_smokes.py
```

### Run Web Agent Demo

```bash
bash scripts/agent/run_agent_web_stack.sh
```

## Tool-Calling Format

The model is trained to emit fixed JSON, for example:

```json
{
  "action": "tool_call",
  "tool_call": {
    "name": "calculator",
    "arguments": {
      "program": "subtract(1200, 1000), divide(#0, 1000), multiply(#1, const_100)"
    }
  }
}
```

The Python controller parses this JSON, executes the whitelisted calculator program, returns an observation, and lets the model produce a final answer.

## Reward Design

The GRPO reward combines:

- JSON validity
- correct `tool_call` action
- correct calculator route
- non-empty program
- program executability
- scaled execution match at 5% and 1% tolerance
- penalties for invalid JSON, wrong tool, empty program, non-executable program, sign flip, or overly long output

Scaled matching is used because financial answers may appear as equivalent forms such as `0.25`, `25%`, or unit-scaled values.

## Notes

- This repository intentionally excludes base model weights, LoRA adapter weights, full datasets, and generated prediction files.
- Metrics and representative logs are included for reproducibility and project review.
- Some scripts assume the original AutoDL layout: `/root/autodl-tmp/gui-grounding-agent`.
- Set `PYTHONPATH=.` when running scripts from the repository root.

## Documentation

See the `docs/` directory for detailed writeups:

- [`AGENT_STAGE1_5_GUIDE.md`](docs/AGENT_STAGE1_5_GUIDE.md)
- [`STAGE6_7_FINQA_CONTROLLER_AND_CALCULATOR.md`](docs/STAGE6_7_FINQA_CONTROLLER_AND_CALCULATOR.md)
- [`GRPO_LONG2400_RUN.md`](docs/GRPO_LONG2400_RUN.md)
- [`CHUNKED_RAG_UPGRADE.md`](docs/CHUNKED_RAG_UPGRADE.md)
- [`WEB_CHUNKED_RAG.md`](docs/WEB_CHUNKED_RAG.md)
- [`WEB_RAG_SMOKE_DEMO.md`](docs/WEB_RAG_SMOKE_DEMO.md)
