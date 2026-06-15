#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/gui-grounding-agent
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONPATH=.
export BGE_RERANK_DEVICE="${BGE_RERANK_DEVICE:-cpu}"
export BGE_RERANKER_PATH="${BGE_RERANKER_PATH:-models/BAAI/bge-reranker-base}"

MERGED_MODEL="${MERGED_MODEL:-saves/agent/qwen25-7b-agent-stage7-merged}"
BASE_MODEL="${BASE_MODEL:-models/Qwen2.5-7B-Instruct}"
ADAPTER="${ADAPTER:-saves/agent/qwen25-7b-agent-stage7-finqa-grpo-r32}"

mkdir -p logs/agent

if [ ! -f "$MERGED_MODEL/config.json" ]; then
  python scripts/agent/merge_stage7_agent_lora.py \
    --base_model "$BASE_MODEL" \
    --adapter "$ADAPTER" \
    --output_dir "$MERGED_MODEL" \
    2>&1 | tee logs/agent/merge_stage7_agent_lora.log
fi

pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
pkill -f "scripts.agent.agent_web_app:app" 2>/dev/null || true

nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MERGED_MODEL" \
  --served-model-name qwen-agent \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.82 \
  --max-model-len 4096 \
  > logs/agent/vllm_agent_web.log 2>&1 &

echo "Waiting for vLLM..."
for i in $(seq 1 120); do
  if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    echo "vLLM ready"
    break
  fi
  sleep 3
done

nohup uvicorn scripts.agent.agent_web_app:app \
  --host 0.0.0.0 \
  --port 7860 \
  > logs/agent/agent_web_fastapi.log 2>&1 &

echo "Waiting for FastAPI..."
for i in $(seq 1 40); do
  if curl -fsS http://127.0.0.1:7860/health >/dev/null 2>&1; then
    echo "FastAPI ready"
    break
  fi
  sleep 1
done

echo "Agent web app: http://127.0.0.1:7860"
