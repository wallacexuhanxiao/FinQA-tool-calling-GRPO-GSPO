#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/gui-grounding-agent
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

MODEL_DIR=${1:-saves/finqa/qwen25-7b-finqa-toolcall-grpo-r32-scaled-full-merged}
SERVED_NAME=${2:-finqa-toolcall-qwen25-7b}
PORT=${3:-8000}

vllm serve "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name "$SERVED_NAME" \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code
