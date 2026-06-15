#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/gui-grounding-agent
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
while true; do
  if ! tmux has-session -t agent_stage1_sft 2>/dev/null; then
    break
  fi
  sleep 60
done
if [ -f saves/agent/qwen25-7b-agent-stage1-sft-r32/adapter_model.safetensors ] || [ -f saves/agent/qwen25-7b-agent-stage1-sft-r32/adapter_model.bin ]; then
  PYTHONPATH=. python scripts/agent/infer_agent_first_action.py \
    --model_path models/Qwen2.5-7B-Instruct \
    --adapter_path saves/agent/qwen25-7b-agent-stage1-sft-r32 \
    --input_jsonl data/agent/processed/stage1/agent_stage1_dev.jsonl \
    --output_jsonl outputs/agent/predictions/stage1_sft_r32_first_action_dev.jsonl \
    --max_new_tokens 160 \
    2>&1 | tee logs/agent/infer_stage1_sft_r32_first_action_dev.log
  python scripts/agent/eval_agent_first_action.py \
    --pred_jsonl outputs/agent/predictions/stage1_sft_r32_first_action_dev.jsonl \
    --out_metrics outputs/agent/metrics/stage1_sft_r32_first_action_dev_metrics.json
else
  echo "adapter not found; training probably failed" | tee logs/agent/stage1_sft_watch_failed.log
fi
