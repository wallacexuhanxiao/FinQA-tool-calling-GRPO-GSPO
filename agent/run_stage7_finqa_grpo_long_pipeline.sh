#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/gui-grounding-agent
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONPATH=.
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

mkdir -p logs/agent outputs/agent/metrics outputs/finqa/predictions outputs/finqa/metrics data/agent/processed/stage7 saves/agent

RUN_NAME="stage7_grpo_long2400_r32"
OUT_DIR="saves/agent/qwen25-7b-agent-stage7-finqa-grpo-r32-long2400"
TRAIN_JSONL="data/agent/processed/stage7/agent_stage7_finqa_grpo_balanced.jsonl"

echo "[1/4] Prepare balanced FinQA GRPO data"
python scripts/agent/prepare_agent_stage7_grpo.py \
  --stage6_train data/agent/processed/stage6/agent_stage6_finqa_train.jsonl \
  --stage6_candidates outputs/finqa/predictions/stage6_agent_finqa_train_for_stage7_dpo.jsonl \
  --out_jsonl "${TRAIN_JSONL}" \
  2>&1 | tee logs/agent/${RUN_NAME}_prepare_data.log

echo "[2/4] Train long Stage7 FinQA GRPO from Stage6 adapter"
rm -rf "${OUT_DIR}"
python scripts/agent/train_stage7_grpo_finqa_agent.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage6-finqa-sft-r32 \
  --train_jsonl "${TRAIN_JSONL}" \
  --output_dir "${OUT_DIR}" \
  --max_steps 2400 \
  --save_steps 400 \
  --save_total_limit 1 \
  --num_generations 8 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_prompt_length 2048 \
  --max_completion_length 128 \
  --learning_rate 2e-7 \
  --temperature 0.9 \
  --top_p 0.95 \
  2>&1 | tee logs/agent/train_${RUN_NAME}.log

echo "[3/4] Evaluate long GRPO on FinQA dev"
python scripts/agent/run_finqa_agent_controller.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path "${OUT_DIR}" \
  --input_jsonl data/agent/processed/stage6/agent_stage6_finqa_dev.jsonl \
  --output_jsonl outputs/finqa/predictions/${RUN_NAME}_finqa_agent_dev.jsonl \
  --out_metrics outputs/finqa/metrics/${RUN_NAME}_finqa_agent_dev_metrics.json \
  --max_new_tokens_tool 160 \
  --max_new_tokens_final 96 \
  2>&1 | tee logs/agent/eval_${RUN_NAME}_finqa_agent_dev.log

echo "[4/4] Write summary"
python - <<'PY'
import json
from pathlib import Path

metrics_path = Path("outputs/finqa/metrics/stage7_grpo_long2400_r32_finqa_agent_dev_metrics.json")
train_log = Path("logs/agent/train_stage7_grpo_long2400_r32.log")
summary_path = Path("outputs/agent/metrics/stage7_grpo_long2400_r32_summary.md")

lines = ["# Stage7 FinQA GRPO Long2400", ""]
lines.append("- base_adapter: saves/agent/qwen25-7b-agent-stage6-finqa-sft-r32")
lines.append("- output_adapter: saves/agent/qwen25-7b-agent-stage7-finqa-grpo-r32-long2400")
lines.append("- max_steps: 2400")
lines.append("- save_steps: 400")
lines.append("- save_total_limit: 1")
lines.append("- num_generations: 8")
lines.append("- max_prompt_length: 2048")
lines.append("- max_completion_length: 128")
lines.append("- learning_rate: 2e-7")
lines.append("")

if metrics_path.exists():
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    for key in [
        "num_samples",
        "tool_json_valid_rate",
        "calculator_call_rate",
        "program_executable_rate",
        "scaled_execution_acc_at_1pct",
        "scaled_execution_acc_at_5pct",
        "final_action_valid_rate",
        "scaled_final_acc_at_1pct",
        "scaled_final_acc_at_5pct",
    ]:
        if key in m:
            lines.append(f"- {key}: {m[key]}")
else:
    lines.append("- metrics: missing")

summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY

echo "Done: ${RUN_NAME}"
