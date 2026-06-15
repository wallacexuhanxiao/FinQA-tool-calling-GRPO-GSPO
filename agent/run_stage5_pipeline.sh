#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/gui-grounding-agent
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

mkdir -p logs/agent outputs/agent/predictions outputs/agent/metrics data/agent/processed/stage5

python scripts/agent/prepare_agent_stage5_grpo.py 2>&1 | tee logs/agent/prepare_stage5_grpo.log

python scripts/agent/train_stage5_grpo_toolroute.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage4-sft-r32 \
  --train_jsonl data/agent/processed/stage5/agent_stage5_grpo_toolroute.jsonl \
  --output_dir saves/agent/qwen25-7b-agent-stage5-grpo-toolroute-r32 \
  --max_steps 600 \
  --save_steps 300 \
  --num_generations 8 \
  --max_prompt_length 2048 \
  --max_completion_length 96 \
  --learning_rate 3e-7 \
  --temperature 0.9 \
  --top_p 0.95 \
  2>&1 | tee logs/agent/train_stage5_grpo_toolroute.log

python scripts/agent/infer_agent_action.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage5-grpo-toolroute-r32 \
  --input_jsonl data/agent/processed/stage3/agent_stage3_bfcl_route_dev.jsonl \
  --output_jsonl outputs/agent/predictions/stage5_grpo_toolroute_bfcl_route_dev.jsonl \
  --max_new_tokens 192 \
  2>&1 | tee logs/agent/infer_stage5_bfcl_route_dev.log

python scripts/agent/eval_agent_first_action.py \
  --pred_jsonl outputs/agent/predictions/stage5_grpo_toolroute_bfcl_route_dev.jsonl \
  --out_metrics outputs/agent/metrics/stage5_grpo_toolroute_bfcl_route_dev_metrics.json

python scripts/agent/infer_agent_action.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage5-grpo-toolroute-r32 \
  --input_jsonl data/agent/processed/stage2/agent_stage2_dev.jsonl \
  --output_jsonl outputs/agent/predictions/stage5_grpo_toolroute_stage2_dev.jsonl \
  --max_new_tokens 192 \
  2>&1 | tee logs/agent/infer_stage5_stage2_dev.log

python scripts/agent/eval_agent_first_action.py \
  --pred_jsonl outputs/agent/predictions/stage5_grpo_toolroute_stage2_dev.jsonl \
  --out_metrics outputs/agent/metrics/stage5_grpo_toolroute_stage2_dev_metrics.json

python scripts/agent/run_finance_agent_controller.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage5-grpo-toolroute-r32 \
  --input_jsonl external/LLaMA-Factory/data/agent_stage1_dev.jsonl \
  --output_jsonl outputs/agent/predictions/stage5_grpo_toolroute_finance_closed_loop_dev.jsonl \
  --out_metrics outputs/agent/metrics/stage5_grpo_toolroute_finance_closed_loop_dev_metrics.json \
  --max_turns 6 \
  --max_new_tokens 192 \
  2>&1 | tee logs/agent/run_stage5_finance_closed_loop_dev.log

python eval/infer_finqa_qwen.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage5-grpo-toolroute-r32 \
  --input_jsonl external/LLaMA-Factory/data/finqa_dev.jsonl \
  --output_jsonl outputs/finqa/predictions/stage5_agent_grpo_toolroute_finqa_dev.jsonl \
  --max_new_tokens 256 \
  2>&1 | tee logs/agent/infer_stage5_finqa_dev.log

python eval/eval_finqa_answer_exec.py \
  --pred_jsonl outputs/finqa/predictions/stage5_agent_grpo_toolroute_finqa_dev.jsonl \
  --out_metrics outputs/finqa/metrics/stage5_agent_grpo_toolroute_finqa_dev_answer_exec_scaled_metrics.json \
  2>&1 | tee logs/agent/eval_stage5_finqa_dev.log

python - <<'PY'
import json
from pathlib import Path
items = [
    ("Stage4 BFCL route", "outputs/agent/metrics/stage4_sft_r32_bfcl_route_dev_metrics.json"),
    ("Stage5 BFCL route", "outputs/agent/metrics/stage5_grpo_toolroute_bfcl_route_dev_metrics.json"),
    ("Stage5 Stage2 replay", "outputs/agent/metrics/stage5_grpo_toolroute_stage2_dev_metrics.json"),
    ("Stage5 finance closed-loop", "outputs/agent/metrics/stage5_grpo_toolroute_finance_closed_loop_dev_metrics.json"),
    ("Stage5 FinQA dev", "outputs/finqa/metrics/stage5_agent_grpo_toolroute_finqa_dev_answer_exec_scaled_metrics.json"),
]
keys = ["num_samples", "json_valid_rate", "action_accuracy", "tool_name_accuracy", "tool_name_accuracy_on_gold_tool_turns", "completion_rate", "tool_success_rate", "final_numeric_acc_1pct", "final_numeric_acc_5pct", "scaled_numeric_accuracy_at_1pct", "scaled_numeric_accuracy_at_5pct", "numeric_accuracy_at_1pct", "numeric_accuracy_at_5pct"]
lines = ["# Agent Stage5 Results", ""]
for title, path in items:
    p = Path(path)
    lines.append(f"## {title}")
    if not p.exists():
        lines += ["- missing", ""]
        continue
    data = json.loads(p.read_text(encoding="utf-8"))
    for k in keys:
        if k in data and data[k] is not None:
            lines.append(f"- {k}: {data[k]}")
    lines.append("")
Path("outputs/agent/metrics/stage5_summary.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
PY
