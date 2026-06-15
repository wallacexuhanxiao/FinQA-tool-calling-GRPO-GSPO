#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/gui-grounding-agent
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONPATH=.
mkdir -p logs/agent outputs/agent/predictions outputs/agent/metrics outputs/finqa/predictions outputs/finqa/metrics

python scripts/agent/run_finqa_agent_controller.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage6-finqa-sft-r32 \
  --input_jsonl data/agent/processed/stage6/agent_stage6_finqa_dev.jsonl \
  --output_jsonl outputs/finqa/predictions/stage6_agent_finqa_sft_r32_finqa_agent_dev.jsonl \
  --out_metrics outputs/finqa/metrics/stage6_agent_finqa_sft_r32_finqa_agent_dev_metrics.json \
  --max_new_tokens_tool 160 \
  --max_new_tokens_final 96 \
  2>&1 | tee logs/agent/eval_stage6_finqa_agent_dev_retry.log

python scripts/agent/infer_agent_action.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage6-finqa-sft-r32 \
  --input_jsonl data/agent/processed/stage3/agent_stage3_bfcl_route_dev.jsonl \
  --output_jsonl outputs/agent/predictions/stage6_finqa_sft_r32_bfcl_route_dev.jsonl \
  --max_new_tokens 192 \
  2>&1 | tee logs/agent/infer_stage6_bfcl_route_dev.log
python scripts/agent/eval_agent_first_action.py \
  --pred_jsonl outputs/agent/predictions/stage6_finqa_sft_r32_bfcl_route_dev.jsonl \
  --out_metrics outputs/agent/metrics/stage6_finqa_sft_r32_bfcl_route_dev_metrics.json

python scripts/agent/infer_agent_action.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage6-finqa-sft-r32 \
  --input_jsonl data/agent/processed/stage2/agent_stage2_dev.jsonl \
  --output_jsonl outputs/agent/predictions/stage6_finqa_sft_r32_stage2_dev.jsonl \
  --max_new_tokens 192 \
  2>&1 | tee logs/agent/infer_stage6_stage2_dev.log
python scripts/agent/eval_agent_first_action.py \
  --pred_jsonl outputs/agent/predictions/stage6_finqa_sft_r32_stage2_dev.jsonl \
  --out_metrics outputs/agent/metrics/stage6_finqa_sft_r32_stage2_dev_metrics.json

python scripts/agent/infer_agent_action.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage6-finqa-sft-r32 \
  --input_jsonl data/agent/processed/stage1/gorilla_hf_eval_agent_sft.sample1000.jsonl \
  --output_jsonl outputs/agent/predictions/stage6_finqa_sft_r32_gorilla_hf_eval1000.jsonl \
  --max_new_tokens 192 \
  2>&1 | tee logs/agent/infer_stage6_gorilla_hf_eval1000.log
python scripts/agent/eval_agent_first_action.py \
  --pred_jsonl outputs/agent/predictions/stage6_finqa_sft_r32_gorilla_hf_eval1000.jsonl \
  --out_metrics outputs/agent/metrics/stage6_finqa_sft_r32_gorilla_hf_eval1000_metrics.json

python scripts/agent/run_finance_agent_controller.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage6-finqa-sft-r32 \
  --input_jsonl external/LLaMA-Factory/data/agent_stage1_dev.jsonl \
  --output_jsonl outputs/agent/predictions/stage6_finqa_sft_r32_finance_closed_loop_dev.jsonl \
  --out_metrics outputs/agent/metrics/stage6_finqa_sft_r32_finance_closed_loop_dev_metrics.json \
  --max_turns 6 \
  --max_new_tokens 192 \
  2>&1 | tee logs/agent/run_stage6_finance_closed_loop_dev.log

python - <<'PY'
import json
from pathlib import Path
items = [
    ("Stage5 BFCL route", "outputs/agent/metrics/stage5_grpo_toolroute_bfcl_route_dev_metrics.json"),
    ("Stage6 FinQA agent dev", "outputs/finqa/metrics/stage6_agent_finqa_sft_r32_finqa_agent_dev_metrics.json"),
    ("Stage6 BFCL route", "outputs/agent/metrics/stage6_finqa_sft_r32_bfcl_route_dev_metrics.json"),
    ("Stage6 Stage2 replay", "outputs/agent/metrics/stage6_finqa_sft_r32_stage2_dev_metrics.json"),
    ("Stage6 Gorilla", "outputs/agent/metrics/stage6_finqa_sft_r32_gorilla_hf_eval1000_metrics.json"),
    ("Stage6 finance closed-loop", "outputs/agent/metrics/stage6_finqa_sft_r32_finance_closed_loop_dev_metrics.json"),
]
keys = ["num_samples", "tool_json_valid_rate", "calculator_call_rate", "program_executable_rate", "scaled_execution_acc_at_1pct", "scaled_execution_acc_at_5pct", "final_action_valid_rate", "scaled_final_acc_at_1pct", "scaled_final_acc_at_5pct", "json_valid_rate", "action_accuracy", "tool_name_accuracy", "tool_name_accuracy_on_gold_tool_turns", "completion_rate", "tool_success_rate", "final_numeric_acc_1pct", "final_numeric_acc_5pct"]
lines = ["# Agent Stage6 FinQA Results", ""]
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
Path("outputs/agent/metrics/stage6_finqa_summary.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
PY
