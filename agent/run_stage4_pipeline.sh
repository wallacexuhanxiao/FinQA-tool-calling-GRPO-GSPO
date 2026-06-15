#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/gui-grounding-agent
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

mkdir -p logs/agent outputs/agent/predictions outputs/agent/metrics data/agent/processed/stage4

python scripts/agent/prepare_agent_stage4_data.py 2>&1 | tee logs/agent/prepare_stage4_data.log

llamafactory-cli train configs/agent/agent_stage4_sft_r32.yaml 2>&1 | tee logs/agent/train_stage4_sft_r32.log

python scripts/agent/build_agent_turn_eval.py \
  --input_jsonl data/agent/processed/stage4/agent_stage4_bfcl_multiturn_dev.jsonl \
  --output_jsonl data/agent/processed/stage4/agent_stage4_bfcl_multiturn_dev_turns.jsonl

python scripts/agent/infer_agent_action.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage4-sft-r32 \
  --input_jsonl data/agent/processed/stage4/agent_stage4_bfcl_multiturn_dev_turns.jsonl \
  --output_jsonl outputs/agent/predictions/stage4_sft_r32_bfcl_multiturn_dev_turns.jsonl \
  --max_new_tokens 192 \
  2>&1 | tee logs/agent/infer_stage4_bfcl_multiturn_dev_turns.log

python scripts/agent/eval_agent_first_action.py \
  --pred_jsonl outputs/agent/predictions/stage4_sft_r32_bfcl_multiturn_dev_turns.jsonl \
  --out_metrics outputs/agent/metrics/stage4_sft_r32_bfcl_multiturn_dev_turns_metrics.json

python scripts/agent/build_agent_turn_eval.py \
  --input_jsonl data/agent/processed/stage4/agent_stage4_finance_multiturn_dev.jsonl \
  --output_jsonl data/agent/processed/stage4/agent_stage4_finance_multiturn_dev_turns.jsonl

python scripts/agent/infer_agent_action.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage4-sft-r32 \
  --input_jsonl data/agent/processed/stage4/agent_stage4_finance_multiturn_dev_turns.jsonl \
  --output_jsonl outputs/agent/predictions/stage4_sft_r32_finance_multiturn_dev_turns.jsonl \
  --max_new_tokens 192 \
  2>&1 | tee logs/agent/infer_stage4_finance_multiturn_dev_turns.log

python scripts/agent/eval_agent_first_action.py \
  --pred_jsonl outputs/agent/predictions/stage4_sft_r32_finance_multiturn_dev_turns.jsonl \
  --out_metrics outputs/agent/metrics/stage4_sft_r32_finance_multiturn_dev_turns_metrics.json

python scripts/agent/infer_agent_action.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage4-sft-r32 \
  --input_jsonl data/agent/processed/stage3/agent_stage3_bfcl_route_dev.jsonl \
  --output_jsonl outputs/agent/predictions/stage4_sft_r32_bfcl_route_dev.jsonl \
  --max_new_tokens 192 \
  2>&1 | tee logs/agent/infer_stage4_bfcl_route_dev.log

python scripts/agent/eval_agent_first_action.py \
  --pred_jsonl outputs/agent/predictions/stage4_sft_r32_bfcl_route_dev.jsonl \
  --out_metrics outputs/agent/metrics/stage4_sft_r32_bfcl_route_dev_metrics.json

python scripts/agent/infer_agent_action.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage4-sft-r32 \
  --input_jsonl data/agent/processed/stage2/agent_stage2_dev.jsonl \
  --output_jsonl outputs/agent/predictions/stage4_sft_r32_stage2_dev.jsonl \
  --max_new_tokens 192 \
  2>&1 | tee logs/agent/infer_stage4_stage2_dev.log

python scripts/agent/eval_agent_first_action.py \
  --pred_jsonl outputs/agent/predictions/stage4_sft_r32_stage2_dev.jsonl \
  --out_metrics outputs/agent/metrics/stage4_sft_r32_stage2_dev_metrics.json

python scripts/agent/infer_agent_action.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage4-sft-r32 \
  --input_jsonl data/agent/processed/stage1/gorilla_hf_eval_agent_sft.sample1000.jsonl \
  --output_jsonl outputs/agent/predictions/stage4_sft_r32_gorilla_hf_eval1000.jsonl \
  --max_new_tokens 192 \
  2>&1 | tee logs/agent/infer_stage4_gorilla_hf_eval1000.log

python scripts/agent/eval_agent_first_action.py \
  --pred_jsonl outputs/agent/predictions/stage4_sft_r32_gorilla_hf_eval1000.jsonl \
  --out_metrics outputs/agent/metrics/stage4_sft_r32_gorilla_hf_eval1000_metrics.json

python scripts/agent/run_finance_agent_controller.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage4-sft-r32 \
  --input_jsonl external/LLaMA-Factory/data/agent_stage1_dev.jsonl \
  --output_jsonl outputs/agent/predictions/stage4_sft_r32_finance_closed_loop_dev.jsonl \
  --out_metrics outputs/agent/metrics/stage4_sft_r32_finance_closed_loop_dev_metrics.json \
  --max_turns 6 \
  --max_new_tokens 192 \
  2>&1 | tee logs/agent/run_stage4_finance_closed_loop_dev.log

python - <<'PY'
import json
from pathlib import Path

items = [
    ("Stage3 BFCL route", "outputs/agent/metrics/stage3_sft_r32_bfcl_route_dev_metrics.json"),
    ("Stage3 finance", "outputs/agent/metrics/stage3_sft_r32_finance_closed_loop_dev_metrics.json"),
    ("Stage4 BFCL multiturn turns", "outputs/agent/metrics/stage4_sft_r32_bfcl_multiturn_dev_turns_metrics.json"),
    ("Stage4 finance multiturn turns", "outputs/agent/metrics/stage4_sft_r32_finance_multiturn_dev_turns_metrics.json"),
    ("Stage4 BFCL route replay", "outputs/agent/metrics/stage4_sft_r32_bfcl_route_dev_metrics.json"),
    ("Stage4 Stage2 replay", "outputs/agent/metrics/stage4_sft_r32_stage2_dev_metrics.json"),
    ("Stage4 Gorilla", "outputs/agent/metrics/stage4_sft_r32_gorilla_hf_eval1000_metrics.json"),
    ("Stage4 finance closed-loop", "outputs/agent/metrics/stage4_sft_r32_finance_closed_loop_dev_metrics.json"),
]
keys = ["num_samples", "json_valid_rate", "action_accuracy", "tool_name_accuracy", "tool_name_accuracy_on_gold_tool_turns", "final_exact_rate", "argument_schema_valid_rate", "completion_rate", "tool_success_rate", "final_numeric_acc_1pct", "final_numeric_acc_5pct"]
lines = ["# Agent Stage4 Results", ""]
for title, path in items:
    p = Path(path)
    lines += [f"## {title}"]
    if not p.exists():
        lines += ["- missing", ""]
        continue
    data = json.loads(p.read_text(encoding="utf-8"))
    for k in keys:
        if k in data and data[k] is not None:
            lines.append(f"- {k}: {data[k]}")
    lines.append("")
Path("outputs/agent/metrics/stage4_summary.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
PY
