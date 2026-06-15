#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/gui-grounding-agent
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONPATH=.
mkdir -p logs/agent outputs/agent/predictions outputs/agent/metrics outputs/finqa/predictions outputs/finqa/metrics data/agent/processed/stage7

echo "[1/6] Prepare balanced FinQA GRPO data"
if [ ! -s outputs/finqa/predictions/stage6_agent_finqa_train_for_stage7_dpo.jsonl ]; then
  echo "Missing Stage6 candidates. Run Stage7 DPO candidate generation first." >&2
  exit 1
fi
python scripts/agent/prepare_agent_stage7_grpo.py \
  --stage6_train data/agent/processed/stage6/agent_stage6_finqa_train.jsonl \
  --stage6_candidates outputs/finqa/predictions/stage6_agent_finqa_train_for_stage7_dpo.jsonl \
  --out_jsonl data/agent/processed/stage7/agent_stage7_finqa_grpo_balanced.jsonl \
  2>&1 | tee logs/agent/stage7_prepare_grpo_data.log

echo "[2/6] Train Stage7 FinQA GRPO from Stage6 adapter"
rm -rf saves/agent/qwen25-7b-agent-stage7-finqa-grpo-r32
python scripts/agent/train_stage7_grpo_finqa_agent.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage6-finqa-sft-r32 \
  --train_jsonl data/agent/processed/stage7/agent_stage7_finqa_grpo_balanced.jsonl \
  --output_dir saves/agent/qwen25-7b-agent-stage7-finqa-grpo-r32 \
  --max_steps 400 \
  --save_steps 200 \
  --num_generations 8 \
  --max_prompt_length 2048 \
  --max_completion_length 128 \
  --learning_rate 2e-7 \
  --temperature 0.9 \
  --top_p 0.95 \
  2>&1 | tee logs/agent/train_stage7_finqa_grpo_r32.log

echo "[3/6] Evaluate Stage7 GRPO FinQA dev"
python scripts/agent/run_finqa_agent_controller.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage7-finqa-grpo-r32 \
  --input_jsonl data/agent/processed/stage6/agent_stage6_finqa_dev.jsonl \
  --output_jsonl outputs/finqa/predictions/stage7_grpo_agent_finqa_r32_finqa_agent_dev.jsonl \
  --out_metrics outputs/finqa/metrics/stage7_grpo_agent_finqa_r32_finqa_agent_dev_metrics.json \
  --max_new_tokens_tool 160 \
  --max_new_tokens_final 96 \
  2>&1 | tee logs/agent/eval_stage7_grpo_finqa_agent_dev.log

echo "[4/6] Evaluate forgetting"
python scripts/agent/infer_agent_action.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage7-finqa-grpo-r32 \
  --input_jsonl data/agent/processed/stage3/agent_stage3_bfcl_route_dev.jsonl \
  --output_jsonl outputs/agent/predictions/stage7_grpo_finqa_r32_bfcl_route_dev.jsonl \
  --max_new_tokens 192 \
  2>&1 | tee logs/agent/infer_stage7_grpo_bfcl_route_dev.log
python scripts/agent/eval_agent_first_action.py \
  --pred_jsonl outputs/agent/predictions/stage7_grpo_finqa_r32_bfcl_route_dev.jsonl \
  --out_metrics outputs/agent/metrics/stage7_grpo_finqa_r32_bfcl_route_dev_metrics.json

python scripts/agent/infer_agent_action.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage7-finqa-grpo-r32 \
  --input_jsonl data/agent/processed/stage2/agent_stage2_dev.jsonl \
  --output_jsonl outputs/agent/predictions/stage7_grpo_finqa_r32_stage2_dev.jsonl \
  --max_new_tokens 192 \
  2>&1 | tee logs/agent/infer_stage7_grpo_stage2_dev.log
python scripts/agent/eval_agent_first_action.py \
  --pred_jsonl outputs/agent/predictions/stage7_grpo_finqa_r32_stage2_dev.jsonl \
  --out_metrics outputs/agent/metrics/stage7_grpo_finqa_r32_stage2_dev_metrics.json

python scripts/agent/infer_agent_action.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage7-finqa-grpo-r32 \
  --input_jsonl data/agent/processed/stage1/gorilla_hf_eval_agent_sft.sample1000.jsonl \
  --output_jsonl outputs/agent/predictions/stage7_grpo_finqa_r32_gorilla_hf_eval1000.jsonl \
  --max_new_tokens 192 \
  2>&1 | tee logs/agent/infer_stage7_grpo_gorilla_hf_eval1000.log
python scripts/agent/eval_agent_first_action.py \
  --pred_jsonl outputs/agent/predictions/stage7_grpo_finqa_r32_gorilla_hf_eval1000.jsonl \
  --out_metrics outputs/agent/metrics/stage7_grpo_finqa_r32_gorilla_hf_eval1000_metrics.json

python scripts/agent/run_finance_agent_controller.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage7-finqa-grpo-r32 \
  --input_jsonl external/LLaMA-Factory/data/agent_stage1_dev.jsonl \
  --output_jsonl outputs/agent/predictions/stage7_grpo_finqa_r32_finance_closed_loop_dev.jsonl \
  --out_metrics outputs/agent/metrics/stage7_grpo_finqa_r32_finance_closed_loop_dev_metrics.json \
  --max_turns 6 \
  --max_new_tokens 192 \
  2>&1 | tee logs/agent/run_stage7_grpo_finance_closed_loop_dev.log

echo "[5/6] Write summary"
python - <<'PY'
import json
from pathlib import Path

items = [
    ("Stage6 FinQA agent dev", "outputs/finqa/metrics/stage6_agent_finqa_sft_r32_finqa_agent_dev_metrics.json"),
    ("Stage7-r2 DPO FinQA dev", "outputs/finqa/metrics/stage7_r2_agent_finqa_dpo_r32_finqa_agent_dev_metrics.json"),
    ("Stage7 GRPO data", "data/agent/processed/stage7/stage7_grpo_data_summary.json"),
    ("Stage7 GRPO FinQA agent dev", "outputs/finqa/metrics/stage7_grpo_agent_finqa_r32_finqa_agent_dev_metrics.json"),
    ("Stage6 BFCL route", "outputs/agent/metrics/stage6_finqa_sft_r32_bfcl_route_dev_metrics.json"),
    ("Stage7 GRPO BFCL route", "outputs/agent/metrics/stage7_grpo_finqa_r32_bfcl_route_dev_metrics.json"),
    ("Stage6 Stage2 replay", "outputs/agent/metrics/stage6_finqa_sft_r32_stage2_dev_metrics.json"),
    ("Stage7 GRPO Stage2 replay", "outputs/agent/metrics/stage7_grpo_finqa_r32_stage2_dev_metrics.json"),
    ("Stage6 Gorilla", "outputs/agent/metrics/stage6_finqa_sft_r32_gorilla_hf_eval1000_metrics.json"),
    ("Stage7 GRPO Gorilla", "outputs/agent/metrics/stage7_grpo_finqa_r32_gorilla_hf_eval1000_metrics.json"),
    ("Stage6 finance closed-loop", "outputs/agent/metrics/stage6_finqa_sft_r32_finance_closed_loop_dev_metrics.json"),
    ("Stage7 GRPO finance closed-loop", "outputs/agent/metrics/stage7_grpo_finqa_r32_finance_closed_loop_dev_metrics.json"),
]
keys = [
    "num_samples", "actual_counts", "tool_json_valid_rate", "calculator_call_rate",
    "program_executable_rate", "scaled_execution_acc_at_1pct", "scaled_execution_acc_at_5pct",
    "final_action_valid_rate", "scaled_final_acc_at_1pct", "scaled_final_acc_at_5pct",
    "json_valid_rate", "action_accuracy", "tool_name_accuracy", "tool_name_accuracy_on_gold_tool_turns",
    "completion_rate", "tool_success_rate", "final_numeric_acc_1pct", "final_numeric_acc_5pct",
]
lines = ["# Agent Stage7 FinQA GRPO Results", ""]
for title, path in items:
    lines.append(f"## {title}")
    p = Path(path)
    if not p.exists():
        lines += ["- missing", ""]
        continue
    data = json.loads(p.read_text(encoding="utf-8"))
    for k in keys:
        if k in data:
            lines.append(f"- {k}: {data[k]}")
    lines.append("")
Path("outputs/agent/metrics/stage7_finqa_grpo_summary.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
PY

echo "[6/6] Done"
