#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/gui-grounding-agent
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONPATH=.
mkdir -p logs/agent outputs/agent/predictions outputs/agent/metrics outputs/finqa/predictions outputs/finqa/metrics

echo "[R2 1/5] Build compact balanced DPO data from existing candidates"
python scripts/agent/build_agent_stage7_dpo_r2_compact.py \
  --finqa_train_jsonl data/agent/processed/stage7/agent_stage7_finqa_train_only.jsonl \
  --pred_jsonl outputs/finqa/predictions/stage6_agent_finqa_train_for_stage7_dpo.jsonl \
  --replay_jsonl data/agent/processed/stage6/agent_stage6_finqa_train.jsonl \
  --out_json data/agent/processed/stage7/agent_stage7_finqa_dpo_r2_compact.json \
  --out_external external/LLaMA-Factory/data/agent_stage7_finqa_dpo_r2_compact.json \
  2>&1 | tee logs/agent/stage7_build_dpo_r2_compact.log

echo "[R2 2/5] Register compact DPO dataset"
python - <<'PY'
import json
from pathlib import Path
p = Path("external/LLaMA-Factory/data/dataset_info.json")
info = json.loads(p.read_text(encoding="utf-8"))
info["agent_stage7_finqa_dpo_r2_compact"] = {
    "file_name": "agent_stage7_finqa_dpo_r2_compact.json",
    "ranking": True,
    "formatting": "sharegpt",
    "columns": {"messages": "conversations", "chosen": "chosen", "rejected": "rejected"}
}
p.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(info["agent_stage7_finqa_dpo_r2_compact"], ensure_ascii=False, indent=2))
PY

echo "[R2 3/5] Train compact low-LR DPO from Stage6 adapter"
rm -rf saves/agent/qwen25-7b-agent-stage7-finqa-dpo-r32-r2
llamafactory-cli train configs/agent/agent_stage7_finqa_dpo_r32_r2.yaml \
  2>&1 | tee logs/agent/train_stage7_finqa_dpo_r32_r2.log

echo "[R2 4/5] Evaluate FinQA and forgetting"
python scripts/agent/run_finqa_agent_controller.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage7-finqa-dpo-r32-r2 \
  --input_jsonl data/agent/processed/stage6/agent_stage6_finqa_dev.jsonl \
  --output_jsonl outputs/finqa/predictions/stage7_r2_agent_finqa_dpo_r32_finqa_agent_dev.jsonl \
  --out_metrics outputs/finqa/metrics/stage7_r2_agent_finqa_dpo_r32_finqa_agent_dev_metrics.json \
  --max_new_tokens_tool 160 \
  --max_new_tokens_final 96 \
  2>&1 | tee logs/agent/eval_stage7_r2_finqa_agent_dev.log

python scripts/agent/infer_agent_action.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage7-finqa-dpo-r32-r2 \
  --input_jsonl data/agent/processed/stage3/agent_stage3_bfcl_route_dev.jsonl \
  --output_jsonl outputs/agent/predictions/stage7_r2_finqa_dpo_r32_bfcl_route_dev.jsonl \
  --max_new_tokens 192 \
  2>&1 | tee logs/agent/infer_stage7_r2_bfcl_route_dev.log
python scripts/agent/eval_agent_first_action.py \
  --pred_jsonl outputs/agent/predictions/stage7_r2_finqa_dpo_r32_bfcl_route_dev.jsonl \
  --out_metrics outputs/agent/metrics/stage7_r2_finqa_dpo_r32_bfcl_route_dev_metrics.json

python scripts/agent/infer_agent_action.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage7-finqa-dpo-r32-r2 \
  --input_jsonl data/agent/processed/stage2/agent_stage2_dev.jsonl \
  --output_jsonl outputs/agent/predictions/stage7_r2_finqa_dpo_r32_stage2_dev.jsonl \
  --max_new_tokens 192 \
  2>&1 | tee logs/agent/infer_stage7_r2_stage2_dev.log
python scripts/agent/eval_agent_first_action.py \
  --pred_jsonl outputs/agent/predictions/stage7_r2_finqa_dpo_r32_stage2_dev.jsonl \
  --out_metrics outputs/agent/metrics/stage7_r2_finqa_dpo_r32_stage2_dev_metrics.json

python scripts/agent/infer_agent_action.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage7-finqa-dpo-r32-r2 \
  --input_jsonl data/agent/processed/stage1/gorilla_hf_eval_agent_sft.sample1000.jsonl \
  --output_jsonl outputs/agent/predictions/stage7_r2_finqa_dpo_r32_gorilla_hf_eval1000.jsonl \
  --max_new_tokens 192 \
  2>&1 | tee logs/agent/infer_stage7_r2_gorilla_hf_eval1000.log
python scripts/agent/eval_agent_first_action.py \
  --pred_jsonl outputs/agent/predictions/stage7_r2_finqa_dpo_r32_gorilla_hf_eval1000.jsonl \
  --out_metrics outputs/agent/metrics/stage7_r2_finqa_dpo_r32_gorilla_hf_eval1000_metrics.json

python scripts/agent/run_finance_agent_controller.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage7-finqa-dpo-r32-r2 \
  --input_jsonl external/LLaMA-Factory/data/agent_stage1_dev.jsonl \
  --output_jsonl outputs/agent/predictions/stage7_r2_finqa_dpo_r32_finance_closed_loop_dev.jsonl \
  --out_metrics outputs/agent/metrics/stage7_r2_finqa_dpo_r32_finance_closed_loop_dev_metrics.json \
  --max_turns 6 \
  --max_new_tokens 192 \
  2>&1 | tee logs/agent/run_stage7_r2_finance_closed_loop_dev.log

echo "[R2 5/5] Write summary"
python - <<'PY'
import json
from pathlib import Path
items = [
    ("Stage6 FinQA agent dev", "outputs/finqa/metrics/stage6_agent_finqa_sft_r32_finqa_agent_dev_metrics.json"),
    ("Stage7 failed DPO FinQA dev", "outputs/finqa/metrics/stage7_agent_finqa_dpo_r32_finqa_agent_dev_metrics.json"),
    ("Stage7-r2 DPO data", "data/agent/processed/stage7/stage7_dpo_r2_data_summary.json"),
    ("Stage7-r2 FinQA agent dev", "outputs/finqa/metrics/stage7_r2_agent_finqa_dpo_r32_finqa_agent_dev_metrics.json"),
    ("Stage6 BFCL route", "outputs/agent/metrics/stage6_finqa_sft_r32_bfcl_route_dev_metrics.json"),
    ("Stage7-r2 BFCL route", "outputs/agent/metrics/stage7_r2_finqa_dpo_r32_bfcl_route_dev_metrics.json"),
    ("Stage6 Stage2 replay", "outputs/agent/metrics/stage6_finqa_sft_r32_stage2_dev_metrics.json"),
    ("Stage7-r2 Stage2 replay", "outputs/agent/metrics/stage7_r2_finqa_dpo_r32_stage2_dev_metrics.json"),
    ("Stage6 Gorilla", "outputs/agent/metrics/stage6_finqa_sft_r32_gorilla_hf_eval1000_metrics.json"),
    ("Stage7-r2 Gorilla", "outputs/agent/metrics/stage7_r2_finqa_dpo_r32_gorilla_hf_eval1000_metrics.json"),
    ("Stage6 finance closed-loop", "outputs/agent/metrics/stage6_finqa_sft_r32_finance_closed_loop_dev_metrics.json"),
    ("Stage7-r2 finance closed-loop", "outputs/agent/metrics/stage7_r2_finqa_dpo_r32_finance_closed_loop_dev_metrics.json"),
]
keys = [
    "num_pairs", "wrong_pairs", "correct_anchor_pairs", "agent_replay_pairs",
    "num_samples", "tool_json_valid_rate", "calculator_call_rate", "program_executable_rate",
    "scaled_execution_acc_at_1pct", "scaled_execution_acc_at_5pct",
    "final_action_valid_rate", "scaled_final_acc_at_1pct", "scaled_final_acc_at_5pct",
    "json_valid_rate", "action_accuracy", "tool_name_accuracy", "tool_name_accuracy_on_gold_tool_turns",
    "completion_rate", "tool_success_rate", "final_numeric_acc_1pct", "final_numeric_acc_5pct",
]
lines = ["# Agent Stage7-r2 FinQA DPO Repair Results", ""]
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
    if "candidate_status_counts" in data:
        lines.append(f"- candidate_status_counts: {data['candidate_status_counts']}")
    if "prompt_style" in data:
        lines.append(f"- prompt_style: {data['prompt_style']}")
    lines.append("")
Path("outputs/agent/metrics/stage7_r2_finqa_dpo_summary.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
PY

echo "[DONE] Stage7-r2 DPO repair complete"
