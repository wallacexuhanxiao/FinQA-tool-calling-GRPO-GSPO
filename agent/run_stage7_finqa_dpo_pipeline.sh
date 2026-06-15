#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/gui-grounding-agent
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONPATH=.

mkdir -p logs/agent data/agent/processed/stage7 outputs/agent/predictions outputs/agent/metrics outputs/finqa/predictions outputs/finqa/metrics

echo "[1/8] Filter FinQA-only Stage6 train rows"
python scripts/agent/filter_stage7_finqa_only.py \
  --input_jsonl data/agent/processed/stage6/agent_stage6_finqa_train.jsonl \
  --output_jsonl data/agent/processed/stage7/agent_stage7_finqa_train_only.jsonl \
  2>&1 | tee logs/agent/stage7_filter_finqa_only.log

echo "[2/8] Generate Stage6 candidates on FinQA train for DPO rejected responses"
python scripts/agent/run_finqa_agent_tool_candidates.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage6-finqa-sft-r32 \
  --input_jsonl data/agent/processed/stage7/agent_stage7_finqa_train_only.jsonl \
  --output_jsonl outputs/finqa/predictions/stage6_agent_finqa_train_for_stage7_dpo.jsonl \
  --out_metrics outputs/finqa/metrics/stage6_agent_finqa_train_for_stage7_dpo_metrics.json \
  --max_new_tokens_tool 160 \
  2>&1 | tee logs/agent/stage7_gen_stage6_train_candidates.log

echo "[3/8] Build balanced DPO pairs"
python scripts/agent/build_agent_stage7_dpo.py \
  --finqa_train_jsonl data/agent/processed/stage7/agent_stage7_finqa_train_only.jsonl \
  --pred_jsonl outputs/finqa/predictions/stage6_agent_finqa_train_for_stage7_dpo.jsonl \
  --replay_jsonl data/agent/processed/stage6/agent_stage6_finqa_train.jsonl \
  --out_json data/agent/processed/stage7/agent_stage7_finqa_dpo_balanced.json \
  --out_external external/LLaMA-Factory/data/agent_stage7_finqa_dpo_balanced.json \
  2>&1 | tee logs/agent/stage7_build_dpo_balanced.log

echo "[4/8] Register LLaMA-Factory DPO dataset"
python - <<'PY'
import json
from pathlib import Path
p = Path("external/LLaMA-Factory/data/dataset_info.json")
info = json.loads(p.read_text(encoding="utf-8"))
info["agent_stage7_finqa_dpo_balanced"] = {
    "file_name": "agent_stage7_finqa_dpo_balanced.json",
    "ranking": True,
    "formatting": "sharegpt",
    "columns": {
        "messages": "conversations",
        "chosen": "chosen",
        "rejected": "rejected"
    }
}
p.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(info["agent_stage7_finqa_dpo_balanced"], ensure_ascii=False, indent=2))
PY

echo "[5/8] Train Stage7 DPO from Stage6 adapter"
set +e
llamafactory-cli train configs/agent/agent_stage7_finqa_dpo_r32.yaml \
  2>&1 | tee logs/agent/train_stage7_finqa_dpo_r32.log
status=${PIPESTATUS[0]}
set -e
if [ "$status" -ne 0 ]; then
  echo "[WARN] DPO training failed, retrying with cutoff_len=2048"
  cp configs/agent/agent_stage7_finqa_dpo_r32.yaml configs/agent/agent_stage7_finqa_dpo_r32.retry3072.yaml
  python - <<'PY'
from pathlib import Path
p = Path("configs/agent/agent_stage7_finqa_dpo_r32.yaml")
text = p.read_text()
text = text.replace("cutoff_len: 3072", "cutoff_len: 2048")
p.write_text(text)
PY
  rm -rf saves/agent/qwen25-7b-agent-stage7-finqa-dpo-r32
  llamafactory-cli train configs/agent/agent_stage7_finqa_dpo_r32.yaml \
    2>&1 | tee logs/agent/train_stage7_finqa_dpo_r32_retry2048.log
fi

echo "[6/8] Evaluate Stage7 FinQA dev"
python scripts/agent/run_finqa_agent_controller.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage7-finqa-dpo-r32 \
  --input_jsonl data/agent/processed/stage6/agent_stage6_finqa_dev.jsonl \
  --output_jsonl outputs/finqa/predictions/stage7_agent_finqa_dpo_r32_finqa_agent_dev.jsonl \
  --out_metrics outputs/finqa/metrics/stage7_agent_finqa_dpo_r32_finqa_agent_dev_metrics.json \
  --max_new_tokens_tool 160 \
  --max_new_tokens_final 96 \
  2>&1 | tee logs/agent/eval_stage7_finqa_agent_dev.log

echo "[7/8] Evaluate forgetting on route/replay/general tool-use"
python scripts/agent/infer_agent_action.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage7-finqa-dpo-r32 \
  --input_jsonl data/agent/processed/stage3/agent_stage3_bfcl_route_dev.jsonl \
  --output_jsonl outputs/agent/predictions/stage7_finqa_dpo_r32_bfcl_route_dev.jsonl \
  --max_new_tokens 192 \
  2>&1 | tee logs/agent/infer_stage7_bfcl_route_dev.log
python scripts/agent/eval_agent_first_action.py \
  --pred_jsonl outputs/agent/predictions/stage7_finqa_dpo_r32_bfcl_route_dev.jsonl \
  --out_metrics outputs/agent/metrics/stage7_finqa_dpo_r32_bfcl_route_dev_metrics.json

python scripts/agent/infer_agent_action.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage7-finqa-dpo-r32 \
  --input_jsonl data/agent/processed/stage2/agent_stage2_dev.jsonl \
  --output_jsonl outputs/agent/predictions/stage7_finqa_dpo_r32_stage2_dev.jsonl \
  --max_new_tokens 192 \
  2>&1 | tee logs/agent/infer_stage7_stage2_dev.log
python scripts/agent/eval_agent_first_action.py \
  --pred_jsonl outputs/agent/predictions/stage7_finqa_dpo_r32_stage2_dev.jsonl \
  --out_metrics outputs/agent/metrics/stage7_finqa_dpo_r32_stage2_dev_metrics.json

python scripts/agent/infer_agent_action.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage7-finqa-dpo-r32 \
  --input_jsonl data/agent/processed/stage1/gorilla_hf_eval_agent_sft.sample1000.jsonl \
  --output_jsonl outputs/agent/predictions/stage7_finqa_dpo_r32_gorilla_hf_eval1000.jsonl \
  --max_new_tokens 192 \
  2>&1 | tee logs/agent/infer_stage7_gorilla_hf_eval1000.log
python scripts/agent/eval_agent_first_action.py \
  --pred_jsonl outputs/agent/predictions/stage7_finqa_dpo_r32_gorilla_hf_eval1000.jsonl \
  --out_metrics outputs/agent/metrics/stage7_finqa_dpo_r32_gorilla_hf_eval1000_metrics.json

python scripts/agent/run_finance_agent_controller.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage7-finqa-dpo-r32 \
  --input_jsonl external/LLaMA-Factory/data/agent_stage1_dev.jsonl \
  --output_jsonl outputs/agent/predictions/stage7_finqa_dpo_r32_finance_closed_loop_dev.jsonl \
  --out_metrics outputs/agent/metrics/stage7_finqa_dpo_r32_finance_closed_loop_dev_metrics.json \
  --max_turns 6 \
  --max_new_tokens 192 \
  2>&1 | tee logs/agent/run_stage7_finance_closed_loop_dev.log

echo "[8/8] Write Stage7 summary"
python - <<'PY'
import json
from pathlib import Path

items = [
    ("Stage6 FinQA agent dev", "outputs/finqa/metrics/stage6_agent_finqa_sft_r32_finqa_agent_dev_metrics.json"),
    ("Stage7 DPO data", "data/agent/processed/stage7/stage7_dpo_data_summary.json"),
    ("Stage7 FinQA agent dev", "outputs/finqa/metrics/stage7_agent_finqa_dpo_r32_finqa_agent_dev_metrics.json"),
    ("Stage6 BFCL route", "outputs/agent/metrics/stage6_finqa_sft_r32_bfcl_route_dev_metrics.json"),
    ("Stage7 BFCL route", "outputs/agent/metrics/stage7_finqa_dpo_r32_bfcl_route_dev_metrics.json"),
    ("Stage6 Stage2 replay", "outputs/agent/metrics/stage6_finqa_sft_r32_stage2_dev_metrics.json"),
    ("Stage7 Stage2 replay", "outputs/agent/metrics/stage7_finqa_dpo_r32_stage2_dev_metrics.json"),
    ("Stage6 Gorilla", "outputs/agent/metrics/stage6_finqa_sft_r32_gorilla_hf_eval1000_metrics.json"),
    ("Stage7 Gorilla", "outputs/agent/metrics/stage7_finqa_dpo_r32_gorilla_hf_eval1000_metrics.json"),
    ("Stage6 finance closed-loop", "outputs/agent/metrics/stage6_finqa_sft_r32_finance_closed_loop_dev_metrics.json"),
    ("Stage7 finance closed-loop", "outputs/agent/metrics/stage7_finqa_dpo_r32_finance_closed_loop_dev_metrics.json"),
]
keys = [
    "num_pairs", "wrong_pairs", "correct_anchor_pairs", "agent_replay_pairs",
    "num_samples", "tool_json_valid_rate", "calculator_call_rate", "program_executable_rate",
    "scaled_execution_acc_at_1pct", "scaled_execution_acc_at_5pct",
    "final_action_valid_rate", "scaled_final_acc_at_1pct", "scaled_final_acc_at_5pct",
    "json_valid_rate", "action_accuracy", "tool_name_accuracy", "tool_name_accuracy_on_gold_tool_turns",
    "completion_rate", "tool_success_rate", "final_numeric_acc_1pct", "final_numeric_acc_5pct",
]
lines = ["# Agent Stage7 FinQA DPO Results", ""]
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
    if title == "Stage7 DPO data" and "candidate_status_counts" in data:
        lines.append(f"- candidate_status_counts: {data['candidate_status_counts']}")
    lines.append("")
Path("outputs/agent/metrics/stage7_finqa_dpo_summary.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
PY

echo "[DONE] Stage7 DPO pipeline complete"
