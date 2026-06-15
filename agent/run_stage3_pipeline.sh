#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/gui-grounding-agent
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
mkdir -p logs/agent outputs/agent/metrics outputs/agent/predictions

python scripts/agent/prepare_agent_stage3_data.py 2>&1 | tee logs/agent/prepare_stage3_data.log

llamafactory-cli train configs/agent/agent_stage3_sft_r32.yaml 2>&1 | tee logs/agent/train_stage3_sft_r32.log

PYTHONPATH=. python scripts/agent/infer_agent_first_action.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage3-sft-r32 \
  --input_jsonl external/LLaMA-Factory/data/agent_stage3_bfcl_route_dev.jsonl \
  --output_jsonl outputs/agent/predictions/stage3_sft_r32_bfcl_route_dev.jsonl \
  --max_new_tokens 128 \
  2>&1 | tee logs/agent/infer_stage3_bfcl_route_dev.log
python scripts/agent/eval_agent_first_action.py \
  --pred_jsonl outputs/agent/predictions/stage3_sft_r32_bfcl_route_dev.jsonl \
  --out_metrics outputs/agent/metrics/stage3_sft_r32_bfcl_route_dev_metrics.json \
  2>&1 | tee logs/agent/eval_stage3_bfcl_route_dev.log

PYTHONPATH=. python scripts/agent/infer_agent_first_action.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage3-sft-r32 \
  --input_jsonl external/LLaMA-Factory/data/agent_stage2_dev.jsonl \
  --output_jsonl outputs/agent/predictions/stage3_sft_r32_stage2_dev.jsonl \
  --max_new_tokens 192 \
  2>&1 | tee logs/agent/infer_stage3_stage2_dev.log
python scripts/agent/eval_agent_first_action.py \
  --pred_jsonl outputs/agent/predictions/stage3_sft_r32_stage2_dev.jsonl \
  --out_metrics outputs/agent/metrics/stage3_sft_r32_stage2_dev_metrics.json \
  2>&1 | tee logs/agent/eval_stage3_stage2_dev.log

PYTHONPATH=. python scripts/agent/infer_agent_first_action.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage3-sft-r32 \
  --input_jsonl data/agent/processed/stage1/gorilla_hf_eval_agent_sft.sample1000.jsonl \
  --output_jsonl outputs/agent/predictions/stage3_sft_r32_gorilla_hf_eval1000.jsonl \
  --max_new_tokens 192 \
  2>&1 | tee logs/agent/infer_stage3_gorilla_hf_eval1000.log
python scripts/agent/eval_agent_first_action.py \
  --pred_jsonl outputs/agent/predictions/stage3_sft_r32_gorilla_hf_eval1000.jsonl \
  --out_metrics outputs/agent/metrics/stage3_sft_r32_gorilla_hf_eval1000_metrics.json \
  2>&1 | tee logs/agent/eval_stage3_gorilla_hf_eval1000.log

PYTHONPATH=. python scripts/agent/run_finance_agent_controller.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/agent/qwen25-7b-agent-stage3-sft-r32 \
  --input_jsonl external/LLaMA-Factory/data/agent_stage1_dev.jsonl \
  --output_jsonl outputs/agent/predictions/stage3_sft_r32_finance_closed_loop_dev.jsonl \
  --out_metrics outputs/agent/metrics/stage3_sft_r32_finance_closed_loop_dev_metrics.json \
  --max_turns 6 \
  --max_new_tokens 192 \
  2>&1 | tee logs/agent/run_stage3_finance_closed_loop_dev.log

python - <<'SUMMARY_STAGE3_PY'
import json
from pathlib import Path
items=[
('Stage2 dev', 'outputs/agent/metrics/stage2_sft_r32_first_action_dev_metrics.json'),
('Stage2 Gorilla', 'outputs/agent/metrics/stage2_sft_r32_gorilla_hf_eval1000_metrics.json'),
('Stage2 finance', 'outputs/agent/metrics/stage2_sft_r32_finance_closed_loop_dev_metrics.json'),
('Stage3 BFCL route', 'outputs/agent/metrics/stage3_sft_r32_bfcl_route_dev_metrics.json'),
('Stage3 stage2-dev replay', 'outputs/agent/metrics/stage3_sft_r32_stage2_dev_metrics.json'),
('Stage3 Gorilla', 'outputs/agent/metrics/stage3_sft_r32_gorilla_hf_eval1000_metrics.json'),
('Stage3 finance', 'outputs/agent/metrics/stage3_sft_r32_finance_closed_loop_dev_metrics.json'),
]
keys=['num_samples','json_valid_rate','no_extra_text_rate','action_valid_rate','action_accuracy','tool_name_accuracy','tool_name_accuracy_on_gold_tool_turns','argument_schema_valid_rate','completion_rate','tool_success_rate','final_numeric_acc_1pct','final_numeric_acc_5pct']
lines=['# Agent Stage3 Results','']
for name,p in items:
    lines.append(f'## {name}')
    try:
        m=json.load(open(p,encoding='utf-8'))
        for k in keys:
            if k in m: lines.append(f'- {k}: {m[k]}')
    except Exception as e:
        lines.append(f'- error: {e}')
    lines.append('')
Path('outputs/agent/metrics/stage3_summary.md').write_text('\n'.join(lines),encoding='utf-8')
print('\n'.join(lines))
SUMMARY_STAGE3_PY

nvidia-smi
