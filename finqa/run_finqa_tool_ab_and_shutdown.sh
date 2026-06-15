#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/gui-grounding-agent
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
mkdir -p logs/finqa outputs/finqa/predictions outputs/finqa/metrics

PYTHONPATH=. python eval/infer_finqa_qwen.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/finqa/qwen25-7b-finqa-dpo-clean-balanced \
  --input_jsonl data/finqa/processed/tool_agent/dpo_clean_balanced_dev_repair_prompts.jsonl \
  --output_jsonl outputs/finqa/predictions/dpo_clean_balanced_dev_repair_preds.jsonl \
  --max_new_tokens 256 \
  2>&1 | tee logs/finqa/infer_dpo_clean_balanced_repair_30.log

python scripts/finqa/merge_finqa_repaired_programs.py \
  --base_postprocessed_jsonl outputs/finqa/predictions/dpo_clean_balanced_dev_exec_answer_like_answer.jsonl \
  --repair_pred_jsonl outputs/finqa/predictions/dpo_clean_balanced_dev_repair_preds.jsonl \
  --output_jsonl outputs/finqa/predictions/dpo_clean_balanced_dev_tool_ab.jsonl \
  2>&1 | tee logs/finqa/merge_dpo_clean_balanced_tool_ab.log

python eval/eval_finqa_answer_exec.py \
  --pred_jsonl outputs/finqa/predictions/dpo_clean_balanced_dev_tool_ab.jsonl \
  --out_metrics outputs/finqa/metrics/dpo_clean_balanced_dev_tool_ab_metrics.json \
  2>&1 | tee logs/finqa/eval_dpo_clean_balanced_tool_ab.log

# Pull-friendly summary file
python - <<'PY'
import json
paths={
 'original':'outputs/finqa/metrics/dpo_clean_balanced_dev_answer_exec_metrics.json',
 'calculator_override':'outputs/finqa/metrics/dpo_clean_balanced_dev_exec_answer_like_answer_metrics.json',
 'tool_ab':'outputs/finqa/metrics/dpo_clean_balanced_dev_tool_ab_metrics.json',
}
keys=['num_samples','json_valid_rate','answer_exact_match','numeric_accuracy_at_1pct','numeric_accuracy_at_5pct','scaled_numeric_accuracy_at_1pct','scaled_numeric_accuracy_at_5pct','program_executable_rate','scaled_execution_accuracy_at_5pct','answer_exec_consistency_at_5pct']
out={}
for name,p in paths.items():
    m=json.load(open(p))
    out[name]={k:m.get(k) for k in keys if k in m}
open('outputs/finqa/metrics/tool_ab_summary.json','w').write(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps(out,ensure_ascii=False,indent=2))
PY

sync
shutdown -h now
