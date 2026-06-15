#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/gui-grounding-agent
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

mkdir -p logs/finqa outputs/finqa/predictions outputs/finqa/metrics

llamafactory-cli train configs/finqa/qwen25_7b_finqa_toolcall_debug100_sft.yaml \
  2>&1 | tee logs/finqa/train_toolcall_debug100_sft.log

llamafactory-cli train configs/finqa/qwen25_7b_finqa_toolcall_full_sft.yaml \
  2>&1 | tee logs/finqa/train_toolcall_full_sft.log

PYTHONPATH=. python eval/infer_finqa_toolcall.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/finqa/qwen25-7b-finqa-toolcall-full-sft \
  --input_jsonl external/LLaMA-Factory/data/finqa_toolcall_dev.jsonl \
  --output_jsonl outputs/finqa/predictions/toolcall_sft_full_dev.jsonl \
  --max_new_tokens 128 \
  2>&1 | tee logs/finqa/infer_toolcall_sft_full_dev.log

python eval/eval_finqa_toolcall.py \
  --pred_jsonl outputs/finqa/predictions/toolcall_sft_full_dev.jsonl \
  --out_metrics outputs/finqa/metrics/toolcall_sft_full_dev_metrics.json \
  2>&1 | tee logs/finqa/eval_toolcall_sft_full_dev.log

python - <<'PY'
import json
from pathlib import Path

rows = []
known = [
    ("Direct SFT-full", "model answer", "outputs/finqa/metrics/sft_full_dev_metrics.json"),
    ("Clean DPO", "model answer", "outputs/finqa/metrics/dpo_clean_balanced_dev_answer_exec_metrics.json"),
    ("Tool-call SFT-full", "calculator result", "outputs/finqa/metrics/toolcall_sft_full_dev_metrics.json"),
]
for method, source, path in known:
    p = Path(path)
    if not p.exists():
        continue
    m = json.loads(p.read_text())
    rows.append([
        method,
        source,
        m.get("tool_json_valid_rate", m.get("json_valid_rate", "-")),
        m.get("program_executable_rate", "-"),
        m.get("final_answer_exact_match", m.get("answer_exact_match", "-")),
        m.get("numeric_accuracy_at_1pct", "-"),
        m.get("numeric_accuracy_at_5pct", "-"),
        m.get("scaled_numeric_accuracy_at_5pct", "-"),
    ])

out = Path("outputs/finqa/metrics/function_calling_comparison.md")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    f.write("| Method | Answer Source | JSON/Tool Valid | Program Exec | Exact | Num@1 | Num@5 | Scaled Num@5 |\\n")
    f.write("|---|---|---:|---:|---:|---:|---:|---:|\\n")
    for row in rows:
        f.write("| " + " | ".join(str(x) for x in row) + " |\\n")
print(out)
PY
