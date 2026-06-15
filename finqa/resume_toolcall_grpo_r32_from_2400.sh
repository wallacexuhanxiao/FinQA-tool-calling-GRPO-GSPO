#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/gui-grounding-agent
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
mkdir -p logs/finqa

PYTHONPATH=. python scripts/finqa/train_grpo_lora_toolcall.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --sft_adapter_path saves/finqa/qwen25-7b-finqa-toolcall-full-sft-r32 \
  --train_jsonl data/finqa/processed/grpo/finqa_toolcall_grpo_full.jsonl \
  --output_dir saves/finqa/qwen25-7b-finqa-toolcall-grpo-r32-scaled-full \
  --resume_from_checkpoint saves/finqa/qwen25-7b-finqa-toolcall-grpo-r32-scaled-full/checkpoint-2400 \
  --max_steps 6000 \
  --save_steps 1200 \
  --save_total_limit 6 \
  --num_generations 8 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_prompt_length 2048 \
  --max_completion_length 96 \
  --learning_rate 5e-7 \
  --temperature 0.9 \
  --top_p 0.95 \
  2>&1 | tee logs/finqa/train_toolcall_grpo_r32_scaled_full_resume2400.log

PYTHONPATH=. python eval/infer_finqa_toolcall.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/finqa/qwen25-7b-finqa-toolcall-grpo-r32-scaled-full \
  --input_jsonl external/LLaMA-Factory/data/finqa_toolcall_dev.jsonl \
  --output_jsonl outputs/finqa/predictions/toolcall_grpo_r32_scaled_full_dev.jsonl \
  --max_new_tokens 128 \
  2>&1 | tee logs/finqa/infer_toolcall_grpo_r32_scaled_full_dev.log

PYTHONPATH=. python eval/eval_finqa_toolcall.py \
  --pred_jsonl outputs/finqa/predictions/toolcall_grpo_r32_scaled_full_dev.jsonl \
  --out_metrics outputs/finqa/metrics/toolcall_grpo_r32_scaled_full_dev_metrics.json \
  2>&1 | tee logs/finqa/eval_toolcall_grpo_r32_scaled_full_dev.log

PYTHONPATH=. python - <<'PY2'
import json
from pathlib import Path
rows = [
    ("Tool-call SFT rank16", "outputs/finqa/metrics/toolcall_sft_full_dev_metrics.json"),
    ("Tool-call SFT rank32", "outputs/finqa/metrics/toolcall_sft_full_r32_dev_metrics.json"),
    ("Tool-call GRPO r32 ckpt1200", "outputs/finqa/metrics/toolcall_grpo_r32_scaled_ckpt1200_dev_metrics.json"),
    ("Tool-call GRPO r32 ckpt2400", "outputs/finqa/metrics/toolcall_grpo_r32_scaled_ckpt2400_dev_metrics.json"),
    ("Tool-call GRPO r32 final6000", "outputs/finqa/metrics/toolcall_grpo_r32_scaled_full_dev_metrics.json"),
]
headers = ["Method", "N", "Tool JSON", "Program Exec", "Exact", "Num@1", "Num@5", "Scaled Num@1", "Scaled Num@5"]
lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
def fmt(v):
    if v is None: return "-"
    if isinstance(v, (int, float)): return f"{v*100:.2f}%"
    return str(v)
for name, path in rows:
    p=Path(path)
    if not p.exists():
        continue
    m=json.loads(p.read_text(encoding='utf-8'))
    lines.append("| " + " | ".join([
        name, str(m.get('num_samples','-')), fmt(m.get('tool_json_valid_rate')),
        fmt(m.get('program_executable_rate')), fmt(m.get('final_answer_exact_match')),
        fmt(m.get('numeric_accuracy_at_1pct')), fmt(m.get('numeric_accuracy_at_5pct')),
        fmt(m.get('scaled_numeric_accuracy_at_1pct')), fmt(m.get('scaled_numeric_accuracy_at_5pct')),
    ]) + " |")
out=Path('outputs/finqa/metrics/toolcall_rank32_grpo_comparison.md')
out.write_text("\n".join(lines)+"\n", encoding='utf-8')
print(out)
print("\n".join(lines))
PY2
