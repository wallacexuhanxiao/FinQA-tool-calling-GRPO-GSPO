#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/gui-grounding-agent
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
mkdir -p logs/finqa outputs/finqa/predictions outputs/finqa/metrics

eval_one() {
  local name="$1"
  local adapter="$2"
  echo "[$(date '+%F %T')] infer/eval ${name}: ${adapter}"
  PYTHONPATH=. python eval/infer_finqa_toolcall.py \
    --model_path models/Qwen2.5-7B-Instruct \
    --adapter_path "$adapter" \
    --input_jsonl external/LLaMA-Factory/data/finqa_toolcall_dev.jsonl \
    --output_jsonl "outputs/finqa/predictions/${name}.jsonl" \
    --max_new_tokens 128 \
    2>&1 | tee "logs/finqa/infer_${name}.log"
  PYTHONPATH=. python eval/eval_finqa_toolcall.py \
    --pred_jsonl "outputs/finqa/predictions/${name}.jsonl" \
    --out_metrics "outputs/finqa/metrics/${name}_metrics.json" \
    2>&1 | tee "logs/finqa/eval_${name}.log"
}

if [ ! -f outputs/finqa/metrics/toolcall_sft_full_r32_dev_metrics.json ]; then
  eval_one toolcall_sft_full_r32_dev saves/finqa/qwen25-7b-finqa-toolcall-full-sft-r32
fi

eval_one toolcall_grpo_r32_scaled_ckpt1200_dev saves/finqa/qwen25-7b-finqa-toolcall-grpo-r32-scaled-full/checkpoint-1200
eval_one toolcall_grpo_r32_scaled_ckpt2400_dev saves/finqa/qwen25-7b-finqa-toolcall-grpo-r32-scaled-full/checkpoint-2400

PYTHONPATH=. python - <<'REMOTE_PY'
import json
from pathlib import Path
rows = [
    ("Tool-call SFT rank16", "outputs/finqa/metrics/toolcall_sft_full_dev_metrics.json"),
    ("Tool-call SFT rank32", "outputs/finqa/metrics/toolcall_sft_full_r32_dev_metrics.json"),
    ("Tool-call GRPO r32 ckpt1200", "outputs/finqa/metrics/toolcall_grpo_r32_scaled_ckpt1200_dev_metrics.json"),
    ("Tool-call GRPO r32 ckpt2400", "outputs/finqa/metrics/toolcall_grpo_r32_scaled_ckpt2400_dev_metrics.json"),
]
headers = ["Method", "N", "Tool JSON", "Program Exec", "Exact", "Num@1", "Num@5", "Scaled Num@1", "Scaled Num@5"]
lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
def fmt(v):
    if v is None: return "-"
    if isinstance(v, (int, float)): return f"{v*100:.2f}%"
    return str(v)
for name, path in rows:
    p = Path(path)
    if not p.exists():
        continue
    m = json.loads(p.read_text(encoding='utf-8'))
    lines.append("| " + " | ".join([
        name,
        str(m.get("num_samples", "-")),
        fmt(m.get("tool_json_valid_rate")),
        fmt(m.get("program_executable_rate")),
        fmt(m.get("final_answer_exact_match")),
        fmt(m.get("numeric_accuracy_at_1pct")),
        fmt(m.get("numeric_accuracy_at_5pct")),
        fmt(m.get("scaled_numeric_accuracy_at_1pct")),
        fmt(m.get("scaled_numeric_accuracy_at_5pct")),
    ]) + " |")
out = Path("outputs/finqa/metrics/toolcall_r32_checkpoint_comparison.md")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out)
print("\n".join(lines))
REMOTE_PY
