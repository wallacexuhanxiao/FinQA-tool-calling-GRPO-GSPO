#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/gui-grounding-agent
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
mkdir -p logs/finqa outputs/finqa/metrics

MERGED=saves/finqa/qwen25-7b-finqa-toolcall-grpo-r32-scaled-full-merged
SERVED=finqa-toolcall-qwen25-7b

if [ ! -d "$MERGED" ]; then
  PYTHONPATH=. python scripts/finqa/merge_toolcall_grpo_r32_for_vllm.py \
    --output_dir "$MERGED" \
    2>&1 | tee logs/finqa/merge_toolcall_grpo_r32_for_vllm.log
fi

python scripts/finqa/benchmark_finqa_hf.py \
  --model_path "$MERGED" \
  --limit 200 \
  --batch_size 1 \
  --out_json outputs/finqa/metrics/hf_benchmark_bs1_limit200.json \
  2>&1 | tee logs/finqa/hf_benchmark_bs1_limit200.log

tmux new -d -s vllm_finqa "bash scripts/finqa/serve_vllm_toolcall_grpo.sh $MERGED $SERVED 8000 2>&1 | tee logs/finqa/vllm_server.log"
sleep 60

for c in 1 4 8 16; do
  python scripts/finqa/benchmark_finqa_vllm.py \
    --model "$SERVED" \
    --limit 200 \
    --concurrency "$c" \
    --out_json "outputs/finqa/metrics/vllm_benchmark_c${c}_limit200.json" \
    2>&1 | tee "logs/finqa/vllm_benchmark_c${c}_limit200.log"
done

tmux kill-session -t vllm_finqa || true
