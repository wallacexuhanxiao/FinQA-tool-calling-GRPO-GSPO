#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/gui-grounding-agent
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=logs/finqa/watch_grpo_r2_full_eval.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
log "watcher started"
while tmux has-session -t finqa_grpo_r2_full 2>/dev/null; do
  sleep 60
done
log "training pipeline session ended; resolve active adapter"
if [ -d saves/finqa/qwen25-7b-finqa-sft-grpo-r2-full-g8 ]; then
  ACTIVE_NAME=r2_full_g8
  ACTIVE_OUT=saves/finqa/qwen25-7b-finqa-sft-grpo-r2-full-g8
else
  ACTIVE_NAME=r2_full_g6_fallback
  ACTIVE_OUT=saves/finqa/qwen25-7b-finqa-sft-grpo-r2-full-g6-fallback
fi
METRICS="outputs/finqa/metrics/${ACTIVE_NAME}_dev_metrics.json"
if [ -s "$METRICS" ]; then
  log "full dev metrics already exists: $METRICS; skip watcher eval"
  exit 0
fi
log "run full dev inference for ${ACTIVE_NAME}"
PYTHONPATH=. python eval/infer_finqa_qwen.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path "$ACTIVE_OUT" \
  --input_jsonl external/LLaMA-Factory/data/finqa_dev.jsonl \
  --output_jsonl "outputs/finqa/predictions/${ACTIVE_NAME}_dev.jsonl" \
  --max_new_tokens 96 2>&1 | tee "logs/finqa/infer_${ACTIVE_NAME}_dev.watcher.log"
log "run full dev eval for ${ACTIVE_NAME}"
python eval/eval_finqa_answer.py \
  --pred_jsonl "outputs/finqa/predictions/${ACTIVE_NAME}_dev.jsonl" \
  --out_metrics "outputs/finqa/metrics/${ACTIVE_NAME}_dev_metrics.json" 2>&1 | tee "logs/finqa/eval_${ACTIVE_NAME}_dev.watcher.log"
log "full dev eval watcher completed"
