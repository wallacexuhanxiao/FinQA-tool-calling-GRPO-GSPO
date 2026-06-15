#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/gui-grounding-agent
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

mkdir -p logs/finqa outputs/finqa/predictions outputs/finqa/metrics data/finqa/processed/dpo saves/finqa

PIPELINE_LOG=logs/finqa/stage2_pipeline.log
AUTO_SHUTDOWN=${AUTO_SHUTDOWN:-1}

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$PIPELINE_LOG"
}

finish() {
  status=$?
  log "pipeline exiting with status ${status}"
  if [[ "$AUTO_SHUTDOWN" == "1" ]]; then
    log "AUTO_SHUTDOWN=1, shutting down now"
    sync
    /sbin/shutdown -h now
  fi
}
trap finish EXIT

log "STEP 1/8 base dev200 inference"
PYTHONPATH=. python eval/infer_finqa_qwen.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --input_jsonl external/LLaMA-Factory/data/finqa_dev.jsonl \
  --output_jsonl outputs/finqa/predictions/base_dev_200.jsonl \
  --limit 200 \
  --max_new_tokens 256 \
  2>&1 | tee logs/finqa/infer_base_dev_200.log

python eval/eval_finqa_answer.py \
  --pred_jsonl outputs/finqa/predictions/base_dev_200.jsonl \
  --out_metrics outputs/finqa/metrics/base_dev_200_metrics.json \
  2>&1 | tee logs/finqa/eval_base_dev_200.log

log "STEP 2/8 sft-full dev200 inference"
PYTHONPATH=. python eval/infer_finqa_qwen.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/finqa/qwen25-7b-finqa-full-sft \
  --input_jsonl external/LLaMA-Factory/data/finqa_dev.jsonl \
  --output_jsonl outputs/finqa/predictions/sft_full_dev_200.jsonl \
  --limit 200 \
  --max_new_tokens 256 \
  2>&1 | tee logs/finqa/infer_sft_full_dev_200.log

python eval/eval_finqa_answer.py \
  --pred_jsonl outputs/finqa/predictions/sft_full_dev_200.jsonl \
  --out_metrics outputs/finqa/metrics/sft_full_dev_200_metrics.json \
  2>&1 | tee logs/finqa/eval_sft_full_dev_200.log

log "STEP 3/8 sft-full dev full inference"
PYTHONPATH=. python eval/infer_finqa_qwen.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/finqa/qwen25-7b-finqa-full-sft \
  --input_jsonl external/LLaMA-Factory/data/finqa_dev.jsonl \
  --output_jsonl outputs/finqa/predictions/sft_full_dev.jsonl \
  --max_new_tokens 256 \
  2>&1 | tee logs/finqa/infer_sft_full_dev.log

python eval/eval_finqa_answer.py \
  --pred_jsonl outputs/finqa/predictions/sft_full_dev.jsonl \
  --out_metrics outputs/finqa/metrics/sft_full_dev_metrics.json \
  2>&1 | tee logs/finqa/eval_sft_full_dev.log

log "STEP 4/8 sft-full train full candidates"
PYTHONPATH=. python eval/infer_finqa_qwen.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/finqa/qwen25-7b-finqa-full-sft \
  --input_jsonl external/LLaMA-Factory/data/finqa_train_full.jsonl \
  --output_jsonl outputs/finqa/predictions/sft_full_train_candidates.jsonl \
  --max_new_tokens 256 \
  2>&1 | tee logs/finqa/gen_sft_full_train_candidates.log

log "STEP 5/8 build DPO pairs"
PYTHONPATH=. python scripts/finqa/build_dpo_pairs.py \
  --src_jsonl external/LLaMA-Factory/data/finqa_train_full.jsonl \
  --pred_jsonl outputs/finqa/predictions/sft_full_train_candidates.jsonl \
  --out_json data/finqa/processed/dpo/finqa_dpo_full.json \
  --dataset_name finqa_dpo_full \
  --copy_to_dataset_dir external/LLaMA-Factory/data/finqa_dpo_full.json \
  2>&1 | tee logs/finqa/build_dpo_pairs_full.log

log "STEP 6/8 DPO train"
llamafactory-cli train configs/finqa/qwen25_7b_finqa_dpo.yaml \
  2>&1 | tee logs/finqa/train_dpo_full.log

log "STEP 7/8 DPO dev inference"
PYTHONPATH=. python eval/infer_finqa_qwen.py \
  --model_path models/Qwen2.5-7B-Instruct \
  --adapter_path saves/finqa/qwen25-7b-finqa-dpo-full \
  --input_jsonl external/LLaMA-Factory/data/finqa_dev.jsonl \
  --output_jsonl outputs/finqa/predictions/dpo_full_dev.jsonl \
  --max_new_tokens 256 \
  2>&1 | tee logs/finqa/infer_dpo_full_dev.log

log "STEP 8/8 DPO dev eval"
python eval/eval_finqa_answer.py \
  --pred_jsonl outputs/finqa/predictions/dpo_full_dev.jsonl \
  --out_metrics outputs/finqa/metrics/dpo_full_dev_metrics.json \
  2>&1 | tee logs/finqa/eval_dpo_full_dev.log

log "stage2 pipeline completed"
