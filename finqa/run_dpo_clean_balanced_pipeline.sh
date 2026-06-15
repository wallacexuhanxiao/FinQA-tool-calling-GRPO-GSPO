#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/gui-grounding-agent
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs/finqa outputs/finqa/predictions outputs/finqa/metrics saves/finqa data/finqa/processed/dpo
LOG=logs/finqa/dpo_clean_balanced_pipeline.log
AUTO_SHUTDOWN=${AUTO_SHUTDOWN:-1}
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
finish(){ status=$?; log "clean-balanced pipeline exiting with status ${status}"; if [[ "$AUTO_SHUTDOWN" == "1" ]]; then log "AUTO_SHUTDOWN=1, shutting down now"; sync; /sbin/shutdown -h now; fi; }
trap finish EXIT
log "STEP 1 build clean-balanced DPO dataset"
PYTHONPATH=. python scripts/finqa/build_dpo_clean_balanced.py 2>&1 | tee logs/finqa/build_dpo_clean_balanced.log
log "STEP 2 train clean-balanced DPO"
llamafactory-cli train configs/finqa/qwen25_7b_finqa_dpo_clean_balanced.yaml 2>&1 | tee logs/finqa/train_dpo_clean_balanced.log
log "STEP 3 infer clean-balanced DPO dev"
PYTHONPATH=. python eval/infer_finqa_qwen.py --model_path models/Qwen2.5-7B-Instruct --adapter_path saves/finqa/qwen25-7b-finqa-dpo-clean-balanced --input_jsonl external/LLaMA-Factory/data/finqa_dev.jsonl --output_jsonl outputs/finqa/predictions/dpo_clean_balanced_dev.jsonl --max_new_tokens 256 2>&1 | tee logs/finqa/infer_dpo_clean_balanced_dev.log
log "STEP 4 eval clean-balanced DPO dev"
python eval/eval_finqa_answer.py --pred_jsonl outputs/finqa/predictions/dpo_clean_balanced_dev.jsonl --out_metrics outputs/finqa/metrics/dpo_clean_balanced_dev_metrics.json 2>&1 | tee logs/finqa/eval_dpo_clean_balanced_dev.log
log "clean-balanced pipeline completed"
