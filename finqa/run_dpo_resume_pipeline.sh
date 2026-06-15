#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/gui-grounding-agent
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs/finqa outputs/finqa/predictions outputs/finqa/metrics saves/finqa
LOG=logs/finqa/dpo_resume_pipeline.log
AUTO_SHUTDOWN=${AUTO_SHUTDOWN:-1}
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
finish(){ status=$?; log "dpo resume exiting with status ${status}"; if [[ "$AUTO_SHUTDOWN" == "1" ]]; then log "AUTO_SHUTDOWN=1, shutting down now"; sync; /sbin/shutdown -h now; fi; }
trap finish EXIT
log "DPO resume STEP 6/8 train with cutoff_len=2048"
llamafactory-cli train configs/finqa/qwen25_7b_finqa_dpo.yaml 2>&1 | tee logs/finqa/train_dpo_full_retry2048.log
log "DPO resume STEP 7/8 DPO dev inference"
PYTHONPATH=. python eval/infer_finqa_qwen.py --model_path models/Qwen2.5-7B-Instruct --adapter_path saves/finqa/qwen25-7b-finqa-dpo-full --input_jsonl external/LLaMA-Factory/data/finqa_dev.jsonl --output_jsonl outputs/finqa/predictions/dpo_full_dev.jsonl --max_new_tokens 256 2>&1 | tee logs/finqa/infer_dpo_full_dev.log
log "DPO resume STEP 8/8 DPO dev eval"
python eval/eval_finqa_answer.py --pred_jsonl outputs/finqa/predictions/dpo_full_dev.jsonl --out_metrics outputs/finqa/metrics/dpo_full_dev_metrics.json 2>&1 | tee logs/finqa/eval_dpo_full_dev.log
log "DPO resume completed"
