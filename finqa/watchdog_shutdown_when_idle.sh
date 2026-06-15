#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/gui-grounding-agent
mkdir -p logs/finqa
LOG=logs/finqa/watchdog_toolcall_r32_grpo_shutdown.log
TARGET=outputs/finqa/metrics/toolcall_rank32_grpo_comparison.md

echo "[$(date '+%F %T')] watchdog started" | tee -a "$LOG"

idle_hits=0
while true; do
  active=0
  pgrep -af 'llamafactory-cli|train_grpo_lora_toolcall.py|infer_finqa_toolcall.py|eval_finqa_toolcall.py' >/tmp/finqa_watchdog_processes.txt || true
  if [ -s /tmp/finqa_watchdog_processes.txt ]; then
    active=1
  fi
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
    active=1
  fi

  if [ "$active" -eq 0 ] && [ -f "$TARGET" ]; then
    idle_hits=$((idle_hits + 1))
    echo "[$(date '+%F %T')] idle hit ${idle_hits}/2, target exists" | tee -a "$LOG"
  else
    idle_hits=0
    echo "[$(date '+%F %T')] active=${active}, target_exists=$([ -f "$TARGET" ] && echo 1 || echo 0)" | tee -a "$LOG"
  fi

  if [ "$idle_hits" -ge 2 ]; then
    echo "[$(date '+%F %T')] shutting down" | tee -a "$LOG"
    sync
    shutdown -h now
    exit 0
  fi

  sleep 180
done
