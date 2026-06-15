# Stage7 FinQA GRPO Long2400 Run

## Why

The previous Stage7 GRPO run used only `400` steps, about `0.09` epoch on the balanced Stage7 dataset. It was useful as a smoke test, but too short for a normal RL post-training run.

## New Run

Remote session:

```bash
tmux attach -t stage7_grpo_long
```

Script:

```bash
bash scripts/agent/run_stage7_finqa_grpo_long_pipeline.sh
```

Main settings:

- Base model: `models/Qwen2.5-7B-Instruct`
- Start adapter: `saves/agent/qwen25-7b-agent-stage6-finqa-sft-r32`
- Output adapter: `saves/agent/qwen25-7b-agent-stage7-finqa-grpo-r32-long2400`
- Training data: `data/agent/processed/stage7/agent_stage7_finqa_grpo_balanced.jsonl`
- `max_steps`: `2400`
- `num_generations`: `8`
- `gradient_accumulation_steps`: `8`
- `max_prompt_length`: `2048`
- `max_completion_length`: `128`
- `learning_rate`: `2e-7`
- `save_steps`: `400`
- `save_total_limit`: `1`

Only the latest checkpoint is kept to avoid filling the 100GB disk.

## Expected Runtime

At roughly `4.7-5.9s/step`, the 2400-step run should take about 3.2-4.0 hours for training, then run a full FinQA dev evaluation.

## Outputs

- Training log: `logs/agent/train_stage7_grpo_long2400_r32.log`
- FinQA predictions: `outputs/finqa/predictions/stage7_grpo_long2400_r32_finqa_agent_dev.jsonl`
- FinQA metrics: `outputs/finqa/metrics/stage7_grpo_long2400_r32_finqa_agent_dev_metrics.json`
- Summary: `outputs/agent/metrics/stage7_grpo_long2400_r32_summary.md`

## Final FinQA Dev Result

- `num_samples`: `883`
- `tool_json_valid_rate`: `1.0`
- `calculator_call_rate`: `1.0`
- `program_executable_rate`: `0.9921`
- `scaled_execution_acc_at_1pct`: `0.6478`
- `scaled_execution_acc_at_5pct`: `0.7271`
- `final_action_valid_rate`: `1.0`
- `scaled_final_acc_at_1pct`: `0.6546`
- `scaled_final_acc_at_5pct`: `0.7339`

This long GRPO run improves over the earlier 400-step GRPO run (`scaled_final_acc_at_5pct` around `0.7123`) and Stage6 FinQA SFT (`scaled_final_acc_at_5pct` around `0.7022`).
