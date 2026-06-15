import argparse
import json
import os
import re
from pathlib import Path

import transformers.utils.hub as hf_hub

if not hasattr(hf_hub, "TRANSFORMERS_CACHE"):
    hf_hub.TRANSFORMERS_CACHE = os.environ.get("TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface/transformers"))

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# TRL optionally imports vLLM inside GRPOTrainer when vLLM is installed. This
# machine has vLLM 0.9.x for serving benchmarks, and that version conflicts with
# the current Transformers AIMv2 config registration during import. GRPO training
# here uses normal Transformers generation, so disable TRL's vLLM path locally.
import trl.import_utils as trl_import_utils

trl_import_utils._vllm_available = False
from trl import GRPOConfig, GRPOTrainer

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def extract_json(text):
    s = str(text or "").strip()
    s = re.sub(r"^```json\s*", "", s)
    s = re.sub(r"^```\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        s = m.group(0)
    try:
        return json.loads(s)
    except Exception:
        return None


def completion_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    return str(completion)


def as_list(value, n):
    return value if isinstance(value, list) else [value for _ in range(n)]


def parse_action(raw):
    obj = extract_json(raw)
    if not isinstance(obj, dict):
        return False, "", ""
    action = str(obj.get("action") or "")
    tool = ""
    if action == "tool_call":
        call = obj.get("tool_call") or {}
        if isinstance(call, dict):
            tool = str(call.get("name") or "")
    return True, action, tool


def route_reward(completions, gold_tool=None, **kwargs):
    n = len(completions)
    gold_tools = as_list(gold_tool, n)
    rewards = []
    for comp, gold in zip(completions, gold_tools):
        raw = completion_text(comp)
        score = 0.0
        json_ok, action, tool = parse_action(raw)
        if json_ok:
            score += 0.08
        else:
            score -= 0.45
        if raw.strip().startswith("{") and raw.strip().endswith("}"):
            score += 0.04
        else:
            score -= 0.10
        if action == "tool_call":
            score += 0.08
        else:
            score -= 0.25
        if tool:
            score += 0.08
        else:
            score -= 0.20
        if tool == gold:
            score += 1.20
        elif tool and gold and tool.split(".")[0] == gold.split(".")[0]:
            score += 0.25
        elif tool:
            score -= 0.15
        if len(raw) > 600:
            score -= 0.20
        rewards.append(float(max(-1.0, min(1.5, score))))
    return rewards


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter_path", default="saves/agent/qwen25-7b-agent-stage4-sft-r32")
    ap.add_argument("--train_jsonl", default="data/agent/processed/stage5/agent_stage5_grpo_toolroute.jsonl")
    ap.add_argument("--output_dir", default="saves/agent/qwen25-7b-agent-stage5-grpo-toolroute-r32")
    ap.add_argument("--max_steps", type=int, default=600)
    ap.add_argument("--save_steps", type=int, default=300)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--max_prompt_length", type=int, default=2048)
    ap.add_argument("--max_completion_length", type=int, default=96)
    ap.add_argument("--learning_rate", type=float, default=3e-7)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=None)
    base.config.use_cache = False
    base.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(base, args.adapter_path, is_trainable=True)
    model.print_trainable_parameters()
    base.warnings_issued = {}
    model.warnings_issued = {}

    ds = Dataset.from_list(load_jsonl(args.train_jsonl))
    cfg = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = GRPOTrainer(model=model, reward_funcs=route_reward, args=cfg, train_dataset=ds, processing_class=tok)
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(json.dumps({"saved": args.output_dir, "max_steps": args.max_steps}, indent=2))


if __name__ == "__main__":
    main()
