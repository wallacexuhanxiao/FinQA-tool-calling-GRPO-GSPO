import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import transformers.utils.hub as hf_hub

if not hasattr(hf_hub, "TRANSFORMERS_CACHE"):
    hf_hub.TRANSFORMERS_CACHE = os.environ.get("TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface/transformers"))

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import trl.import_utils as trl_import_utils

trl_import_utils._vllm_available = False
from trl import GRPOConfig, GRPOTrainer

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.finqa_calculator import execute_program, normalize_number, numeric_match_scaled


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


def parse_agent_tool(raw):
    obj = extract_json(raw)
    if not isinstance(obj, dict):
        return False, "", "", ""
    action = str(obj.get("action") or "")
    call = obj.get("tool_call") or {}
    if not isinstance(call, dict):
        return True, action, "", ""
    args = call.get("arguments") or {}
    if not isinstance(args, dict):
        args = {}
    program = str(args.get("program") or args.get("expression") or "")
    return True, action, str(call.get("name") or ""), program


def sign_flip(pred_value, gold):
    p = normalize_number(pred_value)
    g = normalize_number(gold)
    if p is None or g is None or abs(p) < 1e-12 or abs(g) < 1e-12:
        return False
    return math.copysign(1.0, p) != math.copysign(1.0, g)


def finqa_reward(raw, gold):
    score = 0.0
    json_ok, action, tool, program = parse_agent_tool(raw)

    if json_ok:
        score += 0.10
    else:
        score -= 0.40

    if action == "tool_call":
        score += 0.05
    else:
        score -= 0.30

    if tool == "calculator":
        score += 0.10
    elif tool:
        score -= 0.30

    if program.strip():
        score += 0.05
    else:
        score -= 0.20

    exec_result = execute_program(program) if program.strip() else {"ok": False, "result": None, "error": "empty program"}
    if exec_result.get("ok"):
        score += 0.20
        value = exec_result.get("result")
        scaled5 = numeric_match_scaled(value, gold, 0.05, pred_is_value=True)
        scaled1 = numeric_match_scaled(value, gold, 0.01, pred_is_value=True)
        if scaled5:
            score += 1.00
        if scaled1:
            score += 0.50
        if sign_flip(value, gold) and not scaled5:
            score -= 0.20
    else:
        score -= 0.30

    stripped = raw.strip()
    if stripped and not stripped.startswith("{"):
        score -= 0.10
    if len(raw) > 900:
        score -= 0.10
    return float(max(-1.2, min(2.0, score)))


def route_replay_reward(raw, gold_tool):
    score = 0.0
    json_ok, action, tool, _ = parse_agent_tool(raw)
    if json_ok:
        score += 0.10
    else:
        score -= 0.40
    if action == "tool_call":
        score += 0.10
    else:
        score -= 0.30
    if tool == gold_tool:
        score += 1.00
    elif tool and gold_tool and tool.split(".")[0] == gold_tool.split(".")[0]:
        score += 0.20
    elif tool:
        score -= 0.20
    else:
        score -= 0.20
    if len(raw) > 700:
        score -= 0.10
    return float(max(-1.0, min(1.3, score)))


def mixed_reward(completions, answer=None, gold_tool=None, task_type=None, **kwargs):
    n = len(completions)
    answers = as_list(answer, n)
    gold_tools = as_list(gold_tool, n)
    task_types = as_list(task_type, n)
    rewards = []
    for comp, gold, tool, typ in zip(completions, answers, gold_tools, task_types):
        raw = completion_text(comp)
        if typ == "route_replay":
            rewards.append(route_replay_reward(raw, tool))
        else:
            rewards.append(finqa_reward(raw, gold))
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
    ap.add_argument("--adapter_path", default="saves/agent/qwen25-7b-agent-stage6-finqa-sft-r32")
    ap.add_argument("--train_jsonl", default="data/agent/processed/stage7/agent_stage7_finqa_grpo_balanced.jsonl")
    ap.add_argument("--output_dir", default="saves/agent/qwen25-7b-agent-stage7-finqa-gspo-r32")
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--max_prompt_length", type=int, default=2048)
    ap.add_argument("--max_completion_length", type=int, default=128)
    ap.add_argument("--learning_rate", type=float, default=2e-7)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--importance_sampling_level", default="sequence", choices=["token", "sequence"])
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
        importance_sampling_level=args.importance_sampling_level,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = GRPOTrainer(model=model, reward_funcs=mixed_reward, args=cfg, train_dataset=ds, processing_class=tok)
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(json.dumps({
        "saved": args.output_dir,
        "max_steps": args.max_steps,
        "importance_sampling_level": args.importance_sampling_level,
        "note": "GSPO-style sequence-level clipping via TRL GRPOConfig.importance_sampling_level='sequence'",
    }, indent=2))


if __name__ == "__main__":
    main()
