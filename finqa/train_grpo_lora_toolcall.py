import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import transformers.utils.hub as hf_hub

if not hasattr(hf_hub, "TRANSFORMERS_CACHE"):
    hf_hub.TRANSFORMERS_CACHE = os.environ.get(
        "TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface/transformers")
    )

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    match = re.search(r"\{.*\}", s, flags=re.S)
    if match:
        s = match.group(0)
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


def parse_tool_call(raw):
    obj = extract_json(raw)
    if not isinstance(obj, dict):
        return False, False, ""
    call = obj.get("tool_call")
    if not isinstance(call, dict):
        return True, False, ""
    name_ok = call.get("name") == "calculator"
    args = call.get("arguments") or {}
    if not isinstance(args, dict):
        args = {}
    return True, name_ok, str(args.get("program") or "")


def sign_flip(pred_value, gold):
    p = normalize_number(pred_value)
    g = normalize_number(gold)
    if p is None or g is None or abs(p) < 1e-12 or abs(g) < 1e-12:
        return False
    return math.copysign(1.0, p) != math.copysign(1.0, g)


def toolcall_scaled_reward(completions, answer=None, **kwargs):
    n = len(completions)
    answers = as_list(answer, n)
    rewards = []

    for comp, gold in zip(completions, answers):
        raw = completion_text(comp)
        score = 0.0

        json_ok, name_ok, program = parse_tool_call(raw)
        if json_ok:
            score += 0.05
        else:
            score -= 0.20

        if name_ok:
            score += 0.05
        elif json_ok:
            score -= 0.10

        if program.strip():
            score += 0.05
        else:
            score -= 0.20

        exec_result = execute_program(program) if program.strip() else {
            "ok": False,
            "result": None,
            "error": "empty program",
        }

        if exec_result["ok"]:
            score += 0.30
            value = exec_result["result"]
            scaled1 = numeric_match_scaled(value, gold, 0.01, pred_is_value=True)
            scaled5 = numeric_match_scaled(value, gold, 0.05, pred_is_value=True)
            if scaled1:
                score += 1.00
            elif scaled5:
                score += 0.50
            if sign_flip(value, gold) and not scaled5:
                score -= 0.30
        else:
            score -= 0.30

        stripped = raw.strip()
        if stripped and not stripped.startswith("{"):
            score -= 0.20
        if len(raw) > 700:
            score -= 0.20

        rewards.append(float(max(-1.0, min(1.7, score))))

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
    ap.add_argument("--sft_adapter_path", default="")
    ap.add_argument("--init_lora_rank", type=int, default=0)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_target", default="q_proj,k_proj,v_proj,o_proj")
    ap.add_argument("--train_jsonl", default="data/finqa/processed/grpo/finqa_toolcall_grpo_full.jsonl")
    ap.add_argument("--output_dir", default="saves/finqa/qwen25-7b-finqa-toolcall-grpo-r32-scaled-full")
    ap.add_argument("--resume_from_checkpoint", default="")
    ap.add_argument("--max_steps", type=int, default=6000)
    ap.add_argument("--save_steps", type=int, default=1200)
    ap.add_argument("--save_total_limit", type=int, default=6)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--max_prompt_length", type=int, default=2048)
    ap.add_argument("--max_completion_length", type=int, default=96)
    ap.add_argument("--learning_rate", type=float, default=5e-7)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=None,
    )
    base_model.config.use_cache = False
    base_model.config.pad_token_id = tokenizer.pad_token_id

    if args.sft_adapter_path:
        model = PeftModel.from_pretrained(base_model, args.sft_adapter_path, is_trainable=True)
        model.print_trainable_parameters()
    elif args.init_lora_rank > 0:
        lora_config = LoraConfig(
            r=args.init_lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[x.strip() for x in args.lora_target.split(",") if x.strip()],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
    else:
        model = base_model

    base_model.warnings_issued = {}
    model.warnings_issued = {}

    ds = Dataset.from_list(load_jsonl(args.train_jsonl))
    training_args = GRPOConfig(
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
        save_total_limit=args.save_total_limit,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=toolcall_scaled_reward,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(json.dumps({"saved": args.output_dir, "max_steps": args.max_steps}, indent=2))


if __name__ == "__main__":
    main()
