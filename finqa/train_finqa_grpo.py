import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

# TRL 0.24 imports llm_blender, and older llm_blender expects this symbol.
import transformers.utils.hub as hf_hub
if not hasattr(hf_hub, 'TRANSFORMERS_CACHE'):
    hf_hub.TRANSFORMERS_CACHE = os.environ.get(
        'TRANSFORMERS_CACHE', os.path.expanduser('~/.cache/huggingface/transformers')
    )

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

os.environ.setdefault('WANDB_DISABLED', 'true')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
sys.path.append(str(Path(__file__).resolve().parent))
from finqa_program_executor import numeric_match_scaled, try_execute_program


def extract_json(text):
    if text is None:
        return None
    s = str(text).strip()
    s = re.sub(r'^```json\s*', '', s)
    s = re.sub(r'^```\s*', '', s)
    s = re.sub(r'\s*```$', '', s)
    m = re.search(r'\{.*\}', s, flags=re.S)
    if m:
        s = m.group(0)
    try:
        return json.loads(s)
    except Exception:
        return None


def norm_text(s):
    if s is None:
        return ''
    s = str(s).strip().lower().replace(',', '')
    s = re.sub(r'\s+', ' ', s)
    return s.strip(' .%$')


def extract_num(s):
    if s is None:
        return None
    s = str(s).replace(',', '')
    m = re.search(r'[-+]?\d*\.?\d+', s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def numeric_close(pred, gold, tol):
    p = extract_num(pred)
    g = extract_num(gold)
    if p is None or g is None:
        return False
    return abs(p - g) / max(abs(g), 1e-6) <= tol


def completion_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get('content', ''))
    return str(completion)


def as_list(x, n):
    if isinstance(x, list):
        return x
    return [x for _ in range(n)]


def is_percent_related(question, gold, pred):
    q = str(question or '').lower()
    return (
        '%' in str(gold)
        or '%' in str(pred)
        or any(w in q for w in ['percent', 'percentage', 'portion', 'rate', 'growth', 'increase', 'decrease', 'decline', 'change'])
    )


def scale_error(pred_value, gold_value):
    if pred_value is None or gold_value is None or abs(gold_value) < 1e-12:
        return False
    for factor in [0.001, 0.01, 0.1, 10.0, 100.0, 1000.0, 1000000.0, 1e-6]:
        if abs(pred_value * factor - gold_value) / max(abs(gold_value), 1e-6) <= 0.05:
            return True
    return False


def sign_flip(pred_value, gold_value):
    if pred_value is None or gold_value is None:
        return False
    if abs(pred_value) < 1e-12 or abs(gold_value) < 1e-12:
        return False
    return math.copysign(1.0, pred_value) != math.copysign(1.0, gold_value)


def finqa_reward(completions, answer=None, question=None, **kwargs):
    n = len(completions)
    answers = as_list(answer, n)
    questions = as_list(question, n)
    rewards = []

    for comp, gold, q in zip(completions, answers, questions):
        raw = completion_text(comp)
        obj = extract_json(raw)
        score = 0.0
        pred_answer = ''
        pred_program = ''

        if isinstance(obj, dict):
            score += 0.05
            pred_answer = str(obj.get('answer', ''))
            pred_program = str(obj.get('program', ''))
        else:
            score -= 0.25
            pred_answer = raw.strip().split('\n')[0]

        if pred_answer.strip():
            score += 0.05
        if pred_program.strip():
            score += 0.05

        exact = norm_text(pred_answer) == norm_text(gold)
        close1 = numeric_close(pred_answer, gold, 0.01)
        close5 = numeric_close(pred_answer, gold, 0.05)

        if exact:
            score += 0.30
        if close1:
            score += 1.00
        elif close5:
            score += 0.60

        p = extract_num(pred_answer)
        g = extract_num(gold)

        if pred_program.strip():
            ok, exec_value = try_execute_program(pred_program)
            if ok:
                score += 0.20
                exec_s = str(exec_value)
                strict_exec_close1 = numeric_close(exec_s, gold, 0.01)
                strict_exec_close5 = numeric_close(exec_s, gold, 0.05)
                scaled_exec_close1 = numeric_match_scaled(exec_value, gold, 0.01, pred_is_value=True)
                scaled_exec_close5 = numeric_match_scaled(exec_value, gold, 0.05, pred_is_value=True)
                answer_exec_close5 = numeric_match_scaled(pred_answer, exec_value, 0.05)

                if strict_exec_close1:
                    score += 1.10
                elif strict_exec_close5:
                    score += 0.85
                elif scaled_exec_close1:
                    score += 0.75
                elif scaled_exec_close5:
                    score += 0.55
                elif scale_error(exec_value, g):
                    score -= 0.25

                if answer_exec_close5:
                    score += 0.35
                elif close5 or scaled_exec_close5:
                    score -= 0.35
            else:
                score -= 0.40

        if p is not None and g is not None:
            if scale_error(p, g) and not close5:
                score -= 0.35
            if sign_flip(p, g):
                score -= 0.30
            if is_percent_related(q, gold, pred_answer):
                if close5:
                    score += 0.10
                elif abs(p * 100 - g) / max(abs(g), 1e-6) <= 0.05 or abs(p / 100 - g) / max(abs(g), 1e-6) <= 0.05:
                    score -= 0.45

        if len(raw) > 900:
            score -= 0.20

        rewards.append(float(max(-1.0, min(2.8, score))))
    return rewards

def load_jsonl(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', default='models/Qwen2.5-7B-Instruct')
    ap.add_argument('--sft_adapter_path', default='saves/finqa/qwen25-7b-finqa-full-sft')
    ap.add_argument('--init_lora_rank', type=int, default=0)
    ap.add_argument('--lora_alpha', type=int, default=32)
    ap.add_argument('--lora_dropout', type=float, default=0.05)
    ap.add_argument('--lora_target', default='q_proj,k_proj,v_proj,o_proj')
    ap.add_argument('--train_jsonl', default='data/finqa/processed/grpo/finqa_grpo_train1500.jsonl')
    ap.add_argument('--output_dir', default='saves/finqa/qwen25-7b-finqa-sft-grpo-r1')
    ap.add_argument('--max_steps', type=int, default=100)
    ap.add_argument('--num_generations', type=int, default=4)
    ap.add_argument('--per_device_train_batch_size', type=int, default=1)
    ap.add_argument('--gradient_accumulation_steps', type=int, default=4)
    ap.add_argument('--max_prompt_length', type=int, default=1792)
    ap.add_argument('--max_completion_length', type=int, default=96)
    ap.add_argument('--learning_rate', type=float, default=5e-7)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--top_p', type=float, default=0.9)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

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
    elif args.init_lora_rank > 0:
        lora_config = LoraConfig(
            r=args.init_lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[x.strip() for x in args.lora_target.split(',') if x.strip()],
            bias='none',
            task_type='CAUSAL_LM',
        )
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
    else:
        model = base_model

    base_model.warnings_issued = {}
    model.warnings_issued = {}

    rows = load_jsonl(args.train_jsonl)
    ds = Dataset.from_list(rows)

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
        save_steps=args.max_steps,
        save_total_limit=1,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=finqa_reward,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(json.dumps({'saved': args.output_dir, 'max_steps': args.max_steps}, indent=2))


if __name__ == '__main__':
    main()
