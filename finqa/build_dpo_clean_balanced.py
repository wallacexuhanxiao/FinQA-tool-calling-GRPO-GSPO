import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from eval.eval_finqa_answer import extract_json, norm_text, numeric_match, extract_num


def role_to_sharegpt(message):
    role = message['role']
    if role == 'system':
        return {'from': 'system', 'value': message['content']}
    if role == 'user':
        return {'from': 'human', 'value': message['content']}
    if role == 'assistant':
        return {'from': 'gpt', 'value': message['content']}
    raise ValueError(role)


def classify(q):
    ql = q.lower()
    if any(w in ql for w in ['percent', 'percentage', 'rate', 'growth', 'margin']):
        return 'percent_rate'
    if any(w in ql for w in ['ratio', 'portion', 'as a percentage']):
        return 'ratio'
    if 'average' in ql:
        return 'average'
    if any(w in ql for w in ['total', 'sum', 'combined']):
        return 'sum_total'
    if any(w in ql for w in ['change', 'increase', 'decrease', 'higher', 'lower', 'difference', 'decline']):
        return 'difference'
    if any(w in ql for w in ['market capitalization', 'product']):
        return 'multiply'
    return 'other'


def parse_float(x):
    try:
        return float(str(x).replace(',', '').replace('%', '').strip())
    except Exception:
        return None


def fmt_num(x):
    if x is None or not math.isfinite(x):
        return None
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f'{x:.6g}'


def synthetic_rejected_from_gold(program, answer, q):
    ql = q.lower()
    if not any(w in ql for w in ['percent', 'percentage', 'rate', 'growth']):
        return None
    m = re.search(r'subtract\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)\s*,\s*divide\(#0,\s*[-+]?\d*\.?\d+\)', program)
    if not m:
        return None
    a = parse_float(m.group(1))
    b = parse_float(m.group(2))
    if a is None or b is None:
        return None
    diff = a - b
    bad_answer = fmt_num(diff)
    if not bad_answer:
        return None
    if norm_text(bad_answer) == norm_text(answer) or numeric_match(bad_answer, answer, 0.05):
        return None
    return json.dumps({'program': f'subtract({m.group(1)}, {m.group(2)})', 'answer': bad_answer}, ensure_ascii=False)


def prompt_messages(ex):
    return [m for m in ex['messages'] if m['role'] != 'assistant']


def token_len(tokenizer, ex, assistant_content):
    messages = prompt_messages(ex) + [{'role': 'assistant', 'content': assistant_content}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def make_pair(ex, rejected, source, category):
    return {
        'conversations': [role_to_sharegpt(m) for m in prompt_messages(ex)],
        'chosen': {'from': 'gpt', 'value': ex['messages'][-1]['content']},
        'rejected': {'from': 'gpt', 'value': rejected},
        'meta': {**(ex.get('meta') or {}), 'dpo_source': source, 'category': category},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src_jsonl', default='external/LLaMA-Factory/data/finqa_train_full.jsonl')
    ap.add_argument('--pred_jsonl', default='outputs/finqa/predictions/sft_full_train_candidates.jsonl')
    ap.add_argument('--model_path', default='models/Qwen2.5-7B-Instruct')
    ap.add_argument('--out_json', default='data/finqa/processed/dpo/finqa_dpo_clean_balanced.json')
    ap.add_argument('--dataset_file', default='external/LLaMA-Factory/data/finqa_dpo_clean_balanced.json')
    ap.add_argument('--dataset_info', default='external/LLaMA-Factory/data/dataset_info.json')
    ap.add_argument('--dataset_name', default='finqa_dpo_clean_balanced')
    ap.add_argument('--max_tokens', type=int, default=2048)
    ap.add_argument('--max_per_category', type=int, default=180)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    src = [json.loads(x) for x in open(args.src_jsonl, encoding='utf-8') if x.strip()]
    pred = [json.loads(x) for x in open(args.pred_jsonl, encoding='utf-8') if x.strip()]

    buckets = defaultdict(list)
    stats = defaultdict(int)

    for ex, pr in zip(src, pred):
        meta = ex.get('meta') or {}
        q = meta.get('question') or ''
        category = classify(q)
        gold_msg = ex['messages'][-1]['content']
        gold_obj = extract_json(gold_msg)
        if not isinstance(gold_obj, dict):
            stats['bad_gold_json'] += 1
            continue
        gold_answer = str(gold_obj.get('answer', ''))
        gold_program = str(gold_obj.get('program', ''))
        raw = str(pr.get('raw_output', '')).strip()
        obj = extract_json(raw)
        pred_answer = ''
        rejected = ''
        pred_json = isinstance(obj, dict)
        if pred_json:
            pred_answer = str(obj.get('answer', ''))
            rejected = json.dumps({'program': str(obj.get('program', '')), 'answer': pred_answer}, ensure_ascii=False)
        else:
            pred_answer = raw.split('\n')[0].strip()
            rejected = raw

        sft_correct = norm_text(pred_answer) == norm_text(gold_answer) or numeric_match(pred_answer, gold_answer, 0.05)

        if not sft_correct:
            if not pred_json:
                stats['wrong_skip_invalid_json'] += 1
                continue
            if extract_num(pred_answer) is None or extract_num(gold_answer) is None:
                stats['wrong_skip_non_numeric'] += 1
                continue
            if numeric_match(pred_answer, gold_answer, 0.10):
                stats['wrong_skip_near_miss'] += 1
                continue
            # Skip decimal-vs-percent equivalent ambiguities.
            ga = extract_num(gold_answer)
            pa = extract_num(pred_answer)
            if '%' in gold_answer and ga is not None and pa is not None and abs(pa * 100 - ga) / max(abs(ga), 1e-6) <= 0.08:
                stats['wrong_skip_decimal_percent_equiv'] += 1
                continue
            if token_len(tokenizer, ex, gold_msg) > args.max_tokens or token_len(tokenizer, ex, rejected) > args.max_tokens:
                stats['wrong_skip_too_long'] += 1
                continue
            buckets[category].append(make_pair(ex, rejected, 'sft_wrong_clean', category))
            stats['wrong_keep'] += 1
        else:
            synthetic = synthetic_rejected_from_gold(gold_program, gold_answer, q)
            if synthetic:
                if token_len(tokenizer, ex, gold_msg) > args.max_tokens or token_len(tokenizer, ex, synthetic) > args.max_tokens:
                    stats['synthetic_skip_too_long'] += 1
                    continue
                buckets[category].append(make_pair(ex, synthetic, 'synthetic_subtract_only', category))
                stats['synthetic_keep'] += 1

    selected = []
    per_cat = {}
    order = ['percent_rate', 'ratio', 'average', 'sum_total', 'difference', 'multiply', 'other']
    for cat in order:
        rows = buckets.get(cat, [])
        # Keep deterministic, with synthetic/wrong naturally interleaved by source order.
        chosen = rows[: args.max_per_category]
        selected.extend(chosen)
        per_cat[cat] = {'available': len(rows), 'selected': len(chosen)}

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding='utf-8')
    Path(args.dataset_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.dataset_file).write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding='utf-8')

    info_path = Path(args.dataset_info)
    info = json.loads(info_path.read_text(encoding='utf-8'))
    info[args.dataset_name] = {
        'file_name': Path(args.dataset_file).name,
        'ranking': True,
        'formatting': 'sharegpt',
        'columns': {'messages': 'conversations', 'chosen': 'chosen', 'rejected': 'rejected'},
    }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')

    summary = {'total_selected': len(selected), 'per_category': per_cat, 'stats': dict(stats), 'dataset_name': args.dataset_name}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
