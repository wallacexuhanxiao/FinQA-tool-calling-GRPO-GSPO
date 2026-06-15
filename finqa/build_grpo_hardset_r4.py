import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.eval_finqa_answer import extract_json, extract_num, norm_text, numeric_match


def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]


def classify(question):
    q = str(question or '').lower()
    if any(w in q for w in ['percent', 'percentage', 'portion', 'rate', 'margin', 'growth', 'increase', 'decrease', 'decline', 'change']):
        return 'percent_rate'
    if any(w in q for w in ['ratio', 'relative to', 'compared to', 'times']):
        return 'ratio'
    if any(w in q for w in ['average', 'mean']):
        return 'average'
    if any(w in q for w in ['total', 'sum', 'combined']):
        return 'sum_total'
    if any(w in q for w in ['difference', 'decline from', 'how much higher', 'how much lower']):
        return 'difference'
    return 'other'


def pred_answer(raw):
    obj = extract_json(raw)
    if isinstance(obj, dict):
        return str(obj.get('answer', ''))
    return str(raw or '').strip().split('\n')[0]


def is_correct(pred, gold):
    return norm_text(pred) == norm_text(gold) or numeric_match(pred, gold, 0.05)


def make_grpo_row(src, reason, category):
    meta = src.get('meta') or {}
    prompt = [m for m in src['messages'] if m.get('role') != 'assistant']
    return {
        'prompt': prompt,
        'answer': str(meta.get('answer', '')),
        'gold_program': str(meta.get('program_re', '')),
        'id': meta.get('id'),
        'question': meta.get('question', ''),
        'category': category,
        'hard_reason': reason,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sft_train_jsonl', default='external/LLaMA-Factory/data/finqa_train_full.jsonl')
    ap.add_argument('--sft_train_pred_jsonl', default='outputs/finqa/predictions/sft_full_train_candidates.jsonl')
    ap.add_argument('--out_jsonl', default='data/finqa/processed/grpo/finqa_grpo_r4_hard2500.jsonl')
    ap.add_argument('--target_size', type=int, default=2500)
    ap.add_argument('--wrong_size', type=int, default=1500)
    ap.add_argument('--category_size', type=int, default=650)
    ap.add_argument('--anchor_size', type=int, default=350)
    ap.add_argument('--seed', type=int, default=43)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    src_rows = load_jsonl(args.sft_train_jsonl)
    pred_rows = load_jsonl(args.sft_train_pred_jsonl)
    by_id = {(r.get('meta') or {}).get('id'): r for r in src_rows}

    wrong = []
    correct = []
    category_pool = []
    stats = Counter()

    for pr in pred_rows:
        sid = pr.get('id') or (pr.get('meta') or {}).get('id')
        src = by_id.get(sid)
        if not src:
            continue
        q = pr.get('question') or (pr.get('meta') or {}).get('question', '')
        gold = str(pr.get('answer') or (pr.get('meta') or {}).get('answer', ''))
        pred = pred_answer(pr.get('raw_output', ''))
        cat = classify(q)
        ok = is_correct(pred, gold)
        row = make_grpo_row(src, 'sft_correct' if ok else 'sft_wrong', cat)
        stats[f'{cat}/' + ('correct' if ok else 'wrong')] += 1
        if ok:
            correct.append(row)
        else:
            wrong.append(row)
        if cat in {'percent_rate', 'ratio', 'average'}:
            category_pool.append(row)

    rng.shuffle(wrong)
    rng.shuffle(category_pool)
    rng.shuffle(correct)

    selected = []
    seen = set()

    def add(rows, limit, reason_override=None):
        added = 0
        for row in rows:
            sid = row.get('id')
            if sid in seen:
                continue
            new_row = dict(row)
            if reason_override:
                new_row['hard_reason'] = reason_override
            selected.append(new_row)
            seen.add(sid)
            added += 1
            if added >= limit:
                break
        return added

    add(wrong, args.wrong_size)
    add(category_pool, args.category_size, 'category_focus')
    add(correct, args.anchor_size, 'anchor_correct')
    if len(selected) < args.target_size:
        add(wrong + category_pool + correct, args.target_size - len(selected))
    selected = selected[:args.target_size]

    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    summary = {
        'output': str(out),
        'num_rows': len(selected),
        'source_stats': stats,
        'selected_category': Counter(r['category'] for r in selected),
        'selected_reason': Counter(r['hard_reason'] for r in selected),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
