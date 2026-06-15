import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_jsonl', default='external/LLaMA-Factory/data/finqa_train_full.jsonl')
    ap.add_argument('--output_jsonl', default='data/finqa/processed/grpo/finqa_grpo_train1500.jsonl')
    ap.add_argument('--limit', type=int, default=1500)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rows = []
    with open(args.input_jsonl, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            meta = ex.get('meta') or {}
            prompt = [m for m in ex['messages'] if m.get('role') != 'assistant']
            rows.append({
                'prompt': prompt,
                'answer': str(meta.get('answer', '')),
                'gold_program': str(meta.get('program_re', '')),
                'id': meta.get('id'),
                'question': meta.get('question', ''),
            })

    random.Random(args.seed).shuffle(rows)
    if args.limit:
        rows = rows[:args.limit]

    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps({'output': str(out), 'num_rows': len(rows), 'seed': args.seed}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
