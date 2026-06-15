import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[1] / 'scripts' / 'finqa'))
from finqa_program_executor import best_scaled_value, extract_num, format_number, numeric_variants, try_execute_program
from eval.eval_finqa_answer import extract_json, get_gold, get_pred_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_jsonl', required=True)
    ap.add_argument('--output_jsonl', required=True)
    ap.add_argument('--scale_to_gold', action='store_true', help='oracle analysis only: choose the scale closest to gold')
    ap.add_argument('--scale_like_answer', action='store_true', help='choose the executable value scale closest to the model original answer')
    args = ap.parse_args()

    total = replaced = exec_ok_count = 0
    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(args.pred_jsonl, encoding='utf-8') as f, out.open('w', encoding='utf-8') as g:
        for line in f:
            if not line.strip():
                continue
            total += 1
            ex = json.loads(line)
            raw = get_pred_text(ex)
            obj = extract_json(raw)
            if not isinstance(obj, dict):
                g.write(json.dumps(ex, ensure_ascii=False) + '\n')
                continue
            program = str(obj.get('program', ''))
            if not program.strip():
                g.write(json.dumps(ex, ensure_ascii=False) + '\n')
                continue
            ok, val = try_execute_program(program)
            if not ok:
                g.write(json.dumps(ex, ensure_ascii=False) + '\n')
                continue
            exec_ok_count += 1
            if args.scale_to_gold:
                new_val = best_scaled_value(val, get_gold(ex))
            elif args.scale_like_answer:
                old_answer = str(obj.get('answer', ''))
                old_num = extract_num(old_answer)
                variants = numeric_variants(val, '', include_scale=True)
                new_val = min(variants, key=lambda x: abs(x - old_num) / max(abs(old_num or 0.0), 1e-6)) if old_num is not None else val
            else:
                new_val = val
            obj['answer'] = format_number(new_val) + ('%' if args.scale_like_answer and '%' in str(obj.get('answer', '')) else '')
            ex['raw_output'] = json.dumps(obj, ensure_ascii=False)
            ex['exec_answer_postprocess'] = {
                'original_raw_output': raw,
                'exec_value': val,
                'scaled_to_gold_oracle': bool(args.scale_to_gold),
            }
            replaced += 1
            g.write(json.dumps(ex, ensure_ascii=False) + '\n')

    print(json.dumps({
        'input': args.pred_jsonl,
        'output': args.output_jsonl,
        'total': total,
        'exec_ok': exec_ok_count,
        'replaced': replaced,
        'scale_to_gold_oracle': bool(args.scale_to_gold),
        'scale_like_answer': bool(args.scale_like_answer),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
