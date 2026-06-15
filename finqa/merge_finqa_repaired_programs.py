import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).resolve().parents[2] / 'scripts' / 'finqa'))
from eval.eval_finqa_answer import extract_json, get_pred_text
from finqa_program_executor import extract_num, format_number, numeric_variants, try_execute_program


def best_like_answer(val, old_answer):
    old_num = extract_num(old_answer)
    variants = numeric_variants(val, '', include_scale=True)
    if old_num is None:
        return val
    return min(variants, key=lambda x: abs(x - old_num) / max(abs(old_num or 0.0), 1e-6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_postprocessed_jsonl', required=True)
    ap.add_argument('--repair_pred_jsonl', required=True)
    ap.add_argument('--output_jsonl', required=True)
    args = ap.parse_args()

    repairs = {}
    repair_total = repair_exec = 0
    for line in open(args.repair_pred_jsonl, encoding='utf-8'):
        if not line.strip():
            continue
        repair_total += 1
        ex = json.loads(line)
        sid = ex.get('id') or (ex.get('meta') or {}).get('id')
        obj = extract_json(get_pred_text(ex))
        if not isinstance(obj, dict):
            continue
        program = str(obj.get('program', ''))
        ok, val = try_execute_program(program)
        if not ok:
            continue
        repair_exec += 1
        repairs[sid] = (program, val, get_pred_text(ex))

    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = merged = 0
    with open(args.base_postprocessed_jsonl, encoding='utf-8') as f, out.open('w', encoding='utf-8') as g:
        for line in f:
            if not line.strip():
                continue
            total += 1
            ex = json.loads(line)
            sid = ex.get('id') or (ex.get('meta') or {}).get('id')
            if sid in repairs:
                old_obj = extract_json(get_pred_text(ex)) or {}
                old_answer = str(old_obj.get('answer', ''))
                program, val, repair_raw = repairs[sid]
                new_val = best_like_answer(val, old_answer)
                new_answer = format_number(new_val) + ('%' if '%' in old_answer else '')
                ex['raw_output'] = json.dumps({'program': program, 'answer': new_answer}, ensure_ascii=False)
                ex['self_repair_postprocess'] = {'repair_raw_output': repair_raw, 'exec_value': val}
                merged += 1
            g.write(json.dumps(ex, ensure_ascii=False) + '\n')
    print(json.dumps({'repair_total': repair_total, 'repair_exec': repair_exec, 'base_total': total, 'merged_repairs': merged, 'output': args.output_jsonl}, indent=2))


if __name__ == '__main__':
    main()
