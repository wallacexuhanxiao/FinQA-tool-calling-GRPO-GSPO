import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).resolve().parents[2] / 'scripts' / 'finqa'))
from eval.eval_finqa_answer import extract_json, get_pred_text
from finqa_program_executor import try_execute_program

SYSTEM = (
    "You are a financial reasoning assistant. Repair invalid FinQA reasoning programs. "
    "Return JSON only with keys: program and answer."
)


def load_sft(path):
    out = {}
    for line in open(path, encoding='utf-8'):
        if not line.strip():
            continue
        ex = json.loads(line)
        sid = (ex.get('meta') or {}).get('id')
        if sid:
            out[sid] = ex
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sft_jsonl', required=True)
    ap.add_argument('--pred_jsonl', required=True)
    ap.add_argument('--output_jsonl', required=True)
    args = ap.parse_args()

    sft = load_sft(args.sft_jsonl)
    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = selected = 0
    with open(args.pred_jsonl, encoding='utf-8') as f, out.open('w', encoding='utf-8') as g:
        for line in f:
            if not line.strip():
                continue
            total += 1
            pred = json.loads(line)
            sid = pred.get('id') or (pred.get('meta') or {}).get('id')
            obj = extract_json(get_pred_text(pred))
            program = str(obj.get('program', '')) if isinstance(obj, dict) else ''
            if program.strip():
                ok, _ = try_execute_program(program)
                if ok:
                    continue
            src = sft.get(sid)
            if not src:
                continue
            prompt_messages = [m for m in src['messages'] if m['role'] != 'assistant']
            repair_instruction = (
                "\n\nThe previous model output had an invalid or non-executable FinQA program. "
                "Repair only the reasoning program using the same financial context and question above.\n"
                "Use only these operations when needed: add, subtract, multiply, divide, exp, greater, table_sum, table_average, average. "
                "Reference previous intermediate values as #0, #1, etc. Return JSON only.\n\n"
                f"Previous invalid output:\n{get_pred_text(pred)}\n\n"
                "Return JSON only:\n{\"program\": \"...\", \"answer\": \"...\"}"
            )
            prompt_messages[-1] = dict(prompt_messages[-1])
            prompt_messages[-1]['content'] = prompt_messages[-1]['content'] + repair_instruction
            row = {
                'messages': [{'role': 'system', 'content': SYSTEM}] + [m for m in prompt_messages if m['role'] != 'system'],
                'meta': {
                    **(src.get('meta') or {}),
                    'original_raw_output': get_pred_text(pred),
                    'repair_target': True,
                },
            }
            g.write(json.dumps(row, ensure_ascii=False) + '\n')
            selected += 1
    print(json.dumps({'total': total, 'selected_for_repair': selected, 'output': args.output_jsonl}, indent=2))


if __name__ == '__main__':
    main()
