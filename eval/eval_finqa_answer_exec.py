import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[1] / 'scripts' / 'finqa'))
from finqa_program_executor import numeric_match_scaled, try_execute_program
from eval.eval_finqa_answer import extract_json, extract_num, norm_text, numeric_match, get_gold, get_pred_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_jsonl', required=True)
    ap.add_argument('--out_metrics', required=True)
    args = ap.parse_args()

    total = json_ok = answer_nonempty = program_nonempty = exact = 0
    numeric_total = num1 = num5 = scaled_num1 = scaled_num5 = 0
    program_total = program_exec = exec1 = exec5 = scaled_exec1 = scaled_exec5 = 0
    answer_exec_consistent1 = answer_exec_consistent5 = 0
    answer_right_exec_wrong = exec_right_answer_wrong = 0
    bad = []

    with open(args.pred_jsonl, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            total += 1
            gold = str(get_gold(ex))
            raw = get_pred_text(ex)
            obj = extract_json(raw)
            pred_answer = ''
            pred_program = ''
            if isinstance(obj, dict):
                json_ok += 1
                pred_answer = str(obj.get('answer', ''))
                pred_program = str(obj.get('program', ''))
            else:
                pred_answer = str(raw).strip().split('\n')[0]

            answer_nonempty += bool(pred_answer.strip())
            program_nonempty += bool(pred_program.strip())
            exact += norm_text(pred_answer) == norm_text(gold)

            answer_num_ok1 = answer_num_ok5 = False
            answer_scaled_ok1 = answer_scaled_ok5 = False
            if extract_num(gold) is not None:
                numeric_total += 1
                answer_num_ok1 = numeric_match(pred_answer, gold, 0.01)
                answer_num_ok5 = numeric_match(pred_answer, gold, 0.05)
                answer_scaled_ok1 = numeric_match_scaled(pred_answer, gold, 0.01)
                answer_scaled_ok5 = numeric_match_scaled(pred_answer, gold, 0.05)
                num1 += answer_num_ok1
                num5 += answer_num_ok5
                scaled_num1 += answer_scaled_ok1
                scaled_num5 += answer_scaled_ok5

            exec_value = None
            exec_ok = False
            exec_num_ok1 = exec_num_ok5 = False
            exec_scaled_ok1 = exec_scaled_ok5 = False
            if pred_program.strip():
                program_total += 1
                ok, val = try_execute_program(pred_program)
                if ok:
                    exec_ok = True
                    exec_value = val
                    program_exec += 1
                    exec_num_ok1 = numeric_match(str(val), gold, 0.01)
                    exec_num_ok5 = numeric_match(str(val), gold, 0.05)
                    exec_scaled_ok1 = numeric_match_scaled(val, gold, 0.01, pred_is_value=True)
                    exec_scaled_ok5 = numeric_match_scaled(val, gold, 0.05, pred_is_value=True)
                    exec1 += exec_num_ok1
                    exec5 += exec_num_ok5
                    scaled_exec1 += exec_scaled_ok1
                    scaled_exec5 += exec_scaled_ok5
                    answer_exec_consistent1 += numeric_match_scaled(pred_answer, val, 0.01)
                    answer_exec_consistent5 += numeric_match_scaled(pred_answer, val, 0.05)

            if answer_scaled_ok5 and exec_ok and not exec_scaled_ok5:
                answer_right_exec_wrong += 1
            if exec_scaled_ok5 and not answer_scaled_ok5:
                exec_right_answer_wrong += 1

            miss = not answer_num_ok5 and norm_text(pred_answer) != norm_text(gold)
            if len(bad) < 30 and miss:
                bad.append({
                    'question': ex.get('question') or (ex.get('meta') or {}).get('question'),
                    'gold': gold,
                    'pred_answer': pred_answer,
                    'exec_value': exec_value,
                    'exec_ok': exec_ok,
                    'exec_scaled_ok5': exec_scaled_ok5,
                    'raw': raw[:500],
                })

    metrics = {
        'num_samples': total,
        'json_valid_rate': json_ok / total if total else 0,
        'answer_nonempty_rate': answer_nonempty / total if total else 0,
        'program_nonempty_rate': program_nonempty / total if total else 0,
        'answer_exact_match': exact / total if total else 0,
        'numeric_total': numeric_total,
        'numeric_accuracy_at_1pct': num1 / numeric_total if numeric_total else None,
        'numeric_accuracy_at_5pct': num5 / numeric_total if numeric_total else None,
        'scaled_numeric_accuracy_at_1pct': scaled_num1 / numeric_total if numeric_total else None,
        'scaled_numeric_accuracy_at_5pct': scaled_num5 / numeric_total if numeric_total else None,
        'program_total': program_total,
        'program_executable_rate': program_exec / program_total if program_total else 0,
        'execution_accuracy_at_1pct': exec1 / program_exec if program_exec else None,
        'execution_accuracy_at_5pct': exec5 / program_exec if program_exec else None,
        'scaled_execution_accuracy_at_1pct': scaled_exec1 / program_exec if program_exec else None,
        'scaled_execution_accuracy_at_5pct': scaled_exec5 / program_exec if program_exec else None,
        'answer_exec_consistency_at_1pct': answer_exec_consistent1 / program_exec if program_exec else None,
        'answer_exec_consistency_at_5pct': answer_exec_consistent5 / program_exec if program_exec else None,
        'answer_right_exec_wrong_scaled5': answer_right_exec_wrong,
        'exec_right_answer_wrong_scaled5': exec_right_answer_wrong,
        'bad_preview': bad,
    }

    out = Path(args.out_metrics)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
