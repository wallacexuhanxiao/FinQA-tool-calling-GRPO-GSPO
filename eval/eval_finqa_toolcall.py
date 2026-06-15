import argparse
import json
from collections import Counter
from pathlib import Path

from eval.finqa_calculator import numeric_match, numeric_match_scaled
from eval.eval_finqa_answer import norm_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", required=True)
    ap.add_argument("--out_metrics", required=True)
    args = ap.parse_args()

    total = valid = name_ok = program_nonempty = executable = exact = 0
    num_total = num1 = num5 = scaled_num1 = scaled_num5 = 0
    unsupported = Counter()
    bad = []

    with open(args.pred_jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            total += 1
            gold = str(ex.get("gold_answer") or (ex.get("meta") or {}).get("answer") or "")
            final_answer = str(ex.get("final_answer") or "")
            valid += bool(ex.get("tool_json_valid"))
            name_ok += ex.get("tool_name") == "calculator"
            program = str(ex.get("program") or "")
            program_nonempty += bool(program.strip())
            executable += bool(ex.get("program_executable"))
            exact += norm_text(final_answer) == norm_text(gold)
            if gold.strip():
                num_total += 1
                num1 += numeric_match(final_answer, gold, 0.01)
                num5 += numeric_match(final_answer, gold, 0.05)
                scaled_num1 += numeric_match_scaled(ex.get("execution_result"), gold, 0.01, pred_is_value=True)
                scaled_num5 += numeric_match_scaled(ex.get("execution_result"), gold, 0.05, pred_is_value=True)
            err = str(ex.get("exec_error") or "")
            if "unsupported op " in err:
                unsupported[err.split("unsupported op ", 1)[1].split()[0]] += 1
            if len(bad) < 30 and not numeric_match_scaled(ex.get("execution_result"), gold, 0.05, pred_is_value=True):
                bad.append({
                    "id": ex.get("id"),
                    "question": ex.get("question"),
                    "gold": gold,
                    "final_answer": final_answer,
                    "program": program,
                    "exec_error": ex.get("exec_error"),
                    "raw_output": str(ex.get("raw_output") or "")[:500],
                })

    metrics = {
        "num_samples": total,
        "tool_json_valid_rate": valid / total if total else 0,
        "tool_name_correct_rate": name_ok / total if total else 0,
        "program_nonempty_rate": program_nonempty / total if total else 0,
        "program_executable_rate": executable / total if total else 0,
        "execution_acc_at_1pct": scaled_num1 / num_total if num_total else None,
        "execution_acc_at_5pct": scaled_num5 / num_total if num_total else None,
        "final_answer_exact_match": exact / total if total else 0,
        "numeric_accuracy_at_1pct": num1 / num_total if num_total else None,
        "numeric_accuracy_at_5pct": num5 / num_total if num_total else None,
        "scaled_numeric_accuracy_at_1pct": scaled_num1 / num_total if num_total else None,
        "scaled_numeric_accuracy_at_5pct": scaled_num5 / num_total if num_total else None,
        "unsupported_op_count": dict(unsupported),
        "bad_examples": bad,
    }
    Path(args.out_metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_metrics).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
