import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.finqa_calculator import execute_program, numeric_match, numeric_match_scaled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--out_metrics", required=True)
    ap.add_argument("--out_bad", required=True)
    args = ap.parse_args()

    total = parse_nonempty = executable = acc1 = acc5 = scaled_acc1 = scaled_acc5 = 0
    unsupported = Counter()
    bad = []

    with open(args.input_jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            ex = json.loads(line)
            meta = ex.get("meta") or {}
            program = str(meta.get("program_re") or "")
            gold = str(meta.get("answer") or "")
            if program.strip():
                parse_nonempty += 1
            result = execute_program(program)
            if result["ok"]:
                executable += 1
                acc1 += numeric_match(result["result"], gold, 0.01)
                acc5 += numeric_match(result["result"], gold, 0.05)
                scaled_acc1 += numeric_match_scaled(result["result"], gold, 0.01, pred_is_value=True)
                scaled_acc5 += numeric_match_scaled(result["result"], gold, 0.05, pred_is_value=True)
            else:
                match = re.search(r"unsupported op ([A-Za-z_][A-Za-z0-9_]*)", result["error"] or "")
                if match:
                    unsupported[match.group(1)] += 1
                if len(bad) < 100:
                    bad.append({
                        "id": meta.get("id"),
                        "question": meta.get("question"),
                        "gold": gold,
                        "program": program,
                        "error": result["error"],
                    })

    metrics = {
        "num_samples": total,
        "program_parse_rate": parse_nonempty / total if total else 0,
        "program_executable_rate": executable / total if total else 0,
        "execution_acc_at_1pct": acc1 / executable if executable else None,
        "execution_acc_at_5pct": acc5 / executable if executable else None,
        "scaled_execution_acc_at_1pct": scaled_acc1 / executable if executable else None,
        "scaled_execution_acc_at_5pct": scaled_acc5 / executable if executable else None,
        "unsupported_op_counts": dict(unsupported),
        "bad_count_previewed": len(bad),
    }

    Path(args.out_metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_bad).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_metrics).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(args.out_bad, "w", encoding="utf-8") as f:
        for row in bad:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
