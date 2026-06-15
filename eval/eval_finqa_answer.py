import argparse
import json
import re
from pathlib import Path


def extract_json(text):
    if text is None:
        return None
    s = str(text).strip()
    s = re.sub(r"^```json\s*", "", s)
    s = re.sub(r"^```\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        s = m.group(0)
    try:
        return json.loads(s)
    except Exception:
        return None


def norm_text(s):
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = s.replace(",", "")
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .%$")


def extract_num(s):
    if s is None:
        return None
    s = str(s).replace(",", "")
    m = re.search(r"[-+]?\d*\.?\d+", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def numeric_match(pred, gold, tol):
    p = extract_num(pred)
    g = extract_num(gold)
    if p is None or g is None:
        return False
    denom = max(abs(g), 1e-6)
    return abs(p - g) / denom <= tol


def get_pred_text(ex):
    for key in ["prediction", "raw_output", "response", "output", "model_output"]:
        if key in ex:
            return ex[key]
    return ""


def get_gold(ex):
    if "answer" in ex:
        return ex["answer"]
    return (ex.get("meta") or {}).get("answer", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", required=True)
    ap.add_argument("--out_metrics", required=True)
    args = ap.parse_args()

    total = 0
    json_ok = 0
    answer_nonempty = 0
    program_nonempty = 0
    exact = 0
    num1 = 0
    num5 = 0
    numeric_total = 0
    bad = []

    with open(args.pred_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            total += 1

            gold = str(get_gold(ex))
            raw = get_pred_text(ex)
            obj = extract_json(raw)

            pred_answer = ""
            pred_program = ""
            if isinstance(obj, dict):
                json_ok += 1
                pred_answer = str(obj.get("answer", ""))
                pred_program = str(obj.get("program", ""))
            else:
                pred_answer = str(raw).strip().split("\n")[0]

            answer_nonempty += bool(pred_answer.strip())
            program_nonempty += bool(pred_program.strip())
            exact += norm_text(pred_answer) == norm_text(gold)

            if extract_num(gold) is not None:
                numeric_total += 1
                num1 += numeric_match(pred_answer, gold, 0.01)
                num5 += numeric_match(pred_answer, gold, 0.05)

            miss = (
                not numeric_match(pred_answer, gold, 0.05)
                and norm_text(pred_answer) != norm_text(gold)
            )
            if len(bad) < 30 and miss:
                bad.append(
                    {
                        "question": ex.get("question")
                        or (ex.get("meta") or {}).get("question"),
                        "gold": gold,
                        "pred_answer": pred_answer,
                        "raw": raw[:500],
                    }
                )

    metrics = {
        "num_samples": total,
        "json_valid_rate": json_ok / total if total else 0,
        "answer_nonempty_rate": answer_nonempty / total if total else 0,
        "program_nonempty_rate": program_nonempty / total if total else 0,
        "answer_exact_match": exact / total if total else 0,
        "numeric_total": numeric_total,
        "numeric_accuracy_at_1pct": num1 / numeric_total if numeric_total else None,
        "numeric_accuracy_at_5pct": num5 / numeric_total if numeric_total else None,
        "bad_preview": bad,
    }

    Path(args.out_metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_metrics).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
