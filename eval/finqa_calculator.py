import math
import re
from collections import Counter


OPS = {
    "add",
    "subtract",
    "multiply",
    "divide",
    "average",
    "max",
    "min",
    "greater",
    "less",
    "exp",
}

CONST = {
    "const_m1": -1.0,
    "const_0": 0.0,
    "const_1": 1.0,
    "const_2": 2.0,
    "const_3": 3.0,
    "const_4": 4.0,
    "const_5": 5.0,
    "const_10": 10.0,
    "const_100": 100.0,
    "const_1000": 1000.0,
    "const_1000000": 1000000.0,
}

SCALE_FACTORS = (1.0, 100.0, 0.01, 1000.0, 0.001, 1000000.0, 0.000001)


class CalculatorError(Exception):
    pass


def split_top_level(text, sep=","):
    parts = []
    cur = []
    depth = 0
    for ch in str(text or ""):
        if ch == sep and depth == 0:
            item = "".join(cur).strip()
            if item:
                parts.append(item)
            cur = []
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise CalculatorError("unbalanced parentheses")
        cur.append(ch)
    if depth != 0:
        raise CalculatorError("unbalanced parentheses")
    item = "".join(cur).strip()
    if item:
        parts.append(item)
    return parts


def normalize_number(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        x = float(value)
        return x if math.isfinite(x) else None

    text = str(value).strip().strip("\"'")
    if not text:
        return None
    if text in CONST:
        return CONST[text]
    const_match = re.fullmatch(r"const_([0-9]+(?:\.[0-9]+)?)", text)
    if const_match:
        return float(const_match.group(1))
    text = text.replace(",", "").replace("$", "").replace("%", "").strip()
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        x = float(match.group(0))
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def parse_arg(token, values):
    token = str(token).strip().strip("\"'")
    if re.fullmatch(r"#\d+", token):
        idx = int(token[1:])
        if idx >= len(values):
            raise CalculatorError(f"bad reference {token}")
        return values[idx]
    x = normalize_number(token)
    if x is None:
        raise CalculatorError(f"bad number {token}")
    return x


def apply_op(op, args):
    if op == "add":
        return sum(args)
    if op == "subtract":
        if len(args) != 2:
            raise CalculatorError("subtract expects 2 args")
        return args[0] - args[1]
    if op == "multiply":
        out = 1.0
        for arg in args:
            out *= arg
        return out
    if op == "divide":
        if len(args) != 2:
            raise CalculatorError("divide expects 2 args")
        if abs(args[1]) < 1e-12:
            raise CalculatorError("division by zero")
        return args[0] / args[1]
    if op == "average":
        if not args:
            raise CalculatorError("average expects args")
        return sum(args) / len(args)
    if op == "max":
        if not args:
            raise CalculatorError("max expects args")
        return max(args)
    if op == "min":
        if not args:
            raise CalculatorError("min expects args")
        return min(args)
    if op == "greater":
        if len(args) != 2:
            raise CalculatorError("greater expects 2 args")
        return 1.0 if args[0] > args[1] else 0.0
    if op == "less":
        if len(args) != 2:
            raise CalculatorError("less expects 2 args")
        return 1.0 if args[0] < args[1] else 0.0
    if op == "exp":
        if len(args) != 2:
            raise CalculatorError("exp expects 2 args")
        return args[0] ** args[1]
    raise CalculatorError(f"unsupported op {op}")


def execute_program(program):
    steps = []
    values = []
    ops = []
    try:
        for raw_step in split_top_level(program):
            match = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*", raw_step)
            if not match:
                raise CalculatorError(f"bad step {raw_step}")
            op = match.group(1)
            if op not in OPS:
                raise CalculatorError(f"unsupported op {op}")
            args = [parse_arg(x, values) for x in split_top_level(match.group(2))]
            result = apply_op(op, args)
            if not math.isfinite(result):
                raise CalculatorError("non-finite result")
            values.append(result)
            ops.append(op)
            steps.append({"op": op, "args": args, "result": result})
        if not values:
            raise CalculatorError("empty program")
        return {"ok": True, "result": values[-1], "steps": steps, "error": None, "ops": ops}
    except Exception as exc:
        return {"ok": False, "result": None, "steps": steps, "error": str(exc), "ops": ops}


def numeric_variants(value, text="", include_scale=True):
    x = normalize_number(value)
    if x is None:
        return []
    vals = {x}
    if "%" in str(text):
        vals.add(x / 100.0)
    if include_scale:
        for factor in SCALE_FACTORS:
            vals.add(x * factor)
    return sorted(vals, key=lambda item: (abs(item), item))


def close(a, b, tol):
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) / max(abs(float(b)), 1e-6) <= tol


def numeric_match(pred, gold, tol=0.05):
    p = normalize_number(pred)
    g = normalize_number(gold)
    return close(p, g, tol)


def numeric_match_scaled(pred, gold, tol=0.05, pred_is_value=False):
    p = pred if pred_is_value else normalize_number(pred)
    g = normalize_number(gold)
    if p is None or g is None:
        return False
    return any(close(pv, gv, tol) for pv in numeric_variants(p, pred) for gv in numeric_variants(g, gold, include_scale=False))


def format_number(value):
    x = normalize_number(value)
    if x is None:
        return ""
    if abs(x) >= 1000 and abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f"{x:.10g}"


def unsupported_op_counts(programs):
    counter = Counter()
    for program in programs:
        for op in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", str(program or "")):
            if op not in OPS:
                counter[op] += 1
    return counter
