import math
import re

CONST = {
    'const_1': 1.0,
    'const_2': 2.0,
    'const_3': 3.0,
    'const_4': 4.0,
    'const_5': 5.0,
    'const_10': 10.0,
    'const_100': 100.0,
    'const_1000': 1000.0,
    'const_1000000': 1000000.0,
    'const_m1': -1.0,
}

OPS = {'add', 'subtract', 'multiply', 'divide', 'exp', 'greater'}
SCALE_FACTORS = (1.0, 100.0, 0.01, 1000.0, 0.001, 1000000.0, 0.000001)


def clean_token(x):
    return str(x).strip().strip('"\'')


def parse_value(token, values):
    token = clean_token(token)
    if token in CONST:
        return CONST[token]
    if re.fullmatch(r'#\d+', token):
        idx = int(token[1:])
        if idx >= len(values):
            raise ValueError(f'missing reference {token}')
        return values[idx]
    token = token.replace(',', '').replace('$', '').replace('%', '')
    if token.startswith('(') and token.endswith(')'):
        token = '-' + token[1:-1]
    return float(token)


def split_args(s):
    args = []
    cur = []
    depth = 0
    for ch in s:
        if ch == ',' and depth == 0:
            args.append(''.join(cur).strip())
            cur = []
            continue
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        cur.append(ch)
    if cur:
        args.append(''.join(cur).strip())
    return args


def split_steps(program):
    steps = []
    cur = []
    depth = 0
    for ch in str(program or ''):
        if ch == ',' and depth == 0:
            text = ''.join(cur).strip()
            if text:
                steps.append(text)
            cur = []
            continue
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        cur.append(ch)
    text = ''.join(cur).strip()
    if text:
        steps.append(text)
    return steps


def apply_op(op, args):
    if op == 'add':
        return args[0] + args[1]
    if op == 'subtract':
        return args[0] - args[1]
    if op == 'multiply':
        return args[0] * args[1]
    if op == 'divide':
        if abs(args[1]) < 1e-12:
            raise ZeroDivisionError('divide by zero')
        return args[0] / args[1]
    if op == 'exp':
        return args[0] ** args[1]
    if op == 'greater':
        return 1.0 if args[0] > args[1] else 0.0
    raise ValueError(f'unsupported op {op}')


def execute_program(program):
    values = []
    for step in split_steps(program):
        m = re.fullmatch(r'([A-Za-z_][A-Za-z0-9_]*)\((.*)\)', step.strip())
        if not m:
            raise ValueError(f'bad step: {step}')
        op = m.group(1)
        if op not in OPS:
            raise ValueError(f'unsupported op {op}')
        args = [parse_value(x, values) for x in split_args(m.group(2))]
        if len(args) != 2:
            raise ValueError(f'{op} expects two args')
        out = apply_op(op, args)
        if not math.isfinite(out):
            raise ValueError('non-finite output')
        values.append(out)
    if not values:
        raise ValueError('empty program')
    return values[-1]


def try_execute_program(program):
    try:
        return True, execute_program(program)
    except Exception as exc:
        return False, str(exc)


def extract_num(s):
    if s is None:
        return None
    text = str(s).replace(',', '')
    m = re.search(r'[-+]?\d*\.?\d+', text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def numeric_variants(value, text=None, include_scale=True):
    if value is None or not math.isfinite(float(value)):
        return []
    value = float(value)
    vals = {value}
    text = str(text or '')
    if '%' in text:
        vals.add(value / 100.0)
    if include_scale:
        for factor in SCALE_FACTORS:
            vals.add(value * factor)
    return sorted(vals, key=lambda x: (abs(x), x))


def close(a, b, tol):
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) / max(abs(float(b)), 1e-6) <= tol


def numeric_match_strict(pred, gold, tol):
    p = extract_num(pred)
    g = extract_num(gold)
    return close(p, g, tol)


def numeric_match_scaled(pred, gold, tol, pred_is_value=False):
    p = float(pred) if pred_is_value and pred is not None else extract_num(pred)
    g = extract_num(gold)
    if p is None or g is None:
        return False
    pred_vars = numeric_variants(p, str(pred), include_scale=True)
    gold_vars = numeric_variants(g, str(gold), include_scale=False)
    return any(close(pv, gv, tol) for pv in pred_vars for gv in gold_vars)


def best_scaled_value(value, gold, tol=0.05):
    g = extract_num(gold)
    if value is None or g is None:
        return None
    candidates = numeric_variants(value, '', include_scale=True)
    return min(candidates, key=lambda x: abs(x - g) / max(abs(g), 1e-6))


def format_number(x):
    if x is None:
        return ''
    x = float(x)
    if abs(x) >= 1000 and abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f'{x:.10g}'
