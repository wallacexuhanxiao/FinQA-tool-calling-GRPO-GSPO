import json
from pathlib import Path
from eval.eval_finqa_answer import extract_json, norm_text, numeric_match, extract_num

metrics_p = Path('outputs/finqa/metrics')
metric_names = (
    'base_dev_200_metrics.json',
    'sft_full_dev_200_metrics.json',
    'sft_full_dev_metrics.json',
    'dpo_full_dev_metrics.json',
)
print('metrics:')
for name in metric_names:
    p = metrics_p / name
    if p.exists():
        d = json.loads(p.read_text())
        keys = (
            'num_samples', 'json_valid_rate', 'program_nonempty_rate',
            'answer_exact_match', 'numeric_accuracy_at_1pct', 'numeric_accuracy_at_5pct'
        )
        print(name, {k: d.get(k) for k in keys})

def parse_pred(raw):
    obj = extract_json(raw)
    if isinstance(obj, dict):
        return str(obj.get('answer', '')), str(obj.get('program', '')), True
    return str(raw).strip().split('\n')[0], '', False

def read(p):
    rows = []
    for line in p.open(encoding='utf-8'):
        if not line.strip():
            continue
        ex = json.loads(line)
        ans, prog, j = parse_pred(ex.get('raw_output', ''))
        gold = str(ex.get('answer') or (ex.get('meta') or {}).get('answer', ''))
        q = ex.get('question') or (ex.get('meta') or {}).get('question', '')
        rows.append({
            **ex,
            'pred_answer': ans,
            'pred_program': prog,
            'json': j,
            'gold': gold,
            'q': q,
            'num5': numeric_match(ans, gold, 0.05),
            'num1': numeric_match(ans, gold, 0.01),
            'exact': norm_text(ans) == norm_text(gold),
        })
    return rows

sft = read(Path('outputs/finqa/predictions/sft_full_dev.jsonl'))
dpo = read(Path('outputs/finqa/predictions/dpo_full_dev.jsonl'))
print('rows', len(sft), len(dpo))
trans = {'both_ok': 0, 'sft_ok_dpo_bad': 0, 'sft_bad_dpo_ok': 0, 'both_bad': 0}
examples = {'sft_ok_dpo_bad': [], 'sft_bad_dpo_ok': [], 'both_bad': []}
for a, b in zip(sft, dpo):
    so = a['num5'] or a['exact']
    do = b['num5'] or b['exact']
    if so and do:
        key = 'both_ok'
    elif so and not do:
        key = 'sft_ok_dpo_bad'
    elif not so and do:
        key = 'sft_bad_dpo_ok'
    else:
        key = 'both_bad'
    trans[key] += 1
    if key in examples and len(examples[key]) < 8:
        examples[key].append({
            'q': a['q'],
            'gold': a['gold'],
            'sft': a['pred_answer'],
            'sft_prog': a['pred_program'][:160],
            'dpo': b['pred_answer'],
            'dpo_prog': b['pred_program'][:160],
        })
print('transition_num5_or_exact', trans)

cats = {k: 0 for k in ('gold_percent','question_percent','dpo_decimal_percent','dpo_difference_not_ratio','scale_large','blank')}
for a, b in zip(sft, dpo):
    so = a['num5'] or a['exact']
    do = b['num5'] or b['exact']
    if not (so and not do):
        continue
    q = a['q'].lower()
    g = a['gold']
    dp = b['pred_answer']
    if '%' in g:
        cats['gold_percent'] += 1
    if any(w in q for w in ('percent','percentage','rate','margin','growth')):
        cats['question_percent'] += 1
    gn = extract_num(g)
    dn = extract_num(dp)
    if '%' in g and dn is not None and gn is not None and abs(dn * 100 - gn) / max(abs(gn), 1e-6) <= 0.08:
        cats['dpo_decimal_percent'] += 1
    if any(w in q for w in ('percent','percentage','rate','growth')) and b['pred_program'].startswith('subtract('):
        cats['dpo_difference_not_ratio'] += 1
    if gn is not None and abs(gn) > 1e6 and dn is not None and abs(dn / gn) < 0.2:
        cats['scale_large'] += 1
    if not dp.strip():
        cats['blank'] += 1
print('sft_ok_dpo_bad_categories_nonexclusive', cats)

print('\nExamples SFT ok DPO bad:')
for e in examples['sft_ok_dpo_bad']:
    print(json.dumps(e, ensure_ascii=False))
print('\nExamples SFT bad DPO ok:')
for e in examples['sft_bad_dpo_ok']:
    print(json.dumps(e, ensure_ascii=False))

p = Path('data/finqa/processed/dpo/finqa_dpo_full.json')
if p.exists():
    pairs = json.loads(p.read_text())
    percent = 0
    rejected_json = 0
    rej_invalid = 0
    chosen_percent = 0
    for ex in pairs:
        txt = ' '.join(m['value'] for m in ex['conversations'])
        if any(w in txt.lower() for w in ('percent','percentage','rate','growth')) or '%' in txt:
            percent += 1
        ro = extract_json(ex['rejected']['value'])
        co = extract_json(ex['chosen']['value'])
        if isinstance(ro, dict):
            rejected_json += 1
        else:
            rej_invalid += 1
        if isinstance(co, dict) and '%' in str(co.get('answer', '')):
            chosen_percent += 1
    print('dpo_pairs', len(pairs), 'pair_percent_like', percent, 'chosen_percent', chosen_percent, 'rejected_json', rejected_json, 'rejected_invalid', rej_invalid)
