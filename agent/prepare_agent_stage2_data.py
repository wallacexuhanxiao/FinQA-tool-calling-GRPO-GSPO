import ast
import json
import random
import re
from pathlib import Path

ROOT = Path('/root/autodl-tmp/gui-grounding-agent')
SYSTEM = (
    'You are a tool-use agent. Decide whether to call a tool or give a final answer. '
    'Return JSON only. For tool use, output {"action":"tool_call","tool_call":{"name":"...","arguments":{...}}}. '
    'For final answers, output {"action":"final","answer":"..."}.'
)

def literal(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        if isinstance(node, ast.Name):
            if node.id == 'true': return True
            if node.id == 'false': return False
            if node.id == 'null': return None
            return node.id
        return None

def parse_api_request(text):
    s = str(text or '').strip()
    m = re.search(r'API-Request:\s*\[(.*)\]\s*$', s, flags=re.S)
    if m:
        s = m.group(1).strip()
    if not s:
        return None
    name_m = re.match(r'([A-Za-z_][A-Za-z0-9_\.:-]*)\s*\((.*)\)\s*$', s, flags=re.S)
    if not name_m:
        return None
    name = name_m.group(1)
    args_s = name_m.group(2).strip()
    try:
        tree = ast.parse('f(' + args_s + ')', mode='eval')
        call = tree.body
        args = {}
        for i, arg in enumerate(call.args):
            args[f'arg{i}'] = literal(arg)
        for kw in call.keywords:
            if kw.arg is not None:
                args[kw.arg] = literal(kw.value)
        return name, args
    except Exception:
        return name, {'raw_arguments': args_s} if args_s else {}

def api_bank_row(obj, source):
    parsed = parse_api_request(obj.get('output', ''))
    if not parsed:
        return None
    name, args = parsed
    user = (
        'Use the available API descriptions and dialogue context to choose the next API call.\n\n'
        f"{obj.get('instruction','').strip()}\n\n{obj.get('input','').strip()}"
    )
    assistant = json.dumps({'action':'tool_call','tool_call':{'name':name,'arguments':args}}, ensure_ascii=False, separators=(',', ':'))
    return {
        'messages': [
            {'role':'system','content':SYSTEM},
            {'role':'user','content':user},
            {'role':'assistant','content':assistant},
        ],
        'meta': {'source':source, 'gold_tool':name, 'gold_args_json':json.dumps(args, ensure_ascii=False, sort_keys=True)}
    }

def read_jsonl(path):
    rows=[]
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

def main():
    random.seed(20260606)
    out_dir = ROOT/'data/agent/processed/stage2'
    api_files = [
        ('apibank_lv1', ROOT/'data/agent/samples/api_bank_lv1-train.sample1000.jsonl'),
        ('apibank_lv2', ROOT/'data/agent/samples/api_bank_lv2-train.sample1000.jsonl'),
        ('apibank_lv3', ROOT/'data/agent/samples/api_bank_lv3-train.sample1000.jsonl'),
    ]
    api_train=[]; api_dev=[]; dropped=0
    for source, path in api_files:
        converted=[]
        for obj in read_jsonl(path):
            row=api_bank_row(obj, source)
            if row: converted.append(row)
            else: dropped += 1
        random.shuffle(converted)
        api_dev.extend(converted[:100])
        api_train.extend(converted[100:])

    stage1 = read_jsonl(ROOT/'data/agent/processed/stage1/agent_stage1_train.jsonl')
    random.shuffle(stage1)
    replay_train = stage1[:1800]
    replay_dev = read_jsonl(ROOT/'data/agent/processed/stage1/agent_stage1_dev.jsonl')[:200]

    train = api_train + replay_train
    dev = api_dev + replay_dev
    random.shuffle(train); random.shuffle(dev)

    write_jsonl(out_dir/'agent_stage2_train.jsonl', train)
    write_jsonl(out_dir/'agent_stage2_dev.jsonl', dev)

    lf_dir = ROOT/'external/LLaMA-Factory/data'
    write_jsonl(lf_dir/'agent_stage2_train.jsonl', train)
    write_jsonl(lf_dir/'agent_stage2_dev.jsonl', dev)

    info_p = lf_dir/'dataset_info.json'
    info = json.loads(info_p.read_text(encoding='utf-8'))
    base = {
        'formatting': 'sharegpt',
        'columns': {'messages': 'messages'},
        'tags': {
            'role_tag': 'role', 'content_tag': 'content',
            'user_tag': 'user', 'assistant_tag': 'assistant', 'system_tag': 'system'
        }
    }
    for name, fn in [('agent_stage2_train','agent_stage2_train.jsonl'), ('agent_stage2_dev','agent_stage2_dev.jsonl')]:
        item = dict(base); item['file_name'] = fn; info[name] = item
    info_p.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')

    counts={}
    for r in train+dev:
        src=(r.get('meta') or {}).get('source','unknown')
        counts[src]=counts.get(src,0)+1
    summary={'train':len(train),'dev':len(dev),'dropped':dropped,'source_counts':counts}
    (out_dir/'stage2_data_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
