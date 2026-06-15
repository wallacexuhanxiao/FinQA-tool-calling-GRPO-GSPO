import json
import random
from pathlib import Path

ROOT = Path('/root/autodl-tmp/gui-grounding-agent')
SYSTEM = (
    'You are a multi-step tool-use planning agent. Given a task context and candidate tools, '
    'choose the next tool to call. Return JSON only: '
    '{"action":"tool_call","tool_call":{"name":"...","arguments":{}}}.'
)

BFCL_FILES = [
    ROOT/'data/agent/raw/bfcl/BFCL_v3_multi_turn_base.json',
    ROOT/'data/agent/raw/bfcl/BFCL_v3_multi_turn_composite.json',
    ROOT/'data/agent/raw/bfcl/BFCL_v3_multi_turn_long_context.json',
    ROOT/'data/agent/raw/bfcl/BFCL_v3_multi_turn_miss_func.json',
    ROOT/'data/agent/raw/bfcl/BFCL_v3_multi_turn_miss_param.json',
]

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

def questions_text(qs):
    lines=[]
    for i, turn in enumerate(qs, 1):
        parts=[]
        for m in turn:
            role=m.get('role','user')
            content=str(m.get('content','')).strip()
            if content:
                parts.append(f'{role}: {content}')
        if parts:
            lines.append(f'Turn {i}: ' + ' '.join(parts))
    return '\n'.join(lines)

def bfcl_rows():
    all_paths=[]; raw=[]
    for path in BFCL_FILES:
        for obj in read_jsonl(path):
            paths=[str(x) for x in (obj.get('path') or []) if x]
            if not paths: continue
            all_paths.extend(paths)
            raw.append((path.stem, obj, paths))
    pool=sorted(set(all_paths))
    rng=random.Random(20260606)
    rows=[]
    for source, obj, paths in raw:
        task=questions_text(obj.get('question') or [])
        involved=', '.join(obj.get('involved_classes') or [])
        unique_path=list(dict.fromkeys(paths))
        distractors=[x for x in pool if x not in unique_path]
        rng.shuffle(distractors)
        candidates=unique_path + distractors[:8]
        rng.shuffle(candidates)
        cand_text='\n'.join(f'- {x}' for x in candidates)
        for step_idx, tool in enumerate(paths):
            prev='\n'.join(f'{j+1}. {p}' for j,p in enumerate(paths[:step_idx])) or 'None'
            user=(
                'Choose the next tool call for this BFCL multi-turn task.\n\n'
                f'Sample id: {obj.get("id", "")}\n'
                f'Involved classes: {involved}\n\n'
                f'Task conversation:\n{task}\n\n'
                f'Previous selected tools:\n{prev}\n\n'
                f'Candidate tools:\n{cand_text}\n\n'
                'Return the next tool call JSON. Use an empty arguments object if parameters are not specified.'
            )
            assistant=json.dumps({'action':'tool_call','tool_call':{'name':tool,'arguments':{}}}, ensure_ascii=False, separators=(',', ':'))
            rows.append({
                'messages': [
                    {'role':'system','content':SYSTEM},
                    {'role':'user','content':user},
                    {'role':'assistant','content':assistant},
                ],
                'meta': {'source':'bfcl_multiturn_route','bfcl_file':source,'bfcl_id':str(obj.get('id','')),'gold_tool':tool,'step_idx':str(step_idx)}
            })
    return rows

def main():
    rng=random.Random(20260606)
    out_dir=ROOT/'data/agent/processed/stage3'
    route=bfcl_rows()
    rng.shuffle(route)
    route_dev=route[:600]
    route_train=route[600:]

    replay=[]
    for p, n in [
        (ROOT/'data/agent/processed/stage2/agent_stage2_train.jsonl', 1800),
        (ROOT/'data/agent/processed/stage1/agent_stage1_train.jsonl', 1000),
    ]:
        rows=read_jsonl(p); rng.shuffle(rows); replay.extend(rows[:n])

    train=route_train + replay
    dev=route_dev
    rng.shuffle(train)

    write_jsonl(out_dir/'agent_stage3_train.jsonl', train)
    write_jsonl(out_dir/'agent_stage3_bfcl_route_dev.jsonl', dev)

    lf_dir=ROOT/'external/LLaMA-Factory/data'
    write_jsonl(lf_dir/'agent_stage3_train.jsonl', train)
    write_jsonl(lf_dir/'agent_stage3_bfcl_route_dev.jsonl', dev)

    info_p=lf_dir/'dataset_info.json'
    info=json.loads(info_p.read_text(encoding='utf-8'))
    base={
        'formatting':'sharegpt',
        'columns': {'messages':'messages'},
        'tags': {'role_tag':'role','content_tag':'content','user_tag':'user','assistant_tag':'assistant','system_tag':'system'}
    }
    for name, fn in [('agent_stage3_train','agent_stage3_train.jsonl'),('agent_stage3_bfcl_route_dev','agent_stage3_bfcl_route_dev.jsonl')]:
        item=dict(base); item['file_name']=fn; info[name]=item
    info_p.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')

    counts={}
    for r in train+dev:
        src=(r.get('meta') or {}).get('source','unknown')
        counts[src]=counts.get(src,0)+1
    summary={'route_total':len(route),'route_train':len(route_train),'route_dev':len(route_dev),'train_total':len(train),'replay_total':len(replay),'source_counts':counts}
    (out_dir/'stage3_data_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
