import argparse, json, re
from pathlib import Path
from collections import Counter
from jsonschema import validate

TOOLS_PATH=Path('/root/autodl-tmp/gui-grounding-agent/configs/agent/tool_schemas.json')

def extract_json(text):
    s=str(text or '').strip()
    s=re.sub(r'^```json\s*','',s); s=re.sub(r'^```\s*','',s); s=re.sub(r'\s*```$','',s)
    m=re.search(r'\{.*\}',s,flags=re.S)
    if m: s=m.group(0)
    try: return json.loads(s)
    except Exception: return None

def tool_name(obj):
    try: return obj.get('tool_call',{}).get('name','')
    except Exception: return ''

def norm_text(s):
    return re.sub(r'\s+', ' ', str(s or '').strip().lower())

def answer_text(obj):
    if not isinstance(obj, dict):
        return ''
    return str(obj.get('answer', ''))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pred_jsonl',required=True)
    ap.add_argument('--out_metrics',required=True)
    args=ap.parse_args()
    schemas={t['name']:t.get('arguments_schema',{}) for t in json.loads(TOOLS_PATH.read_text())}
    allowed=set(schemas)|{'api_call'}
    total=0; json_ok=0; action_ok=0; action_acc=0; tool_valid=0; tool_acc=0; schema_ok=0; no_extra=0
    gold_tool_turns=0; pred_tool_turns=0; final_turns=0; final_exact=0
    pred_tools=Counter(); gold_tools=Counter(); bad=[]
    with open(args.pred_jsonl,encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            total+=1; rec=json.loads(line)
            pred=rec.get('raw_output',''); gold=rec.get('gold_output','')
            po=extract_json(pred); go=extract_json(gold)
            if pred.strip().startswith('{') and pred.strip().endswith('}'):
                no_extra+=1
            gt=tool_name(go) if isinstance(go,dict) else ''
            gold_action=go.get('action') if isinstance(go,dict) else ''
            if gt:
                gold_tools[gt]+=1
            if gold_action == 'tool_call':
                gold_tool_turns += 1
            if gold_action == 'final':
                final_turns += 1
            if isinstance(po,dict):
                json_ok+=1
                pred_action=po.get('action')
                if pred_action in {'tool_call','final','clarify'}: action_ok+=1
                if pred_action == gold_action: action_acc += 1
                pt=tool_name(po)
                if pt: pred_tools[pt]+=1
                if po.get('action')=='tool_call':
                    pred_tool_turns += 1
                if po.get('action')=='tool_call' and pt in allowed: tool_valid+=1
                if pt and gt and pt==gt: tool_acc+=1
                if gold_action == 'final' and pred_action == 'final' and norm_text(answer_text(po)) == norm_text(answer_text(go)):
                    final_exact += 1
                if po.get('action')=='tool_call' and pt in schemas:
                    try:
                        validate(po.get('tool_call',{}).get('arguments',{}), schemas[pt]); schema_ok+=1
                    except Exception: pass
                elif po.get('action')=='tool_call' and pt=='api_call':
                    if isinstance(po.get('tool_call',{}).get('arguments',{}),dict): schema_ok+=1
            pt=tool_name(po) if isinstance(po,dict) else ''
            if len(bad)<30 and (not isinstance(po,dict) or (gt and pt!=gt)):
                bad.append({'gold_tool':gt,'pred_tool':pt,'gold':gold[:300],'pred':pred[:300]})
    metrics={
        'num_samples':total,
        'json_valid_rate':json_ok/total if total else 0,
        'no_extra_text_rate':no_extra/total if total else 0,
        'action_valid_rate':action_ok/total if total else 0,
        'action_accuracy':action_acc/total if total else 0,
        'tool_name_valid_rate':tool_valid/total if total else 0,
        'tool_name_accuracy':tool_acc/total if total else 0,
        'gold_tool_turns':gold_tool_turns,
        'pred_tool_turns':pred_tool_turns,
        'tool_name_accuracy_on_gold_tool_turns':tool_acc/gold_tool_turns if gold_tool_turns else None,
        'argument_schema_valid_rate':schema_ok/total if total else 0,
        'argument_schema_valid_rate_on_pred_tool_turns':schema_ok/pred_tool_turns if pred_tool_turns else None,
        'final_turns':final_turns,
        'final_exact_rate':final_exact/final_turns if final_turns else None,
        'pred_tool_counts':dict(pred_tools),
        'gold_tool_counts':dict(gold_tools),
        'bad_preview':bad,
    }
    Path(args.out_metrics).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out_metrics).write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(metrics,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
