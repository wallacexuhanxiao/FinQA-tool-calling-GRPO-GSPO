import argparse,json,re
from pathlib import Path
from eval.eval_finqa_answer import extract_json, norm_text, numeric_match

def load_preds(path):
    out={}
    for line in open(path,encoding='utf-8'):
        if not line.strip(): continue
        ex=json.loads(line); sid=ex.get('id') or (ex.get('meta') or {}).get('id')
        out[sid]=ex
    return out

def get_pred_ans(ex):
    raw=ex.get('raw_output') or ex.get('prediction') or ''
    obj=extract_json(raw)
    if isinstance(obj,dict): return str(obj.get('answer','')), raw
    return str(raw).strip().split('\n')[0], raw

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pred_jsonl',required=True)
    ap.add_argument('--generation_jsonl',required=True)
    ap.add_argument('--out_json',required=True)
    args=ap.parse_args()
    preds=load_preds(args.pred_jsonl)
    cats={k:0 for k in ['total','correct_scaled5','wrong_scaled5','all_gold_in_evidence','missing_gold','wrong_but_all_gold','wrong_and_missing_gold','correct_with_missing_gold','no_gold_label']}
    examples={k:[] for k in cats}
    for line in open(args.generation_jsonl,encoding='utf-8'):
        if not line.strip(): continue
        g=json.loads(line); meta=g.get('meta') or {}; sid=meta.get('id')
        p=preds.get(sid)
        if not p: continue
        gold=str(meta.get('answer',''))
        pred,raw=get_pred_ans(p)
        ok=numeric_match(pred,gold,0.05) or norm_text(pred)==norm_text(gold)
        gold_ids=set(meta.get('gold_chunk_ids') or [])
        ev_ids=set(meta.get('evidence_chunk_ids') or meta.get('retrieved_chunk_ids') or [])
        all_in= bool(gold_ids) and gold_ids <= ev_ids
        cats['total']+=1
        cats['correct_scaled5' if ok else 'wrong_scaled5']+=1
        if not gold_ids:
            cats['no_gold_label']+=1; key='no_gold_label'
        elif all_in:
            cats['all_gold_in_evidence']+=1
            key='wrong_but_all_gold' if not ok else 'correct_scaled5'
            if not ok: cats['wrong_but_all_gold']+=1
        else:
            cats['missing_gold']+=1
            key='wrong_and_missing_gold' if not ok else 'correct_with_missing_gold'
            cats[key]+=1
        if len(examples.get(key,[]))<15:
            examples.setdefault(key,[]).append({'id':sid,'question':meta.get('question'),'gold':gold,'pred':pred,'missing_gold':list(gold_ids-ev_ids),'gold_n':len(gold_ids),'ev_n':len(ev_ids),'raw':raw[:400]})
    result={'counts':cats,'rates':{k:round(v/cats['total']*100,2) for k,v in cats.items() if k!='total'},'examples':examples}
    Path(args.out_json).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out_json).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'counts':cats,'rates':result['rates']},ensure_ascii=False,indent=2))
    print('saved',args.out_json)
if __name__=='__main__': main()
