import argparse,json,statistics
from pathlib import Path
from transformers import AutoTokenizer

def pct(xs, q):
    if not xs: return None
    xs=sorted(xs); idx=min(len(xs)-1, max(0, int(q*len(xs))-1)); return xs[idx]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model_path',default='models/Qwen2.5-7B-Instruct'); ap.add_argument('--inputs',nargs='+',required=True); ap.add_argument('--out_json',required=True); args=ap.parse_args()
    tok=AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    res={}
    for spec in args.inputs:
        name,path=spec.split('=',1)
        lens=[]; chars=[]; rows=0
        for line in open(path,encoding='utf-8'):
            if not line.strip(): continue
            ex=json.loads(line); msgs=[m for m in ex['messages'] if m['role']!='assistant']
            text=tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids=tok(text, add_special_tokens=False).input_ids
            lens.append(len(ids)); chars.append(len(text)); rows+=1
        res[name]={'path':path,'n':rows,'tokens_avg':round(statistics.mean(lens),1),'tokens_p50':pct(lens,.50),'tokens_p90':pct(lens,.90),'tokens_p95':pct(lens,.95),'tokens_p99':pct(lens,.99),'tokens_max':max(lens),'chars_avg':round(statistics.mean(chars),1)}
    Path(args.out_json).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out_json).write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(res,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
