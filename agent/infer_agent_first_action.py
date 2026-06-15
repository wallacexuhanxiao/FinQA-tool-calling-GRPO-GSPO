import argparse, json
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def first_prompt(messages):
    out=[]
    for m in messages:
        if m.get('role')=='assistant':
            break
        out.append(m)
    return out

def first_gold(messages):
    for m in messages:
        if m.get('role')=='assistant':
            return m.get('content','')
    return ''

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model_path',required=True)
    ap.add_argument('--adapter_path',default='')
    ap.add_argument('--input_jsonl',required=True)
    ap.add_argument('--output_jsonl',required=True)
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--max_new_tokens',type=int,default=160)
    args=ap.parse_args()
    tok=AutoTokenizer.from_pretrained(args.model_path,trust_remote_code=True)
    model=AutoModelForCausalLM.from_pretrained(args.model_path,torch_dtype=torch.bfloat16,device_map='auto',trust_remote_code=True)
    if args.adapter_path:
        model=PeftModel.from_pretrained(model,args.adapter_path)
    model.eval()
    rows=[]
    with open(args.input_jsonl,encoding='utf-8') as f:
        for i,line in enumerate(f):
            if args.limit and i>=args.limit: break
            if line.strip(): rows.append(json.loads(line))
    Path(args.output_jsonl).parent.mkdir(parents=True,exist_ok=True)
    with open(args.output_jsonl,'w',encoding='utf-8') as out:
        for ex in tqdm(rows):
            msgs=ex['messages']
            prompt_msgs=first_prompt(msgs)
            text=tok.apply_chat_template(prompt_msgs,tokenize=False,add_generation_prompt=True)
            inputs=tok([text],return_tensors='pt').to(model.device)
            with torch.no_grad():
                gen=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=False,temperature=None,top_p=None)
            pred=tok.decode(gen[0][inputs.input_ids.shape[1]:],skip_special_tokens=True)
            rec={'raw_output':pred,'gold_output':first_gold(msgs),'prompt_messages':prompt_msgs,'meta':ex.get('meta',{})}
            out.write(json.dumps(rec,ensure_ascii=False)+'\n'); out.flush()
if __name__=='__main__': main()
