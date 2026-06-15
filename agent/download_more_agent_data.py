import os, json, traceback, random, math
from pathlib import Path
from datetime import datetime

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HOME', '/root/autodl-tmp/gui-grounding-agent/data/agent/cache/hf')

from huggingface_hub import snapshot_download, hf_hub_download
from datasets import load_dataset

ROOT=Path('/root/autodl-tmp/gui-grounding-agent')
RAW=ROOT/'data/agent/raw'
SAMPLES=ROOT/'data/agent/samples'
PROC=ROOT/'data/agent/processed'
LOG=ROOT/'logs/agent/download_more_agent_data.log'
for p in [RAW,SAMPLES,PROC,LOG.parent]: p.mkdir(parents=True, exist_ok=True)

def log(x):
    print(x, flush=True)
    with LOG.open('a',encoding='utf-8') as f: f.write(str(x)+'\n')

def write_jsonl(rows,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    n=0
    with path.open('w',encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r,ensure_ascii=False)+'\n'); n+=1
    return n

def sample_json_file(path, out, max_rows=1000):
    p=Path(path)
    try:
        obj=json.loads(p.read_text(encoding='utf-8'))
        rows=[]
        if isinstance(obj,list): rows=obj[:max_rows]
        elif isinstance(obj,dict):
            # flatten likely split/list containers
            for k,v in obj.items():
                if isinstance(v,list):
                    rows += [{'split_or_key':k, 'item':x} for x in v[:max_rows-len(rows)]]
                    if len(rows)>=max_rows: break
            if not rows: rows=[obj]
        n=write_jsonl(rows,out)
        return n
    except Exception as e:
        log(f'FAILED sample_json_file {path}: {e}')
        return 0

def snapshot_api_bank():
    repo='liminghao1630/API-Bank'
    out=RAW/'api_bank'
    try:
        snapshot_download(repo_id=repo, repo_type='dataset', local_dir=str(out), local_dir_use_symlinks=False, allow_patterns=['*.json','README.md'])
        files=[str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()]
        (out/'download_summary.json').write_text(json.dumps({'repo_id':repo,'files':files},ensure_ascii=False,indent=2),encoding='utf-8')
        previews=[]
        for jf in sorted(out.rglob('*.json')):
            n=sample_json_file(jf, SAMPLES/f'api_bank_{jf.stem}.sample1000.jsonl', max_rows=1000)
            previews.append({'file':str(jf.relative_to(out)),'sample_rows':n})
        log('API-Bank done '+json.dumps(previews,ensure_ascii=False))
        return True
    except Exception as e:
        log(f'FAILED API-Bank {type(e).__name__}: {e}')
        traceback.print_exc(); return False

def download_xlam_direct():
    repo='Salesforce/xlam-function-calling-60k'
    out=RAW/'xlam_function_calling_60k'
    out.mkdir(parents=True,exist_ok=True)
    try:
        f=hf_hub_download(repo_id=repo, repo_type='dataset', filename='xlam_function_calling_60k.json', local_dir=str(out), local_dir_use_symlinks=False)
        sample_json_file(f, SAMPLES/'xlam_function_calling_60k.sample2000.jsonl', max_rows=2000)
        (out/'download_summary.json').write_text(json.dumps({'repo_id':repo,'file':str(f)},ensure_ascii=False,indent=2),encoding='utf-8')
        log('xLAM direct done')
        return True
    except Exception as e:
        log(f'FAILED xLAM direct {type(e).__name__}: {e}')
        traceback.print_exc(); return False

def load_toolbench_sample():
    repo='tuandunghcmut/toolbench-v1'
    try:
        # The dataset card mentions config='benchmark'. Try benchmark then default.
        try:
            ds=load_dataset(repo, 'benchmark', trust_remote_code=True)
        except Exception as e:
            log(f'ToolBench benchmark config failed, try default: {e}')
            ds=load_dataset(repo, trust_remote_code=True)
        summary={}
        for split,d in ds.items():
            summary[split]={'num_rows':len(d),'columns':list(d.column_names)}
            rows=[]
            for i in range(min(len(d),3000)):
                row=d[i]
                keep={k:row[k] for k in row.keys() if k in ['id','conversations','messages','tools','query','answer']}
                rows.append(keep or row)
            n=write_jsonl(rows, RAW/'toolbench_v1'/f'{split}.sample3000.jsonl')
            write_jsonl(rows[:100], SAMPLES/f'toolbench_v1_{split}.sample100.jsonl')
            log(f'ToolBench {split}: {n}')
        (RAW/'toolbench_v1'/'dataset_summary.json').parent.mkdir(parents=True,exist_ok=True)
        (RAW/'toolbench_v1'/'dataset_summary.json').write_text(json.dumps({'repo_id':repo,'summary':summary},ensure_ascii=False,indent=2),encoding='utf-8')
        return True
    except Exception as e:
        log(f'FAILED ToolBench {type(e).__name__}: {e}')
        traceback.print_exc(); return False

SYSTEM=("You are a financial research agent. You can call tools: web_search, fetch_url, finance_api, and calculator. Return JSON only.")
FIN_DATA={
    'AAPL': {'name':'Apple', 'revenue': {'FY2022':394.3,'FY2023':383.3,'FY2024':391.0}, 'net_income': {'FY2022':99.8,'FY2023':97.0,'FY2024':93.7}, 'market_cap': {'latest':3000.0}},
    'MSFT': {'name':'Microsoft', 'revenue': {'FY2022':198.3,'FY2023':211.9,'FY2024':245.1}, 'net_income': {'FY2022':72.7,'FY2023':72.4,'FY2024':88.1}, 'market_cap': {'latest':3200.0}},
    'NVDA': {'name':'NVIDIA', 'revenue': {'FY2023':27.0,'FY2024':60.9,'FY2025':130.5}, 'net_income': {'FY2023':4.4,'FY2024':29.8,'FY2025':72.9}, 'market_cap': {'latest':3500.0}},
    'TSLA': {'name':'Tesla', 'revenue': {'FY2022':81.5,'FY2023':96.8,'FY2024':97.7}, 'net_income': {'FY2022':12.6,'FY2023':15.0,'FY2024':7.1}, 'market_cap': {'latest':900.0}},
    'AMZN': {'name':'Amazon', 'revenue': {'FY2022':514.0,'FY2023':574.8,'FY2024':638.0}, 'net_income': {'FY2022':-2.7,'FY2023':30.4,'FY2024':59.2}, 'market_cap': {'latest':2100.0}},
}

def tool_call(name,args):
    return json.dumps({'action':'tool_call','tool_call':{'name':name,'arguments':args}},ensure_ascii=False)
def final(answer,sources=None):
    return json.dumps({'action':'final','answer':answer,'sources':sources or [{'title':'Synthetic cached finance table','url':'local://data/agent/finance_synthetic'}]},ensure_ascii=False)
def obs(name,obj):
    return f'Tool observation from {name}:\n'+json.dumps(obj,ensure_ascii=False)

def make_finance_traces(n=2000):
    rows=[]; syms=list(FIN_DATA.keys())
    for idx in range(n):
        typ=idx%4
        sym=random.choice(syms); info=FIN_DATA[sym]
        periods=list(info['revenue'].keys())
        if typ==0:
            p1,p2=periods[0],periods[-1]
            v1=info['revenue'][p1]; v2=info['revenue'][p2]
            expr=f'({v2}-{v1})/{v1}*100'
            ans=(v2-v1)/v1*100
            q=f"What was {info['name']}'s revenue growth from {p1} to {p2}? Use finance_api and calculator."
            messages=[{'role':'system','content':SYSTEM},{'role':'user','content':q},
                {'role':'assistant','content':tool_call('finance_api',{'symbol':sym,'metric':'revenue','period':p1})},
                {'role':'user','content':obs('finance_api',{'symbol':sym,'metric':'revenue','period':p1,'value':v1,'unit':'billion USD','source':'synthetic cache'})},
                {'role':'assistant','content':tool_call('finance_api',{'symbol':sym,'metric':'revenue','period':p2})},
                {'role':'user','content':obs('finance_api',{'symbol':sym,'metric':'revenue','period':p2,'value':v2,'unit':'billion USD','source':'synthetic cache'})},
                {'role':'assistant','content':tool_call('calculator',{'expression':expr})},
                {'role':'user','content':obs('calculator',{'ok':True,'result':round(ans,4)})},
                {'role':'assistant','content':final(f"{info['name']}'s revenue growth from {p1} to {p2} was about {ans:.2f}%.")}]
        elif typ==1:
            p=periods[-1]; rev=info['revenue'][p]; ni=info['net_income'][p]
            ans=ni/rev*100; expr=f'{ni}/{rev}*100'
            q=f"Calculate {info['name']}'s net margin in {p}."
            messages=[{'role':'system','content':SYSTEM},{'role':'user','content':q},
                {'role':'assistant','content':tool_call('finance_api',{'symbol':sym,'metric':'revenue','period':p})},
                {'role':'user','content':obs('finance_api',{'symbol':sym,'metric':'revenue','period':p,'value':rev,'unit':'billion USD'})},
                {'role':'assistant','content':tool_call('finance_api',{'symbol':sym,'metric':'net_income','period':p})},
                {'role':'user','content':obs('finance_api',{'symbol':sym,'metric':'net_income','period':p,'value':ni,'unit':'billion USD'})},
                {'role':'assistant','content':tool_call('calculator',{'expression':expr})},
                {'role':'user','content':obs('calculator',{'ok':True,'result':round(ans,4)})},
                {'role':'assistant','content':final(f"{info['name']}'s net margin in {p} was about {ans:.2f}%.")}]
        elif typ==2:
            sym2=random.choice([s for s in syms if s!=sym]); v1=info['market_cap']['latest']; v2=FIN_DATA[sym2]['market_cap']['latest']; ans=v1/v2
            q=f"Compare {info['name']}'s latest market cap to {FIN_DATA[sym2]['name']}'s and calculate the ratio."
            messages=[{'role':'system','content':SYSTEM},{'role':'user','content':q},
                {'role':'assistant','content':tool_call('finance_api',{'symbol':sym,'metric':'market_cap','period':'latest'})},
                {'role':'user','content':obs('finance_api',{'symbol':sym,'metric':'market_cap','period':'latest','value':v1,'unit':'billion USD','timestamp':'synthetic'})},
                {'role':'assistant','content':tool_call('finance_api',{'symbol':sym2,'metric':'market_cap','period':'latest'})},
                {'role':'user','content':obs('finance_api',{'symbol':sym2,'metric':'market_cap','period':'latest','value':v2,'unit':'billion USD','timestamp':'synthetic'})},
                {'role':'assistant','content':tool_call('calculator',{'expression':f'{v1}/{v2}'})},
                {'role':'user','content':obs('calculator',{'ok':True,'result':round(ans,4)})},
                {'role':'assistant','content':final(f"{info['name']}'s market cap is about {ans:.2f}x {FIN_DATA[sym2]['name']}'s.")}]
        else:
            p1,p2=periods[0],periods[-1]; years=max(len(periods)-1,1); v1=info['revenue'][p1]; v2=info['revenue'][p2]
            ans=(v2/v1)**(1/years)-1; expr=f'(({v2}/{v1})**(1/{years})-1)*100'
            q=f"Compute {info['name']}'s revenue CAGR from {p1} to {p2}."
            messages=[{'role':'system','content':SYSTEM},{'role':'user','content':q},
                {'role':'assistant','content':tool_call('finance_api',{'symbol':sym,'metric':'revenue','period':p1})},
                {'role':'user','content':obs('finance_api',{'symbol':sym,'metric':'revenue','period':p1,'value':v1,'unit':'billion USD'})},
                {'role':'assistant','content':tool_call('finance_api',{'symbol':sym,'metric':'revenue','period':p2})},
                {'role':'user','content':obs('finance_api',{'symbol':sym,'metric':'revenue','period':p2,'value':v2,'unit':'billion USD'})},
                {'role':'assistant','content':tool_call('calculator',{'expression':expr})},
                {'role':'user','content':obs('calculator',{'ok':True,'result':round(ans*100,4)})},
                {'role':'assistant','content':final(f"{info['name']}'s revenue CAGR from {p1} to {p2} was about {ans*100:.2f}%.")}]
        rows.append({'messages':messages,'meta':{'source':'synthetic_finance_agent','task_type':typ,'gold_scaled_answer':messages[-2]['content']}})
    return rows

def make_finqa_agent_sample(limit=1000):
    src=Path('/root/autodl-tmp/gui-grounding-agent/external/LLaMA-Factory/data/finqa_toolcall_train_full.jsonl')
    if not src.exists():
        log('FinQA toolcall train not found, skip')
        return False
    rows=[]
    with src.open(encoding='utf-8') as f:
        for i,line in enumerate(f):
            if i>=limit: break
            ex=json.loads(line); msgs=ex.get('messages',[]); meta=ex.get('meta',{})
            user=next((m['content'] for m in msgs if m.get('role')=='user'), '')
            assistant=msgs[-1]['content'] if msgs else ''
            try:
                obj=json.loads(assistant); program=obj['tool_call']['arguments']['program']
            except Exception:
                program=meta.get('program_re','')
            answer=meta.get('answer','')
            agent_msgs=[{'role':'system','content':SYSTEM},{'role':'user','content':user},
                {'role':'assistant','content':tool_call('calculator',{'program':program})},
                {'role':'user','content':obs('calculator',{'ok':True,'result':answer})},
                {'role':'assistant','content':final(str(answer),[])}]
            rows.append({'messages':agent_msgs,'meta':{'source':'finqa_toolcall_agent','id':meta.get('id'),'gold_answer':answer,'gold_program':program}})
    write_jsonl(rows, PROC/'finqa_toolcall_agent_sft.sample1000.jsonl')
    write_jsonl(rows[:100], SAMPLES/'finqa_toolcall_agent_sft.sample100.jsonl')
    log(f'FinQA agent sample {len(rows)}')
    return True

def main():
    LOG.write_text('', encoding='utf-8')
    results={}
    results['api_bank']=snapshot_api_bank()
    results['xlam_direct']=download_xlam_direct()
    results['toolbench_v1']=load_toolbench_sample()
    finance=make_finance_traces(2000)
    write_jsonl(finance, PROC/'finance_api_calculator_synthetic_agent_sft.jsonl')
    write_jsonl(finance[:100], SAMPLES/'finance_api_calculator_synthetic_agent_sft.sample100.jsonl')
    results['finance_synthetic']=len(finance)
    results['finqa_agent_sample']=make_finqa_agent_sample(1000)
    (RAW/'download_more_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    log('RESULTS '+json.dumps(results,ensure_ascii=False))

if __name__=='__main__': main()
