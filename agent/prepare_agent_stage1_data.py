import json, random
from pathlib import Path

ROOT = Path('/root/autodl-tmp/gui-grounding-agent')
SRC_FILES = [
    ROOT/'data/agent/processed/gorilla_apibench_hf_train_agent_sft.sample2000.jsonl',
    ROOT/'data/agent/processed/finance_api_calculator_synthetic_agent_sft.jsonl',
    ROOT/'data/agent/processed/finqa_toolcall_agent_sft.sample1000.jsonl',
]
OUT_DIR = ROOT/'data/agent/processed/stage1'
LF_DIR = ROOT/'external/LLaMA-Factory/data'
random.seed(42)

def read_jsonl(p):
    rows=[]
    with open(p,encoding='utf-8') as f:
        for line in f:
            if line.strip():
                row=json.loads(line)
                row.setdefault('meta',{})['source_file']=p.name
                rows.append(row)
    return rows

def write_jsonl(rows,p):
    p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,'w',encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(p, len(rows))

def main():
    all_rows=[]
    for p in SRC_FILES:
        rows=read_jsonl(p)
        print('read',p,len(rows))
        all_rows.extend(rows)
    random.shuffle(all_rows)
    n_dev=max(500, int(len(all_rows)*0.1))
    dev=all_rows[:n_dev]
    train=all_rows[n_dev:]
    write_jsonl(train, OUT_DIR/'agent_stage1_train.jsonl')
    write_jsonl(dev, OUT_DIR/'agent_stage1_dev.jsonl')
    write_jsonl(train, LF_DIR/'agent_stage1_train.jsonl')
    write_jsonl(dev, LF_DIR/'agent_stage1_dev.jsonl')
    info_path=LF_DIR/'dataset_info.json'
    info=json.loads(info_path.read_text(encoding='utf-8'))
    base={
        'formatting':'sharegpt',
        'columns':{'messages':'messages'},
        'tags':{
            'role_tag':'role','content_tag':'content',
            'user_tag':'user','assistant_tag':'assistant','system_tag':'system'
        }
    }
    for name,file_name in [('agent_stage1_train','agent_stage1_train.jsonl'),('agent_stage1_dev','agent_stage1_dev.jsonl')]:
        item=dict(base); item['file_name']=file_name; info[name]=item
    info_path.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print('registered datasets')

if __name__=='__main__': main()
