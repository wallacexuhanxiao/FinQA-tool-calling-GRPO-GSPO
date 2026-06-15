import argparse,json,re
from pathlib import Path
from collections import defaultdict
SYSTEM=("You are a financial reasoning assistant. Solve numerical reasoning questions over gold financial evidence. Return JSON only with keys: program and answer.")

def load_sft_meta(path):
    by_id={}
    for line in open(path,encoding='utf-8'):
        if not line.strip(): continue
        ex=json.loads(line); meta=ex.get('meta') or {}; sid=meta.get('id')
        if sid: by_id[sid]=meta
    return by_id

def load_corpus(path):
    by_chunk={}; by_sample_type=defaultdict(dict)
    for line in open(path,encoding='utf-8'):
        if not line.strip(): continue
        c=json.loads(line); by_chunk[c['chunk_id']]=c; by_sample_type[(c['sample_id'],c['chunk_type'])][int(c.get('chunk_index',0))]=c
    return by_chunk, by_sample_type

def add(out, seen, c, reason):
    if not c: return
    cid=c.get('chunk_id')
    if cid in seen: return
    x=dict(c); x['oracle_reason']=reason; out.append(x); seen.add(cid)

def expand(c, by_sample_type, text_window=1, table_window=1):
    sid=c.get('sample_id'); typ=c.get('chunk_type'); idx=int(c.get('chunk_index',0)); out=[]
    if typ=='table_row':
        h=by_sample_type.get((sid,'table_header'),{}).get(0)
        if h: out.append((h,'table_header'))
        rows=by_sample_type.get((sid,'table_row'),{})
        for j in range(idx-table_window, idx+table_window+1):
            if j in rows: out.append((rows[j],'neighbor_table_row' if j!=idx else 'gold_table_row'))
    elif typ=='table_header':
        rows=by_sample_type.get((sid,'table_row'),{})
        for j in range(1, table_window+2):
            if j in rows: out.append((rows[j],'header_following_row'))
    elif typ in {'pre_text','post_text'}:
        rows=by_sample_type.get((sid,typ),{})
        for j in range(idx-text_window, idx+text_window+1):
            if j in rows: out.append((rows[j],'neighbor_text' if j!=idx else 'gold_text'))
    return out

def fmt(c,i):
    txt=re.sub(r'\s+',' ',c.get('text','')).strip()
    return f"[{i}] ({c.get('oracle_reason','gold')}; {c.get('chunk_type')}) {txt}"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--retrieval_jsonl',required=True); ap.add_argument('--sft_jsonl',required=True); ap.add_argument('--corpus_jsonl',required=True); ap.add_argument('--output_jsonl',required=True); ap.add_argument('--text_window',type=int,default=1); ap.add_argument('--table_window',type=int,default=1); ap.add_argument('--max_evidence_chunks',type=int,default=40); args=ap.parse_args()
    meta_by_id=load_sft_meta(args.sft_jsonl); by_chunk,by_sample_type=load_corpus(args.corpus_jsonl)
    Path(args.output_jsonl).parent.mkdir(parents=True,exist_ok=True)
    n=0; total_ev=0; missing_gold=0
    with open(args.output_jsonl,'w',encoding='utf-8') as out:
        for line in open(args.retrieval_jsonl,encoding='utf-8'):
            if not line.strip(): continue
            r=json.loads(line); sid=r.get('sample_id') or r.get('id'); meta=dict(meta_by_id.get(sid,{})); question=r.get('question') or meta.get('question')
            ev=[]; seen=set()
            for gid in r.get('gold_chunk_ids') or []:
                c=by_chunk.get(gid)
                if not c: missing_gold+=1; continue
                add(ev,seen,c,'gold')
                for ec,reason in expand(c,by_sample_type,args.text_window,args.table_window): add(ev,seen,ec,reason)
            ev=ev[:args.max_evidence_chunks]; total_ev+=len(ev)
            evidence='\n'.join(fmt(c,i+1) for i,c in enumerate(ev))
            user=("Given the gold financial evidence and question, generate a short executable reasoning program and final answer. Use only the evidence.\n\nReturn JSON only:\n{\"program\": \"...\", \"answer\": \"...\"}\n\n"+f"Gold and expanded financial evidence:\n{evidence}\n\nQuestion:\n{question}")
            row={'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':user}], 'meta':{**meta,'id':sid,'question':question,'answer':meta.get('answer') or r.get('answer'),'program_re':meta.get('program_re') or r.get('program_re'),'oracle_gold_evidence':True,'gold_chunk_ids':r.get('gold_chunk_ids') or [],'evidence_chunk_ids':[c.get('chunk_id') for c in ev]}}
            out.write(json.dumps(row,ensure_ascii=False)+'\n'); n+=1
    print(json.dumps({'written':n,'avg_evidence_chunks':total_ev/n if n else 0,'missing_gold_refs':missing_gold,'output':args.output_jsonl},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
