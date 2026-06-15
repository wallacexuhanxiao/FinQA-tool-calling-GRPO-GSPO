import argparse, json, re, statistics
from collections import defaultdict, Counter
from pathlib import Path

def load_corpus(path):
    by_chunk = {}
    by_sample_type = defaultdict(dict)
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            c=json.loads(line)
            by_chunk[c['chunk_id']]=c
            by_sample_type[(c['sample_id'], c['chunk_type'])][int(c.get('chunk_index',0))]=c
    return by_chunk, by_sample_type

def expand(c, by_sample_type, text_window=1, table_window=1):
    sid=c.get('sample_id'); typ=c.get('chunk_type'); idx=int(c.get('chunk_index',0))
    out=[]
    if typ=='table_row':
        h=by_sample_type.get((sid,'table_header'),{}).get(0)
        if h: out.append(h)
        rows=by_sample_type.get((sid,'table_row'),{})
        for j in range(idx-table_window, idx+table_window+1):
            if j in rows: out.append(rows[j])
    elif typ=='table_header':
        rows=by_sample_type.get((sid,'table_row'),{})
        for j in range(1, table_window+2):
            if j in rows: out.append(rows[j])
    elif typ in {'pre_text','post_text'}:
        rows=by_sample_type.get((sid,typ),{})
        for j in range(idx-text_window, idx+text_window+1):
            if j in rows: out.append(rows[j])
    return out

def ids_for_retrieved(r, by_chunk, by_sample_type, top_k, text_window, table_window, max_chunks):
    seen=[]; s=set(); primary=[]
    for rc in (r.get('retrieved') or [])[:top_k]:
        cid=rc.get('chunk_id')
        c=by_chunk.get(cid, rc)
        if cid not in s:
            seen.append(cid); s.add(cid); primary.append(cid)
        for ec in expand(c, by_sample_type, text_window, table_window):
            eid=ec['chunk_id']
            if eid not in s:
                seen.append(eid); s.add(eid)
        if len(seen)>=max_chunks:
            break
    return seen[:max_chunks], primary

def pct(x): return round(100*x,2)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--retrieval_jsonl', required=True)
    ap.add_argument('--corpus_jsonl', required=True)
    ap.add_argument('--out_json', required=True)
    ap.add_argument('--topks', default='5,10,15,20')
    ap.add_argument('--max_chunks', type=int, default=40)
    args=ap.parse_args()
    by_chunk, by_sample_type=load_corpus(args.corpus_jsonl)
    rows=[json.loads(x) for x in open(args.retrieval_jsonl,encoding='utf-8') if x.strip()]
    result={'retrieval_jsonl':args.retrieval_jsonl,'num_queries':len(rows),'settings':{}}
    for top_k in [int(x) for x in args.topks.split(',') if x.strip()]:
        for tw, tabw in [(0,0),(1,1),(1,2),(2,2)]:
            key=f'top{top_k}_text{tw}_table{tabw}'
            any_hit=all_hit=0; covs=[]; n_gold=[]; n_ev=[]; gold_type=Counter(); miss_type=Counter(); evidence_type=Counter()
            bad=[]
            for r in rows:
                gold=set(r.get('gold_chunk_ids') or [])
                ids, primary = ids_for_retrieved(r, by_chunk, by_sample_type, top_k, tw, tabw, args.max_chunks)
                got=set(ids)
                hit=gold & got
                if hit: any_hit+=1
                if gold and gold <= got: all_hit+=1
                covs.append(len(hit)/len(gold) if gold else 0)
                n_gold.append(len(gold)); n_ev.append(len(ids))
                for cid in gold:
                    typ=by_chunk.get(cid,{}).get('chunk_type','unknown'); gold_type[typ]+=1
                    if cid not in got: miss_type[typ]+=1
                for cid in ids:
                    evidence_type[by_chunk.get(cid,{}).get('chunk_type','unknown')]+=1
                if len(bad)<20 and gold and not (gold <= got):
                    bad.append({'sample_id':r.get('sample_id'),'question':r.get('question'),'gold':list(gold),'got_gold':list(hit),'missing':list(gold-got),'primary_top_ids':primary[:top_k], 'evidence_n':len(ids)})
            result['settings'][key]={
                'top_k':top_k,'text_window':tw,'table_window':tabw,'max_chunks':args.max_chunks,
                'any_recall':pct(any_hit/len(rows)),
                'all_recall':pct(all_hit/len(rows)),
                'evidence_coverage':pct(sum(covs)/len(covs)),
                'avg_gold_chunks':round(statistics.mean(n_gold),2),
                'avg_evidence_chunks':round(statistics.mean(n_ev),2),
                'p95_evidence_chunks':sorted(n_ev)[int(0.95*len(n_ev))-1],
                'gold_type_counts':dict(gold_type),
                'missing_type_counts':dict(miss_type),
                'evidence_type_counts':dict(evidence_type),
                'bad_preview':bad,
            }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result['settings'],ensure_ascii=False,indent=2)[:6000])
    print('saved',args.out_json)
if __name__=='__main__': main()
