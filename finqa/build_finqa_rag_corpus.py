import argparse
import json
import re
from pathlib import Path
from datasets import load_dataset, load_from_disk


def norm_text(s):
    s = str(s or '').lower()
    # FinQA text often uses 2019 as a broken apostrophe marker; remove it only when attached to letters.
    s = re.sub(r'(?<=[a-z])2019(?=[a-z])', "'", s)
    s = re.sub(r'[^a-z0-9.%-]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def token_set(s):
    return set(re.findall(r'[a-z]+|\d+(?:\.\d+)?%?', norm_text(s)))


def numbers(s):
    vals = []
    for x in re.findall(r'[-+]?\$?\s*\d+(?:\.\d+)?%?', str(s or '').replace(',', '')):
        x = x.replace('$', '').replace(' ', '')
        if x:
            vals.append(x)
    return vals


def text_match(gold, candidates):
    ng = norm_text(gold)
    if not ng:
        return False
    for cand in candidates:
        nc = norm_text(cand)
        if not nc:
            continue
        if ng == nc:
            return True
        if len(ng) >= 20 and ng in nc:
            return True
        if len(nc) >= 20 and nc in ng:
            return True
    return False


def table_fuzzy_match(gold, row, header, rendered):
    gold_norm = norm_text(gold)
    if not gold_norm:
        return False
    row = [str(x) for x in row]
    header = [str(x) for x in header]
    label = row[0] if row else ''
    label_tokens = {t for t in token_set(label) if not re.fullmatch(r'\d+(?:\.\d+)?%?', t)}
    rendered_tokens = token_set(rendered)
    gold_tokens = token_set(gold)

    # Direct text containment still catches text-like gold evidence.
    if text_match(gold, [rendered, ' | '.join(row)]):
        return True

    row_nums = numbers(' '.join(row))
    matched_nums = sum(1 for x in row_nums if x in gold_norm)

    # Most official FinQA table gold strings preserve the row label and at least one value.
    if label_tokens:
        label_overlap = len(label_tokens & gold_tokens) / max(len(label_tokens), 1)
        if label_overlap >= 0.6 and matched_nums >= 1:
            return True

    # Some two-column tables have useful information in the first row/header-like row.
    if matched_nums >= 2 and len(rendered_tokens & gold_tokens) >= 3:
        return True

    # Header/column names plus values can identify rows where labels are short, e.g. "total".
    header_tokens = {t for t in token_set(' '.join(header)) if not re.fullmatch(r'\d+(?:\.\d+)?%?', t)}
    if matched_nums >= 1 and len((header_tokens | label_tokens) & gold_tokens) >= 2:
        return True

    return False


def is_match(gold, candidates):
    return text_match(gold, candidates)

def row_to_pairs(row, header):
    row = [str(x) for x in row]
    header = [str(x) for x in header]
    if not header:
        return ' | '.join(row)
    cells = []
    label = row[0] if row else ''
    if label:
        cells.append(f'metric: {label}')
    for i, val in enumerate(row[1:], start=1):
        key = header[i] if i < len(header) else f'col_{i}'
        cells.append(f'{key}: {val}')
    return ' ; '.join(cells)


def make_chunks(ex, split):
    sid = ex['id']
    gold_inds = ex.get('gold_inds') or []
    chunks = []

    def add(chunk_type, idx, text, source_text, extra=None, match_fn=None):
        candidates = [source_text, text]
        if extra:
            candidates.extend(extra)
        if match_fn is None:
            matched = [g for g in gold_inds if is_match(g, candidates)]
        else:
            matched = [g for g in gold_inds if match_fn(g)]
        chunks.append({
            'chunk_id': f'{sid}::{chunk_type}::{idx}',
            'sample_id': sid,
            'split': split,
            'chunk_type': chunk_type,
            'chunk_index': idx,
            'text': text,
            'source_text': source_text,
            'is_gold': bool(matched),
            'matched_gold_inds': matched,
        })

    for i, line in enumerate(ex.get('pre_text') or []):
        add('pre_text', i, f'[pre_text] {line}', line)

    table = ex.get('table') or []
    header = [str(x) for x in table[0]] if table else []
    for i, row in enumerate(table):
        row_join = ' | '.join(map(str, row))
        pair_text = row_to_pairs(row, header)
        if i == 0:
            text = f'[table_header] {row_join}'
            ctype = 'table_header'
        else:
            text = f'[table_row] {pair_text}'
            ctype = 'table_row'
        add(ctype, i, text, row_join, extra=[pair_text], match_fn=lambda g, row=row, header=header, text=text: table_fuzzy_match(g, row, header, text))

    for i, line in enumerate(ex.get('post_text') or []):
        add('post_text', i, f'[post_text] {line}', line)

    return chunks


def load_finqa(raw_dir):
    p = Path(raw_dir)
    if p.exists():
        return load_from_disk(str(p))
    return load_dataset('ibm-research/finqa', trust_remote_code=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_dir', default='data/finqa/raw/hf_finqa')
    ap.add_argument('--out_dir', default='data/finqa/processed/rag')
    args = ap.parse_args()
    ds = load_finqa(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    split_map = {'train': 'train', 'validation': 'dev', 'test': 'test'}
    for hf_split, out_split in split_map.items():
        corpus_path = out_dir / f'corpus_{out_split}.jsonl'
        query_path = out_dir / f'queries_{out_split}.jsonl'
        total_chunks = gold_chunks = total_gold_inds = covered_gold_inds = 0
        with corpus_path.open('w', encoding='utf-8') as cf, query_path.open('w', encoding='utf-8') as qf:
            for ex in ds[hf_split]:
                chunks = make_chunks(ex, out_split)
                gold_chunk_ids = [c['chunk_id'] for c in chunks if c['is_gold']]
                matched_gold = set()
                for c in chunks:
                    cf.write(json.dumps(c, ensure_ascii=False) + '\n')
                    total_chunks += 1
                    gold_chunks += int(c['is_gold'])
                    matched_gold.update(c['matched_gold_inds'])
                gold_inds = ex.get('gold_inds') or []
                total_gold_inds += len(gold_inds)
                covered_gold_inds += len(matched_gold)
                qf.write(json.dumps({
                    'id': ex['id'],
                    'sample_id': ex['id'],
                    'split': out_split,
                    'question': ex['question'],
                    'answer': str(ex.get('final_result') or ex.get('answer') or ''),
                    'program_re': str(ex.get('program_re') or ''),
                    'gold_inds': gold_inds,
                    'gold_chunk_ids': gold_chunk_ids,
                }, ensure_ascii=False) + '\n')
        summary[out_split] = {
            'samples': len(ds[hf_split]),
            'chunks': total_chunks,
            'gold_chunks': gold_chunks,
            'total_gold_inds': total_gold_inds,
            'covered_gold_inds': covered_gold_inds,
            'gold_ind_coverage_by_chunks': covered_gold_inds / total_gold_inds if total_gold_inds else 0,
            'corpus_path': str(corpus_path),
            'query_path': str(query_path),
        }
    (out_dir / 'build_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
