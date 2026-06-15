from pathlib import Path
import json, os, sys, traceback
from itertools import islice

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HOME', '/root/autodl-tmp/gui-grounding-agent/data/agent/cache/hf')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')

from huggingface_hub import snapshot_download
from datasets import load_dataset

ROOT = Path('/root/autodl-tmp/gui-grounding-agent')
RAW = ROOT / 'data/agent/raw'
SAMPLES = ROOT / 'data/agent/samples'
PROCESSED = ROOT / 'data/agent/processed'
LOG = ROOT / 'logs/agent/download_agent_datasets.log'
for p in [RAW, SAMPLES, PROCESSED, LOG.parent]:
    p.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(msg, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(str(msg) + '\n')


def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            n += 1
    return n


def simplify_messages(row):
    # Normalize common chat/tool rows into a light inspection format.
    out = {}
    for key in ['conversation_id', 'id', 'system', 'tools', 'messages', 'query', 'question', 'answer']:
        if key in row:
            out[key] = row[key]
    return out or row


def sample_dataset(repo_id, out_name, max_rows=5000):
    log(f'== load_dataset {repo_id} ==')
    try:
        ds = load_dataset(repo_id, trust_remote_code=True)
    except Exception as e:
        log(f'FAILED load_dataset {repo_id}: {type(e).__name__}: {e}')
        traceback.print_exc()
        return False
    info = {}
    for split, d in ds.items():
        info[split] = {'num_rows': len(d), 'columns': list(d.column_names)}
        rows = [simplify_messages(d[i]) for i in range(min(len(d), max_rows))]
        n = write_jsonl(rows, RAW / out_name / f'{split}.sample{max_rows}.jsonl')
        n2 = write_jsonl(rows[:100], SAMPLES / f'{out_name}_{split}.sample100.jsonl')
        log(f'wrote {repo_id} split={split}: sample_rows={n}, preview_rows={n2}')
    (RAW / out_name / 'dataset_summary.json').write_text(json.dumps({'repo_id': repo_id, 'info': info}, ensure_ascii=False, indent=2), encoding='utf-8')
    return True


def download_bfcl():
    repo_id = 'gorilla-llm/Berkeley-Function-Calling-Leaderboard'
    out = RAW / 'bfcl'
    log(f'== snapshot_download {repo_id} ==')
    try:
        path = snapshot_download(
            repo_id=repo_id,
            repo_type='dataset',
            local_dir=str(out),
            local_dir_use_symlinks=False,
            allow_patterns=['*.json', '*.jsonl', 'README.md', '*.yaml'],
        )
        files = sorted(str(p.relative_to(out)) for p in out.rglob('*') if p.is_file())
        (out / 'download_summary.json').write_text(json.dumps({'repo_id': repo_id, 'files': files}, ensure_ascii=False, indent=2), encoding='utf-8')
        # Build a small preview across BFCL json files.
        preview = []
        for jf in sorted(out.glob('*.json')):
            try:
                obj = json.loads(jf.read_text(encoding='utf-8'))
                if isinstance(obj, list):
                    for item in obj[:20]:
                        preview.append({'source_file': jf.name, 'item': item})
                elif isinstance(obj, dict):
                    preview.append({'source_file': jf.name, 'item': obj})
            except Exception as e:
                preview.append({'source_file': jf.name, 'error': str(e)})
        write_jsonl(preview[:300], SAMPLES / 'bfcl_mixed.sample300.jsonl')
        log(f'BFCL files={len(files)}, preview={len(preview[:300])}')
        return True
    except Exception as e:
        log(f'FAILED BFCL: {type(e).__name__}: {e}')
        traceback.print_exc()
        return False


def main():
    LOG.write_text('', encoding='utf-8')
    results = {}
    results['bfcl'] = download_bfcl()
    results['gorilla_apibench'] = sample_dataset('Post-training-Data-Flywheel/gorilla-apibench', 'gorilla_apibench', max_rows=5000)
    # These are useful but optional. If mirror/network is slow or schema changes, keep failure log and continue.
    results['xlam_function_calling_60k'] = sample_dataset('Post-training-Data-Flywheel/Salesforce-xlam-function-calling-60k', 'xlam_function_calling_60k', max_rows=5000)
    results['gorilla_openfunctions_v1'] = sample_dataset('Post-training-Data-Flywheel/gorilla-openfunctions-v1', 'gorilla_openfunctions_v1', max_rows=5000)
    (RAW / 'download_results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    log('RESULTS ' + json.dumps(results, ensure_ascii=False))

if __name__ == '__main__':
    main()
