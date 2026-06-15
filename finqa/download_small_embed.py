import os
from huggingface_hub import snapshot_download
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
repo='sentence-transformers/all-MiniLM-L6-v2'
local='models/sentence-transformers/all-MiniLM-L6-v2'
print(f'DOWNLOAD {repo} -> {local}', flush=True)
snapshot_download(repo_id=repo, local_dir=local, max_workers=4, resume_download=True)
print('DONE', flush=True)
