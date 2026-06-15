import os
from huggingface_hub import snapshot_download
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
patterns = ['*.json', '*.model', '*.txt', '*.safetensors', 'pytorch_model.bin']
models = [
    ('BAAI/bge-reranker-v2-m3', 'models/BAAI/bge-reranker-v2-m3'),
    ('BAAI/bge-m3', 'models/BAAI/bge-m3'),
]
for repo, local in models:
    print(f'DOWNLOAD {repo} -> {local}', flush=True)
    snapshot_download(
        repo_id=repo,
        local_dir=local,
        allow_patterns=patterns,
        max_workers=2,
        resume_download=True,
    )
    print(f'DONE {repo}', flush=True)
