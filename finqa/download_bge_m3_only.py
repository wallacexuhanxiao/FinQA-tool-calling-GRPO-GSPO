import os
from huggingface_hub import snapshot_download
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
print('DOWNLOAD BAAI/bge-m3 -> models/BAAI/bge-m3', flush=True)
snapshot_download(
    repo_id='BAAI/bge-m3',
    local_dir='models/BAAI/bge-m3',
    allow_patterns=['*.json','*.model','*.txt','*.safetensors','pytorch_model.bin'],
    max_workers=2,
    resume_download=True,
)
print('DONE BAAI/bge-m3', flush=True)
