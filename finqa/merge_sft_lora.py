import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_model', default='models/Qwen2.5-7B-Instruct')
    ap.add_argument('--adapter', default='saves/finqa/qwen25-7b-finqa-full-sft')
    ap.add_argument('--output_dir', default='models/Qwen2.5-7B-Instruct-FinQA-SFT-Merged')
    args = ap.parse_args()

    out = Path(args.output_dir)
    if (out / 'config.json').exists():
        print(f'merged model already exists: {out}')
        return

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map='auto',
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    print(f'saved merged model to {out}')


if __name__ == '__main__':
    main()
