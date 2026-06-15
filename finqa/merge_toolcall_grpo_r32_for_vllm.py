import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="models/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--adapter",
        default="saves/finqa/qwen25-7b-finqa-toolcall-grpo-r32-scaled-full",
    )
    parser.add_argument(
        "--output_dir",
        default="saves/finqa/qwen25-7b-finqa-toolcall-grpo-r32-scaled-full-merged",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    print(f"saved merged model to {out}")


if __name__ == "__main__":
    main()
