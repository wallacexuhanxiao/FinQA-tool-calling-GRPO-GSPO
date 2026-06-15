import json
import re
from collections import Counter, defaultdict
from pathlib import Path


PATHS = [
    "data/finqa/processed/sft/train.jsonl",
    "data/finqa/processed/sft/dev.jsonl",
    "data/finqa/processed/sft/test.jsonl",
]


def main():
    ops = Counter()
    examples = defaultdict(list)

    for path in PATHS:
        p = Path(path)
        if not p.exists():
            print("missing:", path)
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                ex = json.loads(line)
                prog = str((ex.get("meta") or {}).get("program_re", ""))
                for op in re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", prog):
                    ops[op] += 1
                    if len(examples[op]) < 3:
                        examples[op].append(prog)

    print("==== OP COUNTS ====")
    for op, count in ops.most_common():
        print(op, count)
        for example in examples[op]:
            print("  ", example)


if __name__ == "__main__":
    main()
