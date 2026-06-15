import argparse
import json
import re
from pathlib import Path

SYSTEM = (
    "You are a tool-using agent. Decide whether to call a tool or provide a final answer. "
    "Return JSON only. Valid actions are tool_call, final, and clarify."
)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def extract_tag(text, tag):
    m = re.search(rf"<<<{re.escape(tag)}>>>\s*:\s*(.*?)(?=\n<<<[a-zA-Z_]+>>>:|\Z)", text, re.S)
    return m.group(1).strip() if m else ""


def guess_query(row):
    messages = row.get("messages") or []
    for msg in messages:
        if msg.get("role") == "user" and msg.get("content"):
            return str(msg["content"]).strip()
    for key in ["question", "query", "instruction", "prompt", "user"]:
        if row.get(key):
            return str(row[key]).strip()
    return json.dumps(row, ensure_ascii=False)[:1000]


def guess_api_call(row):
    for key in ["api_call", "answer", "output", "function_call", "call"]:
        if row.get(key):
            return str(row[key]).strip()

    messages = row.get("messages") or []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("content"):
            call = extract_tag(str(msg["content"]), "api_call")
            if call:
                return call

    tools = row.get("tools") or []
    for tool in tools:
        try:
            obj = json.loads(tool) if isinstance(tool, str) else tool
        except Exception:
            obj = {}
        if obj.get("api_call"):
            return str(obj["api_call"]).strip()
    return ""


def convert_one(row):
    q = guess_query(row)
    call = guess_api_call(row)
    if not call:
        return None
    assistant = {
        "action": "tool_call",
        "tool_call": {
            "name": "api_call",
            "arguments": {"call": call},
        },
    }
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": q},
            {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False)},
        ],
        "meta": {"source": "gorilla_apibench", "conversation_id": row.get("conversation_id")},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped = 0
    with open(args.output_jsonl, "w", encoding="utf-8") as out:
        for row in read_jsonl(args.input_jsonl):
            if args.limit and kept >= args.limit:
                break
            ex = convert_one(row)
            if ex is None:
                skipped += 1
                continue
            out.write(json.dumps(ex, ensure_ascii=False) + "\n")
            kept += 1
    print("wrote", kept, args.output_jsonl, "skipped", skipped)


if __name__ == "__main__":
    main()
