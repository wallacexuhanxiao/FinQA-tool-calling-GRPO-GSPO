import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from scripts.agent.langchain_tool_agent import build_tools, extract_json


MODEL_NAME = "qwen-agent"
VLLM_CHAT_URL = "http://127.0.0.1:8000/v1/chat/completions"


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = None
    max_turns: int = 8


class AgentController:
    def __init__(self):
        self.tools = build_tools()

    def tool_schema_text(self) -> str:
        chunks = []
        for name, tool in self.tools.items():
            schema = tool.args_schema.model_json_schema() if tool.args_schema else {}
            chunks.append(
                f"- {name}: {tool.description}\n"
                f"  args_schema: {json.dumps(schema.get('properties', {}), ensure_ascii=False)}"
            )
        return "\n".join(chunks)

    def system_prompt(self) -> str:
        return (
            "You are a production API-planning financial agent. Return JSON only.\n"
            "Valid actions:\n"
            '{"action":"tool_call","tool_call":{"name":"calculator|finance_api|market_data|web_search","arguments":{...}}}\n'
            '{"action":"final","answer":"..."}\n\n'
            "Available tools:\n"
            f"{self.tool_schema_text()}\n\n"
            "Plan step by step. Call one tool at a time. Prefer market_data for current stock market capitalization questions. "
            "Use web_search for open-ended web questions, and rely on rag_context/reranked_results from search observations. "
            "Use calculator for arithmetic after collecting numeric observations. If the user asks for a difference without specifying order, report the absolute difference. "
            "Use calculator observation fields such as abs_trillion and abs_formatted_usd when formatting large USD values. "
            "After receiving a Tool observation, either call another tool or return final JSON."
        )

    def generate(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 256,
        }
        resp = requests.post(VLLM_CHAT_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def run(self, task: str, history: Optional[List[Dict[str, str]]] = None, max_turns: int = 8) -> Dict[str, Any]:
        messages = [{"role": "system", "content": self.system_prompt()}]
        if history:
            for item in history[-8:]:
                role = item.get("role")
                content = item.get("content")
                if role in {"user", "assistant"} and content:
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": task})

        trace = []
        for turn in range(max_turns):
            raw = self.generate(messages)
            obj = extract_json(raw)
            record = {"turn": turn, "raw_model_output": raw, "parsed": obj}
            if not isinstance(obj, dict):
                record["error"] = "invalid_json"
                trace.append(record)
                messages += [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "Invalid JSON. Return a valid JSON action only."},
                ]
                continue

            action = obj.get("action")
            if action == "final":
                answer = str(obj.get("answer", ""))
                record["final_answer"] = answer
                trace.append(record)
                return {"ok": True, "answer": answer, "trace": trace}

            if action != "tool_call":
                record["error"] = f"unknown action {action}"
                trace.append(record)
                messages += [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "Unknown action. Use tool_call or final."},
                ]
                continue

            call = obj.get("tool_call") or {}
            tool_name = call.get("name")
            args = call.get("arguments") or {}
            if not isinstance(args, dict):
                args = {}
            tool = self.tools.get(str(tool_name))
            if not tool:
                obs = {"ok": False, "error": f"unknown tool {tool_name}", "available_tools": list(self.tools)}
            else:
                try:
                    obs = tool.invoke(args)
                except Exception as exc:
                    obs = {"ok": False, "error": str(exc), "tool": tool_name}

            record["tool_name"] = tool_name
            record["tool_args"] = args
            record["observation"] = obs
            trace.append(record)

            followup = f"Tool observation from {tool_name}:\n{json.dumps(obs, ensure_ascii=False)}"
            if tool_name == "web_search":
                followup += "\n\nUse rag_context and reranked_results to answer the user now. Return final JSON."
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": followup},
            ]

        return {"ok": False, "answer": "", "trace": trace, "error": "max_turns_exceeded"}


app = FastAPI(title="Qwen Financial Tool Agent")
controller = AgentController()
static_dir = Path(__file__).resolve().parent / "web_static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    return (static_dir / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    try:
        r = requests.get("http://127.0.0.1:8000/v1/models", timeout=5)
        vllm_ok = r.status_code == 200
    except Exception:
        vllm_ok = False
    return {"ok": True, "vllm_ok": vllm_ok, "tools": list(controller.tools)}


@app.post("/api/chat")
def chat(req: ChatRequest):
    return controller.run(req.message, history=req.history, max_turns=req.max_turns)
