import argparse
import ast
import html
import json
import math
import operator
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from langchain_core.tools import StructuredTool
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.finqa_calculator import execute_program, format_number

try:
    from scripts.agent.web_chunked_rag import web_chunked_rag
except Exception:
    web_chunked_rag = None


FINANCE_DB = {
    ("NVDA", "revenue", "FY2023"): {"value": 27.0, "unit": "billion USD"},
    ("NVDA", "revenue", "FY2024"): {"value": 60.9, "unit": "billion USD"},
    ("NVDA", "revenue", "FY2025"): {"value": 130.5, "unit": "billion USD"},
    ("AAPL", "revenue", "FY2023"): {"value": 383.3, "unit": "billion USD"},
    ("AAPL", "revenue", "FY2024"): {"value": 391.0, "unit": "billion USD"},
    ("MSFT", "revenue", "FY2024"): {"value": 245.1, "unit": "billion USD"},
}

_RERANKER = None
_RERANKER_TOKENIZER = None
_RERANKER_NAME = None

FALLBACK_SEARCH_CORPUS = {
    "langchain": [
        {
            "title": "LangChain overview",
            "href": "local://fallback/langchain-overview",
            "body": "LangChain is a framework for building applications powered by large language models, including chains, agents, tools, retrieval, and orchestration.",
        },
        {
            "title": "LangChain agents",
            "href": "local://fallback/langchain-agents",
            "body": "LangChain agents use a language model to choose actions, call tools, observe results, and continue until a final answer is produced.",
        },
    ],
    "qwen": [
        {
            "title": "Qwen model family",
            "href": "local://fallback/qwen",
            "body": "Qwen is a family of large language models that can be adapted for tool use, reasoning, and instruction following.",
        }
    ],
}


class CalculatorArgs(BaseModel):
    expression: Optional[str] = Field(default=None, description="A safe arithmetic expression, for example ((130.5/27.0)**(1/2)-1)*100.")
    program: Optional[str] = Field(default=None, description="A FinQA calculator program, for example divide(subtract(1200, 1000), 1000).")


class FinanceArgs(BaseModel):
    symbol: str = Field(description="Stock ticker, for example NVDA.")
    metric: str = Field(description="Financial metric, for example revenue.")
    period: str = Field(description="Fiscal period, for example FY2025.")


class MarketDataArgs(BaseModel):
    symbol: str = Field(description="Stock ticker or company name, for example AAPL, Apple, TSLA, Tesla.")
    metric: str = Field(default="market_cap", description="Market metric to fetch. Currently supports market_cap.")


class WebSearchArgs(BaseModel):
    query: str = Field(description="Search query.")
    max_results: int = Field(default=5, description="Number of retrieved search results before reranking.")
    top_k: int = Field(default=3, description="Number of reranked page chunks to keep as RAG context.")


ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
ALLOWED_EXPR_FUNCS = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "divide": lambda a, b: a / b,
    "max": lambda *xs: max(xs),
    "min": lambda *xs: min(xs),
    "average": lambda *xs: sum(xs) / len(xs),
}


def safe_eval_expr(expr: str) -> float:
    def eval_node(node):
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_BINOPS:
            return ALLOWED_BINOPS[type(node.op)](eval_node(node.left), eval_node(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_UNARY:
            return ALLOWED_UNARY[type(node.op)](eval_node(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ALLOWED_EXPR_FUNCS:
            args = [eval_node(arg) for arg in node.args]
            return ALLOWED_EXPR_FUNCS[node.func.id](*args)
        raise ValueError(f"unsupported expression node: {type(node).__name__}")

    value = eval_node(ast.parse(str(expr), mode="eval"))
    if not math.isfinite(value):
        raise ValueError("non-finite result")
    return float(value)


def calculator_tool(expression: Optional[str] = None, program: Optional[str] = None) -> Dict[str, Any]:
    if program:
        result = execute_program(program)
        if result.get("ok"):
            value = float(result.get("result"))
            return {
                "ok": True,
                "result": format_number(value),
                "raw_result": value,
                "trillion": value / 1_000_000_000_000,
                "billion": value / 1_000_000_000,
                "abs_trillion": abs(value) / 1_000_000_000_000,
                "abs_billion": abs(value) / 1_000_000_000,
                "formatted_usd": compact_usd(value),
                "abs_formatted_usd": compact_usd(abs(value)),
                "mode": "program",
            }
        return {"ok": False, "error": result.get("error"), "mode": "program"}
    if not expression:
        return {"ok": False, "error": "missing expression or program"}
    try:
        value = safe_eval_expr(expression)
        return {
            "ok": True,
            "result": format_number(value),
            "raw_result": value,
            "trillion": value / 1_000_000_000_000,
            "billion": value / 1_000_000_000,
            "abs_trillion": abs(value) / 1_000_000_000_000,
            "abs_billion": abs(value) / 1_000_000_000,
            "formatted_usd": compact_usd(value),
            "abs_formatted_usd": compact_usd(abs(value)),
            "mode": "expression",
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "mode": "expression"}


def finance_api_tool(symbol: str, metric: str, period: str) -> Dict[str, Any]:
    key = (str(symbol).upper(), str(metric).lower(), str(period).upper())
    item = FINANCE_DB.get(key)
    if not item:
        return {"ok": False, "error": f"not found: {key}", "available_examples": list(map(str, list(FINANCE_DB)[:5]))}
    return {"ok": True, "symbol": key[0], "metric": key[1], "period": key[2], **item}


MARKET_DATA_SLUGS = {
    "AAPL": ("apple", "Apple"),
    "APPLE": ("apple", "Apple"),
    "苹果": ("apple", "Apple"),
    "TSLA": ("tesla", "Tesla"),
    "TESLA": ("tesla", "Tesla"),
    "特斯拉": ("tesla", "Tesla"),
    "MSFT": ("microsoft", "Microsoft"),
    "MICROSOFT": ("microsoft", "Microsoft"),
    "微软": ("microsoft", "Microsoft"),
    "NVDA": ("nvidia", "NVIDIA"),
    "NVIDIA": ("nvidia", "NVIDIA"),
    "英伟达": ("nvidia", "NVIDIA"),
    "GOOGL": ("alphabet-google", "Alphabet (Google)"),
    "GOOG": ("alphabet-google", "Alphabet (Google)"),
    "GOOGLE": ("alphabet-google", "Alphabet (Google)"),
    "AMZN": ("amazon", "Amazon"),
    "AMAZON": ("amazon", "Amazon"),
    "META": ("meta-platforms", "Meta Platforms"),
    "META PLATFORMS": ("meta-platforms", "Meta Platforms"),
}


def parse_market_cap_value(text: str) -> Optional[float]:
    match = re.search(r"Market cap:\s*<span[^>]*>\$([0-9.,]+)\s*(Trillion|Billion|Million)\s*USD", text, flags=re.I)
    if not match:
        match = re.search(r"market cap of <strong>\$([0-9.,]+)\s*(Trillion|Billion|Million)\s*USD", text, flags=re.I)
    if not match:
        return None
    value = float(match.group(1).replace(",", ""))
    unit = match.group(2).lower()
    multiplier = {"trillion": 1_000_000_000_000, "billion": 1_000_000_000, "million": 1_000_000}[unit]
    return value * multiplier


def compact_usd(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.3f} trillion"
    if abs_value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.3f} billion"
    if abs_value >= 1_000_000:
        return f"${value / 1_000_000:.3f} million"
    return f"${value:.2f}"


def market_data_tool(symbol: str, metric: str = "market_cap") -> Dict[str, Any]:
    metric = str(metric or "market_cap").lower()
    if metric not in {"market_cap", "market capitalization", "marketcap"}:
        return {"ok": False, "error": f"unsupported metric: {metric}", "supported_metrics": ["market_cap"]}
    key = str(symbol or "").strip().upper()
    slug_item = MARKET_DATA_SLUGS.get(key) or MARKET_DATA_SLUGS.get(str(symbol or "").strip())
    if not slug_item:
        return {"ok": False, "error": f"unknown symbol/company: {symbol}", "supported_examples": sorted(MARKET_DATA_SLUGS)[:12]}
    slug, company_name = slug_item
    url = f"https://companiesmarketcap.com/{slug}/marketcap/"
    try:
        import requests

        resp = requests.get(
            url,
            timeout=15,
            verify=False,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.8,zh-CN;q=0.6",
            },
        )
        if resp.status_code != 200:
            return {"ok": False, "error": f"HTTP {resp.status_code}", "source": url}
        value = parse_market_cap_value(resp.text)
        if value is None:
            return {"ok": False, "error": "could not parse market cap", "source": url}
        as_of_match = re.search(r"As of ([A-Za-z]+ \d{4})", resp.text)
        return {
            "ok": True,
            "symbol": key,
            "company": company_name,
            "metric": "market_cap",
            "value": value,
            "unit": "USD",
            "formatted": compact_usd(value),
            "as_of": as_of_match.group(1) if as_of_match else "",
            "source": url,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "source": url}


def web_search_tool(query: str, max_results: int = 5, top_k: int = 3) -> Dict[str, Any]:
    max_results = max(1, min(int(max_results or 5), 8))
    top_k = max(1, min(int(top_k or 3), max_results))
    if web_chunked_rag is not None:
        try:
            return web_chunked_rag(
                query=query,
                max_search_results=max_results,
                max_pages_to_fetch=max_results,
                retrieve_top_n=max(20, top_k * 6),
                final_top_k=top_k,
                reranker_model=os.environ.get("BGE_RERANKER_PATH", "models/BAAI/bge-reranker-base"),
            )
        except Exception as exc:
            fallback_error = str(exc)
    else:
        fallback_error = "web_chunked_rag module unavailable"

    bing_result = bing_search(query, max_results)
    if bing_result:
        obs = build_rag_observation(query, bing_result, top_k, live=True, backend="bing_html")
        obs["chunked_rag_error"] = fallback_error
        return obs

    try:
        from duckduckgo_search import DDGS

        rows = []
        with DDGS() as ddgs:
            for row in ddgs.text(query, max_results=max_results):
                rows.append({
                    "title": row.get("title"),
                    "href": row.get("href"),
                    "body": row.get("body"),
                })
        if rows:
            return build_rag_observation(query, rows, top_k, live=True, backend="duckduckgo")
    except Exception as exc:
        # The AutoDL network is sometimes restricted. Return a deterministic
        # observation so the agent loop can still be tested end-to-end.
        fallback = fallback_search(query, max_results)
        obs = build_rag_observation(query, fallback, top_k, live=False, backend="fallback")
        obs["error"] = str(exc)
        obs["chunked_rag_error"] = fallback_error
        obs["fallback_note"] = "Live search backend unavailable on this machine; returned local fallback search results for controller verification."
        return obs
    fallback = fallback_search(query, max_results)
    obs = build_rag_observation(query, fallback, top_k, live=False, backend="fallback")
    obs["error"] = "no live results"
    obs["chunked_rag_error"] = fallback_error
    return obs


STOPWORDS = {
    "a", "an", "and", "are", "as", "be", "by", "for", "from", "in", "is", "it",
    "of", "on", "or", "that", "the", "then", "to", "use", "what", "with",
    "look", "up", "summarize", "official",
}


def tokenize_for_rank(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_+-]*", str(text or "").lower())
    return [tok for tok in tokens if tok not in STOPWORDS and len(tok) > 1]


def rerank_search_results(query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    model_ranked = bge_rerank_search_results(query, results)
    if model_ranked:
        return model_ranked
    return lexical_rerank_search_results(query, results)


def lexical_rerank_search_results(query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    query_terms = tokenize_for_rank(query)
    query_set = set(query_terms)
    ranked: List[Tuple[float, int, Dict[str, Any]]] = []
    for idx, row in enumerate(results):
        title = str(row.get("title") or "")
        body = str(row.get("body") or "")
        href = str(row.get("href") or "")
        title_terms = tokenize_for_rank(title)
        body_terms = tokenize_for_rank(body)
        href_terms = tokenize_for_rank(href)
        title_overlap = len(query_set.intersection(title_terms))
        body_overlap = len(query_set.intersection(body_terms))
        href_overlap = len(query_set.intersection(href_terms))
        phrase_bonus = 0.0
        q_lower = str(query or "").lower().strip()
        text_lower = f"{title} {body} {href}".lower()
        if q_lower and q_lower in text_lower:
            phrase_bonus += 4.0
        domain_bonus = 0.0
        if "official" in str(query or "").lower() and any(x in href.lower() for x in [".com", ".org", "docs."]):
            domain_bonus += 0.5
        score = 3.0 * title_overlap + 1.0 * body_overlap + 0.6 * href_overlap + phrase_bonus + domain_bonus
        enriched = dict(row)
        enriched["rerank_score"] = round(score, 4)
        enriched["original_rank"] = idx + 1
        enriched["reranker"] = "lexical_title_body_overlap_v1"
        ranked.append((score, -idx, enriched))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in ranked]


def load_bge_reranker():
    global _RERANKER, _RERANKER_TOKENIZER, _RERANKER_NAME
    if _RERANKER is not None and _RERANKER_TOKENIZER is not None:
        return _RERANKER_TOKENIZER, _RERANKER, _RERANKER_NAME
    model_path = os.environ.get("BGE_RERANKER_PATH", "models/BAAI/bge-reranker-base")
    if not Path(model_path).exists():
        return None, None, None
    try:
        device = os.environ.get("BGE_RERANK_DEVICE", "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
        model.to(device)
        model.eval()
        _RERANKER_TOKENIZER = tokenizer
        _RERANKER = model
        _RERANKER_NAME = str(model_path)
        return tokenizer, model, _RERANKER_NAME
    except Exception:
        return None, None, None


def source_prior_score(query: str, row: Dict[str, Any]) -> float:
    query_lower = str(query or "").lower()
    if "official" not in query_lower:
        return 0.0
    href = str(row.get("href") or "").lower()
    title = str(row.get("title") or "").lower()
    host_match = re.search(r"https?://([^/]+)", href)
    host = host_match.group(1) if host_match else href
    host = host.replace("www.", "")
    q_terms = [tok for tok in tokenize_for_rank(query_lower) if tok not in {"framework", "documentation"}]
    score = 0.0
    for term in q_terms:
        if host == f"{term}.com" or host == f"{term}.org" or host.startswith(f"{term}."):
            score += 6.0
        elif term in host and "doc" not in host:
            score += 2.0
    if "official" in title or "official" in href:
        score += 1.0
    return score


def bge_rerank_search_results(query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tokenizer, model, model_name = load_bge_reranker()
    if model is None or tokenizer is None or not results:
        return []
    try:
        passages = [
            f"{row.get('title') or ''}\nURL: {row.get('href') or ''}\n{row.get('body') or ''}"
            for row in results
        ]
        pairs = [[str(query or ""), passage] for passage in passages]
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        if logits.ndim == 2 and logits.shape[1] > 1:
            scores = logits[:, -1].detach().float().cpu().tolist()
        else:
            scores = logits.reshape(-1).detach().float().cpu().tolist()
        ranked: List[Tuple[float, int, Dict[str, Any]]] = []
        for idx, (score, row) in enumerate(zip(scores, results)):
            prior = source_prior_score(query, row)
            final_score = float(score) + prior
            enriched = dict(row)
            enriched["rerank_score"] = round(final_score, 4)
            enriched["model_rerank_score"] = round(float(score), 4)
            enriched["source_prior_score"] = round(prior, 4)
            enriched["original_rank"] = idx + 1
            enriched["reranker"] = model_name
            ranked.append((final_score, -idx, enriched))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in ranked]
    except Exception:
        return []


def build_rag_context(results: List[Dict[str, Any]], top_k: int) -> str:
    chunks = []
    for i, row in enumerate(results[:top_k], start=1):
        title = str(row.get("title") or "").strip()
        href = str(row.get("href") or "").strip()
        body = str(row.get("body") or "").strip()
        chunks.append(f"[{i}] {title}\nURL: {href}\nSnippet: {body}")
    return "\n\n".join(chunks)


def build_rag_observation(query: str, results: List[Dict[str, Any]], top_k: int, live: bool, backend: str) -> Dict[str, Any]:
    reranked = rerank_search_results(query, results)
    kept = reranked[:top_k]
    return {
        "ok": bool(results),
        "query": query,
        "results": results,
        "reranked_results": kept,
        "rag_context": build_rag_context(kept, top_k),
        "retrieved_count": len(results),
        "rerank_top_k": top_k,
        "reranker": kept[0].get("reranker") if kept else "none",
        "live": live,
        "backend": backend,
    }


def strip_tags(text: str) -> str:
    text = re.sub(r"<script.*?</script>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def bing_search(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    try:
        import requests

        resp = requests.get(
            "https://www.bing.com/search",
            params={"q": query, "mkt": "en-US", "setlang": "en-US", "cc": "US"},
            timeout=12,
            verify=False,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.8,zh-CN;q=0.6",
            },
        )
        if resp.status_code != 200:
            return []
        text = resp.text
        rows = []
        blocks = re.findall(r'<li class="b_algo".*?</li>', text, flags=re.S | re.I)
        if not blocks:
            blocks = re.findall(r"<h2.*?</h2>.*?(?:<p.*?</p>)?", text, flags=re.S | re.I)
        for block in blocks:
            link = re.search(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', block, flags=re.S | re.I)
            if not link:
                continue
            href = html.unescape(link.group(1))
            title = strip_tags(link.group(2))
            snippet_match = re.search(r'<p[^>]*>(.*?)</p>', block, flags=re.S | re.I)
            body = strip_tags(snippet_match.group(1)) if snippet_match else ""
            if title and href:
                rows.append({"title": title, "href": href, "body": body})
            if len(rows) >= max_results:
                break
        return rows
    except Exception:
        return []


def fallback_search(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    q = str(query or "").lower()
    rows = []
    for key, docs in FALLBACK_SEARCH_CORPUS.items():
        if key in q:
            rows.extend(docs)
    return rows[:max_results]


def build_tools() -> Dict[str, StructuredTool]:
    tools = [
        StructuredTool.from_function(
            name="calculator",
            description="Run arithmetic or FinQA calculator programs.",
            func=calculator_tool,
            args_schema=CalculatorArgs,
        ),
        StructuredTool.from_function(
            name="finance_api",
            description="Look up a small local financial metric table by symbol, metric, and fiscal period.",
            func=finance_api_tool,
            args_schema=FinanceArgs,
        ),
        StructuredTool.from_function(
            name="market_data",
            description="Fetch current public-market metrics such as market capitalization from live web sources.",
            func=market_data_tool,
            args_schema=MarketDataArgs,
        ),
        StructuredTool.from_function(
            name="web_search",
            description="Search the web, rerank retrieved pages, and return a compact RAG context.",
            func=web_search_tool,
            args_schema=WebSearchArgs,
        ),
    ]
    return {tool.name: tool for tool in tools}


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    s = str(text or "").strip()
    s = re.sub(r"^```json\s*", "", s)
    s = re.sub(r"^```\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    match = re.search(r"\{.*\}", s, flags=re.S)
    if match:
        s = match.group(0)
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


class LangChainQwenAgent:
    def __init__(self, model_path: str, adapter_path: str = "", max_new_tokens: int = 192):
        self.tools = build_tools()
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.do_sample = False
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p = None
            self.model.generation_config.top_k = None

    def tool_schema_text(self) -> str:
        chunks = []
        for name, tool in self.tools.items():
            schema = tool.args_schema.model_json_schema() if tool.args_schema else {}
            chunks.append(f"- {name}: {tool.description}\n  args_schema: {json.dumps(schema.get('properties', {}), ensure_ascii=False)}")
        return "\n".join(chunks)

    def system_prompt(self) -> str:
        return (
            "You are a LangChain-style API planning agent. You can call external tools by returning JSON only.\n"
            "Valid actions:\n"
            '{"action":"tool_call","tool_call":{"name":"calculator|finance_api|market_data|web_search","arguments":{...}}}\n'
            '{"action":"final","answer":"..."}\n\n'
            "Available tools:\n"
            f"{self.tool_schema_text()}\n\n"
            "Plan step by step. Call one tool at a time. Prefer market_data for current stock market capitalization questions. "
            "Use calculator for arithmetic after collecting numeric observations. If the user asks for a difference without specifying order, report the absolute difference. "
            "Use calculator observation fields such as abs_trillion and abs_formatted_usd when formatting large USD values. "
            "After receiving a Tool observation, either call another tool or return final JSON."
        )

    def generate(self, messages: List[Dict[str, str]]) -> str:
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        gen = out[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)

    def run(self, task: str, max_turns: int = 5) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": task},
        ]
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
                    {"role": "user", "content": 'Invalid JSON. Return a valid JSON action only.'},
                ]
                continue

            action = obj.get("action")
            if action == "final":
                record["final_answer"] = obj.get("answer", "")
                trace.append(record)
                return {"ok": True, "task": task, "final_answer": obj.get("answer", ""), "trace": trace}

            if action != "tool_call":
                record["error"] = f"unknown action {action}"
                trace.append(record)
                messages += [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": 'Unknown action. Use tool_call or final.'},
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
                followup += "\n\nUse rag_context and reranked_results to answer the user now. Return final JSON. Do not call calculator unless the user explicitly asks for arithmetic."
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": followup},
            ]
        return {"ok": False, "task": task, "final_answer": "", "trace": trace, "error": "max_turns_exceeded"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter_path", default="saves/agent/qwen25-7b-agent-stage7-finqa-grpo-r32")
    ap.add_argument("--task", default="")
    ap.add_argument("--input_jsonl", default="")
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--max_turns", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=192)
    args = ap.parse_args()

    tasks = []
    if args.task:
        tasks.append(args.task)
    if args.input_jsonl:
        with open(args.input_jsonl, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    tasks.append(obj["task"] if isinstance(obj, dict) else str(obj))
    if not tasks:
        raise SystemExit("Provide --task or --input_jsonl")

    agent = LangChainQwenAgent(args.model_path, args.adapter_path, args.max_new_tokens)
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as out:
        for task in tasks:
            result = agent.run(task, max_turns=args.max_turns)
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            print(json.dumps(result, ensure_ascii=False, indent=2)[:3000])


if __name__ == "__main__":
    main()
