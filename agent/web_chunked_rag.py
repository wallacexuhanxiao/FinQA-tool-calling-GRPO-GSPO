import html
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


STOPWORDS = {
    "a", "an", "and", "are", "as", "be", "by", "for", "from", "in", "is", "it",
    "of", "on", "or", "that", "the", "then", "to", "use", "what", "with", "how",
    "much", "many", "difference", "between", "please", "search", "find",
}

_RERANKER = None
_RERANKER_TOKENIZER = None
_RERANKER_NAME = None

COMPANIES_MARKET_CAP_SLUGS = {
    "tesla": ("tesla", "Tesla market cap"),
    "tsla": ("tesla", "Tesla market cap"),
    "特斯拉": ("tesla", "Tesla market cap"),
    "apple": ("apple", "Apple market cap"),
    "aapl": ("apple", "Apple market cap"),
    "苹果": ("apple", "Apple market cap"),
    "microsoft": ("microsoft", "Microsoft market cap"),
    "msft": ("microsoft", "Microsoft market cap"),
    "nvidia": ("nvidia", "NVIDIA market cap"),
    "nvda": ("nvidia", "NVIDIA market cap"),
    "amazon": ("amazon", "Amazon market cap"),
    "amzn": ("amazon", "Amazon market cap"),
    "meta": ("meta-platforms", "Meta Platforms market cap"),
    "google": ("alphabet-google", "Alphabet Google market cap"),
    "alphabet": ("alphabet-google", "Alphabet Google market cap"),
    "googl": ("alphabet-google", "Alphabet Google market cap"),
    "goog": ("alphabet-google", "Alphabet Google market cap"),
}


@dataclass
class WebPage:
    title: str
    url: str
    snippet: str
    rank: int


@dataclass
class WebChunk:
    chunk_id: str
    url: str
    title: str
    source_rank: int
    chunk_index: int
    text: str
    bm25_score: float = 0.0
    rerank_score: float = 0.0
    reranker: str = "none"


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def strip_tags(text: str) -> str:
    text = re.sub(r"<script.*?</script>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<noscript.*?</noscript>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return normalize_space(html.unescape(text))


def tokenize(text: str) -> List[str]:
    toks = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_+.$%-]*", str(text or "").lower())
    return [tok for tok in toks if tok not in STOPWORDS and len(tok) > 1]


def bing_search(query: str, max_results: int) -> List[WebPage]:
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
        blocks = re.findall(r'<li class="b_algo".*?</li>', resp.text, flags=re.S | re.I)
        pages: List[WebPage] = []
        for block in blocks:
            link = re.search(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', block, flags=re.S | re.I)
            if not link:
                continue
            url = html.unescape(link.group(1))
            title = strip_tags(link.group(2))
            snippet_match = re.search(r"<p[^>]*>(.*?)</p>", block, flags=re.S | re.I)
            snippet = strip_tags(snippet_match.group(1)) if snippet_match else ""
            if title and url.startswith("http"):
                pages.append(WebPage(title=title, url=url, snippet=snippet, rank=len(pages) + 1))
            if len(pages) >= max_results:
                break
        return pages
    except Exception:
        return []


def duckduckgo_search(query: str, max_results: int) -> List[WebPage]:
    try:
        from duckduckgo_search import DDGS

        pages: List[WebPage] = []
        with DDGS() as ddgs:
            for row in ddgs.text(query, max_results=max_results):
                pages.append(WebPage(
                    title=str(row.get("title") or ""),
                    url=str(row.get("href") or ""),
                    snippet=str(row.get("body") or ""),
                    rank=len(pages) + 1,
                ))
        return [p for p in pages if p.url.startswith("http")]
    except Exception:
        return []


def search_web(query: str, max_results: int) -> Tuple[str, List[WebPage]]:
    search_query = rewrite_search_query(query)
    pages = bing_search(search_query, max_results)
    pages = inject_domain_candidates(query, pages)
    if pages:
        return "bing_html", pages[:max_results + 4]
    pages = duckduckgo_search(search_query, max_results)
    pages = inject_domain_candidates(query, pages)
    if pages:
        return "duckduckgo", pages[:max_results + 4]
    return "fallback", [
        WebPage(
            title="Fallback web RAG observation",
            url="local://fallback/web-rag",
            snippet=f"Live search is unavailable. Query was: {query}",
            rank=1,
        )
    ]


def inject_domain_candidates(query: str, pages: List[WebPage]) -> List[WebPage]:
    low = str(query or "").lower()
    if not any(term in low for term in ["market cap", "market capitalization", "市值"]):
        return pages
    existing = {p.url for p in pages}
    injected: List[WebPage] = []
    for key, (slug, title) in COMPANIES_MARKET_CAP_SLUGS.items():
        if key in low:
            url = f"https://companiesmarketcap.com/{slug}/marketcap/"
            if url not in existing:
                injected.append(WebPage(
                    title=title,
                    url=url,
                    snippet=f"Market capitalization page for {title}.",
                    rank=1,
                ))
                existing.add(url)
    # Put curated financial data sources before generic search results.
    for idx, page in enumerate(injected + pages, 1):
        page.rank = idx
    return injected + pages


def rewrite_search_query(query: str) -> str:
    q = str(query or "").strip()
    low = q.lower()
    finance_terms = ["market cap", "market capitalization", "市值"]
    if any(term in low for term in finance_terms) and "companiesmarketcap" not in low:
        return f"{q} companiesmarketcap market capitalization"
    return q


def fetch_page_text(url: str, timeout: int = 12) -> Dict[str, Any]:
    if url.startswith("local://"):
        return {"ok": True, "url": url, "text": "", "error": None}
    try:
        import requests

        resp = requests.get(
            url,
            timeout=timeout,
            verify=False,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.8,zh-CN;q=0.6",
            },
        )
        ctype = resp.headers.get("content-type", "")
        if resp.status_code != 200:
            return {"ok": False, "url": url, "text": "", "error": f"HTTP {resp.status_code}"}
        if "text/html" not in ctype and "text/plain" not in ctype and not resp.text.lstrip().startswith("<"):
            return {"ok": False, "url": url, "text": "", "error": f"unsupported content-type: {ctype}"}
        return {"ok": True, "url": url, "text": strip_tags(resp.text), "error": None}
    except Exception as exc:
        return {"ok": False, "url": url, "text": "", "error": str(exc)}


def chunk_text(text: str, chunk_words: int, overlap_words: int) -> List[str]:
    words = normalize_space(text).split()
    if not words:
        return []
    chunks = []
    step = max(1, chunk_words - overlap_words)
    for start in range(0, len(words), step):
        piece = words[start:start + chunk_words]
        if len(piece) < 25 and chunks:
            break
        chunks.append(" ".join(piece))
    return chunks


def build_chunks(pages: List[WebPage], chunk_words: int, overlap_words: int, max_pages_to_fetch: int) -> Tuple[List[WebChunk], List[Dict[str, Any]]]:
    chunks: List[WebChunk] = []
    fetch_reports: List[Dict[str, Any]] = []
    for page in pages[:max_pages_to_fetch]:
        fetched = fetch_page_text(page.url)
        text = fetched.get("text") or page.snippet
        fetch_reports.append({
            "url": page.url,
            "title": page.title,
            "ok": fetched.get("ok"),
            "error": fetched.get("error"),
            "text_chars": len(text),
        })
        pieces = chunk_text(text, chunk_words, overlap_words) or [page.snippet]
        for idx, piece in enumerate(pieces):
            if not piece.strip():
                continue
            chunks.append(WebChunk(
                chunk_id=f"src{page.rank}-chunk{idx}",
                url=page.url,
                title=page.title,
                source_rank=page.rank,
                chunk_index=idx,
                text=piece,
            ))
    return chunks, fetch_reports


def lexical_retrieve(query: str, chunks: List[WebChunk], top_n: int) -> List[WebChunk]:
    q_terms = tokenize(query)
    q_set = set(q_terms)
    ranked = []
    for chunk in chunks:
        terms = tokenize(f"{chunk.title} {chunk.text} {urlparse(chunk.url).netloc}")
        term_set = set(terms)
        overlap = len(q_set & term_set)
        coverage = overlap / max(len(q_set), 1)
        phrase_bonus = 2.0 if normalize_space(query).lower() in f"{chunk.title} {chunk.text}".lower() else 0.0
        source_prior = 1.0 / max(chunk.source_rank, 1)
        domain = urlparse(chunk.url).netloc.lower()
        domain_bonus = 0.0
        if "market cap" in query.lower() or "market capitalization" in query.lower() or "市值" in query:
            if "companiesmarketcap.com" in domain:
                domain_bonus += 2.0
            if any(x in domain for x in ["tesla.com", "apple.com"]) and "marketcap" not in chunk.url.lower():
                domain_bonus -= 0.4
        score = 3.0 * coverage + 0.15 * overlap + phrase_bonus + 0.2 * source_prior + domain_bonus
        chunk.bm25_score = float(score)
        ranked.append((score, -chunk.source_rank, -chunk.chunk_index, chunk))
    ranked.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return [x[3] for x in ranked[:top_n]]


def load_reranker(model_path: str):
    global _RERANKER, _RERANKER_TOKENIZER, _RERANKER_NAME
    if not model_path:
        return None, None, None
    if _RERANKER is not None and _RERANKER_TOKENIZER is not None and _RERANKER_NAME == model_path:
        return _RERANKER_TOKENIZER, _RERANKER, _RERANKER_NAME
    try:
        from pathlib import Path

        if not Path(model_path).exists():
            return None, None, None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
        model.to(device)
        model.eval()
        _RERANKER_TOKENIZER = tok
        _RERANKER = model
        _RERANKER_NAME = model_path
        return tok, model, model_path
    except Exception:
        return None, None, None


def rerank_chunks(query: str, chunks: List[WebChunk], top_k: int, reranker_model: str) -> List[WebChunk]:
    tok, model, model_name = load_reranker(reranker_model)
    if tok is None or model is None:
        for c in chunks:
            c.rerank_score = c.bm25_score
            c.reranker = "lexical_chunk_overlap_v1"
        return sorted(chunks, key=lambda c: c.rerank_score, reverse=True)[:top_k]
    try:
        pairs = [[query, f"{c.title}\nURL: {c.url}\n{c.text}"] for c in chunks]
        inputs = tok(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        if logits.ndim == 2 and logits.shape[-1] > 1:
            scores = logits[:, -1].detach().float().cpu().tolist()
        else:
            scores = logits.reshape(-1).detach().float().cpu().tolist()
        for c, s in zip(chunks, scores):
            c.rerank_score = float(s) + 0.05 * c.bm25_score
            c.reranker = model_name
        return sorted(chunks, key=lambda c: c.rerank_score, reverse=True)[:top_k]
    except Exception:
        for c in chunks:
            c.rerank_score = c.bm25_score
            c.reranker = "lexical_chunk_overlap_v1"
        return sorted(chunks, key=lambda c: c.rerank_score, reverse=True)[:top_k]


def build_context(chunks: List[WebChunk], max_context_words: int) -> str:
    blocks = []
    used = 0
    for i, c in enumerate(chunks, 1):
        words = c.text.split()
        if blocks and used + len(words) > max_context_words:
            continue
        used += len(words)
        blocks.append(
            f"[{i}] {c.title}\nURL: {c.url}\nChunk: {c.text}"
        )
        if used >= max_context_words:
            break
    return "\n\n".join(blocks)


def web_chunked_rag(
    query: str,
    max_search_results: int = 6,
    max_pages_to_fetch: int = 5,
    retrieve_top_n: int = 30,
    final_top_k: int = 5,
    chunk_words: int = 160,
    overlap_words: int = 40,
    max_context_words: int = 900,
    reranker_model: str = "models/BAAI/bge-reranker-base",
) -> Dict[str, Any]:
    backend, pages = search_web(query, max_search_results)
    chunks, fetch_reports = build_chunks(pages, chunk_words, overlap_words, max_pages_to_fetch)
    candidates = lexical_retrieve(query, chunks, retrieve_top_n)
    selected = rerank_chunks(query, candidates, final_top_k, reranker_model)
    return {
        "ok": bool(selected),
        "query": query,
        "backend": backend,
        "search_results": [asdict(p) for p in pages],
        "fetch_reports": fetch_reports,
        "chunk_count": len(chunks),
        "retrieved_count": len(candidates),
        "selected_count": len(selected),
        "rag_context": build_context(selected, max_context_words),
        "selected_chunks": [asdict(c) for c in selected],
        "citations": [{"title": c.title, "url": c.url, "chunk_id": c.chunk_id} for c in selected],
        "reranker": selected[0].reranker if selected else "none",
    }


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--max_search_results", type=int, default=6)
    ap.add_argument("--final_top_k", type=int, default=5)
    ap.add_argument("--reranker_model", default="models/BAAI/bge-reranker-base")
    args = ap.parse_args()
    print(json.dumps(web_chunked_rag(
        query=args.query,
        max_search_results=args.max_search_results,
        final_top_k=args.final_top_k,
        reranker_model=args.reranker_model,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
