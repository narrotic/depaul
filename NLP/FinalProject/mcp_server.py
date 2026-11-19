"""Cleantech MCP server: exposes the research tools over the Model Context Protocol (stdio).

Any MCP-capable client (our LangGraph agents, Claude Desktop, an IDE) can discover and
call these tools without importing this code. stdout carries JSON-RPC frames, so the
server must never print; all logging goes to stderr.
"""

from __future__ import annotations

import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import hashlib
import json

import anthropic
import httpx
from dotenv import load_dotenv
from mcp.server.mcpserver import Context, MCPServer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval import HybridRetriever, clean_article_text  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
SUMMARY_MODEL = os.environ.get("CLEANTECH_SUMMARY_MODEL", "claude-haiku-5-5")
OPENALEX = "https://api.openalex.org/works"
CACHE_DIR = Path(__file__).resolve().parent / "artifacts" / "tool_cache"


def _cached(kind: str, key: dict):
    """Tiny content-addressed disk cache so re-running the notebook replays tool calls exactly."""
    path = CACHE_DIR / kind / (hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest() + ".json")
    return path, (json.loads(path.read_text()) if path.exists() else None)


def _store(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False))


@asynccontextmanager
async def lifespan(_server: MCPServer):
    """Load the indexes once per server process, not once per tool call."""
    yield {
        "retriever": HybridRetriever(use_dense=os.environ.get("CLEANTECH_DENSE", "1") == "1"),
        "llm": anthropic.AsyncAnthropic(),
        "http": httpx.AsyncClient(timeout=20.0),
        # Sync tools run in worker threads; PyTorch's MPS backend is not thread-safe,
        # so parallel researchers must take turns on the GPU encoders.
        "gpu_lock": threading.Lock(),
    }


mcp = MCPServer("cleantech-research", lifespan=lifespan)


def _state(ctx: Context) -> dict:
    return ctx.request_context.lifespan_context


@mcp.tool()
def search_corpus(query: str, k: int = 6, ctx: Context = None) -> list[dict]:
    """Hybrid search (BM25 + dense + cross-encoder rerank) over 20k cleantech news articles.

    Returns up to k passages, each with chunk_id, doc_id, title, date, url and text.
    Write the query as a short keyword-rich phrase; call several times for different aspects.
    """
    state = _state(ctx)
    with state["gpu_lock"]:
        hits = state["retriever"].search(query, k=max(1, min(k, 12)))
    return [{"chunk_id": h.chunk_id, "doc_id": h.doc_id, "title": h.title, "date": h.date,
             "url": h.url, "score": round(h.score, 3), "text": h.text} for h in hits]


@mcp.tool()
def get_article(doc_id: int, max_chars: int = 6000, ctx: Context = None) -> dict:
    """Return the full cleaned text of one article (by doc_id) for close reading."""
    chunks = _state(ctx)["retriever"].chunks
    rows = chunks[chunks["doc_id"] == doc_id]
    if rows.empty:
        return {"error": f"doc_id {doc_id} not found"}
    first = rows.iloc[0]
    return {"doc_id": doc_id, "title": first.title, "date": first.date, "url": first.url,
            "text": clean_article_text(rows)[:max_chars]}


@mcp.tool()
async def summarize_article(doc_id: int, focus: str, ctx: Context = None) -> dict:
    """Query-focused summary (<= 5 sentences) of one article, keeping every number and name
    that is relevant to `focus`. Use it to compress long articles before synthesis."""
    article = get_article(doc_id, max_chars=12000, ctx=ctx)
    if "error" in article:
        return article
    path, hit = _cached("summaries", {"doc_id": doc_id, "focus": focus, "model": SUMMARY_MODEL})
    if hit is not None:
        return hit
    msg = await _state(ctx)["llm"].messages.create(
        model=SUMMARY_MODEL,
        max_tokens=1024,
        output_config={"effort": "low"},
        system="You summarize news articles for an analyst. Use only facts stated in the article.",
        messages=[{"role": "user", "content": (
            f"Focus: {focus}\n\nArticle title: {article['title']}\n\n{article['text']}\n\n"
            "Write at most 5 sentences on the focus. Keep every relevant figure, date, "
            "organisation and place. If the article does not cover the focus, say so in one sentence."
        )}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    result = {"doc_id": doc_id, "title": article["title"], "summary": text.strip(),
              # reported back so the client can account for the server's own LLM spend
              "usage": {"model": SUMMARY_MODEL, "input_tokens": msg.usage.input_tokens,
                        "output_tokens": msg.usage.output_tokens}}
    _store(path, result)
    return result


def _abstract(inverted: dict | None, limit: int = 60) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; rebuild the first `limit` words."""
    if not inverted:
        return ""
    words = sorted((p, w) for w, ps in inverted.items() for p in ps)
    return " ".join(w for _, w in words[:limit])


@mcp.tool()
async def search_openalex(query: str, n: int = 4, ctx: Context = None) -> list[dict]:
    """Search OpenAlex (250M+ scholarly works) for peer-reviewed evidence that can confirm
    or contextualise a claim from the news corpus. Returns title, year, venue, citations, abstract."""
    path, hit = _cached("openalex", {"query": query, "n": n})
    if hit is not None:
        return hit
    resp = await _state(ctx)["http"].get(OPENALEX, params={
        "search": query, "per-page": max(1, min(n, 8)), "sort": "relevance_score:desc",
        "filter": "type:article,has_abstract:true",
        "select": "id,doi,display_name,publication_year,primary_location,cited_by_count,abstract_inverted_index",
    })
    resp.raise_for_status()
    out = []
    for w in resp.json().get("results", []):
        venue = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
        out.append({"openalex_id": w["id"].rsplit("/", 1)[-1], "title": w["display_name"],
                    "year": w.get("publication_year"), "venue": venue, "doi": w.get("doi"),
                    "cited_by": w.get("cited_by_count"), "abstract": _abstract(w.get("abstract_inverted_index"))})
    _store(path, out)
    return out


if __name__ == "__main__":
    mcp.run(transport="stdio")
