"""Hybrid retrieval engine for the Cleantech corpus.

Three retrievers share one chunk table:
    * DenseRetriever   - bge-base embeddings stored in a persistent Chroma collection
    * BM25Retriever    - Okapi BM25 implemented from scratch on a sparse matrix
    * HybridRetriever  - Reciprocal Rank Fusion of the two, plus optional
                         cross-encoder reranking and article-level de-duplication
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from nltk.stem import PorterStemmer

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"  # v2-m3 was 9x slower for no recall gain (see notebook)
# bge models are trained with this instruction on the query side only
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass
class Hit:
    """One retrieved chunk with its score and the retriever that produced it."""

    chunk_id: str
    doc_id: int
    title: str
    date: str
    url: str
    text: str
    score: float
    source: str


def load_chunks() -> pd.DataFrame:
    return pd.read_parquet(ARTIFACTS / "chunks.parquet")


def clean_article_text(rows: pd.DataFrame) -> str:
    """Rebuild an article from its chunks in order (a short overlap paragraph may repeat once)."""
    order = rows["chunk_id"].str.rsplit("-", n=1).str[-1].astype(int)
    return "\n\n".join(rows.assign(_o=order).sort_values("_o")["text"])


def _to_hits(chunks: pd.DataFrame, rows: np.ndarray, scores: np.ndarray, source: str) -> list[Hit]:
    out = []
    for r, s in zip(rows, scores):
        c = chunks.iloc[int(r)]
        out.append(Hit(c.chunk_id, int(c.doc_id), c.title, c.date, c.url, c.text, float(s), source))
    return out


# -------------------------------------------------------------------------- Dense
class DenseRetriever:
    """Approximate nearest-neighbour search over a persistent Chroma collection."""

    def __init__(self, chunks: pd.DataFrame, persist_dir: Path = ARTIFACTS / "chroma",
                 collection: str = "cleantech_bge_base", device: str = "mps"):
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks
        self.row_of = {cid: i for i, cid in enumerate(chunks["chunk_id"])}
        self.encoder = SentenceTransformer(EMBED_MODEL, device=device)
        client = chromadb.PersistentClient(path=str(persist_dir))
        self.col = client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})
        if self.col.count() == 0:
            self._populate()

    def _populate(self, batch: int = 5000) -> None:
        """Load the pre-computed embeddings (build_index.py) into Chroma once."""
        emb = np.load(ARTIFACTS / "embeddings.npy").astype(np.float32)
        c = self.chunks
        for s in range(0, len(c), batch):
            part = c.iloc[s:s + batch]
            self.col.add(
                ids=part["chunk_id"].tolist(),
                embeddings=emb[s:s + batch],
                metadatas=[{"doc_id": int(d), "date": str(t)} for d, t in zip(part["doc_id"], part["date"])],
            )

    def search(self, query: str, k: int = 50) -> list[Hit]:
        q = self.encoder.encode(BGE_QUERY_PREFIX + query, normalize_embeddings=True)
        res = self.col.query(query_embeddings=[q.tolist()], n_results=k, include=["distances"])
        rows = np.array([self.row_of[i] for i in res["ids"][0]])
        sims = 1.0 - np.array(res["distances"][0])  # cosine distance -> similarity
        return _to_hits(self.chunks, rows, sims, "dense")


# --------------------------------------------------------------------------- BM25
_STEM = PorterStemmer()
_STOP = frozenset(
    "a an and are as at be been but by for from has have he her his i in into is it its "
    "of on or our she that the their them there these they this to was were which while "
    "who will with would you your what how why when where do does did can could should "
    "than then so such not no also about over after before between more most other".split()
)
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")


@lru_cache(maxsize=200_000)
def _stem(word: str) -> str:
    return _STEM.stem(word)


def tokenize(text: str) -> list[str]:
    """Lowercase, keep alphanumerics (incl. '2.3', 'co2', 'long-duration'), drop stop words, stem."""
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 1]


class BM25Retriever:
    """Okapi BM25 (Robertson & Zaragoza, 2009), vectorised with scipy.

    score(q, d) = sum_{t in q} idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
    idf(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

    The whole fraction is pre-computed once per (doc, term) pair, so a query is
    just a sum of a few sparse columns.
    """

    def __init__(self, chunks: pd.DataFrame, k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        docs = [tokenize(f"{t} {x}") for t, x in zip(chunks["title"], chunks["text"])]
        self.vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        for i, toks in enumerate(docs):
            for term, tf in Counter(toks).items():
                j = self.vocab.setdefault(term, len(self.vocab))
                rows.append(i), cols.append(j), vals.append(tf)
        tf = sp.csr_matrix((vals, (rows, cols)), shape=(len(docs), len(self.vocab)), dtype=np.float32)

        n_docs = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        norm = k1 * (1 - b + b * doc_len / doc_len.mean())  # per-document length penalty

        # Rewrite every stored tf value in place with its full BM25 weight.
        w = tf.tocoo()
        w.data = self.idf[w.col] * w.data * (k1 + 1) / (w.data + norm[w.row])
        self.weights = w.tocsc()  # column slicing (one column per query term) is fast in CSC

    def search(self, query: str, k: int = 50) -> list[Hit]:
        cols = [self.vocab[t] for t in tokenize(query) if t in self.vocab]
        if not cols:
            return []
        scores = np.asarray(self.weights[:, cols].sum(axis=1)).ravel()
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return _to_hits(self.chunks, top, scores[top], "bm25")


# ------------------------------------------------------------------ Fusion/rerank
def reciprocal_rank_fusion(ranked_lists: list[list[Hit]], k: int = 60) -> list[Hit]:
    """RRF (Cormack et al., 2009): score(d) = sum over lists of 1 / (k + rank(d)).

    Uses only ranks, so BM25 scores (unbounded) and cosine similarities ([-1, 1])
    never have to be put on the same scale.
    """
    fused: dict[str, float] = {}
    best: dict[str, Hit] = {}
    for hits in ranked_lists:
        for rank, h in enumerate(hits, start=1):
            fused[h.chunk_id] = fused.get(h.chunk_id, 0.0) + 1.0 / (k + rank)
            best.setdefault(h.chunk_id, h)
    order = sorted(fused, key=fused.get, reverse=True)
    return [Hit(**{**best[c].__dict__, "score": fused[c], "source": "rrf"}) for c in order]


class CrossEncoderReranker:
    """Re-scores (query, chunk) pairs jointly; far more precise than bi-encoder cosine."""

    def __init__(self, model: str = RERANK_MODEL, device: str = "mps"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model, device=device, max_length=512)

    def rerank(self, query: str, hits: list[Hit], top_n: int | None = None) -> list[Hit]:
        if not hits:
            return []
        scores = self.model.predict([(query, f"{h.title}\n{h.text}") for h in hits], batch_size=16)
        order = np.argsort(-scores)[: top_n or len(hits)]
        return [Hit(**{**hits[i].__dict__, "score": float(scores[i]), "source": "rerank"}) for i in order]


def dedupe_by_article(hits: list[Hit], per_article: int = 2) -> list[Hit]:
    """Keep at most `per_article` chunks per article so one long story cannot crowd out others."""
    seen: Counter = Counter()
    out = []
    for h in hits:
        if seen[h.doc_id] < per_article:
            out.append(h)
            seen[h.doc_id] += 1
    return out


class HybridRetriever:
    """BM25 + dense -> RRF -> (optional) cross-encoder rerank -> article de-dup."""

    def __init__(self, chunks: pd.DataFrame | None = None, use_dense: bool = True, use_reranker: bool = True):
        self.chunks = load_chunks() if chunks is None else chunks
        self.bm25 = BM25Retriever(self.chunks)
        self.dense = DenseRetriever(self.chunks) if use_dense else None
        self.reranker = CrossEncoderReranker() if use_reranker else None

    def search(self, query: str, k: int = 8, candidates: int = 30, rerank: bool = True) -> list[Hit]:
        lists = [self.bm25.search(query, candidates)]
        if self.dense is not None:
            lists.append(self.dense.search(query, candidates))
        fused = reciprocal_rank_fusion(lists)[:candidates]
        if rerank and self.reranker is not None:
            fused = self.reranker.rerank(query, fused)
        return dedupe_by_article(fused)[:k]
