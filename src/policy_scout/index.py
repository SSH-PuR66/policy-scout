"""Hybrid retrieval: TF-IDF cosine + sparse BM25, RRF fusion, MMR diversification.

v0.3 scalability fix: BM25 no longer materializes a dense term matrix. Scoring
densifies only the query's columns — memory is O(docs × query_terms), not
O(docs × vocabulary) — so the same code path holds from 15 chunks to Advana-scale
corpora until a dedicated vector store takes over.

Persistence stays pickle-free (JSON manifest + deterministic rebuild) — see v0.2
critique finding #1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .query import expand

RRF_K = 60  # standard reciprocal-rank-fusion constant
MMR_LAMBDA = 0.7  # relevance vs diversity trade-off


class BM25:
    """Okapi BM25 over a sparse term matrix; densifies only query columns."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.vectorizer = CountVectorizer(stop_words="english")

    def fit(self, texts: list[str]) -> BM25:
        self.tf = self.vectorizer.fit_transform(texts).tocsc()  # sparse, column-sliceable
        self.doc_len = np.asarray(self.tf.sum(axis=1)).ravel()
        self.avg_len = float(self.doc_len.mean()) or 1.0
        n_docs = self.tf.shape[0]
        df = np.asarray((self.tf > 0).sum(axis=0)).ravel()
        self.idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        return self

    def score(self, query: str) -> np.ndarray:
        q_cols = np.asarray(
            self.vectorizer.transform([query]).tocoo().col, dtype=int
        )
        if q_cols.size == 0:
            return np.zeros(self.tf.shape[0])
        tf_q = self.tf[:, q_cols].toarray()  # docs × |query terms| — small by construction
        idf_q = self.idf[q_cols]
        denom = tf_q + self.k1 * (1 - self.b + self.b * (self.doc_len / self.avg_len))[:, None]
        return (idf_q * tf_q * (self.k1 + 1) / denom).sum(axis=1)


class SearchIndex:
    """Hybrid TF-IDF + BM25 with RRF fusion, query expansion, and MMR."""

    def __init__(self) -> None:
        self.chunks: list[dict] = []
        self.tfidf: TfidfVectorizer | None = None
        self.tfidf_matrix = None
        self.bm25: BM25 | None = None

    def build(self, chunks_path: Path) -> SearchIndex:
        with chunks_path.open(encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]
        if not self.chunks:
            raise ValueError(f"No chunks found in {chunks_path}")
        texts = [c["text"] for c in self.chunks]
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.tfidf_matrix = self.tfidf.fit_transform(texts)
        self.bm25 = BM25().fit(texts)
        return self

    def _ranks(self, scores: np.ndarray) -> dict[int, int]:
        order = np.argsort(scores)[::-1]
        return {int(idx): rank for rank, idx in enumerate(order) if scores[idx] > 0}

    def _mmr(self, candidates: list[int], fused: dict[int, float], top_k: int) -> list[int]:
        """Maximal Marginal Relevance: relevant but non-redundant results."""
        if len(candidates) <= top_k:
            return candidates
        sims = cosine_similarity(self.tfidf_matrix[candidates])
        pos = {c: i for i, c in enumerate(candidates)}
        selected: list[int] = []
        remaining = list(candidates)
        max_rel = max(fused[c] for c in candidates) or 1.0
        while remaining and len(selected) < top_k:
            best, best_score = None, -np.inf
            for c in remaining:
                rel = fused[c] / max_rel
                red = max((sims[pos[c], pos[s]] for s in selected), default=0.0)
                score = MMR_LAMBDA * rel - (1 - MMR_LAMBDA) * red
                if score > best_score:
                    best, best_score = c, score
            selected.append(best)
            remaining.remove(best)
        return selected

    def search(self, query: str, top_k: int = 5, expand_query: bool = True, mmr: bool = True) -> list[dict]:
        if self.tfidf is None:
            raise RuntimeError("Index not built. Call build() or load() first.")
        q = expand(query) if expand_query else query
        tfidf_scores = cosine_similarity(self.tfidf.transform([q]), self.tfidf_matrix).ravel()
        bm25_scores = self.bm25.score(q)
        fused: dict[int, float] = {}
        for ranks in (self._ranks(tfidf_scores), self._ranks(bm25_scores)):
            for idx, rank in ranks.items():
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank)
        if not fused:
            return []
        candidates = sorted(fused, key=fused.get, reverse=True)[: max(top_k * 3, top_k)]
        order = self._mmr(candidates, fused, top_k) if mmr else candidates[:top_k]
        return [
            {
                **self.chunks[i],
                "score": round(fused[i], 5),
                "tfidf": round(float(tfidf_scores[i]), 4),
                "bm25": round(float(bm25_scores[i]), 4),
                "expanded_query": q if q != query else None,
            }
            for i in order
        ]

    # -- persistence (no pickle, by design) --------------------------------

    def save(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        with (dir_path / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c) + "\n")
        (dir_path / "manifest.json").write_text(
            json.dumps({"version": 3, "n_chunks": len(self.chunks), "backend": "hybrid-tfidf-sparsebm25-mmr"})
        )

    @classmethod
    def load(cls, dir_path: Path) -> SearchIndex:
        manifest = dir_path / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(f"No index manifest at {dir_path}")
        return cls().build(dir_path / "chunks.jsonl")
