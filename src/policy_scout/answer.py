"""Extractive QA with citations, a pluggable composer seam, and a reranker seam.

The Composer protocol is the LLM integration point: `ExtractiveComposer` (default)
is fully offline and hallucination-free by construction; `LLMComposer` shows exactly
where Azure OpenAI Gov synthesis plugs in inside an authorized boundary. Retrieval,
sentence attribution, and citation plumbing are identical for both — the LLM only
ever sees retrieved, cited passages (grounded generation).

v0.4 adds the Reranker protocol — the second-stage (question, sentence) pair
scorer that closed the citation-precision leak (v0.3 known weakness #2):

* `LexicalReranker` (default, offline, auditable): IDF-weighted term overlap so
  generic words ("control", "information") stop qualifying sentences on their
  own; a hard requirement that at least one ORIGINAL query term matches (thesaurus
  expansions can support a sentence, never carry it); and an exact-phrase bonus.
* `CrossEncoderReranker` (optional, in-boundary): a true cross-encoder via
  sentence-transformers for deployments where a local model is authorized. Same
  interface, one config change — the same pattern as the embedding backend.
"""

from __future__ import annotations

import re
from typing import Protocol

from .index import SearchIndex
from .query import expand

_STOP = frozenset(
    ["the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "by", "with",
     "is", "are", "shall", "be", "that", "this", "it", "as", "from", "at", "what",
     "how", "which", "must", "may", "can", "does", "do"]
)


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.;])\s+(?=[A-Z(])", text)
    return [p.strip() for p in parts if len(p.split()) >= 5]


def _terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9&]+", text.lower()) if w not in _STOP and len(w) > 1}


def _stem(term: str) -> str:
    """Suffix-stripping stemmer sufficient for policy prose ("marking" == "marked").

    Deliberately not Porter: four auditable rules beat an algorithm nobody in the
    accreditation meeting can recite. Order matters; stems stay >= 4 chars.
    """
    for suffix in ("ings", "ing", "ied", "ies", "ed", "es", "s"):
        if term.endswith(suffix) and len(term) - len(suffix) >= 4:
            stem = term[: -len(suffix)]
            return stem + "y" if suffix in ("ied", "ies") else stem
    return term


class Composer(Protocol):
    def compose(self, question: str, picked: list[tuple[float, str, str]]) -> str: ...


class Reranker(Protocol):
    """Second-stage scorer for (question, candidate sentence) pairs."""

    def score(self, question: str, sentence: str, base: float) -> float: ...


class LexicalReranker:
    """Cross-encoder-shaped lexical reranker: IDF-weighted, expansion-skeptical.

    Three principles, each auditable line-by-line (the accreditation argument
    that justified the thesaurus over embeddings applies here too):

    1. **Original terms are load-bearing.** A sentence sharing zero terms with
       the user's actual question — only with thesaurus expansions — is vetoed.
       This single rule closed the v0.3 leak where a CI/CD sentence rode into
       CUI-marking answers on the back of "access control" -> "control".
    2. **Rarity is evidence.** Shared terms are weighted by corpus IDF, so
       "cui" and "safeguarding" count for far more than "information".
    3. **Phrases beat bags of words.** An exact multi-word phrase from the
       question appearing verbatim in the sentence earns a fixed bonus.
    """

    MIN_COVERAGE = 0.5  # sentence must cover >=50% of the question's IDF mass

    def __init__(self, idf: dict[str, float] | None = None) -> None:
        # IDF keyed by stem; max across morphological variants (favor rarity)
        self.idf: dict[str, float] = {}
        for term, val in (idf or {}).items():
            s = _stem(term)
            self.idf[s] = max(self.idf.get(s, 0.0), val)
        self.default_idf = (sum(self.idf.values()) / len(self.idf)) if self.idf else 1.0

    def _weight(self, term: str) -> float:
        return self.idf.get(term, self.default_idf)

    def score(self, question: str, sentence: str, base: float) -> float:
        q_orig = {_stem(t) for t in _terms(question)}
        q_exp = {_stem(t) for t in _terms(expand(question))} - q_orig
        s_terms = {_stem(t) for t in _terms(sentence)}

        shared_orig = q_orig & s_terms
        if not shared_orig:
            return 0.0  # expansion-only match: veto (principle 1)

        shared_exp = q_exp & s_terms
        w_orig = sum(self._weight(t) for t in shared_orig)
        w_query = sum(self._weight(t) for t in q_orig) or 1.0
        idf_overlap = w_orig / w_query
        # Topicality veto: mentioning one rare query term in passing is not an
        # answer. Multi-term questions demand majority IDF-mass coverage —
        # this is the rule that closed the v0.3 CI/CD-into-CUI leak.
        if len(q_orig) >= 3 and idf_overlap < self.MIN_COVERAGE:
            return 0.0
        exp_support = min(len(shared_exp), 3) * 0.03  # expansions support, capped

        phrase_bonus = 0.0
        q_words = [w for w in re.findall(r"[a-z0-9&]+", question.lower()) if w not in _STOP]
        s_lower = sentence.lower()
        for n in (3, 2):
            if any(" ".join(q_words[i:i + n]) in s_lower for i in range(len(q_words) - n + 1)):
                phrase_bonus = 0.10 if n == 2 else 0.15
                break

        return 0.55 * idf_overlap + exp_support + phrase_bonus + 0.30 * base


class CrossEncoderReranker:  # pragma: no cover — requires an in-boundary local model
    """True cross-encoder reranking via sentence-transformers (local model).

    For deployments where a local model is authorized inside the boundary
    (models run offline; no data leaves). Same seam as LexicalReranker —
    one config change, identical citation plumbing.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder  # lazy: optional heavy dep

        self.model = CrossEncoder(model_name)

    def score(self, question: str, sentence: str, base: float) -> float:
        import math

        logit = float(self.model.predict([(question, sentence)])[0])
        return 1.0 / (1.0 + math.exp(-logit))  # sigmoid -> [0, 1]


class ExtractiveComposer:
    """Default: verbatim cited sentences. Cannot hallucinate — it can only quote."""

    def compose(self, question: str, picked: list[tuple[float, str, str]]) -> str:
        if not picked:
            return "No supporting passages found in the indexed corpus."
        return " ".join(f"{sent} [{cite}]" for _, sent, cite in picked)


class LLMComposer:  # pragma: no cover — requires Azure OpenAI Gov inside the boundary
    """Grounded synthesis via Azure OpenAI (Gov). Sees ONLY retrieved cited passages."""

    SYSTEM = (
        "Answer strictly from the provided passages. Cite every claim with the "
        "bracketed passage id. If the passages do not answer the question, say so."
    )

    def __init__(self, endpoint: str, api_key: str, deployment: str) -> None:
        self.endpoint, self.api_key, self.deployment = endpoint, api_key, deployment

    def compose(self, question: str, picked: list[tuple[float, str, str]]) -> str:
        import httpx

        context = "\n".join(f"[{cite}] {sent}" for _, sent, cite in picked)
        url = (
            f"{self.endpoint}/openai/deployments/{self.deployment}"
            "/chat/completions?api-version=2024-02-01"
        )
        resp = httpx.post(
            url,
            headers={"api-key": self.api_key},
            json={
                "messages": [
                    {"role": "system", "content": self.SYSTEM},
                    {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {question}"},
                ],
                "temperature": 0.0,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def _index_idf(index: SearchIndex) -> dict[str, float]:
    """Per-term IDF from the index's fitted TF-IDF vocabulary (free — already computed)."""
    if index.tfidf is None or not hasattr(index.tfidf, "idf_"):
        return {}
    vocab = index.tfidf.vocabulary_
    idf = index.tfidf.idf_
    return {term: float(idf[i]) for term, i in vocab.items() if " " not in term}


def answer(
    index: SearchIndex,
    question: str,
    top_k_chunks: int = 4,
    top_sentences: int = 3,
    composer: Composer | None = None,
    reranker: Reranker | None = None,
) -> dict:
    composer = composer or ExtractiveComposer()
    reranker = reranker or LexicalReranker(idf=_index_idf(index))
    hits = index.search(question, top_k=top_k_chunks)
    q_terms = _terms(expand(question))  # expansion still gates candidacy cheaply
    scored: list[tuple[float, str, str]] = []
    top_chunk_score = hits[0]["score"] if hits else 1.0
    min_shared = 2 if len(q_terms) >= 3 else 1
    for h in hits:
        cite = f"{h['doc_id']}#{h['chunk_id']}"
        rel = h["score"] / top_chunk_score  # relative retrieval confidence in [0, 1]
        for sent in _sentences(h["text"]):
            s_terms = _terms(sent)
            if len(q_terms & s_terms) < min_shared:
                continue  # cheap first-stage gate
            score = reranker.score(question, sent, rel)  # second-stage pair scoring
            if score <= 0:
                continue  # vetoed (e.g. expansion-only match)
            scored.append((score, sent, cite))
    scored.sort(key=lambda t: t[0], reverse=True)
    picked = scored[:top_sentences]
    return {
        "question": question,
        "answer": composer.compose(question, picked),
        "composer": type(composer).__name__,
        "reranker": type(reranker).__name__,
        "sentences": [{"text": s, "citation": c, "score": round(sc, 4)} for sc, s, c in picked],
        "retrieved_chunks": [
            {"citation": f"{h['doc_id']}#{h['chunk_id']}", "section": h.get("section", ""), "score": h["score"]}
            for h in hits
        ],
    }
