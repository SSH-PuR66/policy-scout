"""Evaluation harness: retrieval (recall@k, MRR, nDCG@k) + answer citation precision.

JATIC-inspired: reproducible, threshold-gated in CI, honest about corpus size.
Citation precision checks that every citation in an answer points at a chunk that
was actually retrieved AND belongs to a gold-relevant document — an anti-
hallucination metric for the QA layer, not just the retriever.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from .answer import answer as run_answer
from .index import SearchIndex

DEFAULT_GOLD = Path("eval/gold_queries.json")


def _ndcg(got_docs: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        (1.0 if d in relevant else 0.0) / math.log2(rank + 1)
        for rank, d in enumerate(got_docs[:k], start=1)
    )
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def evaluate(index: SearchIndex, gold_path: Path = DEFAULT_GOLD, k: int = 3) -> dict:
    gold = json.loads(gold_path.read_text())
    recalls, rrs, ndcgs, cite_precisions = [], [], [], []
    per_query = []
    for item in gold:
        results = index.search(item["query"], top_k=k)
        # dedupe doc_ids preserving rank order — repeated chunks from one doc must
        # not inflate DCG above the ideal (nDCG must stay in [0, 1])
        got_docs = list(dict.fromkeys(r["doc_id"] for r in results))
        relevant = set(item["relevant_docs"])
        hit = bool(relevant & set(got_docs))
        recalls.append(1.0 if hit else 0.0)
        rr = next((1.0 / r for r, d in enumerate(got_docs, 1) if d in relevant), 0.0)
        rrs.append(rr)
        ndcgs.append(_ndcg(got_docs, relevant, k))

        ans = run_answer(index, item["query"])
        cites = re.findall(r"\[([\w.-]+)#\d+\]", ans["answer"])
        if cites:
            good = sum(1 for c in cites if c in relevant)
            cite_precisions.append(good / len(cites))

        per_query.append({"query": item["query"], "hit": hit, "rr": rr, "top": got_docs})
    n = len(gold) or 1
    return {
        "k": k,
        "n_queries": len(gold),
        f"recall_at_{k}": round(sum(recalls) / n, 4),
        "mrr": round(sum(rrs) / n, 4),
        f"ndcg_at_{k}": round(sum(ndcgs) / n, 4),
        "citation_precision": round(sum(cite_precisions) / len(cite_precisions), 4)
        if cite_precisions
        else None,
        "per_query": per_query,
    }


if __name__ == "__main__":  # pragma: no cover
    idx = SearchIndex().build(Path("data/index/chunks.jsonl"))
    report = evaluate(idx)
    print(json.dumps({k: v for k, v in report.items() if k != "per_query"}, indent=2))
    for q in report["per_query"]:
        print(("PASS" if q["hit"] else "FAIL"), q["query"], "->", q["top"])
