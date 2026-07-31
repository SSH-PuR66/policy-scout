"""CLI: python -m policy_scout.cli index | search "query" | answer "question" | eval"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .answer import answer as extract_answer
from .evaluate import evaluate
from .index import SearchIndex
from .ingest import ingest_directory

INDEX_DIR = Path("data/index")


def _load() -> SearchIndex:
    return SearchIndex().build(INDEX_DIR / "chunks.jsonl")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "index":
        chunks = ingest_directory(Path("data/raw"), INDEX_DIR / "chunks.jsonl")
        idx = SearchIndex().build(INDEX_DIR / "chunks.jsonl")
        idx.save(INDEX_DIR)
        print(f"Indexed {len(chunks)} chunks -> {INDEX_DIR}")
    elif cmd == "search":
        for r in _load().search(" ".join(sys.argv[2:])):
            sec = f" ({r['section']})" if r.get("section") else ""
            print(f"[rrf {r['score']:.4f}] {r['doc_id']}#{r['chunk_id']}{sec}: {r['text'][:120]}...")
    elif cmd == "answer":
        result = extract_answer(_load(), " ".join(sys.argv[2:]))
        print(result["answer"])
    elif cmd == "eval":
        report = evaluate(_load())
        print(json.dumps({k: v for k, v in report.items() if k != "per_query"}, indent=2))
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
