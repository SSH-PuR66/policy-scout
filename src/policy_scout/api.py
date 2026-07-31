"""FastAPI service: search, answer, graph, eval, metrics, and a built-in UI.

Hardening (v0.2) + observability (v0.3) + Prometheus export (v0.4): per-endpoint request counts and latency
percentiles at /metrics, timing middleware, lifespan-managed state, optional
X-API-Key auth (constant-time compare; CAC/mTLS at the gateway in IL5 production),
strict input validation.
"""

from __future__ import annotations

import logging
import os
import secrets
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from .answer import answer as extract_answer
from .evaluate import evaluate
from .index import SearchIndex
from .ingest import ingest_directory, reference_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("policy_scout.api")

DATA_RAW = Path(os.environ.get("POLICY_SCOUT_RAW", "data/raw"))
INDEX_DIR = Path(os.environ.get("POLICY_SCOUT_INDEX", "data/index"))
GOLD = Path(os.environ.get("POLICY_SCOUT_GOLD", "eval/gold_queries.json"))
STATIC = Path(__file__).resolve().parent / "static" / "index.html"


class State:
    def __init__(self) -> None:
        self.index: SearchIndex | None = None
        self.counts: dict[str, int] = defaultdict(int)
        self.latencies: dict[str, list[float]] = defaultdict(list)


state = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    chunks = INDEX_DIR / "chunks.jsonl"
    if chunks.exists():
        state.index = SearchIndex().build(chunks)
        log.info("index loaded: %d chunks", len(state.index.chunks))
    else:
        log.warning("no index at %s — POST /reindex to build", chunks)
    yield
    state.index = None


app = FastAPI(
    title="PolicyScout",
    description="Hybrid semantic search + cited extractive QA over DoD policy documents",
    version="0.4.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    path = request.url.path
    state.counts[path] += 1
    buf = state.latencies[path]
    buf.append(elapsed)
    if len(buf) > 1000:  # bounded memory
        del buf[: len(buf) - 1000]
    response.headers["X-Response-Time-Ms"] = f"{elapsed:.1f}"
    return response


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get("POLICY_SCOUT_API_KEY")
    if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
        raise HTTPException(401, "invalid or missing X-API-Key")


def get_index() -> SearchIndex:
    if state.index is None:
        raise HTTPException(503, "Index not built. POST /reindex first.")
    return state.index


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui() -> str:
    return STATIC.read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": "0.4.0",
        "index_loaded": state.index is not None,
        "chunks": len(state.index.chunks) if state.index else 0,
    }


def _pct(v: list[float], p: float) -> float:
    s = sorted(v)
    return round(s[min(int(len(s) * p), len(s) - 1)], 2) if s else 0.0


@app.get("/metrics")
def metrics(format: str = Query("json", pattern="^(json|prometheus)$")):
    """In-process metrics. `?format=prometheus` emits the standard text
    exposition format so a Prometheus scraper (or OTel collector with a
    prometheus receiver) ingests it with zero extra dependencies — the v0.3
    'metrics reset on restart' weakness answered at the export layer first.
    """
    if format == "prometheus":
        lines = [
            "# HELP policyscout_requests_total Requests per endpoint since process start",
            "# TYPE policyscout_requests_total counter",
        ]
        for path, count in state.counts.items():
            lines.append(f'policyscout_requests_total{{path="{path}"}} {count}')
        lines += [
            "# HELP policyscout_latency_ms Request latency quantiles (in-process window)",
            "# TYPE policyscout_latency_ms summary",
        ]
        for path, lat in state.latencies.items():
            lines.append(f'policyscout_latency_ms{{path="{path}",quantile="0.5"}} {_pct(lat, 0.50)}')
            lines.append(f'policyscout_latency_ms{{path="{path}",quantile="0.95"}} {_pct(lat, 0.95)}')
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

    return {
        path: {
            "count": state.counts[path],
            "p50_ms": _pct(lat, 0.50),
            "p95_ms": _pct(lat, 0.95),
        }
        for path, lat in state.latencies.items()
    }


@app.post("/reindex", dependencies=[Depends(require_api_key)])
def reindex() -> dict:
    chunks = ingest_directory(DATA_RAW, INDEX_DIR / "chunks.jsonl")
    state.index = SearchIndex().build(INDEX_DIR / "chunks.jsonl")
    state.index.save(INDEX_DIR)
    log.info("reindexed: %d chunks", len(chunks))
    return {"chunks_indexed": len(chunks)}


@app.get("/search", dependencies=[Depends(require_api_key)])
def search(
    q: str = Query(..., min_length=2, max_length=500),
    top_k: int = Query(5, ge=1, le=50),
    expand: bool = Query(True, description="apply domain-thesaurus query expansion"),
    mmr: bool = Query(True, description="apply MMR result diversification"),
) -> dict:
    return {"query": q, "results": get_index().search(q, top_k=top_k, expand_query=expand, mmr=mmr)}


@app.get("/answer", dependencies=[Depends(require_api_key)])
def answer(q: str = Query(..., min_length=5, max_length=500)) -> dict:
    return extract_answer(get_index(), q)


@app.get("/graph", dependencies=[Depends(require_api_key)])
def graph() -> dict:
    return {"reference_graph": reference_graph(get_index().chunks)}


@app.get("/eval", dependencies=[Depends(require_api_key)])
def run_eval() -> dict:
    if not GOLD.exists():
        raise HTTPException(404, f"no gold query set at {GOLD}")
    report = evaluate(get_index(), GOLD)
    report.pop("per_query", None)
    return report
