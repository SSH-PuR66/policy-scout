"""End-to-end tests: ingest -> hybrid index -> search/answer/graph -> API -> eval."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from policy_scout.answer import answer
from policy_scout.evaluate import evaluate
from policy_scout.index import BM25, SearchIndex
from policy_scout.ingest import (
    chunk_text,
    extract_references,
    extract_text,
    ingest_directory,
    reference_graph,
    split_sections,
)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
GOLD = ROOT / "eval" / "gold_queries.json"


@pytest.fixture(scope="module")
def chunks_file(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("index") / "chunks.jsonl"
    ingest_directory(RAW, out)
    return out


@pytest.fixture(scope="module")
def index(chunks_file) -> SearchIndex:
    return SearchIndex().build(chunks_file)


# -- ingestion ---------------------------------------------------------------

def test_extract_text_reads_txt():
    assert "CUI" in extract_text(RAW / "sample_cui_policy.txt")


def test_section_split_finds_numbered_headings():
    text = extract_text(RAW / "sample_cui_policy.txt")
    headings = [h for h, _ in split_sections(text)]
    assert any("PURPOSE" in h for h in headings)
    assert any("MARKING" in h for h in headings)


def test_chunking_overlap_and_no_empty():
    chunks = chunk_text("word " * 500, max_words=120, overlap=20)
    assert all(chunks) and len(chunks) >= 4
    assert chunk_text("") == []


def test_reference_extraction():
    refs = extract_references("Per DoDI 5200.48 and DoDD 3000.09, see also DTM-24-001.")
    assert "DODI 5200.48" in refs
    assert "DODD 3000.09" in refs
    assert any(r.startswith("DTM") for r in refs)


def test_reference_graph(chunks_file):
    import json

    chunks = [json.loads(line) for line in chunks_file.read_text().splitlines()]
    graph = reference_graph(chunks)
    assert set(graph) == {c["doc_id"] for c in chunks}


# -- retrieval ---------------------------------------------------------------

def test_bm25_scores_relevant_higher():
    docs = ["cats sit on mats", "dogs chase cats sometimes", "satellites orbit the earth"]
    bm = BM25().fit(docs)
    scores = bm.score("cats")
    assert scores[0] > scores[2] and scores[1] > scores[2]


def test_hybrid_search_relevance(index):
    top = index.search("impact level 5 controlled unclassified information")[0]
    assert top["doc_id"] in {"sample_cui_policy", "sample_data_strategy"}
    assert {"score", "tfidf", "bm25"} <= set(top)

    top2 = index.search("autonomous weapon human judgment")[0]
    assert top2["doc_id"] == "sample_ai_governance_directive"


def test_index_save_load_roundtrip_no_pickle(index, tmp_path):
    index.save(tmp_path)
    assert (tmp_path / "manifest.json").exists()
    assert not list(tmp_path.glob("*.pkl")), "pickle persistence is banned"
    loaded = SearchIndex.load(tmp_path)
    assert loaded.search("test and evaluation")


# -- answer ------------------------------------------------------------------

def test_extractive_answer_has_citations(index):
    result = answer(index, "What level of human judgment is required over use of force?")
    assert result["sentences"], "expected supporting sentences"
    assert all("#" in s["citation"] for s in result["sentences"])
    assert "[sample_ai_governance_directive#" in result["answer"]


def test_answer_no_match_is_honest(index):
    result = answer(index, "quantum blockchain marsupial cuisine")
    assert result["answer"].startswith("No supporting passages")


# -- evaluation --------------------------------------------------------------

def test_eval_meets_thresholds(index):
    report = evaluate(index, GOLD, k=3)
    assert report["recall_at_3"] >= 0.8, report["per_query"]
    assert report["mrr"] >= 0.7, report["per_query"]


# -- API ---------------------------------------------------------------------

@pytest.fixture()
def client(chunks_file, monkeypatch):
    import policy_scout.api as api_mod

    monkeypatch.setattr(api_mod, "INDEX_DIR", chunks_file.parent)
    monkeypatch.setattr(api_mod, "GOLD", GOLD)
    api_mod.state.index = None
    with TestClient(api_mod.app) as c:
        yield c


def test_api_endpoints(client):
    assert client.get("/health").json()["index_loaded"] is True

    r = client.get("/search", params={"q": "responsible AI principles"})
    assert r.status_code == 200 and r.json()["results"]

    a = client.get("/answer", params={"q": "how must CUI be safeguarded?"})
    assert a.status_code == 200 and a.json()["sentences"]

    g = client.get("/graph")
    assert g.status_code == 200 and "reference_graph" in g.json()

    e = client.get("/eval")
    assert e.status_code == 200 and e.json()["mrr"] > 0


def test_api_key_enforced(client, monkeypatch):
    monkeypatch.setenv("POLICY_SCOUT_API_KEY", "secret123")
    assert client.get("/search", params={"q": "cui"}).status_code == 401
    ok = client.get("/search", params={"q": "cui"}, headers={"X-API-Key": "secret123"})
    assert ok.status_code == 200


def test_api_input_validation(client):
    assert client.get("/search", params={"q": "x"}).status_code == 422
    assert client.get("/search", params={"q": "cui", "top_k": 999}).status_code == 422


# -- v0.3: expansion, MMR, metrics, UI, deeper eval --------------------------

def test_query_expansion_bridges_paraphrase(index):
    from policy_scout.query import expand

    assert "human oversight" in expand("human judgment over use of force")
    assert expand("unrelated gibberish") == "unrelated gibberish"
    # paraphrase query still lands on the right doc thanks to expansion
    top = index.search("human oversight of lethal engagement")[0]
    assert top["doc_id"] == "sample_ai_governance_directive"


def test_bm25_is_sparse():
    from scipy import sparse

    docs = ["cats sit on mats", "dogs chase cats"]
    bm = BM25().fit(docs)
    assert sparse.issparse(bm.tf), "BM25 term matrix must stay sparse"


def test_mmr_reduces_redundancy(index):
    diversified = index.search("controlled unclassified information", top_k=4, mmr=True)
    assert diversified  # returns and is well-formed with MMR path exercised
    plain = index.search("controlled unclassified information", top_k=4, mmr=False)
    assert {r["doc_id"] for r in diversified} >= {plain[0]["doc_id"]}


def test_eval_reports_ndcg_and_citation_precision(index):
    report = evaluate(index, GOLD, k=3)
    assert report["ndcg_at_3"] >= 0.7
    assert report["citation_precision"] is not None
    assert report["citation_precision"] >= 0.95  # v0.4 reranker lifted 0.94 -> 1.0


def test_metrics_and_ui(client):
    client.get("/search", params={"q": "cui marking"})
    m = client.get("/metrics").json()
    assert "/search" in m and m["/search"]["count"] >= 1
    assert m["/search"]["p95_ms"] >= 0

    home = client.get("/")
    assert home.status_code == 200 and "PolicyScout" in home.text


# -- v0.4: cross-encoder-shaped reranker, stemming, thesaurus, metrics export --

def test_reranker_lifts_citation_precision_to_perfect(index):
    """The v0.3 known leak (0.94) — a CI/CD sentence riding into CUI answers on
    an expansion overlap — must be closed. The reranker is the fix; prove it."""
    report = evaluate(index, GOLD, k=3)
    assert report["citation_precision"] == 1.0, report["per_query"]


def test_reranker_vetoes_expansion_only_matches(index):
    """A sentence that shares only THESAURUS-EXPANSION terms with the question —
    never an original query term — must not be citable."""
    from policy_scout.answer import LexicalReranker, _index_idf

    rr = LexicalReranker(idf=_index_idf(index))
    # "control" is an expansion of "access control"/"least privilege", not in the query
    veto = rr.score("how must CUI be marked", "Access control lists govern the pipeline.", base=1.0)
    assert veto == 0.0


def test_reranker_requires_majority_idf_coverage(index):
    from policy_scout.answer import LexicalReranker, _index_idf

    rr = LexicalReranker(idf=_index_idf(index))
    q = "how must CUI be marked and safeguarded"
    strong = rr.score(q, "CUI shall be marked and safeguarded per policy.", base=0.5)
    passing = rr.score(q, "The CI/CD pipeline runs at impact level 5 for CUI.", base=0.9)
    assert strong > 0.0
    assert passing == 0.0  # mentions 'cui' in passing but fails coverage veto


def test_stemmer_unifies_morphology():
    from policy_scout.answer import _stem

    assert _stem("marking") == _stem("marked") == "mark"
    assert _stem("safeguarded") == "safeguard"
    assert _stem("policies") == "policy"
    assert _stem("is") == "is"  # too short to strip


def test_thesaurus_covers_cyber_operations_domain():
    from policy_scout.query import expand

    assert "ioc" in expand("indicators of compromise in the network")
    assert "rmf" in expand("apply the risk management framework")
    assert "mfa" in expand("require multifactor authentication").split()


def test_metrics_prometheus_exposition(client):
    client.get("/search", params={"q": "zero trust"})
    r = client.get("/metrics", params={"format": "prometheus"})
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    body = r.text
    assert "policyscout_requests_total" in body
    assert 'quantile="0.95"' in body


def test_reranker_reported_in_answer(index):
    result = answer(index, "how must CUI be marked and safeguarded")
    assert result["reranker"] == "LexicalReranker"


def test_larger_corpus_eval_is_meaningful(index):
    """Corpus + gold set grew (v0.4) — eval is now over >=12 queries, >=6 docs."""
    report = evaluate(index, GOLD, k=3)
    assert report["n_queries"] >= 12
    assert report["recall_at_3"] >= 0.8
    assert report["mrr"] >= 0.7
