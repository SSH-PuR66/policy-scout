# Engineering Critique Log

Every version gets an adversarial review; findings drive the next version. This is the
JATIC mindset applied to our own code: assume it's broken until proven otherwise.

## v0.1 findings → fixed in v0.2

| # | Severity | Finding | Fix |
|---|---|---|---|
| 1 | **Critical** | `pickle.load()` for index persistence = arbitrary code execution if the file is tampered with. Disqualifying for an accreditation boundary. | Pickle removed entirely; JSON manifest + deterministic rebuild. A test (`test_index_save_load_roundtrip_no_pickle`) bans regressions. |
| 2 | High | TF-IDF-only retrieval; word-window chunking shreds the numbered-section structure policy documents are built on. | Section-aware chunker + hybrid BM25/TF-IDF with reciprocal rank fusion. |
| 3 | High | No retrieval evaluation — claimed JATIC-style reproducibility, measured nothing. | Gold-query harness (recall@3, MRR) run in tests; CI fails below recall 0.8 / MRR 0.7. |
| 4 | Medium | No answer layer — retrieval alone is a search box, not an analytic tool. | Offline extractive QA with per-sentence `[doc#chunk]` citations; honest "no supporting passages" on no-match (hallucination guard). |
| 5 | Medium | No cross-reference graph — issuance citation networks are GAMECHANGER's core navigational feature. | Regex reference extractor (DoDD/DoDI/DoDM/DTM/CJCSI) per chunk + `/graph` endpoint. |
| 6 | Medium | API: mutable global index in handlers, no auth, no logging, unbounded inputs. | Lifespan-managed state, optional `X-API-Key` (constant-time compare; gateway CAC/mTLS in production), structured logging, strict Query bounds (422 on abuse). |
| 7 | Medium | No container, no SAST, no dependency scanning, no SBOM. | Non-root multi-stage Dockerfile (Iron Bank rebase path documented), bandit + pip-audit + CycloneDX SBOM + container smoke test in CI. |
| 8 | Low | Sample corpus only ~6 chunks — eval numbers are not statistically meaningful. | Acknowledged; `fetch_corpus.py` swaps in real issuances. Thresholds re-baseline on the real corpus. |

## Known remaining weaknesses (v0.2 — be honest)

1. **Extractive QA is shallow.** Term-overlap sentence scoring misses paraphrase
   ("use of force" vs "lethal engagement"). Real fix: LLM synthesis via Azure OpenAI
   Gov inside the boundary, retrieval + citations unchanged.
2. **BM25 uses a dense term matrix** (`toarray()`), O(docs × vocab) memory — fine to
   ~10⁴ chunks, wrong at Advana scale. Fix: sparse BM25 or Azure AI Search/pgvector.
3. **Rebuild-on-load persistence** trades startup latency for safety — correct at this
   scale, needs a real vector store beyond it.
4. **Reference extractor is regex-only**: misses prose references ("the CUI
   instruction") and titles. Fix: NER pass or LLM extraction.
5. **API-key auth is a placeholder**, not IL5 identity. Production = CAC/PIV via
   gateway + Azure AD Gov; the dependency hook is where that lands.
6. **Eval set is tiny and author-written** — gold labels should come from SMEs
   (expert-driven development, literally).

## v0.2 findings → fixed in v0.3

| # | Severity | Finding | Fix |
|---|---|---|---|
| 1 | High | Dense BM25 term matrix — O(docs × vocab) memory, dies at scale. | Sparse CSC matrix; scoring densifies only the query's columns: O(docs × query terms). Regression test asserts sparsity. |
| 2 | High | Lexical retrieval blind to paraphrase ("use of force" ≠ "lethal engagement"). | Curated DoD-domain thesaurus expansion — auditable line-by-line, unlike embeddings, which matters when an accreditor asks *why* a document surfaced. Verified by a paraphrase-query test. |
| 3 | Medium | Top-k results often redundant (3 chunks of the same section). | MMR diversification (λ=0.7) over the RRF candidate pool. |
| 4 | Medium | No LLM integration seam — "swap in an LLM" was hand-waving. | `Composer` protocol: `ExtractiveComposer` (default, quote-only, cannot hallucinate) / `LLMComposer` (Azure OpenAI Gov, grounded: sees ONLY retrieved cited passages, temperature 0). |
| 5 | Medium | Zero observability — no latency data, no counters. | Timing middleware + `/metrics` (per-endpoint count, p50/p95 ms, bounded buffers) + `X-Response-Time-Ms` header. |
| 6 | Medium | Eval measured retrieval only; the answer layer was untested against gold. | nDCG@k added; **citation precision** — do answer citations point at gold-relevant docs? — added as an anti-hallucination gate (≥0.9 in CI). |
| 7 | Low | API-only; nothing demoable to a non-engineer. | Self-contained single-file web UI at `/` (no build step, no external CDNs — deployable in disconnected environments). |

### Bugs caught by the new tests during v0.3 development (verification works)

- **Thesaurus substring bug:** "cui" expanded inside "cuisine", making gibberish
  queries retrieve CUI policy. Fixed with word-boundary regex. Caught by the
  no-match honesty test.
- **nDCG > 1.0:** repeated chunks from one document inflated DCG above the ideal.
  Fixed by rank-order dedup of doc_ids. Caught by the nDCG bound test.
- **Gold-label narrowness:** the T&E query legitimately matches two documents; the
  label, not the system, was wrong. Labels revised — eval sets need review too.

## v0.3 findings → fixed in v0.4

| # | Severity | Finding | Fix |
|---|---|---|---|
| 1 | High | Citation precision stuck at 0.94 — a CI/CD-pipeline sentence leaked into a CUI-marking answer because thesaurus expansion ("access control" → "control") let it share terms with the query. | **Cross-encoder-shaped `LexicalReranker`** as a second stage. Three auditable rules: (a) a sentence sharing only *expansion* terms — never an original query term — is vetoed; (b) shared terms are IDF-weighted so rare terms dominate; (c) multi-term questions must cover ≥50% of the query's IDF mass (a topicality veto). Citation precision **0.94 → 1.00**, proven by `test_reranker_lifts_citation_precision_to_perfect`. |
| 2 | Medium | Paraphrase coverage thin outside the AI/CUI core; nothing for the cyber-operations vocabulary the tool is pitched against. | Thesaurus expanded from 17 to ~50 curated entries — zero trust, RMF/800-53, ICAM/MFA, IOC/CTI, SBOM/SCRM, incident response, DevSecOps. Still line-by-line auditable. |
| 3 | Medium | Morphology blind: "marking" ≠ "marked", "policies" ≠ "policy" — the reranker double-counted variants and missed matches. | Four-rule suffix stemmer (`_stem`) applied in reranking; deliberately not Porter, so an accreditor can recite it. Verified by `test_stemmer_unifies_morphology`. |
| 4 | Medium | Metrics in-process only — no path to a monitoring stack. | `/metrics?format=prometheus` emits the standard text exposition format (OTel-collector compatible) with zero new dependencies. Counters stay in-process; **export** is the part production needed first. |
| 5 | Low | Eval set (6 queries / 3 docs) too small for the numbers to mean much. | Corpus doubled to 6 synthetic issuances (28 chunks); gold set doubled to 12 queries. recall@3 1.0 · MRR 1.0 · nDCG@3 0.97 · citation precision 1.0 — now over a set with enough spread to be worth reporting. |

### Bugs caught by the new tests during v0.4 development

- **Reranker over-corrected first (0.94 → 0.92):** a pure IDF-overlap reranker still
  admitted the CI/CD sentence (it shares the rare term "cui") *and* dropped a legitimate
  title-line citation. The IDF-mass **coverage veto** — not just overlap weighting — was
  the actual fix. Two wrong hypotheses before the right one; the eval caught both.
- **Stray corpus file:** an empty `sample_zero_trust_strategy.txt` shadowed the real
  zero-trust doc during authoring — caught immediately because its chunk count was wrong.

## Known remaining weaknesses (v0.4 — still honest)

1. The `LexicalReranker` is a lexical stand-in for a true cross-encoder. `CrossEncoderReranker`
   (sentence-transformers) ships behind the same seam, but the default stays offline and
   deterministic; the real model needs an in-boundary authorization to run.
2. The IDF-coverage veto is tuned on a 12-query set. On the real corpus the 0.5 threshold
   re-baselines with the rest — it is a knob, not a law.
3. Metrics export is Prometheus text; full OTel traces/spans are the next rung.
4. Stemmer is suffix-only — it will mis-stem irregulars ("analyses"→"analys"). Acceptable
   at this scale; a real morphological analyzer is the scalable fix.
5. Thesaurus is still hand-curated — coverage grows only as SMEs contribute. The in-boundary
   embedding model remains the endgame for paraphrase.
6. Sample corpus is synthetic; all thresholds re-baseline on real issuances via `fetch_corpus.py`.
