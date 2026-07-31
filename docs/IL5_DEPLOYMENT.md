# IL5 Deployment Architecture

How PolicyScout would deploy for real CUI workloads, using only platforms from the
approved tools list.

## Impact-level boundary

| Data | Where it may live | PolicyScout posture |
|---|---|---|
| Public / unclassified | Public GitHub, commercial cloud | This repo (code + public issuances) |
| CUI | IL5-authorized platforms only (GenAI.mil, CamoGPT, Azure Government IL5 enclave) | Deployed instance + its corpus |
| Classified | SIPR / IL6 (CamoGPT only, per approved list) | Out of scope |

**Rule applied:** "CUI is authorized only on IL5 platforms." Code is unclassified and
public; data never ships with the code. The corpus is mounted at deploy time inside
the boundary.

## Target architecture (Azure Government, FedRAMP High)

```
   CAC-authenticated users
          │  (EAMS-A / Azure AD Gov, CAC as auth factor)
          ▼
   App Gateway + WAF  ──►  App Service / AKS (FastAPI container)
          │                        │
          │                Azure OpenAI (Gov) embeddings   ← AzureEmbeddingBackend
          │                        │
          ▼                        ▼
   Azure Storage (CUI corpus,  Azure AI Search or pgvector
   encrypted at rest, CMK)     (vector index)
```

- **Compute:** container on Azure Government App Service or AKS; DoD Container
  Hardening (Iron Bank base images) for production.
- **Embeddings:** `AzureEmbeddingBackend` pointed at an Azure OpenAI deployment inside
  the Gov tenant — no CUI transits commercial endpoints. The offline TF-IDF backend
  remains the fallback for disconnected/edge environments.
- **Identity:** CAC authentication fronting all endpoints (matching the pattern of every
  CDAO Advana tool).
- **Encryption:** TLS 1.2+ in transit; customer-managed keys at rest.
- **PII:** corpus is policy text (no PII); any user-generated content would require
  anonymization or a completed Privacy Impact Assessment before ingestion.

## DevSecOps path

1. Public GitHub → CI (lint + tests) on every PR — this repo, today.
2. Mirror to the sponsoring org's internal repo (e.g., Platform One / GitLab).
3. Pipeline adds SAST, dependency scanning (SBOM), container scan against Iron Bank.
4. Deploy through the enclave's cATO/CtF process into the IL5 environment.

## What changes between demo and IL5

| Concern | Demo (this repo) | IL5 production |
|---|---|---|
| Corpus | Synthetic samples / public issuances | CUI policy libraries |
| Embeddings | TF-IDF (offline) | Azure OpenAI Gov |
| Index | Pickle on disk | Azure AI Search / pgvector, backed up |
| Auth | None | CAC via Azure AD Gov |
| Hosting | localhost / any container host | Azure Government IL5 enclave |
