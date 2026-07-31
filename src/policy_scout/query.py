"""Query expansion with a DoD-domain thesaurus.

Lexical retrieval fails on paraphrase ("use of force" vs "lethal engagement").
Until an embedding model runs inside the boundary, a curated domain thesaurus
closes most of the gap — and unlike embeddings it is auditable line-by-line,
which matters when an accreditor asks *why* a document was retrieved.
"""

from __future__ import annotations

import re

# Curated, auditable expansion map. Extend with SME input (expert-driven, literally).
THESAURUS: dict[str, list[str]] = {
    "use of force": ["lethal engagement", "engagement", "weapons employment"],
    "human judgment": ["human oversight", "human control", "operator control"],
    "autonomous": ["autonomy", "semi-autonomous", "unmanned"],
    "cui": ["controlled unclassified information"],
    "controlled unclassified information": ["cui"],
    "il5": ["impact level 5"],
    "impact level": ["il4", "il5", "authorization level"],
    "test and evaluation": ["t&e", "verification and validation", "v&v"],
    "artificial intelligence": ["ai", "machine learning", "ml"],
    "machine learning": ["ml", "artificial intelligence", "data science"],
    "cybersecurity": ["information security", "risk management framework", "rmf"],
    "workforce": ["talent", "upskilling", "training"],
    "governance": ["oversight", "policy", "responsibilities"],
    "marking": ["banner marking", "dissemination control"],
    "safeguarding": ["protection", "access control", "encryption"],
    "responsible ai": ["ethical ai", "traceable", "governable", "equitable"],
    # -- v0.4 coverage expansion: cyber operations & accreditation ---------
    "zero trust": ["ztna", "never trust always verify", "microsegmentation"],
    "authorization to operate": ["ato", "accreditation", "authorization boundary"],
    "ato": ["authorization to operate", "accreditation"],
    "rmf": ["risk management framework", "nist 800-53", "security controls"],
    "risk management framework": ["rmf", "security controls", "continuous monitoring"],
    "incident response": ["incident handling", "cyber incident", "containment", "eradication"],
    "insider threat": ["insider risk", "user activity monitoring", "personnel security"],
    "supply chain": ["scrm", "software supply chain", "vendor risk"],
    "sbom": ["software bill of materials", "component inventory"],
    "vulnerability": ["cve", "weakness", "exposure", "flaw remediation"],
    "threat intelligence": ["cyber threat intelligence", "cti", "indicators of compromise", "ioc"],
    "indicators of compromise": ["ioc", "observables", "artifacts"],
    "ransomware": ["extortion malware", "data encryption attack"],
    "malware": ["malicious software", "malicious code"],
    "phishing": ["social engineering", "spearphishing", "credential harvesting"],
    "encryption": ["cryptographic protection", "cryptography", "fips 140"],
    "identity": ["icam", "authentication", "credentialing", "piv", "cac"],
    "multifactor": ["mfa", "two-factor", "phishing-resistant authentication"],
    "continuous monitoring": ["conmon", "ongoing authorization", "security posture"],
    "penetration testing": ["red team", "adversarial assessment", "security assessment"],
    "red team": ["penetration testing", "adversarial emulation", "opfor"],
    "audit": ["logging", "audit trail", "accountability records"],
    "logging": ["audit records", "audit trail", "telemetry"],
    "cloud": ["cloud computing", "cloud service offering", "cso", "fedramp"],
    "fedramp": ["federal risk and authorization management program", "cloud authorization"],
    "interoperability": ["data sharing", "coalition", "mission partner environment"],
    "resilience": ["resiliency", "survivability", "continuity of operations", "coop"],
    "spillage": ["unauthorized disclosure", "data spill", "compromise"],
    "sanitization": ["media sanitization", "destruction", "degaussing", "purging"],
    "least privilege": ["access control", "need to know", "separation of duties"],
    "devsecops": ["secure software development", "ci/cd security", "software factory"],
    "software factory": ["devsecops", "platform one", "continuous delivery"],
}


def expand(query: str, max_terms: int = 12) -> str:
    """Append thesaurus expansions for any phrase present in the query."""
    q_lower = re.sub(r"\s+", " ", query.lower())
    additions: list[str] = []
    for phrase, synonyms in THESAURUS.items():
        if re.search(rf"\b{re.escape(phrase)}\b", q_lower):
            additions.extend(s for s in synonyms if s not in q_lower)
    additions = list(dict.fromkeys(additions))[:max_terms]
    return f"{query} {' '.join(additions)}".strip() if additions else query
