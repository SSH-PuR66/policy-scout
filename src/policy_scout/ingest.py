"""Document ingestion: extract text, section-aware chunking, cross-reference extraction.

Policy documents are structured (numbered sections, enclosures). Chunking on section
boundaries preserves semantic units; a word-window fallback handles unstructured text.
Cross-references to other issuances (e.g. "DoDI 5200.48") are extracted per chunk to
build a citation graph — the core navigational feature of GAMECHANGER.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None

# Matches "DoDD 3000.09", "DoDI 5200.48", "DoDM 5200.01, Volume 1", "DTM-24-001", etc.
REFERENCE_RE = re.compile(
    r"\b(DoD[DIM]?|DTM|CJCSI|SecDef Memo)[\s-]*(\d{3,5}\.\d{1,2}|\d{2}-\d{3})\b",
    re.IGNORECASE,
)

# Numbered top-level section headings: "1. PURPOSE.", "2.3. Policy", "SECTION 4:"
SECTION_RE = re.compile(r"(?m)^\s*(?:SECTION\s+\d+|(?:\d+\.){1,3})\s*[A-Z][A-Z /,&-]{2,}\.?")


@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str
    source_path: str
    section: str = ""
    references: list[str] = field(default_factory=list)


def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        if PdfReader is None:
            raise RuntimeError("pypdf is required for PDF ingestion")
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="replace")


def extract_references(text: str) -> list[str]:
    """Normalized issuance identifiers referenced in text (e.g. 'DODI 5200.48')."""
    return sorted({f"{m.group(1).upper()} {m.group(2)}" for m in REFERENCE_RE.finditer(text)})


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split on numbered section headings. Returns (heading, body) pairs."""
    matches = list(SECTION_RE.finditer(text))
    if len(matches) < 2:
        return [("", text)]
    sections = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("PREAMBLE", preamble))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading = " ".join(m.group(0).split()).rstrip(".")
        body = text[m.end() : end].strip()
        sections.append((heading, body))
    return sections


def chunk_text(text: str, max_words: int = 120, overlap: int = 20) -> list[str]:
    """Overlapping word windows — fallback splitter for long or unstructured bodies."""
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split(" ")
    if not words or words == [""]:
        return []
    chunks = []
    step = max_words - overlap
    for start in range(0, len(words), step):
        window = words[start : start + max_words]
        if len(window) < 15 and chunks:
            chunks[-1] = chunks[-1] + " " + " ".join(window)
            break
        chunks.append(" ".join(window))
    return chunks


def chunk_document(text: str, max_words: int = 160) -> list[tuple[str, str]]:
    """Section-aware chunking: one chunk per section, window-split when oversized."""
    out: list[tuple[str, str]] = []
    for heading, body in split_sections(text):
        full = f"{heading}. {body}" if heading else body
        if len(full.split()) <= max_words:
            if full.strip():
                out.append((heading, re.sub(r"\s+", " ", full).strip()))
        else:
            out.extend((heading, piece) for piece in chunk_text(full, max_words=max_words))
    return out


def ingest_directory(raw_dir: Path, out_path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(raw_dir.iterdir()):
        if path.suffix.lower() not in {".pdf", ".txt"}:
            continue
        text = extract_text(path)
        for i, (section, piece) in enumerate(chunk_document(text)):
            chunks.append(
                Chunk(
                    doc_id=path.stem,
                    chunk_id=i,
                    text=piece,
                    source_path=str(path),
                    section=section,
                    references=extract_references(piece),
                )
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c)) + "\n")
    return chunks


def reference_graph(chunks: list[Chunk] | list[dict]) -> dict[str, list[str]]:
    """doc_id -> sorted unique issuances it references."""
    graph: dict[str, set] = {}
    for c in chunks:
        d = c if isinstance(c, dict) else asdict(c)
        graph.setdefault(d["doc_id"], set()).update(d.get("references", []))
    return {k: sorted(v) for k, v in graph.items()}


if __name__ == "__main__":  # pragma: no cover
    import sys

    raw = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/index/chunks.jsonl")
    result = ingest_directory(raw, out)
    print(f"Ingested {len(result)} chunks from {raw} -> {out}")
