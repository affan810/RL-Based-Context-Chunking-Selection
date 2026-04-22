"""Load documents and Q&A pairs from the data directory."""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

import config as cfg


@dataclass
class QAPair:
    question: str
    golden_answer: str


@dataclass
class Document:
    id: str
    title: str
    domain: str
    text: str
    qa_pairs: list[QAPair]

    def __repr__(self) -> str:
        return f"Document(id={self.id!r}, domain={self.domain!r}, words={len(self.text.split())})"


def load_documents(path: Path | None = None) -> list[Document]:
    """Load all documents from the sample docs JSON file."""
    p = path or (cfg.SAMPLE_DOCS_DIR / "documents.json")
    if not p.exists():
        return []
    raw = json.loads(p.read_text())
    docs = []
    for item in raw:
        docs.append(Document(
            id=item["id"],
            title=item["title"],
            domain=item["domain"],
            text=item["text"],
            qa_pairs=[
                QAPair(question=pair["question"], golden_answer=pair["golden_answer"])
                for pair in item.get("qa_pairs", [])
            ],
        ))
    return docs


def get_document_by_id(doc_id: str) -> Document | None:
    docs = load_documents()
    for d in docs:
        if d.id == doc_id:
            return d
    return None


def get_document_titles() -> list[dict]:
    """Return lightweight list for UI dropdowns."""
    return [
        {"id": d.id, "title": d.title, "domain": d.domain, "qa_count": len(d.qa_pairs)}
        for d in load_documents()
    ]
