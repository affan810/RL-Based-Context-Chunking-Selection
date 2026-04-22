"""Abstract base for all chunking strategies."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    text: str
    index: int                          # original position in doc
    token_count: int = 0
    score: float = 0.0                  # relevance score (retrieval methods)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk(idx={self.index}, tokens={self.token_count}, score={self.score:.3f}, '{preview}...')"


@dataclass
class ChunkerParams:
    """Unified parameter bag — each chunker reads only what it needs."""
    max_token_budget: int = 1024
    chunk_size: int = 256
    overlap: float = 0.15           # fraction
    top_k: int = 5
    embedding_model: str = "all-MiniLM-L6-v2"
    similarity_metric: str = "cosine"
    hybrid_alpha: float = 0.6
    hybrid_beta: float = 0.4
    min_chunk_size: int = 50
    max_chunk_size: int = 512
    semantic_threshold: float = 0.5
    window_size: int = 256
    stride: int = 128
    mmr_lambda: float = 0.5
    domain: str = "general"
    preserve_order: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class BaseChunker(ABC):
    """All chunkers implement this interface."""

    name: str = "base"
    description: str = ""

    def __init__(self, params: ChunkerParams | None = None):
        self.params = params or ChunkerParams()

    @abstractmethod
    def chunk(self, text: str, query: str = "") -> list[Chunk]:
        """
        Split/select from text and return an ordered list of Chunk objects.
        The total token count of returned chunks should respect params.max_token_budget.
        """
        ...

    # ── Helpers shared by subclasses ─────────────────────────────────────────

    def _count_tokens(self, text: str) -> int:
        """Approximate token count (word-level proxy — fast, no model required)."""
        return max(1, len(text.split()))

    def _split_into_sentences(self, text: str) -> list[str]:
        """Naive sentence splitter (avoids heavy NLP dependency at import time)."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if s.strip()]

    def _assign_token_counts(self, chunks: list[Chunk]) -> list[Chunk]:
        for c in chunks:
            c.token_count = self._count_tokens(c.text)
        return chunks

    def _trim_to_budget(self, chunks: list[Chunk], budget: int) -> list[Chunk]:
        """Keep chunks (in order) until budget is exhausted."""
        selected, used = [], 0
        for c in chunks:
            if used + c.token_count > budget:
                break
            selected.append(c)
            used += c.token_count
        return selected

    def _token_window(self, text: str, size: int, stride: int) -> list[str]:
        """Split text into overlapping token windows."""
        words = text.split()
        windows = []
        i = 0
        while i < len(words):
            window = words[i: i + size]
            windows.append(" ".join(window))
            if i + size >= len(words):
                break
            i += stride
        return windows
