"""Baseline chunking strategies: full context, truncated head/tail."""

from __future__ import annotations
from .base import BaseChunker, Chunk, ChunkerParams


class FullContextChunker(BaseChunker):
    name = "full_context"
    description = "Returns the entire document as a single chunk (baseline)."

    def chunk(self, text: str, query: str = "") -> list[Chunk]:
        c = Chunk(text=text, index=0)
        c.token_count = self._count_tokens(text)
        # Hard truncate if over budget
        if c.token_count > self.params.max_token_budget:
            words = text.split()[: self.params.max_token_budget]
            c.text = " ".join(words)
            c.token_count = len(words)
        return [c]


class TruncateHeadChunker(BaseChunker):
    name = "truncate_head"
    description = "Keep only the first N tokens of the document."

    def chunk(self, text: str, query: str = "") -> list[Chunk]:
        words = text.split()[: self.params.max_token_budget]
        c = Chunk(text=" ".join(words), index=0, token_count=len(words))
        return [c]


class TruncateTailChunker(BaseChunker):
    name = "truncate_tail"
    description = "Keep only the last N tokens of the document."

    def chunk(self, text: str, query: str = "") -> list[Chunk]:
        words = text.split()
        words = words[-self.params.max_token_budget :]
        c = Chunk(text=" ".join(words), index=0, token_count=len(words))
        return [c]


class TruncateHeadTailChunker(BaseChunker):
    name = "truncate_head_tail"
    description = "Keep first and last N/2 tokens of the document."

    def chunk(self, text: str, query: str = "") -> list[Chunk]:
        budget = self.params.max_token_budget
        half = budget // 2
        words = text.split()
        if len(words) <= budget:
            selected = words
        else:
            selected = words[:half] + words[-half:]
        c = Chunk(text=" ".join(selected), index=0, token_count=len(selected))
        return [c]
