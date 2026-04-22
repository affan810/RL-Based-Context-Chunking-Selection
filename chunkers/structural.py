"""Structural chunking: sliding window, semantic, recursive."""

from __future__ import annotations
import re
from .base import BaseChunker, Chunk, ChunkerParams


class SlidingWindowChunker(BaseChunker):
    name = "sliding_window"
    description = "Overlapping fixed-size windows; selects top-K by TF-IDF score."

    def chunk(self, text: str, query: str = "") -> list[Chunk]:
        p = self.params
        windows = self._token_window(text, p.window_size, p.stride)
        chunks = [
            Chunk(text=w, index=i, token_count=self._count_tokens(w))
            for i, w in enumerate(windows)
        ]
        if not query:
            return self._trim_to_budget(chunks, p.max_token_budget)
        # Score by query-term overlap
        q_terms = set(query.lower().split())
        for c in chunks:
            terms = set(c.text.lower().split())
            c.score = len(q_terms & terms) / (len(q_terms) + 1e-9)
        chunks.sort(key=lambda c: c.score, reverse=True)
        selected = self._trim_to_budget(chunks[: p.top_k], p.max_token_budget)
        if p.preserve_order:
            selected.sort(key=lambda c: c.index)
        return selected


class SemanticChunker(BaseChunker):
    """Split on paragraph / sentence boundaries, then merge small chunks."""

    name = "semantic_chunking"
    description = "Paragraph-aware split; merges under-sized chunks; selects by similarity."

    def chunk(self, text: str, query: str = "") -> list[Chunk]:
        p = self.params
        raw_chunks = self._split_paragraphs(text)
        raw_chunks = self._merge_small(raw_chunks, p.min_chunk_size)

        chunks = [
            Chunk(text=c, index=i, token_count=self._count_tokens(c))
            for i, c in enumerate(raw_chunks)
        ]

        if not query:
            return self._trim_to_budget(chunks, p.max_token_budget)

        chunks = self._score_by_overlap(chunks, query)
        chunks.sort(key=lambda c: c.score, reverse=True)
        selected = self._trim_to_budget(chunks[: p.top_k], p.max_token_budget)
        if p.preserve_order:
            selected.sort(key=lambda c: c.index)
        return selected

    def _split_paragraphs(self, text: str) -> list[str]:
        paras = re.split(r"\n{2,}", text.strip())
        result = []
        for para in paras:
            para = para.strip()
            if not para:
                continue
            if self._count_tokens(para) > self.params.max_chunk_size:
                # further split long paragraphs at sentence level
                sents = self._split_into_sentences(para)
                result.extend(sents)
            else:
                result.append(para)
        return result

    def _merge_small(self, chunks: list[str], min_size: int) -> list[str]:
        merged, buf = [], ""
        for c in chunks:
            buf = (buf + " " + c).strip()
            if self._count_tokens(buf) >= min_size:
                merged.append(buf)
                buf = ""
        if buf:
            merged.append(buf)
        return merged

    def _score_by_overlap(self, chunks: list[Chunk], query: str) -> list[Chunk]:
        q_terms = set(query.lower().split())
        for c in chunks:
            terms = set(c.text.lower().split())
            c.score = len(q_terms & terms) / (len(q_terms) + 1e-9)
        return chunks


class RecursiveChunker(BaseChunker):
    """Hierarchical splitting: try paragraph → sentence → word boundaries."""

    name = "recursive"
    description = "Recursively splits on paragraph → sentence → word boundaries."

    SEPARATORS = ["\n\n", "\n", ". ", " "]

    def chunk(self, text: str, query: str = "") -> list[Chunk]:
        p = self.params
        raw = self._recursive_split(text, self.SEPARATORS, p.chunk_size)
        chunks = [
            Chunk(text=c, index=i, token_count=self._count_tokens(c))
            for i, c in enumerate(raw)
        ]
        if not query:
            return self._trim_to_budget(chunks, p.max_token_budget)

        q_terms = set(query.lower().split())
        for c in chunks:
            terms = set(c.text.lower().split())
            c.score = len(q_terms & terms) / (len(q_terms) + 1e-9)
        chunks.sort(key=lambda c: c.score, reverse=True)
        selected = self._trim_to_budget(chunks[: p.top_k], p.max_token_budget)
        if p.preserve_order:
            selected.sort(key=lambda c: c.index)
        return selected

    def _recursive_split(self, text: str, seps: list[str], target: int) -> list[str]:
        if not seps:
            # word-level fallback
            words = text.split()
            return [" ".join(words[i: i + target]) for i in range(0, len(words), target)]

        sep = seps[0]
        pieces = text.split(sep)
        result, buf = [], ""
        for piece in pieces:
            candidate = (buf + sep + piece).strip() if buf else piece.strip()
            if self._count_tokens(candidate) <= target:
                buf = candidate
            else:
                if buf:
                    result.append(buf)
                if self._count_tokens(piece) > target:
                    result.extend(self._recursive_split(piece, seps[1:], target))
                    buf = ""
                else:
                    buf = piece.strip()
        if buf:
            result.append(buf)
        return [r for r in result if r.strip()]
