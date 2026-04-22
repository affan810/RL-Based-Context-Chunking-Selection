"""Retrieval-based chunkers: TF-IDF, semantic top-K, hybrid."""

from __future__ import annotations
import math
import re
from collections import Counter
from .base import BaseChunker, Chunk, ChunkerParams
from .structural import RecursiveChunker


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


# ── TF-IDF ───────────────────────────────────────────────────────────────────

class TFIDFChunker(BaseChunker):
    name = "tfidf"
    description = "Scores chunks via TF-IDF against query; selects top-K."

    def chunk(self, text: str, query: str = "") -> list[Chunk]:
        p = self.params
        splitter = RecursiveChunker(p)
        raw_chunks = splitter.chunk(text)  # get all chunks, no query filter

        if not query:
            return self._trim_to_budget(raw_chunks, p.max_token_budget)

        corpus = [_tokenize(c.text) for c in raw_chunks]
        q_tokens = _tokenize(query)

        # IDF
        N = len(corpus)
        df: Counter[str] = Counter()
        for doc in corpus:
            for term in set(doc):
                df[term] += 1
        idf = {t: math.log((N + 1) / (df[t] + 1)) + 1 for t in df}

        for chunk, doc_tokens in zip(raw_chunks, corpus):
            tf = Counter(doc_tokens)
            total = len(doc_tokens) + 1
            score = sum(
                (tf[t] / total) * idf.get(t, 0) for t in q_tokens
            )
            chunk.score = score

        raw_chunks.sort(key=lambda c: c.score, reverse=True)
        selected = self._trim_to_budget(raw_chunks[: p.top_k], p.max_token_budget)
        if p.preserve_order:
            selected.sort(key=lambda c: c.index)
        return selected


# ── Semantic Top-K ─────────────────────────────────────────────────────────────

class SemanticTopKChunker(BaseChunker):
    name = "semantic_topk"
    description = "Embeds chunks + query; selects top-K by cosine similarity."

    def chunk(self, text: str, query: str = "") -> list[Chunk]:
        p = self.params
        splitter = RecursiveChunker(p)
        raw_chunks = splitter.chunk(text)

        if not query:
            return self._trim_to_budget(raw_chunks, p.max_token_budget)

        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except ImportError:
            # Graceful degradation to TF-IDF
            return TFIDFChunker(p).chunk(text, query)

        model = SentenceTransformer(p.embedding_model)
        texts = [c.text for c in raw_chunks]
        corpus_embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        query_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

        scores = corpus_embs @ query_emb  # cosine (normalized)
        for chunk, score in zip(raw_chunks, scores):
            chunk.score = float(score)

        raw_chunks.sort(key=lambda c: c.score, reverse=True)
        selected = self._trim_to_budget(raw_chunks[: p.top_k], p.max_token_budget)
        if p.preserve_order:
            selected.sort(key=lambda c: c.index)
        return selected


# ── Hybrid Retrieval ──────────────────────────────────────────────────────────

class HybridRetrievalChunker(BaseChunker):
    name = "hybrid_retrieval"
    description = "Fuses BM25 + semantic scores (α·sem + β·bm25); selects top-K."

    def chunk(self, text: str, query: str = "") -> list[Chunk]:
        p = self.params
        splitter = RecursiveChunker(p)
        raw_chunks = splitter.chunk(text)

        if not query:
            return self._trim_to_budget(raw_chunks, p.max_token_budget)

        bm25_scores = self._bm25_scores(raw_chunks, query)
        semantic_scores = self._semantic_scores(raw_chunks, query, p)

        # Normalize then fuse
        bm25_n = _normalize(bm25_scores)
        sem_n = _normalize(semantic_scores)

        for i, chunk in enumerate(raw_chunks):
            chunk.score = p.hybrid_alpha * sem_n[i] + p.hybrid_beta * bm25_n[i]

        raw_chunks.sort(key=lambda c: c.score, reverse=True)
        selected = self._trim_to_budget(raw_chunks[: p.top_k], p.max_token_budget)
        if p.preserve_order:
            selected.sort(key=lambda c: c.index)
        return selected

    def _bm25_scores(self, chunks: list[Chunk], query: str) -> list[float]:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return [0.0] * len(chunks)
        corpus = [_tokenize(c.text) for c in chunks]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(_tokenize(query))
        return scores.tolist()

    def _semantic_scores(self, chunks: list[Chunk], query: str, p: ChunkerParams) -> list[float]:
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            model = SentenceTransformer(p.embedding_model)
            texts = [c.text for c in chunks]
            embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            qemb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
            return (embs @ qemb).tolist()
        except ImportError:
            return [0.0] * len(chunks)


def _normalize(scores: list[float]) -> list[float]:
    mn, mx = min(scores, default=0), max(scores, default=1)
    rng = mx - mn if mx != mn else 1.0
    return [(s - mn) / rng for s in scores]
