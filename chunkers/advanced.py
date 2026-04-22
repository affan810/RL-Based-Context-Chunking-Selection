"""Advanced chunking: MMR, query-aware dynamic, parent-child (unused slot)."""

from __future__ import annotations
import numpy as np
from .base import BaseChunker, Chunk, ChunkerParams
from .structural import RecursiveChunker
from .retrieval import _normalize, _tokenize


# ── MMR ───────────────────────────────────────────────────────────────────────

class MMRChunker(BaseChunker):
    """Maximal Marginal Relevance: balances relevance and diversity."""

    name = "mmr"
    description = "MMR selection: λ·relevance − (1−λ)·max_sim_to_selected."

    def chunk(self, text: str, query: str = "") -> list[Chunk]:
        p = self.params
        splitter = RecursiveChunker(p)
        raw_chunks = splitter.chunk(text)

        if not query or len(raw_chunks) <= 1:
            return self._trim_to_budget(raw_chunks, p.max_token_budget)

        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(p.embedding_model)
            texts = [c.text for c in raw_chunks]
            embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            qemb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
            rel_scores = (embs @ qemb).tolist()
        except ImportError:
            # Fallback: term overlap scores
            q_terms = set(_tokenize(query))
            rel_scores = [
                len(q_terms & set(_tokenize(c.text))) / (len(q_terms) + 1e-9)
                for c in raw_chunks
            ]
            embs = None

        selected_indices: list[int] = []
        remaining = list(range(len(raw_chunks)))
        lam = p.mmr_lambda

        while remaining and self._budget_ok(selected_indices, raw_chunks, p.max_token_budget):
            if not selected_indices:
                # Pick highest relevance first
                best = max(remaining, key=lambda i: rel_scores[i])
            else:
                def mmr_score(i: int) -> float:
                    rel = rel_scores[i]
                    if embs is not None:
                        sim_to_sel = max(
                            float(embs[i] @ embs[j]) for j in selected_indices
                        )
                    else:
                        sel_texts = set(
                            w for j in selected_indices for w in _tokenize(raw_chunks[j].text)
                        )
                        cur_terms = set(_tokenize(raw_chunks[i].text))
                        overlap = len(sel_texts & cur_terms) / (len(cur_terms) + 1e-9)
                        sim_to_sel = overlap
                    return lam * rel - (1 - lam) * sim_to_sel

                best = max(remaining, key=mmr_score)

            next_tokens = raw_chunks[best].token_count
            used = sum(raw_chunks[j].token_count for j in selected_indices)
            if used + next_tokens > p.max_token_budget:
                break

            raw_chunks[best].score = rel_scores[best]
            selected_indices.append(best)
            remaining.remove(best)

            if len(selected_indices) >= p.top_k:
                break

        result = [raw_chunks[i] for i in selected_indices]
        if p.preserve_order:
            result.sort(key=lambda c: c.index)
        return result

    def _budget_ok(self, indices: list[int], chunks: list[Chunk], budget: int) -> bool:
        return sum(chunks[i].token_count for i in indices) < budget


# ── Query-Aware Dynamic Chunker ───────────────────────────────────────────────

class QueryAwareChunker(BaseChunker):
    """Adjusts chunk size based on query complexity then selects via hybrid score."""

    name = "query_aware"
    description = "Adapts chunk size to query length/complexity; hybrid selection."

    def chunk(self, text: str, query: str = "") -> list[Chunk]:
        p = self.params
        # Heuristic: longer query → smaller, more precise chunks
        q_words = len(query.split()) if query else 10
        if q_words <= 5:
            target_size = p.chunk_size
        elif q_words <= 15:
            target_size = max(p.min_chunk_size, p.chunk_size // 2)
        else:
            target_size = p.min_chunk_size

        adapted_params = ChunkerParams(
            **{k: v for k, v in p.__dict__.items() if k != "chunk_size"},
            chunk_size=target_size,
        )
        from .retrieval import HybridRetrievalChunker
        return HybridRetrievalChunker(adapted_params).chunk(text, query)
