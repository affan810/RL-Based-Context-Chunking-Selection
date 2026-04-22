"""Strategy registry — import by name."""

from __future__ import annotations
from .base import BaseChunker, Chunk, ChunkerParams
from .baselines import FullContextChunker, TruncateHeadChunker, TruncateTailChunker, TruncateHeadTailChunker
from .retrieval import TFIDFChunker, SemanticTopKChunker, HybridRetrievalChunker
from .structural import SlidingWindowChunker, SemanticChunker, RecursiveChunker
from .advanced import MMRChunker, QueryAwareChunker
from .rl_bandit import RLBanditChunker

REGISTRY: dict[str, type[BaseChunker]] = {
    "full_context":        FullContextChunker,
    "truncate_head":       TruncateHeadChunker,
    "truncate_tail":       TruncateTailChunker,
    "truncate_head_tail":  TruncateHeadTailChunker,
    "tfidf":               TFIDFChunker,
    "semantic_topk":       SemanticTopKChunker,
    "hybrid_retrieval":    HybridRetrievalChunker,
    "sliding_window":      SlidingWindowChunker,
    "semantic_chunking":   SemanticChunker,
    "recursive":           RecursiveChunker,
    "mmr":                 MMRChunker,
    "query_aware":         QueryAwareChunker,
    "rl_bandit":           RLBanditChunker,
}


def get_chunker(name: str, params: ChunkerParams | None = None) -> BaseChunker:
    if name not in REGISTRY:
        raise ValueError(f"Unknown chunker: {name!r}. Available: {list(REGISTRY)}")
    return REGISTRY[name](params)


__all__ = [
    "BaseChunker", "Chunk", "ChunkerParams",
    "REGISTRY", "get_chunker",
    "FullContextChunker", "TruncateHeadChunker", "TruncateTailChunker", "TruncateHeadTailChunker",
    "TFIDFChunker", "SemanticTopKChunker", "HybridRetrievalChunker",
    "SlidingWindowChunker", "SemanticChunker", "RecursiveChunker",
    "MMRChunker", "QueryAwareChunker", "RLBanditChunker",
]
