"""Metric helpers for the evaluation pipeline."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class RunMetrics:
    strategy: str
    tokens_used: int
    token_budget: int
    token_efficiency: float    # quality / token_fraction
    quality_score: float       # [0, 1] from evaluator
    correctness: float
    completeness: float
    factual_alignment: float
    redundancy: float          # [0, 1] pairwise Jaccard
    coverage: float            # [0, 1]
    latency: float             # seconds
    answer: str
    chunks_selected: int
    reward: float              # RL reward (computed even for non-RL methods)
    evaluator_reasoning: str = ""


def token_efficiency(quality: float, tokens_used: int, budget: int) -> float:
    """Quality per unit token fraction used (higher = better)."""
    frac = tokens_used / max(budget, 1)
    return quality / max(frac, 1e-6)


def pairwise_redundancy(texts: list[str]) -> float:
    if len(texts) < 2:
        return 0.0
    sets = [set(t.lower().split()) for t in texts]
    sims = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            inter = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            sims.append(inter / union if union else 0.0)
    return sum(sims) / len(sims) if sims else 0.0


def coverage_score(texts: list[str]) -> float:
    if not texts:
        return 0.0
    all_terms: set[str] = set()
    total = 0
    for t in texts:
        terms = set(t.lower().split())
        all_terms |= terms
        total += len(terms)
    return len(all_terms) / max(total, 1)
