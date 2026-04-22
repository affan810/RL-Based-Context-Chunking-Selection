"""
Reward signal computation.

R = α·quality − β·token_cost − γ·redundancy + δ·coverage − ε·latency
"""

from __future__ import annotations
import time
from dataclasses import dataclass

import config as cfg


@dataclass
class RewardComponents:
    quality: float = 0.0       # [0, 1]
    token_cost: float = 0.0    # [0, 1]
    redundancy: float = 0.0    # [0, 1]
    coverage: float = 0.0      # [0, 1]
    latency: float = 0.0       # [0, 1]
    total: float = 0.0


def compute_reward(
    quality_score: float,       # 0..1 from ChatGPT evaluator
    tokens_used: int,
    token_budget: int,
    chunk_texts: list[str],
    latency_seconds: float,
    max_latency: float = 30.0,
    alpha: float = cfg.REWARD_ALPHA,
    beta: float = cfg.REWARD_BETA,
    gamma: float = cfg.REWARD_GAMMA,
    delta: float = cfg.REWARD_DELTA,
    epsilon: float = cfg.REWARD_EPSILON,
) -> RewardComponents:
    """
    Compute multi-component reward.

    Args:
        quality_score: ChatGPT evaluation score normalized to [0,1]
        tokens_used: actual tokens in selected chunks
        token_budget: maximum allowed tokens
        chunk_texts: list of selected chunk text strings
        latency_seconds: wall-clock time for full pipeline step
        max_latency: normalization constant for latency
    """
    r = RewardComponents()

    # Quality (primary)
    r.quality = float(quality_score)

    # Token cost: fraction of budget used (lower = better, penalty)
    r.token_cost = min(tokens_used / max(token_budget, 1), 1.0)

    # Redundancy: mean pairwise Jaccard similarity between chunks
    r.redundancy = _pairwise_redundancy(chunk_texts)

    # Coverage: fraction of unique terms across chunks vs. all possible
    r.coverage = _coverage_score(chunk_texts)

    # Latency: normalized
    r.latency = min(latency_seconds / max_latency, 1.0)

    r.total = (
        alpha * r.quality
        - beta * r.token_cost
        - gamma * r.redundancy
        + delta * r.coverage
        - epsilon * r.latency
    )
    return r


def _pairwise_redundancy(texts: list[str]) -> float:
    if len(texts) < 2:
        return 0.0
    sets = [set(t.lower().split()) for t in texts]
    sims = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            inter = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            sims.append(inter / union if union else 0)
    return sum(sims) / len(sims) if sims else 0.0


def _coverage_score(texts: list[str]) -> float:
    if not texts:
        return 0.0
    all_terms = set()
    per_chunk = []
    for t in texts:
        terms = set(t.lower().split())
        all_terms |= terms
        per_chunk.append(terms)
    # Coverage: ratio of unique terms covered
    return min(len(all_terms) / max(sum(len(s) for s in per_chunk), 1), 1.0)
