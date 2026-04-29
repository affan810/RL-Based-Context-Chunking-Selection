"""RL Bandit chunker — delegates to rl/bandit_agent.py for chunk selection."""

from __future__ import annotations
from .base import BaseChunker, Chunk, ChunkerParams
from .structural import RecursiveChunker


class RLBanditChunker(BaseChunker):
    """
    Uses the RL bandit policy to select chunks.

    Each candidate chunk is an "arm". The bandit uses learned Q-values
    (per domain) to greedily select the highest-value subset within budget.
    """

    name = "rl_bandit"
    description = "Multi-armed bandit selects chunks by learned Q-values per domain."

    def chunk(self, text: str, query: str = "") -> list[Chunk]:
        p = self.params
        # First get candidate chunks via recursive splitter
        splitter = RecursiveChunker(p)
        candidates = splitter.chunk(text)

        if not candidates:
            return []

        from rl.bandit_agent import BanditAgent
        agent = BanditAgent.load_or_create(domain=p.domain)

        # Build feature vectors for each chunk
        q_terms = set(query.lower().split()) if query else set()
        features = []
        for c in candidates:
            c_terms = set(c.text.lower().split())
            overlap = len(q_terms & c_terms) / (len(q_terms) + 1e-9)
            density = c.token_count / max(p.chunk_size, 1)
            position = c.index / max(len(candidates) - 1, 1)
            features.append([overlap, density, position])

        # Agent selects arm indices greedily within budget
        selected_indices = agent.select(
            n_arms=len(candidates),
            features=features,
            budget=p.max_token_budget,
            token_counts=[c.token_count for c in candidates],
            top_k=p.top_k,
        )
        agent.save()

        selected = [candidates[i] for i in selected_indices]
        if p.preserve_order:
            selected.sort(key=lambda c: c.index)
        return selected
