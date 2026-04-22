"""
Multi-Armed Bandit for chunk selection.

Each document position is an "arm". The agent maintains Q-values per
domain and uses UCB or ε-greedy to trade off exploration/exploitation.
Supports incremental online updates via a reward signal.
"""

from __future__ import annotations
import json
import math
import random
from pathlib import Path
from dataclasses import dataclass, field

import config as cfg


@dataclass
class ArmStats:
    q_value: float = 0.0   # estimated value
    n_pulls: int = 0        # times selected
    # Thompson Sampling — Beta distribution parameters
    alpha: float = 1.0
    beta: float = 1.0


class BanditAgent:
    """
    Context-free multi-armed bandit with per-domain Q-tables.

    Arms are indexed 0..N-1 where N = number of candidate chunks.
    Because N varies per document, we store rolling stats in a
    fixed-size table (max 200 arms) indexed by *relative position bucket*.
    """

    N_BUCKETS = 200  # fixed arm table size; positions mapped to buckets

    def __init__(
        self,
        domain: str = "general",
        algorithm: str | None = None,
        epsilon: float | None = None,
        lr: float | None = None,
        ucb_c: float | None = None,
    ):
        self.domain = domain
        self.algorithm = algorithm or cfg.BANDIT_ALGORITHM
        self.epsilon = epsilon if epsilon is not None else cfg.BANDIT_EPSILON
        self.lr = lr if lr is not None else cfg.BANDIT_LR
        self.ucb_c = ucb_c if ucb_c is not None else cfg.BANDIT_UCB_C

        # Per-bucket arm statistics
        self.arms: list[ArmStats] = [ArmStats() for _ in range(self.N_BUCKETS)]
        self.total_steps: int = 0

        # History for dashboard
        self.reward_history: list[float] = []
        self.selected_history: list[list[int]] = []

    # ── Persistence ───────────────────────────────────────────────────────────

    @classmethod
    def load_or_create(cls, domain: str = "general", **kwargs) -> "BanditAgent":
        path = cls._state_path(domain)
        agent = cls(domain=domain, **kwargs)
        if path.exists():
            try:
                agent._load(path)
            except Exception:
                pass  # corrupt state → fresh agent
        return agent

    def save(self) -> None:
        path = self._state_path(self.domain)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "domain": self.domain,
            "algorithm": self.algorithm,
            "epsilon": self.epsilon,
            "lr": self.lr,
            "ucb_c": self.ucb_c,
            "total_steps": self.total_steps,
            "arms": [
                {"q": a.q_value, "n": a.n_pulls, "alpha": a.alpha, "beta": a.beta}
                for a in self.arms
            ],
            "reward_history": self.reward_history[-500:],   # keep last 500
            "selected_history": self.selected_history[-100:],
        }
        path.write_text(json.dumps(state, indent=2))

    def _load(self, path: Path) -> None:
        state = json.loads(path.read_text())
        self.algorithm = state.get("algorithm", self.algorithm)
        self.epsilon = state.get("epsilon", self.epsilon)
        self.lr = state.get("lr", self.lr)
        self.ucb_c = state.get("ucb_c", self.ucb_c)
        self.total_steps = state.get("total_steps", 0)
        for i, a in enumerate(state.get("arms", [])):
            if i < self.N_BUCKETS:
                self.arms[i] = ArmStats(
                    q_value=a["q"], n_pulls=a["n"],
                    alpha=a.get("alpha", 1.0), beta=a.get("beta", 1.0),
                )
        self.reward_history = state.get("reward_history", [])
        self.selected_history = state.get("selected_history", [])

    @staticmethod
    def _state_path(domain: str) -> Path:
        return cfg.RL_STATE_DIR / f"bandit_{domain}.json"

    # ── Core API ──────────────────────────────────────────────────────────────

    def select(
        self,
        n_arms: int,
        features: list[list[float]],
        budget: int,
        token_counts: list[int],
        top_k: int,
    ) -> list[int]:
        """
        Select a subset of arm indices within the token budget.

        Returns arm indices (chunk positions) chosen by the policy.
        """
        buckets = self._to_buckets(list(range(n_arms)), n_arms)
        arm_scores = [self._score(b) for b in buckets]

        # Rank by score, then greedily fill budget
        order = sorted(range(n_arms), key=lambda i: arm_scores[i], reverse=True)
        selected, used = [], 0
        for i in order:
            tc = token_counts[i]
            if used + tc <= budget:
                selected.append(i)
                used += tc
            if len(selected) >= top_k:
                break

        self.selected_history.append(selected)
        return selected

    def update(self, selected_indices: list[int], n_arms: int, reward: float) -> None:
        """
        Online update: adjust Q-values for selected arms toward the observed reward.
        """
        self.total_steps += 1
        buckets = self._to_buckets(selected_indices, n_arms)

        for b in buckets:
            arm = self.arms[b]
            arm.n_pulls += 1
            # Incremental mean update
            arm.q_value += self.lr * (reward - arm.q_value)
            # Thompson sampling parameters (treat reward as Bernoulli-like)
            r_clamped = max(0.0, min(1.0, (reward + 1) / 2))
            arm.alpha += r_clamped
            arm.beta += 1 - r_clamped

        self.reward_history.append(reward)

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score(self, bucket: int) -> float:
        arm = self.arms[bucket]
        if self.algorithm == "ucb":
            if arm.n_pulls == 0:
                return float("inf")
            bonus = self.ucb_c * math.sqrt(math.log(self.total_steps + 1) / arm.n_pulls)
            return arm.q_value + bonus
        elif self.algorithm == "thompson":
            return random.betavariate(arm.alpha, arm.beta)
        else:  # epsilon_greedy
            if random.random() < self.epsilon:
                return random.random()
            return arm.q_value

    # ── Utils ─────────────────────────────────────────────────────────────────

    def _to_buckets(self, indices: list[int], n_arms: int) -> list[int]:
        """Map arm indices (variable-length) to fixed bucket indices."""
        if n_arms <= 1:
            return [0] * len(indices)
        return [
            int(i / n_arms * (self.N_BUCKETS - 1))
            for i in indices
        ]

    # ── Dashboard helpers ─────────────────────────────────────────────────────

    def get_q_values(self) -> list[float]:
        return [a.q_value for a in self.arms]

    def get_pull_counts(self) -> list[int]:
        return [a.n_pulls for a in self.arms]

    def smoothed_rewards(self, window: int = 20) -> list[float]:
        h = self.reward_history
        if not h:
            return []
        result = []
        for i in range(len(h)):
            start = max(0, i - window + 1)
            result.append(sum(h[start: i + 1]) / (i - start + 1))
        return result

    def reset(self) -> None:
        self.arms = [ArmStats() for _ in range(self.N_BUCKETS)]
        self.total_steps = 0
        self.reward_history.clear()
        self.selected_history.clear()
