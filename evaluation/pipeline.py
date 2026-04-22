"""
Main evaluation pipeline.

For a given (document, query, golden_answer) triple, runs ALL or a subset
of chunking strategies, generates TinyLlama answers, evaluates with ChatGPT,
and returns RunMetrics for each strategy.

Also handles RL bandit updates after each run.
"""

from __future__ import annotations
import time
from typing import Optional

import config as cfg
from chunkers import get_chunker, ChunkerParams, REGISTRY
from llm.tinyllama import TinyLlamaModel
from llm.openai_evaluator import evaluate_answer
from rl.bandit_agent import BanditAgent
from rl.reward import compute_reward
from evaluation.metrics import RunMetrics, pairwise_redundancy, coverage_score, token_efficiency


class EvaluationPipeline:
    """
    Orchestrates one full evaluation pass.

    Usage:
        pipe = EvaluationPipeline()
        results = pipe.run(
            document="...",
            query="...",
            golden_answer="...",
            domain="article",
            strategies=["full_context", "rl_bandit", ...],
            params=ChunkerParams(max_token_budget=512),
        )
    """

    def __init__(
        self,
        llm: Optional[TinyLlamaModel] = None,
        update_rl: bool = True,
    ):
        self._llm = llm  # lazy-loaded on first call
        self.update_rl = update_rl

    def _get_llm(self) -> TinyLlamaModel:
        if self._llm is None:
            self._llm = TinyLlamaModel()
        return self._llm

    def run(
        self,
        document: str,
        query: str,
        golden_answer: str,
        domain: str = "general",
        strategies: list[str] | None = None,
        params: ChunkerParams | None = None,
    ) -> list[RunMetrics]:
        """Run all requested strategies and return metrics list."""
        strategies = strategies or list(REGISTRY.keys())
        base_params = params or ChunkerParams()
        # Apply domain hints
        hints = cfg.DOMAIN_CHUNK_HINTS.get(domain, {})
        if hints:
            base_params = ChunkerParams(
                **{**base_params.__dict__,
                   "chunk_size": hints.get("chunk_size", base_params.chunk_size),
                   "top_k": hints.get("top_k", base_params.top_k),
                   "preserve_order": hints.get("preserve_order", base_params.preserve_order),
                   "domain": domain,
                   }
            )

        llm = self._get_llm()
        results: list[RunMetrics] = []

        for strategy in strategies:
            try:
                metrics = self._run_single(
                    strategy=strategy,
                    document=document,
                    query=query,
                    golden_answer=golden_answer,
                    domain=domain,
                    params=base_params,
                    llm=llm,
                )
                results.append(metrics)
            except Exception as e:
                # Don't let one failure kill the whole run
                results.append(RunMetrics(
                    strategy=strategy,
                    tokens_used=0, token_budget=base_params.max_token_budget,
                    token_efficiency=0, quality_score=0,
                    correctness=0, completeness=0, factual_alignment=0,
                    redundancy=0, coverage=0, latency=0,
                    answer=f"ERROR: {e}", chunks_selected=0, reward=-1,
                ))

        return results

    def _run_single(
        self,
        strategy: str,
        document: str,
        query: str,
        golden_answer: str,
        domain: str,
        params: ChunkerParams,
        llm: TinyLlamaModel,
    ) -> RunMetrics:

        # 1. Chunk
        t0 = time.time()
        chunker = get_chunker(strategy, params)
        chunks = chunker.chunk(document, query)
        chunk_texts = [c.text for c in chunks]
        context = llm.chunks_to_context(chunks)
        tokens_used = sum(c.token_count for c in chunks)

        # 2. Generate answer
        resp = llm.generate(context, query)
        latency = time.time() - t0

        # 3. Evaluate
        eval_result = evaluate_answer(query, golden_answer, resp.answer)

        # 4. Compute reward components
        reward_comps = compute_reward(
            quality_score=eval_result.score,
            tokens_used=tokens_used,
            token_budget=params.max_token_budget,
            chunk_texts=chunk_texts,
            latency_seconds=latency,
        )

        # 5. Update RL bandit if this was the bandit strategy
        if strategy == "rl_bandit" and self.update_rl:
            agent = BanditAgent.load_or_create(domain=domain)
            if agent.selected_history:
                last_selected = agent.selected_history[-1]
                n_candidates = len(chunk_texts) + max(0, len(last_selected) - len(chunk_texts))
                agent.update(last_selected, n_arms=max(n_candidates, len(chunks)), reward=reward_comps.total)
                agent.save()

        redundancy = pairwise_redundancy(chunk_texts)
        coverage = coverage_score(chunk_texts)
        eff = token_efficiency(eval_result.score, tokens_used, params.max_token_budget)

        return RunMetrics(
            strategy=strategy,
            tokens_used=tokens_used,
            token_budget=params.max_token_budget,
            token_efficiency=round(eff, 4),
            quality_score=round(eval_result.score, 4),
            correctness=round(eval_result.correctness, 4),
            completeness=round(eval_result.completeness, 4),
            factual_alignment=round(eval_result.factual_alignment, 4),
            redundancy=round(redundancy, 4),
            coverage=round(coverage, 4),
            latency=round(latency, 3),
            answer=resp.answer,
            chunks_selected=len(chunks),
            reward=round(reward_comps.total, 4),
            evaluator_reasoning=eval_result.reasoning,
        )
