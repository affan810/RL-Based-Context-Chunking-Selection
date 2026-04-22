"""
RL bandit training loop — runs many episodes on the dataset to pre-train the policy.

Usage:
    python scripts/train_rl.py --episodes 50
    python scripts/train_rl.py --episodes 100 --domain article
"""

from __future__ import annotations
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config as cfg
from data.loader import load_documents
from chunkers import get_chunker, ChunkerParams, REGISTRY
from rl.bandit_agent import BanditAgent
from rl.reward import compute_reward
from llm.tinyllama import TinyLlamaModel
from llm.openai_evaluator import evaluate_answer
from evaluation.metrics import pairwise_redundancy, coverage_score
import time


def run_episode(
    agent: BanditAgent,
    llm: TinyLlamaModel,
    document: str,
    query: str,
    golden_answer: str,
    domain: str,
    params: ChunkerParams,
) -> float:
    """One RL training episode. Returns total reward."""
    from chunkers.structural import RecursiveChunker

    # Get candidate chunks
    base_chunker = RecursiveChunker(params)
    candidates = base_chunker.chunk(document)
    if not candidates:
        return 0.0

    # Compute features
    q_terms = set(query.lower().split())
    features = []
    for c in candidates:
        c_terms = set(c.text.lower().split())
        overlap = len(q_terms & c_terms) / (len(q_terms) + 1e-9)
        density = c.token_count / max(params.chunk_size, 1)
        position = c.index / max(len(candidates) - 1, 1)
        features.append([overlap, density, position])

    # Agent selects chunks
    selected_indices = agent.select(
        n_arms=len(candidates),
        features=features,
        budget=params.max_token_budget,
        token_counts=[c.token_count for c in candidates],
        top_k=params.top_k,
    )

    selected_chunks = [candidates[i] for i in selected_indices]
    context = llm.chunks_to_context(selected_chunks)
    tokens_used = sum(c.token_count for c in selected_chunks)
    chunk_texts = [c.text for c in selected_chunks]

    # Generate and evaluate
    t0 = time.time()
    resp = llm.generate(context, query)
    latency = time.time() - t0

    eval_result = evaluate_answer(query, golden_answer, resp.answer)

    reward_comps = compute_reward(
        quality_score=eval_result.score,
        tokens_used=tokens_used,
        token_budget=params.max_token_budget,
        chunk_texts=chunk_texts,
        latency_seconds=latency,
    )

    agent.update(selected_indices, n_arms=len(candidates), reward=reward_comps.total)
    return reward_comps.total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--domain", default=None, help="Filter by domain")
    parser.add_argument("--max-tokens", type=int, default=cfg.DEFAULT_MAX_TOKENS)
    parser.add_argument("--chunk-size", type=int, default=cfg.DEFAULT_CHUNK_SIZE)
    parser.add_argument("--top-k", type=int, default=cfg.DEFAULT_TOP_K)
    args = parser.parse_args()

    docs = load_documents()
    if args.domain:
        docs = [d for d in docs if d.domain == args.domain]

    qa_items = [
        (doc.text, qa.question, qa.golden_answer, doc.domain)
        for doc in docs
        for qa in doc.qa_pairs
    ]

    if not qa_items:
        print("No QA items found. Check dataset.")
        return

    llm = TinyLlamaModel()
    params = ChunkerParams(
        max_token_budget=args.max_tokens,
        chunk_size=args.chunk_size,
        top_k=args.top_k,
    )

    # Load agents per domain
    agents: dict[str, BanditAgent] = {}
    for _, _, _, domain in qa_items:
        if domain not in agents:
            agents[domain] = BanditAgent.load_or_create(domain=domain)

    print(f"\nTraining RL bandit for {args.episodes} episodes across {len(qa_items)} Q&A pairs\n")

    for ep in range(args.episodes):
        text, query, golden, domain = random.choice(qa_items)
        agent = agents[domain]
        reward = run_episode(agent, llm, text, query, golden, domain, params)
        agent.save()

        if (ep + 1) % 5 == 0:
            recent = agent.reward_history[-5:]
            avg = sum(recent) / len(recent) if recent else 0
            print(f"  Episode {ep+1:3d}/{args.episodes} | domain={domain:<8} | "
                  f"reward={reward:.3f} | avg(last5)={avg:.3f}")

    print("\nTraining complete. Policies saved.")
    for domain, agent in agents.items():
        print(f"  {domain}: {agent.total_steps} total steps")


if __name__ == "__main__":
    main()
