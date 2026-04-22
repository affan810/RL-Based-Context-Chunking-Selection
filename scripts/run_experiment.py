"""
CLI experiment runner — runs all strategies on all documents and prints a report.

Usage:
    python scripts/run_experiment.py
    python scripts/run_experiment.py --strategies semantic_topk rl_bandit full_context
    python scripts/run_experiment.py --doc article_001 --max-tokens 512
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config as cfg
from data.loader import load_documents
from chunkers import ChunkerParams, REGISTRY
from evaluation.pipeline import EvaluationPipeline
from llm.tinyllama import TinyLlamaModel


def main():
    parser = argparse.ArgumentParser(description="Run chunking strategy experiment")
    parser.add_argument("--strategies", nargs="+", default=list(REGISTRY.keys()),
                        help="Strategies to evaluate")
    parser.add_argument("--doc", default=None, help="Document ID to run on (default: all)")
    parser.add_argument("--max-tokens", type=int, default=cfg.DEFAULT_MAX_TOKENS)
    parser.add_argument("--chunk-size", type=int, default=cfg.DEFAULT_CHUNK_SIZE)
    parser.add_argument("--top-k", type=int, default=cfg.DEFAULT_TOP_K)
    parser.add_argument("--no-rl-update", action="store_true", default=False)
    args = parser.parse_args()

    docs = load_documents()
    if args.doc:
        docs = [d for d in docs if d.id == args.doc]
        if not docs:
            print(f"[ERROR] Document '{args.doc}' not found.")
            sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  RL Chunking Experiment")
    print(f"  Strategies: {args.strategies}")
    print(f"  Token budget: {args.max_tokens} | Chunk size: {args.chunk_size} | Top-K: {args.top_k}")
    print(f"{'='*70}\n")

    llm = TinyLlamaModel()
    pipeline = EvaluationPipeline(llm=llm, update_rl=not args.no_rl_update)
    params = ChunkerParams(
        max_token_budget=args.max_tokens,
        chunk_size=args.chunk_size,
        top_k=args.top_k,
    )

    for doc in docs:
        print(f"\nDocument: {doc.title} [{doc.domain}]")
        print(f"  Words: {len(doc.text.split()):,}")

        for qa in doc.qa_pairs:
            print(f"\n  Query: {qa.question[:80]}...")
            results = pipeline.run(
                document=doc.text,
                query=qa.question,
                golden_answer=qa.golden_answer,
                domain=doc.domain,
                strategies=args.strategies,
                params=params,
            )

            # Print table
            print(f"\n  {'Strategy':<22} {'Quality':>8} {'Tokens':>8} {'Effic':>8} {'Reward':>8}")
            print(f"  {'-'*58}")
            results_sorted = sorted(results, key=lambda r: r.quality_score, reverse=True)
            for r in results_sorted:
                marker = " ★" if r == results_sorted[0] else ""
                print(f"  {r.strategy:<22} {r.quality_score:>8.3f} {r.tokens_used:>8} "
                      f"{r.token_efficiency:>8.3f} {r.reward:>8.3f}{marker}")
            print()


if __name__ == "__main__":
    main()
