"""Global configuration — all tunable parameters in one place."""

from __future__ import annotations
import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed yet — fall back to env vars

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
RL_STATE_DIR = DATA_DIR / "rl_state"

# ── Models ───────────────────────────────────────────────────────────────────
TINYLLAMA_MODEL = os.getenv("TINYLLAMA_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ── Token Budget ─────────────────────────────────────────────────────────────
DEFAULT_MAX_TOKENS = 1024          # max context tokens fed to TinyLlama
TINYLLAMA_MAX_NEW_TOKENS = 256     # max generation tokens

# ── Chunking defaults ─────────────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE = 256           # tokens per chunk
DEFAULT_CHUNK_OVERLAP = 0.15       # 15% overlap
DEFAULT_TOP_K = 5                  # retrieval top-k
MIN_CHUNK_SIZE = 50
MAX_CHUNK_SIZE = 512

# ── Retrieval ─────────────────────────────────────────────────────────────────
SIMILARITY_METRIC = "cosine"       # cosine | dot | l2
HYBRID_ALPHA = 0.6                 # weight for semantic score
HYBRID_BETA = 0.4                  # weight for BM25 score

# ── Sliding Window ────────────────────────────────────────────────────────────
WINDOW_SIZE = 256
STRIDE = 128

# ── MMR ───────────────────────────────────────────────────────────────────────
MMR_LAMBDA = 0.5                   # 0 = max diversity, 1 = max relevance

# ── RL / Bandit ───────────────────────────────────────────────────────────────
BANDIT_EPSILON = 0.15              # ε-greedy exploration rate
BANDIT_LR = 0.1                    # learning rate for value update
BANDIT_UCB_C = 1.4                 # UCB exploration constant
BANDIT_ALGORITHM = "ucb"           # ucb | epsilon_greedy | thompson

# ── Reward weights ────────────────────────────────────────────────────────────
REWARD_ALPHA = 1.0    # answer quality weight
REWARD_BETA = 0.3     # token cost penalty weight
REWARD_GAMMA = 0.2    # redundancy penalty weight
REWARD_DELTA = 0.3    # coverage bonus weight
REWARD_EPSILON = 0.1  # latency penalty weight

# ── Domain types ──────────────────────────────────────────────────────────────
DOMAIN_TYPES = ["story", "article", "news", "sports", "law", "general"]

DOMAIN_CHUNK_HINTS: dict[str, dict] = {
    "story":   {"chunk_size": 384, "top_k": 4, "preserve_order": True},
    "law":     {"chunk_size": 128, "top_k": 7, "preserve_order": False},
    "news":    {"chunk_size": 256, "top_k": 5, "preserve_order": True},
    "sports":  {"chunk_size": 256, "top_k": 5, "preserve_order": True},
    "article": {"chunk_size": 256, "top_k": 5, "preserve_order": False},
    "general": {"chunk_size": 256, "top_k": 5, "preserve_order": False},
}

# ── Strategy names (registry keys) ───────────────────────────────────────────
ALL_STRATEGIES = [
    "full_context",
    "truncate_head",
    "truncate_tail",
    "truncate_head_tail",
    "tfidf",
    "semantic_topk",
    "hybrid_retrieval",
    "sliding_window",
    "semantic_chunking",
    "recursive",
    "mmr",
    "query_aware",
    "rl_bandit",
]
