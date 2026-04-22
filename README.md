# RL-Based Context Chunking & Selection System

> **Minimum context → Maximum answer quality**

A research-grade system demonstrating that intelligent RL-based chunk selection outperforms naive full-context prompting in LLMs.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

> If no API key is provided, the evaluator falls back to a token-overlap heuristic.

### 3. Launch the UI

```bash
streamlit run app.py
```

### 4. Run CLI experiment

```bash
python scripts/run_experiment.py
python scripts/run_experiment.py --doc article_001 --max-tokens 512
```

### 5. Pre-train the RL bandit

```bash
python scripts/train_rl.py --episodes 50
```

---

## Architecture

```
NLP2/
├── app.py                  # Streamlit UI entry point
├── config.py               # All tunable parameters
├── chunkers/               # 13 chunking strategies
│   ├── baselines.py        # Full, truncate head/tail
│   ├── retrieval.py        # TF-IDF, semantic, hybrid
│   ├── structural.py       # Sliding window, semantic, recursive
│   ├── advanced.py         # MMR, query-aware
│   └── rl_bandit.py        # RL bandit selector
├── rl/
│   ├── bandit_agent.py     # UCB / Thompson / ε-greedy bandit
│   └── reward.py           # Multi-component reward function
├── llm/
│   ├── tinyllama.py        # Local TinyLlama inference
│   └── openai_evaluator.py # ChatGPT quality evaluation
├── evaluation/
│   ├── pipeline.py         # Orchestration layer
│   └── metrics.py          # Quality, token, latency, redundancy
├── data/
│   ├── sample_docs/        # 5 long documents (article/story/sports/news/law)
│   └── rl_state/           # Persisted bandit policies (auto-created)
├── ui/
│   └── components.py       # Plotly charts, tables, dashboards
└── scripts/
    ├── run_experiment.py   # CLI batch runner
    └── train_rl.py         # RL pre-training loop
```

---

## Chunking Strategies

| # | Strategy | Category |
|---|----------|----------|
| 1 | `full_context` | Baseline |
| 2 | `truncate_head` | Baseline |
| 3 | `truncate_tail` | Baseline |
| 4 | `truncate_head_tail` | Baseline |
| 5 | `tfidf` | Retrieval |
| 6 | `semantic_topk` | Retrieval |
| 7 | `hybrid_retrieval` | Retrieval |
| 8 | `sliding_window` | Structural |
| 9 | `semantic_chunking` | Structural |
| 10 | `recursive` | Structural |
| 11 | `mmr` | Advanced |
| 12 | `query_aware` | Advanced |
| 13 | `rl_bandit` | RL |

---

## Reward Function

```
R = α·Quality − β·TokenCost − γ·Redundancy + δ·Coverage − ε·Latency
```

Weights configurable in `config.py`.

---

## RL Bandit

- **Algorithms**: UCB (default), Thompson Sampling, ε-greedy
- **Per-domain policies**: learns separately for story/article/news/sports/law
- **Online updates**: reward fed back after each evaluation
- **Persistence**: saved to `data/rl_state/bandit_{domain}.json`

---

## Stack

- Python 3.10+
- PyTorch + HuggingFace Transformers (TinyLlama)
- sentence-transformers (embeddings)
- FAISS (vector search)
- rank-bm25 (BM25)
- OpenAI SDK (evaluator)
- Streamlit + Plotly (UI)
