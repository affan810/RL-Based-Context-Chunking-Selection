"""
RL-Based Context Chunking & Selection System — Streamlit UI
Run: streamlit run app.py
"""

from __future__ import annotations
import sys
from pathlib import Path

# Ensure project root is on sys.path when run from any directory
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd

import config as cfg
from data.loader import load_documents, Document, QAPair
from chunkers import get_chunker, ChunkerParams, REGISTRY
from llm.tinyllama import TinyLlamaModel
from llm.openai_evaluator import evaluate_answer
from rl.bandit_agent import BanditAgent
from rl.reward import compute_reward
from evaluation.pipeline import EvaluationPipeline
from evaluation.metrics import RunMetrics, pairwise_redundancy, coverage_score
from ui.components import (
    render_comparison_table,
    render_quality_chart,
    render_token_quality_scatter,
    render_radar_chart,
    render_redundancy_heatmap,
    render_chunk_viewer,
    render_reward_breakdown,
    render_rl_dashboard,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RL Chunking System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ────────────────────────────────────────────────────

if "results" not in st.session_state:
    st.session_state.results = []
if "last_chunks" not in st.session_state:
    st.session_state.last_chunks = {}
if "llm" not in st.session_state:
    st.session_state.llm = None
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading TinyLlama…")
def load_llm() -> TinyLlamaModel:
    return TinyLlamaModel()


@st.cache_data(show_spinner=False)
def get_documents() -> list[Document]:
    return load_documents()


def get_pipeline() -> EvaluationPipeline:
    if st.session_state.pipeline is None:
        st.session_state.pipeline = EvaluationPipeline(llm=load_llm(), update_rl=True)
    return st.session_state.pipeline


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧠 RL Chunking")
    st.markdown("---")

    # ── Document selection
    docs = get_documents()
    doc_titles = {f"{d.title} [{d.domain}]": d for d in docs}
    doc_key = st.selectbox("Document", list(doc_titles.keys()))
    selected_doc: Document = doc_titles[doc_key]

    st.caption(f"Words: {len(selected_doc.text.split()):,} | Domain: {selected_doc.domain}")

    # ── Custom text input
    use_custom = st.checkbox("Use custom text instead")
    if use_custom:
        custom_text = st.text_area("Paste document text:", height=200)
        custom_domain = st.selectbox("Domain type", cfg.DOMAIN_TYPES)
        document_text = custom_text or selected_doc.text
        domain = custom_domain
    else:
        document_text = selected_doc.text
        domain = selected_doc.domain

    st.markdown("---")

    # ── Query selection
    qa_options = [qa.question for qa in selected_doc.qa_pairs]
    if not use_custom and qa_options:
        q_choice = st.selectbox("Query (golden Q&A)", qa_options)
        selected_qa = next(qa for qa in selected_doc.qa_pairs if qa.question == q_choice)
        query = selected_qa.question
        golden_answer = selected_qa.golden_answer
    else:
        query = st.text_input("Custom query")
        golden_answer = st.text_area("Golden answer (for evaluation)", height=80)

    st.markdown("---")

    # ── Parameters
    st.subheader("Parameters")
    max_tokens = st.slider("Token budget", 128, 2048, cfg.DEFAULT_MAX_TOKENS, 64)
    chunk_size = st.slider("Chunk size (tokens)", 64, 512, cfg.DEFAULT_CHUNK_SIZE, 32)
    top_k = st.slider("Top-K chunks", 1, 12, cfg.DEFAULT_TOP_K)
    overlap_pct = st.slider("Overlap %", 0, 50, int(cfg.DEFAULT_CHUNK_OVERLAP * 100))
    mmr_lambda = st.slider("MMR λ (relevance↔diversity)", 0.0, 1.0, cfg.MMR_LAMBDA, 0.05)

    st.markdown("---")

    # ── Strategy selection
    st.subheader("Strategies")
    all_strats = list(REGISTRY.keys())
    selected_strategies = st.multiselect(
        "Run these strategies",
        all_strats,
        default=all_strats,
    )

    st.markdown("---")

    # ── RL settings
    st.subheader("RL Bandit Config")
    bandit_algo = st.selectbox("Algorithm", ["ucb", "epsilon_greedy", "thompson"],
                                index=["ucb", "epsilon_greedy", "thompson"].index(cfg.BANDIT_ALGORITHM))
    bandit_epsilon = st.slider("ε (exploration)", 0.0, 0.5, cfg.BANDIT_EPSILON, 0.01)
    bandit_lr = st.slider("Learning rate", 0.01, 0.5, cfg.BANDIT_LR, 0.01)
    ucb_c = st.slider("UCB c", 0.5, 3.0, cfg.BANDIT_UCB_C, 0.1)

    reset_rl = st.button("🔄 Reset RL policy")
    if reset_rl:
        agent = BanditAgent.load_or_create(domain=domain)
        agent.reset()
        agent.save()
        st.success("RL policy reset.")

    st.markdown("---")
    run_btn = st.button("▶ Run Evaluation", type="primary", use_container_width=True)


# ── Main area ────────────────────────────────────────────────────────────────

st.title("🧠 RL-Based Context Chunking & Selection")
st.markdown(
    "> **Goal:** Minimum context → Maximum answer quality. "
    "Compare all chunking strategies side-by-side."
)

tab_compare, tab_viz, tab_chunks, tab_rl, tab_about = st.tabs([
    "📊 Comparison", "📈 Visualizations", "📄 Chunk Viewer", "🤖 RL Dashboard", "ℹ️ About"
])

# ── Run evaluation ────────────────────────────────────────────────────────────

if run_btn:
    if not query.strip():
        st.warning("Please enter a query.")
    elif not selected_strategies:
        st.warning("Select at least one strategy.")
    else:
        params = ChunkerParams(
            max_token_budget=max_tokens,
            chunk_size=chunk_size,
            top_k=top_k,
            overlap=overlap_pct / 100,
            mmr_lambda=mmr_lambda,
            domain=domain,
        )

        pipeline = get_pipeline()

        with st.spinner(f"Running {len(selected_strategies)} strategies…"):
            results = pipeline.run(
                document=document_text,
                query=query,
                golden_answer=golden_answer,
                domain=domain,
                strategies=selected_strategies,
                params=params,
            )

        # Also store per-strategy chunks for the chunk viewer
        chunk_map = {}
        for strat in selected_strategies:
            try:
                chunker = get_chunker(strat, params)
                chunks = chunker.chunk(document_text, query)
                chunk_map[strat] = chunks
            except Exception:
                chunk_map[strat] = []

        st.session_state.results = results
        st.session_state.last_chunks = chunk_map
        st.success(f"Done! Ran {len(results)} strategies.")

results: list[RunMetrics] = st.session_state.results
chunk_map: dict = st.session_state.last_chunks

# ── Tab: Comparison ───────────────────────────────────────────────────────────

with tab_compare:
    if results:
        st.subheader("Query")
        st.info(query)
        if golden_answer:
            with st.expander("Golden answer"):
                st.write(golden_answer)

        render_comparison_table(results)

        st.markdown("---")
        st.subheader("Per-Strategy Answers")
        cols = st.columns(2)
        for i, r in enumerate(results):
            with cols[i % 2]:
                with st.expander(f"`{r.strategy}` — Q={r.quality_score:.3f} | T={r.tokens_used}"):
                    st.markdown(f"**Answer:** {r.answer}")
                    if r.evaluator_reasoning:
                        st.caption(f"Evaluator: {r.evaluator_reasoning}")
    else:
        st.info("Run an evaluation from the sidebar to see results here.")

# ── Tab: Visualizations ───────────────────────────────────────────────────────

with tab_viz:
    if results:
        col1, col2 = st.columns(2)
        with col1:
            render_quality_chart(results)
        with col2:
            render_token_quality_scatter(results)

        render_radar_chart(results)
        render_reward_breakdown(results)

        # Redundancy heatmap for selected strategy
        st.markdown("---")
        st.subheader("Redundancy Heatmap")
        strat_for_heatmap = st.selectbox(
            "Strategy for heatmap",
            [r.strategy for r in results],
            key="heatmap_strat",
        )
        hm_chunks = chunk_map.get(strat_for_heatmap, [])
        if hm_chunks:
            render_redundancy_heatmap([c.text for c in hm_chunks], strat_for_heatmap)
    else:
        st.info("Run an evaluation to see visualizations.")

# ── Tab: Chunk Viewer ────────────────────────────────────────────────────────

with tab_chunks:
    if chunk_map:
        strat_for_chunks = st.selectbox(
            "View chunks for strategy",
            list(chunk_map.keys()),
            key="chunk_viewer_strat",
        )
        chunks = chunk_map.get(strat_for_chunks, [])
        if chunks:
            render_chunk_viewer(chunks, strat_for_chunks)
            # Show overlap visualization
            if len(chunks) >= 2:
                st.markdown("**Token counts per chunk:**")
                df = pd.DataFrame([
                    {"Chunk": f"#{i+1}", "Tokens": c.token_count, "Score": round(c.score, 3)}
                    for i, c in enumerate(chunks)
                ])
                import plotly.express as px
                fig = px.bar(df, x="Chunk", y="Tokens", color="Score",
                             color_continuous_scale="Blues",
                             title="Chunk Sizes and Relevance Scores")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No chunks available for this strategy.")
    else:
        st.info("Run an evaluation to see chunk details.")

# ── Tab: RL Dashboard ────────────────────────────────────────────────────────

with tab_rl:
    rl_domain = domain if results else "general"
    agent = BanditAgent.load_or_create(domain=rl_domain)
    render_rl_dashboard(agent)

    st.markdown("---")
    st.subheader("Multi-Domain Comparison")
    domain_stats = []
    for d in cfg.DOMAIN_TYPES:
        a = BanditAgent.load_or_create(domain=d)
        if a.total_steps > 0:
            avg_reward = sum(a.reward_history[-50:]) / max(len(a.reward_history[-50:]), 1)
            domain_stats.append({
                "Domain": d, "Steps": a.total_steps,
                "Avg Reward (last 50)": round(avg_reward, 3),
            })
    if domain_stats:
        st.dataframe(pd.DataFrame(domain_stats), hide_index=True)
    else:
        st.info("No RL training data yet. Run evaluations to train the bandit.")

# ── Tab: About ────────────────────────────────────────────────────────────────

with tab_about:
    st.markdown("""
## RL-Based Context Chunking & Selection System

### Core Idea
> **Minimum context → Maximum answer quality**

This system optimizes *which* parts of a document to give to an LLM, not just how to split it.

### Architecture
- **TinyLlama** — local LLM generating answers under token constraints
- **ChatGPT** — gold standard evaluator scoring correctness, completeness, factual alignment
- **13 chunking strategies** — from naive baselines to RL-optimized selection
- **Multi-Armed Bandit (UCB / Thompson / ε-greedy)** — learns per-domain chunk selection policies

### Reward Function
```
R = α·Quality − β·TokenCost − γ·Redundancy + δ·Coverage − ε·Latency
```

### Strategies
| Category | Strategies |
|----------|-----------|
| Baseline | Full context, Truncate head, Truncate tail, Head+tail |
| Retrieval | TF-IDF, Semantic Top-K, Hybrid (BM25+semantic) |
| Structural | Sliding window, Semantic chunking, Recursive |
| Advanced | MMR (diversity), Query-aware dynamic |
| RL | Bandit selector (learns from reward signal) |

### Domain Awareness
The RL bandit learns *different policies* per domain type:
- **Story** → larger chunks, preserve order
- **Law** → small, precise chunks
- **News/Article** → moderate size, hybrid retrieval
- **Sports** → moderate size, TF-IDF heavy

### Key Insight
Full context ≠ best performance. Smart selection improves quality AND reduces tokens.
    """)
