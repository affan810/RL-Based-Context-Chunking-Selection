"""
Reusable UI components for the Streamlit dashboard.
All functions receive data (plain Python objects) and render Streamlit widgets.
"""

from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from evaluation.metrics import RunMetrics


# ── Colour palette ─────────────────────────────────────────────────────────

STRATEGY_COLORS = {
    "full_context":       "#636EFA",
    "truncate_head":      "#EF553B",
    "truncate_tail":      "#00CC96",
    "truncate_head_tail": "#AB63FA",
    "tfidf":              "#FFA15A",
    "semantic_topk":      "#19D3F3",
    "hybrid_retrieval":   "#FF6692",
    "sliding_window":     "#B6E880",
    "semantic_chunking":  "#FF97FF",
    "recursive":          "#FECB52",
    "mmr":                "#1CFFCE",
    "query_aware":        "#F8A19F",
    "rl_bandit":          "#FF4136",  # highlight RL
}


# ── Comparison table ──────────────────────────────────────────────────────

def render_comparison_table(results: list[RunMetrics]) -> None:
    if not results:
        st.info("No results yet. Run an evaluation first.")
        return

    rows = []
    for r in results:
        rows.append({
            "Strategy": r.strategy,
            "Quality ↑": f"{r.quality_score:.3f}",
            "Tokens": r.tokens_used,
            "Efficiency ↑": f"{r.token_efficiency:.3f}",
            "Redundancy ↓": f"{r.redundancy:.3f}",
            "Coverage ↑": f"{r.coverage:.3f}",
            "Latency (s) ↓": f"{r.latency:.2f}",
            "Reward ↑": f"{r.reward:.3f}",
            "Chunks": r.chunks_selected,
        })

    df = pd.DataFrame(rows)
    # Highlight best quality row
    best_idx = max(range(len(results)), key=lambda i: results[i].quality_score)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )
    best = results[best_idx]
    st.success(f"Best quality: **{best.strategy}** (score {best.quality_score:.3f})")


# ── Quality bar chart ────────────────────────────────────────────────────

def render_quality_chart(results: list[RunMetrics]) -> None:
    if not results:
        return
    df = pd.DataFrame([
        {"Strategy": r.strategy, "Quality": r.quality_score,
         "Tokens": r.tokens_used, "Reward": r.reward}
        for r in results
    ])
    fig = px.bar(
        df, x="Strategy", y="Quality", color="Strategy",
        color_discrete_map=STRATEGY_COLORS,
        title="Answer Quality by Strategy",
        text_auto=".3f",
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)


# ── Token vs Quality scatter ─────────────────────────────────────────────

def render_token_quality_scatter(results: list[RunMetrics]) -> None:
    if not results:
        return
    df = pd.DataFrame([
        {"Strategy": r.strategy, "Tokens Used": r.tokens_used,
         "Quality": r.quality_score, "Reward": r.reward}
        for r in results
    ])
    fig = px.scatter(
        df, x="Tokens Used", y="Quality", color="Strategy", text="Strategy",
        size=[15] * len(df),
        color_discrete_map=STRATEGY_COLORS,
        title="Quality vs Token Usage  (upper-left = ideal)",
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ── Radar chart per strategy ──────────────────────────────────────────────

def render_radar_chart(results: list[RunMetrics]) -> None:
    if not results:
        return
    categories = ["Quality", "Efficiency", "Coverage", "1-Redundancy", "1-Latency_norm"]
    fig = go.Figure()
    for r in results:
        latency_norm = min(r.latency / 30.0, 1.0)
        values = [
            r.quality_score,
            min(r.token_efficiency / 3, 1.0),
            r.coverage,
            1 - r.redundancy,
            1 - latency_norm,
        ]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            name=r.strategy,
            line_color=STRATEGY_COLORS.get(r.strategy, "#888"),
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Multi-Metric Radar",
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Redundancy heatmap ────────────────────────────────────────────────────

def render_redundancy_heatmap(chunk_texts: list[str], strategy_name: str) -> None:
    if len(chunk_texts) < 2:
        st.info("Need ≥2 chunks for redundancy heatmap.")
        return
    sets = [set(t.lower().split()) for t in chunk_texts]
    n = len(sets)
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            inter = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            row.append(inter / union if union else 0.0)
        matrix.append(row)

    labels = [f"Chunk {i+1}" for i in range(n)]
    fig = px.imshow(
        matrix,
        x=labels, y=labels,
        color_continuous_scale="Blues",
        title=f"Chunk Similarity Heatmap — {strategy_name}",
        zmin=0, zmax=1,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Chunk viewer ─────────────────────────────────────────────────────────

def render_chunk_viewer(chunks: list, strategy_name: str) -> None:
    st.markdown(f"### Selected chunks — `{strategy_name}`")
    for i, c in enumerate(chunks):
        with st.expander(f"Chunk {i+1} | tokens≈{c.token_count} | score={c.score:.3f}"):
            st.text(c.text)


# ── Reward breakdown bar ───────────────────────────────────────────────────

def render_reward_breakdown(results: list[RunMetrics]) -> None:
    if not results:
        return
    df = pd.DataFrame([
        {
            "Strategy": r.strategy,
            "Quality": r.quality_score,
            "−Token Cost": -r.tokens_used / max(r.token_budget, 1) * 0.3,
            "+Coverage": r.coverage * 0.3,
            "−Redundancy": -r.redundancy * 0.2,
            "Total Reward": r.reward,
        }
        for r in results
    ])
    fig = px.bar(
        df, x="Strategy", y=["Quality", "−Token Cost", "+Coverage", "−Redundancy"],
        barmode="relative",
        title="Reward Component Breakdown",
    )
    fig.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)


# ── RL Dashboard ─────────────────────────────────────────────────────────

def render_rl_dashboard(agent) -> None:
    """Render RL bandit training stats."""
    st.subheader("RL Bandit — Training State")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Steps", agent.total_steps)
    col2.metric("Domain", agent.domain)
    col3.metric("Algorithm", agent.algorithm.upper())

    # Smoothed reward over time
    smoothed = agent.smoothed_rewards(window=20)
    if smoothed:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=agent.reward_history, mode="lines",
            name="Raw reward", line=dict(color="#aaa", width=1),
        ))
        fig.add_trace(go.Scatter(
            y=smoothed, mode="lines",
            name="Smoothed (w=20)", line=dict(color="#FF4136", width=2),
        ))
        fig.update_layout(title="Reward Over Training Steps", xaxis_title="Step", yaxis_title="Reward")
        st.plotly_chart(fig, use_container_width=True)

    # Q-value heatmap (first 50 buckets)
    q_vals = agent.get_q_values()[:50]
    pulls = agent.get_pull_counts()[:50]
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=list(range(len(q_vals))), y=q_vals,
        name="Q-Value", marker_color="#19D3F3",
    ))
    fig2.add_trace(go.Bar(
        x=list(range(len(pulls))), y=[p / max(max(pulls), 1) for p in pulls],
        name="Pull Fraction", marker_color="#FFA15A", opacity=0.6,
    ))
    fig2.update_layout(
        barmode="overlay",
        title="Q-Values and Pull Counts (first 50 position buckets)",
        xaxis_title="Position Bucket", yaxis_title="Value",
    )
    st.plotly_chart(fig2, use_container_width=True)

    if agent.selected_history:
        st.markdown("**Last 10 selection patterns:**")
        for i, sel in enumerate(agent.selected_history[-10:]):
            st.code(f"Step -{10-i}: selected indices {sel}")
