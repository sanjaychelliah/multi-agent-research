"""
Streamlit Dashboard for Multi-Agent Research Assistant

Screens:
  1. Research   — submit a query, watch agent activity, read final report
  2. Metrics    — interactive charts of run history
  3. A2A Log    — message flow explorer per run

Run:
    streamlit run ui/app.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import cfg
from metrics import MetricsStore
from pipeline import PipelineResult, run_pipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔬",
    layout="wide",
)

store = MetricsStore(cfg.METRICS_DB_PATH)


# ── Helpers ───────────────────────────────────────────────────────────────────
def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def confidence_badge(score: float) -> str:
    if score >= 0.75:
        return f"🟢 {score:.0%}"
    elif score >= 0.45:
        return f"🟡 {score:.0%}"
    return f"🔴 {score:.0%}"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Research Assistant")
    st.caption("Multi-Agent · MCP · A2A · LangChain")
    st.divider()
    page = st.radio("Navigation", ["Research", "Metrics Dashboard", "A2A Explorer"])
    st.divider()
    st.caption(f"Model: `{cfg.LLM_MODEL}`")
    st.caption(f"Max subtasks: `{cfg.MAX_SUBTASKS}`")
    st.caption(f"Search: {'Tavily' if cfg.has_tavily() else 'DuckDuckGo'}")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — Research
# ══════════════════════════════════════════════════════════════════════════════
if page == "Research":
    st.title("Multi-Agent Research Assistant")
    st.markdown(
        "Ask any research question. Four specialized agents collaborate via "
        "**A2A messaging** and **MCP tools** to produce a cited report."
    )

    query = st.text_input(
        "Research Query",
        placeholder="e.g. What are the latest breakthroughs in quantum computing?",
    )

    col1, col2 = st.columns([1, 5])
    run_btn = col1.button("▶ Run", type="primary", disabled=not query)

    if run_btn and query:
        # Status placeholders
        status_bar = st.empty()
        progress = st.progress(0, text="Starting pipeline…")

        agent_cols = st.columns(4)
        agent_states = {
            "orchestrator": agent_cols[0].empty(),
            "search_agent": agent_cols[1].empty(),
            "summarizer_agent": agent_cols[2].empty(),
            "critic_agent": agent_cols[3].empty(),
        }
        agent_labels = {
            "orchestrator": "🧠 Orchestrator",
            "search_agent": "🔍 Search",
            "summarizer_agent": "📝 Summarizer",
            "critic_agent": "⚖️ Critic",
        }
        for k, placeholder in agent_states.items():
            placeholder.info(f"{agent_labels[k]}\nWaiting…")

        try:
            cfg.validate()

            # Update UI as steps complete
            agent_states["orchestrator"].warning(f"{agent_labels['orchestrator']}\nRunning…")
            progress.progress(10, text="Orchestrator decomposing query…")

            result: PipelineResult = run_async(run_pipeline(query, store=store))

            agent_states["orchestrator"].success(
                f"{agent_labels['orchestrator']}\n"
                f"{len(result.plan.get('subtasks', []))} subtasks"
            )
            progress.progress(40, text="Search agent fetching results…")
            agent_states["search_agent"].success(
                f"{agent_labels['search_agent']}\n"
                f"{sum(len(r.get('results',[])) for r in result.search_results)} results"
            )
            progress.progress(65, text="Summarizer synthesising…")
            agent_states["summarizer_agent"].success(
                f"{agent_labels['summarizer_agent']}\n"
                f"{len(result.summaries)} summaries"
            )
            progress.progress(90, text="Critic reviewing…")
            agent_states["critic_agent"].success(
                f"{agent_labels['critic_agent']}\n"
                f"Confidence: {result.confidence_score:.0%}"
            )
            progress.progress(100, text="Done!")
            status_bar.success("Pipeline complete!")

            # ── Report ────────────────────────────────────────────────────────
            st.divider()
            report = result.final_report
            st.subheader(report.get("title", "Research Report"))

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Confidence", confidence_badge(result.confidence_score))
            m_col2.metric("Total Tokens", f"{result.metrics['total_tokens']:,}")
            m_col3.metric("A2A Messages", result.metrics["a2a_message_count"])
            m_col4.metric("Cost (USD)", f"${result.metrics['total_cost_usd']:.5f}")

            st.info(report.get("executive_summary", ""))

            tabs = st.tabs(["Full Report", "Subtask Summaries", "A2A Log", "Raw JSON"])

            with tabs[0]:
                for section in report.get("sections", []):
                    st.markdown(f"### {section.get('heading')}")
                    st.write(section.get("content", ""))

                if report.get("key_takeaways"):
                    st.markdown("**Key Takeaways**")
                    for t in report["key_takeaways"]:
                        st.markdown(f"- {t}")

                if report.get("all_sources"):
                    with st.expander("Sources"):
                        for url in report["all_sources"]:
                            st.markdown(f"- {url}")

            with tabs[1]:
                for s in result.summaries:
                    with st.expander(f"Subtask {s.get('subtask_id')}: {s.get('query', '')}"):
                        for f_ in s.get("key_findings", []):
                            st.markdown(f"• {f_}")
                        st.write(s.get("summary", ""))

            with tabs[2]:
                if result.a2a_log:
                    df = pd.DataFrame(result.a2a_log)
                    st.dataframe(df, use_container_width=True)

            with tabs[3]:
                st.json(result.metrics)

            if result.critique.get("gaps"):
                with st.expander("⚠️ Identified Gaps"):
                    for g in result.critique["gaps"]:
                        st.warning(g)

        except ValueError as e:
            st.error(f"Configuration error: {e}\n\nCopy `.env.example` → `.env` and add your API keys.")
        except Exception as e:
            st.error(f"Pipeline error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — Metrics Dashboard
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Metrics Dashboard":
    st.title("Metrics Dashboard")

    rows = store.fetch_all()
    if not rows:
        st.info("No runs yet. Run a research query first.")
        st.stop()

    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["agents"] = df["agents_json"].apply(lambda x: json.loads(x) if isinstance(x, str) else {})

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Runs", len(df))
    k2.metric("Avg Confidence", f"{df['confidence'].mean():.0%}" if "confidence" in df else "N/A")
    k3.metric("Avg Tokens", f"{df['total_tokens'].mean():,.0f}")
    k4.metric("Avg Latency", f"{df['latency_ms'].mean():,.0f} ms")
    k5.metric("Total Cost", f"${df['total_cost'].sum():.4f}")

    st.divider()

    col_l, col_r = st.columns(2)

    with col_l:
        fig = px.line(
            df.sort_values("created_at"),
            x="created_at", y="confidence",
            title="Confidence Score Over Time",
            markers=True, range_y=[0, 1],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig2 = px.bar(
            df.sort_values("created_at").tail(10),
            x="created_at", y="total_tokens",
            title="Token Usage (Last 10 Runs)",
            color="total_tokens", color_continuous_scale="Blues",
        )
        st.plotly_chart(fig2, use_container_width=True)

    col_l2, col_r2 = st.columns(2)

    with col_l2:
        # Per-agent token breakdown from latest run
        latest = df.iloc[0]
        agents_data = latest.get("agents", {})
        if agents_data:
            agent_names = list(agents_data.keys())
            agent_tokens = [agents_data[a].get("total_tokens", 0) for a in agent_names]
            fig3 = go.Figure(go.Pie(
                labels=agent_names, values=agent_tokens, hole=0.4,
            ))
            fig3.update_layout(title="Token Share by Agent (Latest Run)")
            st.plotly_chart(fig3, use_container_width=True)

    with col_r2:
        fig4 = px.scatter(
            df, x="latency_ms", y="confidence",
            size="total_tokens", color="status",
            title="Latency vs Confidence",
            hover_data=["query"],
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Run History")
    display_cols = ["created_at", "query", "status", "confidence", "total_tokens", "latency_ms", "total_cost", "a2a_count"]
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available].rename(columns={
        "created_at": "Time", "total_tokens": "Tokens",
        "latency_ms": "Latency (ms)", "total_cost": "Cost ($)",
        "a2a_count": "A2A Msgs",
    }), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — A2A Explorer
# ══════════════════════════════════════════════════════════════════════════════
elif page == "A2A Explorer":
    st.title("A2A Message Flow Explorer")
    st.markdown(
        "Visualise the **Agent-to-Agent** message log for any run to understand "
        "how agents hand off tasks to each other."
    )

    rows = store.fetch_all()
    if not rows:
        st.info("No runs yet. Run a research query first.")
        st.stop()

    run_labels = {r["run_id"]: f"{r['created_at'][:16]}  —  {r['query'][:60]}" for r in rows}
    selected_id = st.selectbox("Select a run", options=list(run_labels.keys()), format_func=lambda x: run_labels[x])

    run_data = store.fetch_run(selected_id)
    if run_data:
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Status", run_data.get("status", ""))
        mc2.metric("Confidence", f"{run_data.get('confidence', 0):.0%}")
        mc3.metric("A2A Messages", run_data.get("a2a_count", 0))

        agents_info = run_data.get("agents", {})
        if agents_info:
            st.subheader("Per-Agent Metrics")
            agent_df = pd.DataFrame([
                {
                    "Agent": name,
                    "Tokens": data.get("total_tokens", 0),
                    "Latency (ms)": data.get("latency_ms", 0),
                    "Cost ($)": data.get("cost_usd", 0),
                    "LLM Calls": data.get("llm_calls", 0),
                }
                for name, data in agents_info.items()
            ])
            st.dataframe(agent_df, use_container_width=True)

            fig = px.bar(
                agent_df, x="Agent", y="Latency (ms)",
                color="Tokens", title="Agent Latency & Token Usage",
                color_continuous_scale="Viridis",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Pipeline Flow")
        flow_steps = [
            ("User", "orchestrator", "Query"),
            ("orchestrator", "search_agent", "Subtask (web_search)"),
            ("search_agent", "summarizer_agent", "Results (summarize)"),
            ("summarizer_agent", "critic_agent", "Summaries (critique)"),
        ]
        mermaid = "graph LR\n"
        for src, dst, label in flow_steps:
            mermaid += f'    {src}["{src}"] -->|"{label}"| {dst}["{dst}"]\n'
        st.code(mermaid, language="mermaid")
        st.caption("Copy the above into mermaid.live to render the flow diagram.")
