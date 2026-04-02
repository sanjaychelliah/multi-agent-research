"""
Streamlit Dashboard — Multi-Agent Research Assistant

Pages:
  1. Research   — live agent feed, tool calls, LLM calls, A2A messages, final report
  2. Metrics    — history charts across all runs
  3. A2A Explorer — per-run message flow

Run:
    streamlit run ui/app.py
"""

from __future__ import annotations

import json
import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import cfg
from events import (
    EventQueue,
    AgentStarted, AgentFinished,
    LLMCallStarted, LLMCallFinished,
    ToolCallStarted, ToolCallFinished,
    A2AMessageSent, PipelineFinished,
    BaseEvent,
)
from metrics import MetricsStore
from pipeline import PipelineResult, run_pipeline_threaded

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Research",
    page_icon="🔬",
    layout="wide",
)

store = MetricsStore(cfg.METRICS_DB_PATH)

# ── Constants ─────────────────────────────────────────────────────────────────
AGENT_ORDER = ["orchestrator", "search_agent", "summarizer_agent", "critic_agent"]
AGENT_META = {
    "orchestrator":    {"label": "🧠 Orchestrator",  "color": "#6366f1"},
    "search_agent":    {"label": "🔍 Search",         "color": "#0ea5e9"},
    "summarizer_agent":{"label": "📝 Summarizer",     "color": "#f59e0b"},
    "critic_agent":    {"label": "⚖️ Critic",          "color": "#10b981"},
}

EVENT_ICONS = {
    "agent_started":    "🟡",
    "agent_finished":   "✅",
    "llm_call_started": "🧠",
    "llm_call_finished":"💬",
    "tool_call_started":"🔧",
    "tool_call_finished":"📦",
    "a2a_message":      "✉️",
    "pipeline_finished":"🎉",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _ts(event: BaseEvent) -> str:
    return datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")


def format_event_md(event: BaseEvent) -> str:
    """Convert an event into a one- or two-line markdown string for the feed."""
    ts = _ts(event)
    icon = EVENT_ICONS.get(event.event_type, "•")

    if isinstance(event, AgentStarted):
        return f"`{ts}` {icon} **{event.agent_label}** — started"

    if isinstance(event, AgentFinished):
        cost = f"${event.cost_usd:.5f}" if event.cost_usd else "—"
        return (
            f"`{ts}` {icon} **{event.agent_label}** — finished &nbsp;"
            f"· {event.tokens:,} tokens · {event.latency_ms:.0f} ms · {cost}"
        )

    # LLM events are rendered by render_llm_call_block(), not as plain markdown.
    # These branches are kept as a text-only fallback (e.g. for the CLI).
    if isinstance(event, LLMCallStarted):
        preview = event.prompt_preview[:100] + "…" if len(event.prompt_preview) > 100 else event.prompt_preview
        return (
            f"`{ts}` 🧠 **LLM call** `{event.agent_name}` · `{event.model}`  \n"
            f"&nbsp;&nbsp;&nbsp;&nbsp;*\"{preview}\"*"
        )

    if isinstance(event, LLMCallFinished):
        cost = f"${event.cost_usd:.5f}" if event.cost_usd else "—"
        return (
            f"`{ts}` 💬 **LLM response** — "
            f"{event.prompt_tokens}↑ {event.completion_tokens}↓ tokens · "
            f"{event.latency_ms:.0f} ms · {cost}"
        )

    if isinstance(event, ToolCallStarted):
        return (
            f"`{ts}` {icon} **Tool** `{event.tool_name}` ({event.agent_name})  \n"
            f"&nbsp;&nbsp;&nbsp;&nbsp;`{event.input_preview}`"
        )

    if isinstance(event, ToolCallFinished):
        noun = "result" if event.result_count == 1 else "results"
        return (
            f"`{ts}` {icon} **Tool done** `{event.tool_name}` — "
            f"{event.result_count} {noun} · {event.latency_ms:.0f} ms"
        )

    if isinstance(event, A2AMessageSent):
        return (
            f"`{ts}` {icon} **A2A** `{event.sender}` → `{event.receiver}` "
            f"· task: `{event.task_type}`"
        )

    if isinstance(event, PipelineFinished):
        if event.success:
            return f"`{ts}` 🎉 **Pipeline complete**"
        return f"`{ts}` ❌ **Pipeline failed** — {event.error}"

    return f"`{ts}` {icon} `{event.event_type}`"


def render_llm_call_block(
    started: LLMCallStarted,
    finished: LLMCallFinished | None,
) -> None:
    """
    Render one LLM call as a single st.expander widget.

    - While the pipeline is still running (finished is None): show a
      condensed 'in progress' line with the prompt preview but no expander.
    - Once finished: show the summary line as the expander label, and
      inside it show full input messages and the response in two tabs.
    """
    ts = _ts(started)

    if finished is None:
        # Still waiting for the response — show a minimal progress line
        preview = (
            started.prompt_preview[:90] + "…"
            if len(started.prompt_preview) > 90
            else started.prompt_preview
        )
        st.markdown(
            f"`{ts}` 🧠 **LLM call** `{started.agent_name}` · *waiting for response…*  \n"
            f"&nbsp;&nbsp;&nbsp;&nbsp;*\"{preview}\"*",
            unsafe_allow_html=True,
        )
        return

    # Build the expander label (summary line)
    cost = f"${finished.cost_usd:.5f}" if finished.cost_usd else "—"
    label = (
        f"`{ts}` 🧠 **LLM call** · `{started.agent_name}` · `{started.model}` · "
        f"{finished.prompt_tokens}↑ {finished.completion_tokens}↓ tokens · "
        f"{finished.latency_ms:.0f} ms · {cost}"
    )

    with st.expander(label, expanded=False):
        in_tab, out_tab = st.tabs(["📥 Input messages", "📤 Response"])

        with in_tab:
            if started.full_messages:
                for msg in started.full_messages:
                    role = msg.get("role", "unknown").capitalize()
                    content = msg.get("content", "")
                    # Colour-code the role badge
                    badge_color = {
                        "System": "#6366f1",
                        "Human": "#0ea5e9",
                        "Assistant": "#10b981",
                    }.get(role, "#64748b")
                    st.markdown(
                        f"<span style='background:{badge_color};color:white;"
                        f"padding:2px 8px;border-radius:4px;font-size:0.8em;"
                        f"font-weight:600'>{role}</span>",
                        unsafe_allow_html=True,
                    )
                    st.code(content, language="text")
            else:
                st.caption("No messages captured.")

        with out_tab:
            if finished.response_text:
                st.code(finished.response_text, language="text")
            else:
                st.caption("No response text captured.")


def build_llm_call_index(events: list[BaseEvent]) -> dict[str, LLMCallFinished | None]:
    """
    Pre-scan the event list and return a dict mapping each call_id
    to its LLMCallFinished event (or None if not yet received).
    """
    index: dict[str, LLMCallFinished | None] = {}
    for e in events:
        if isinstance(e, LLMCallStarted):
            if e.call_id not in index:
                index[e.call_id] = None
        elif isinstance(e, LLMCallFinished):
            index[e.call_id] = e
    return index


def render_feed(events: list[BaseEvent]) -> None:
    """
    Render the activity feed.

    LLMCallStarted events are rendered as expandable blocks (paired with
    their matching LLMCallFinished by call_id). Every other event is
    rendered as a plain markdown line.  LLMCallFinished events are skipped
    here because they are consumed inside their Started block.
    """
    llm_index = build_llm_call_index(events)

    for event in events:
        if isinstance(event, LLMCallStarted):
            render_llm_call_block(event, llm_index.get(event.call_id))
        elif isinstance(event, LLMCallFinished):
            pass  # already rendered inside its LLMCallStarted block
        else:
            st.markdown(format_event_md(event), unsafe_allow_html=True)


def agent_status_card(container, agent_id: str, state: str, metrics: dict) -> None:
    """Render one agent status card inside the given st container."""
    meta = AGENT_META[agent_id]
    label = meta["label"]

    if state == "waiting":
        container.markdown(
            f"<div style='padding:12px;border-radius:8px;background:#1e293b;"
            f"border:1px solid #334155;text-align:center'>"
            f"<div style='font-size:1.1em;font-weight:600;color:#94a3b8'>{label}</div>"
            f"<div style='color:#475569;font-size:0.8em;margin-top:4px'>⏸ waiting</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    elif state == "running":
        border_color = meta["color"]
        text_color = meta["color"]
        container.markdown(
            f"<div style='padding:12px;border-radius:8px;"
            f"background:linear-gradient(135deg,#1e3a5f,#1e293b);"
            f"border:2px solid {border_color};text-align:center'>"
            f"<div style='font-size:1.1em;font-weight:700;color:{text_color}'>{label}</div>"
            f"<div style='color:#38bdf8;font-size:0.8em;margin-top:4px'>⚙️ running…</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    elif state == "done":
        tokens = metrics.get(agent_id, {}).get("total_tokens", 0)
        latency = metrics.get(agent_id, {}).get("latency_ms", 0)
        container.markdown(
            f"<div style='padding:12px;border-radius:8px;background:#052e16;"
            f"border:1px solid #166534;text-align:center'>"
            f"<div style='font-size:1.1em;font-weight:700;color:#4ade80'>{label}</div>"
            f"<div style='color:#86efac;font-size:0.75em;margin-top:4px'>"
            f"✅ {tokens:,} tok · {latency:.0f} ms</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    elif state == "error":
        container.markdown(
            f"<div style='padding:12px;border-radius:8px;background:#2d0a0a;"
            f"border:1px solid #7f1d1d;text-align:center'>"
            f"<div style='font-size:1.1em;font-weight:700;color:#f87171'>{label}</div>"
            f"<div style='color:#fca5a5;font-size:0.8em;margin-top:4px'>❌ error</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def init_session_state() -> None:
    defaults = {
        "running": False,
        "all_events": [],
        "agent_states": {a: "waiting" for a in AGENT_ORDER},
        "agent_metrics": {},
        "live_tokens": 0,
        "live_cost": 0.0,
        "pipeline_result": None,
        "pipeline_error": None,
        "result_holder": [],
        "event_queue": None,
        "pipeline_thread": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_run_state() -> None:
    st.session_state.running = False
    st.session_state.all_events = []
    st.session_state.agent_states = {a: "waiting" for a in AGENT_ORDER}
    st.session_state.agent_metrics = {}
    st.session_state.live_tokens = 0
    st.session_state.live_cost = 0.0
    st.session_state.pipeline_result = None
    st.session_state.pipeline_error = None
    st.session_state.result_holder = []
    st.session_state.event_queue = None
    st.session_state.pipeline_thread = None


def process_events(new_events: list[BaseEvent]) -> None:
    """Update session state from a batch of new events."""
    for e in new_events:
        st.session_state.all_events.append(e)

        if isinstance(e, AgentStarted):
            st.session_state.agent_states[e.agent_name] = "running"

        elif isinstance(e, AgentFinished):
            st.session_state.agent_states[e.agent_name] = "done"
            if e.agent_name not in st.session_state.agent_metrics:
                st.session_state.agent_metrics[e.agent_name] = {}
            st.session_state.agent_metrics[e.agent_name]["total_tokens"] = e.tokens
            st.session_state.agent_metrics[e.agent_name]["latency_ms"] = e.latency_ms

        elif isinstance(e, LLMCallFinished):
            st.session_state.live_tokens += e.prompt_tokens + e.completion_tokens
            st.session_state.live_cost += e.cost_usd

        elif isinstance(e, PipelineFinished):
            st.session_state.running = False
            # Pull result from holder
            holder = st.session_state.result_holder
            if holder:
                status, data = holder[0]
                if status == "ok":
                    st.session_state.pipeline_result = data
                else:
                    st.session_state.pipeline_error = data
                    for a in AGENT_ORDER:
                        if st.session_state.agent_states[a] == "running":
                            st.session_state.agent_states[a] = "error"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Research Assistant")
    st.caption("Multi-Agent · MCP · A2A · LangChain")
    st.divider()
    page = st.radio("Navigation", ["🔬 Research", "📊 Metrics", "✉️ A2A Explorer"])
    st.divider()
    st.caption(f"**Provider:** `{cfg.LLM_PROVIDER}`")
    st.caption(f"**Model:** `{cfg.LLM_MODEL}`")
    st.caption(f"**Search:** {'Tavily' if cfg.has_tavily() else 'DuckDuckGo'}")
    st.caption(f"**Max subtasks:** `{cfg.MAX_SUBTASKS}`")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — Research (Live Feed)
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔬 Research":
    init_session_state()

    st.title("Multi-Agent Research Assistant")
    st.markdown(
        "Type a research question and watch four agents collaborate in real-time "
        "via **A2A messaging** and **MCP tools**."
    )

    # ── Query input ───────────────────────────────────────────────────────────
    col_q, col_btn = st.columns([6, 1])
    query = col_q.text_input(
        "Research Query",
        placeholder="e.g. What are the latest breakthroughs in quantum computing?",
        disabled=st.session_state.running,
        label_visibility="collapsed",
    )
    run_btn = col_btn.button(
        "▶ Run",
        type="primary",
        disabled=st.session_state.running or not query,
        use_container_width=True,
    )

    # ── Kick off pipeline ─────────────────────────────────────────────────────
    if run_btn and query:
        reset_run_state()
        eq = EventQueue()
        result_holder: list = []
        st.session_state.event_queue = eq
        st.session_state.result_holder = result_holder
        st.session_state.running = True
        st.session_state.pipeline_thread = run_pipeline_threaded(
            query=query,
            result_holder=result_holder,
            store=store,
            event_queue=eq,
        )
        st.rerun()

    # ── Poll for new events while running ─────────────────────────────────────
    if st.session_state.running and st.session_state.event_queue:
        new_events = st.session_state.event_queue.drain()
        if new_events:
            process_events(new_events)

    # ── Render pipeline UI whenever there's something to show ─────────────────
    if st.session_state.all_events or st.session_state.running:
        st.divider()

        # Agent pipeline cards
        agent_cols = st.columns(4)
        for col, agent_id in zip(agent_cols, AGENT_ORDER):
            with col:
                placeholder = st.empty()
                agent_status_card(
                    placeholder,
                    agent_id,
                    st.session_state.agent_states[agent_id],
                    st.session_state.agent_metrics,
                )

        # Arrow connector between cards (cosmetic)
        st.markdown(
            "<div style='text-align:center;color:#475569;font-size:0.8em;margin:-8px 0 8px'>"
            "──────────────── A2A Message Flow ────────────────"
            "</div>",
            unsafe_allow_html=True,
        )

        # Live metrics bar
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tokens so far", f"{st.session_state.live_tokens:,}")
        m2.metric("Est. Cost", f"${st.session_state.live_cost:.5f}")
        m3.metric("A2A Messages",
                  sum(1 for e in st.session_state.all_events if isinstance(e, A2AMessageSent)))
        m4.metric("Tool Calls",
                  sum(1 for e in st.session_state.all_events if isinstance(e, ToolCallStarted)))

        # Live activity feed
        st.markdown("#### Live Activity Feed")
        st.caption("🧠 LLM calls are expandable — click to see full prompts & responses.")
        feed_container = st.container(border=True)
        with feed_container:
            events_to_show = st.session_state.all_events[-120:]  # cap display
            render_feed(events_to_show)

            if st.session_state.running:
                st.markdown("*⏳ Pipeline running…*")

        # Keep polling
        if st.session_state.running:
            time.sleep(0.35)
            st.rerun()

    # ── Final report (shown after pipeline finishes) ──────────────────────────
    if st.session_state.pipeline_error:
        st.error(f"Pipeline failed: {st.session_state.pipeline_error}")

    if st.session_state.pipeline_result:
        result: PipelineResult = st.session_state.pipeline_result
        report = result.final_report
        m = result.metrics

        st.divider()
        st.subheader(report.get("title", "Research Report"))

        # KPI row
        k1, k2, k3, k4, k5 = st.columns(5)
        score = result.confidence_score
        color = "green" if score >= 0.7 else "orange" if score >= 0.4 else "red"
        k1.metric("Confidence", f"{score:.0%}")
        k2.metric("Total Tokens", f"{m['total_tokens']:,}")
        k3.metric("Total Cost", f"${m['total_cost_usd']:.5f}")
        k4.metric("A2A Messages", m["a2a_message_count"])
        k5.metric("Latency", f"{m['total_latency_ms']/1000:.1f}s")

        # Confidence bar
        bar_filled = int(score * 20)
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        st.markdown(
            f"<div style='font-family:monospace;color:{'#4ade80' if score>=0.7 else '#fbbf24' if score>=0.4 else '#f87171'}'>"
            f"Confidence {bar} {score:.0%}</div>",
            unsafe_allow_html=True,
        )

        st.info(report.get("executive_summary", ""))

        tabs = st.tabs(["📄 Full Report", "🔍 Subtask Summaries", "📊 Agent Metrics", "🗃️ Raw JSON"])

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

            critique = result.critique
            if critique.get("gaps"):
                with st.expander("⚠️ Identified Gaps"):
                    for g in critique["gaps"]:
                        st.warning(g)

        with tabs[1]:
            for s in result.summaries:
                with st.expander(f"Subtask {s.get('subtask_id')}: {s.get('query', '')}"):
                    for f_ in s.get("key_findings", []):
                        st.markdown(f"• {f_}")
                    st.write(s.get("summary", ""))

        with tabs[2]:
            if m.get("agents"):
                agent_df = pd.DataFrame([
                    {
                        "Agent": name,
                        "Tokens": data["total_tokens"],
                        "Prompt Tokens": data["prompt_tokens"],
                        "Completion Tokens": data["completion_tokens"],
                        "Latency (ms)": data["latency_ms"],
                        "Cost ($)": data["cost_usd"],
                        "LLM Calls": data["llm_calls"],
                    }
                    for name, data in m["agents"].items()
                ])
                st.dataframe(agent_df, use_container_width=True)

                fig = px.bar(
                    agent_df, x="Agent", y="Tokens",
                    color="Latency (ms)", color_continuous_scale="Blues",
                    title="Token Usage by Agent",
                )
                st.plotly_chart(fig, use_container_width=True)

        with tabs[3]:
            st.json(m)

        # New query button
        st.divider()
        if st.button("🔄 New Query"):
            reset_run_state()
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — Metrics Dashboard
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Metrics":
    st.title("Metrics Dashboard")

    rows = store.fetch_all()
    if not rows:
        st.info("No runs yet. Run a research query first.")
        st.stop()

    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["agents"] = df["agents_json"].apply(lambda x: json.loads(x) if isinstance(x, str) else {})

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Runs", len(df))
    k2.metric("Avg Confidence", f"{df['confidence'].mean():.0%}" if "confidence" in df.columns else "—")
    k3.metric("Avg Tokens", f"{df['total_tokens'].mean():,.0f}" if "total_tokens" in df.columns else "—")
    k4.metric("Avg Latency", f"{df['latency_ms'].mean():,.0f} ms" if "latency_ms" in df.columns else "—")
    k5.metric("Total Cost", f"${df['total_cost'].sum():.4f}" if "total_cost" in df.columns else "—")

    st.divider()

    col_l, col_r = st.columns(2)

    with col_l:
        if "confidence" in df.columns:
            fig = px.line(
                df.sort_values("created_at"),
                x="created_at", y="confidence",
                title="Confidence Score Over Time",
                markers=True, range_y=[0, 1],
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        if "total_tokens" in df.columns:
            fig2 = px.bar(
                df.sort_values("created_at").tail(10),
                x="created_at", y="total_tokens",
                title="Token Usage (Last 10 Runs)",
                color="total_tokens", color_continuous_scale="Blues",
            )
            st.plotly_chart(fig2, use_container_width=True)

    col_l2, col_r2 = st.columns(2)

    with col_l2:
        latest_agents = df.iloc[0].get("agents", {})
        if latest_agents:
            agent_names = list(latest_agents.keys())
            agent_tokens = [latest_agents[a].get("total_tokens", 0) for a in agent_names]
            fig3 = go.Figure(go.Pie(labels=agent_names, values=agent_tokens, hole=0.4))
            fig3.update_layout(title="Token Share by Agent (Latest Run)")
            st.plotly_chart(fig3, use_container_width=True)

    with col_r2:
        if "latency_ms" in df.columns and "confidence" in df.columns:
            fig4 = px.scatter(
                df, x="latency_ms", y="confidence",
                size="total_tokens", color="status",
                title="Latency vs Confidence",
                hover_data=["query"],
            )
            st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Run History")
    display_cols = ["created_at", "query", "status", "confidence",
                    "total_tokens", "latency_ms", "total_cost", "a2a_count"]
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df[available].rename(columns={
            "created_at": "Time", "total_tokens": "Tokens",
            "latency_ms": "Latency (ms)", "total_cost": "Cost ($)",
            "a2a_count": "A2A Msgs",
        }),
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — A2A Explorer
# ══════════════════════════════════════════════════════════════════════════════
elif page == "✉️ A2A Explorer":
    st.title("A2A Message Flow Explorer")
    st.markdown(
        "Inspect the **Agent-to-Agent** message log for any past run. "
        "See exactly which task was handed off, when, and in what state."
    )

    rows = store.fetch_all()
    if not rows:
        st.info("No runs yet. Run a research query first.")
        st.stop()

    run_labels = {
        r["run_id"]: f"{r['created_at'][:16]}  —  {r['query'][:60]}"
        for r in rows
    }
    selected_id = st.selectbox(
        "Select a run",
        options=list(run_labels.keys()),
        format_func=lambda x: run_labels[x],
    )

    run_data = store.fetch_run(selected_id)
    if run_data:
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Status", run_data.get("status", ""))
        mc2.metric("Confidence", f"{run_data.get('confidence', 0):.0%}")
        mc3.metric("A2A Messages", run_data.get("a2a_count", 0))
        mc4.metric("Tokens", f"{run_data.get('total_tokens', 0):,}")

        agents_info = run_data.get("agents", {})
        if agents_info:
            st.subheader("Per-Agent Breakdown")
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

            col_a, col_b = st.columns(2)
            with col_a:
                fig = px.bar(
                    agent_df, x="Agent", y="Latency (ms)",
                    color="Tokens", title="Latency & Tokens per Agent",
                    color_continuous_scale="Viridis",
                )
                st.plotly_chart(fig, use_container_width=True)
            with col_b:
                fig2 = px.pie(
                    agent_df, names="Agent", values="Tokens",
                    title="Token Distribution", hole=0.35,
                )
                st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Pipeline Flow (Mermaid)")
        flow_steps = [
            ("User", "orchestrator", "query"),
            ("orchestrator", "search_agent", "web_search ×N"),
            ("search_agent", "summarizer_agent", "summarize ×N"),
            ("summarizer_agent", "critic_agent", "critique"),
        ]
        mermaid = "graph LR\n"
        for src, dst, label in flow_steps:
            mermaid += f'    {src}["{src}"] -->|"{label}"| {dst}["{dst}"]\n'
        st.code(mermaid, language="mermaid")
        st.caption("Paste into [mermaid.live](https://mermaid.live) to render interactively.")
