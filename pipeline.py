"""
Research Pipeline

Wires together all agents, MCP tool connections, and the A2A message bus
into a single end-to-end async pipeline.

Flow:
    User Query
        │
        ▼
    OrchestratorAgent  ──(A2A: web_search tasks)──►  SearchAgent
                                                          │
                                                          │ (A2A: summarize tasks)
                                                          ▼
                                                   SummarizerAgent
                                                          │
                                                          │ (A2A: critique task)
                                                          ▼
                                                     CriticAgent
                                                          │
                                                          ▼
                                                   Final Report + Metrics
"""

from __future__ import annotations

import asyncio
import sys
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain_mcp_adapters.sessions import StdioConnection
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI

from a2a import MessageBus
from agents import CriticAgent, OrchestratorAgent, SearchAgent, SummarizerAgent
from config import cfg
from metrics import MetricsStore, MetricsTracker

if TYPE_CHECKING:
    from events import EventQueue


@dataclass
class PipelineResult:
    query: str
    plan: dict[str, Any]
    search_results: list[dict[str, Any]]
    summaries: list[dict[str, Any]]
    critique: dict[str, Any]
    final_report: dict[str, Any]
    confidence_score: float
    metrics: dict[str, Any]
    a2a_log: list[dict[str, Any]]


def _build_llm():
    """Instantiate the LLM based on LLM_PROVIDER env var.

    Supported providers:
      openai      — OpenAI API (default)
      anthropic   — Anthropic API (Claude models)
      openrouter  — OpenRouter (any model via OpenAI-compatible API)
      ollama      — Local Ollama instance (OpenAI-compatible API)
    """
    provider = cfg.LLM_PROVIDER

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # type: ignore
        return ChatAnthropic(  # type: ignore
            model=cfg.LLM_MODEL,
            api_key=cfg.ANTHROPIC_API_KEY,
            temperature=0.2,
        )

    if provider == "openrouter":
        return ChatOpenAI(
            model=cfg.LLM_MODEL,
            api_key=cfg.OPENROUTER_API_KEY,
            base_url=cfg.OPENROUTER_BASE_URL,
            temperature=0.2,
            default_headers={
                "HTTP-Referer": "https://github.com/multi-agent-research",
                "X-Title": "Multi-Agent Research Assistant",
            },
        )

    if provider == "ollama":
        return ChatOpenAI(
            model=cfg.LLM_MODEL,
            api_key="ollama",
            base_url=cfg.OLLAMA_BASE_URL,
            temperature=0.2,
        )

    # Default: OpenAI
    return ChatOpenAI(
        model=cfg.LLM_MODEL,
        api_key=cfg.OPENAI_API_KEY,
        temperature=0.2,
    )


async def run_pipeline(
    query: str,
    store: MetricsStore | None = None,
    event_queue: "EventQueue | None" = None,
) -> PipelineResult:
    """Execute the full multi-agent research pipeline for a given query."""
    cfg.validate()

    tracker = MetricsTracker(query=query, model=cfg.LLM_MODEL, event_queue=event_queue)
    bus = MessageBus(event_queue=event_queue)
    llm = _build_llm()

    # ── Connect to MCP Servers via StdioConnection ────────────────────────────
    memory_connection: StdioConnection = {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["mcp_servers/memory_server.py"],
    }
    search_connection: StdioConnection = {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["mcp_servers/search_server.py"],
    }

    memory_tools = await load_mcp_tools(None, connection=memory_connection)
    search_tools = await load_mcp_tools(None, connection=search_connection)

    search_tool = next(t for t in search_tools if t.name == "web_search")
    memory_write_tool = next(t for t in memory_tools if t.name == "memory_write")

    # ── Instantiate Agents ────────────────────────────────────────────────────
    orchestrator = OrchestratorAgent(llm=llm, tracker=tracker, bus=bus)
    search_agent = SearchAgent(
        llm=llm, tracker=tracker, bus=bus,
        search_tool=search_tool,
        memory_write_tool=memory_write_tool,
    )
    summarizer = SummarizerAgent(llm=llm, tracker=tracker, bus=bus)
    critic = CriticAgent(llm=llm, tracker=tracker, bus=bus)

    # ── Run Pipeline ──────────────────────────────────────────────────────────
    plan = await orchestrator.run(query)
    search_results = await search_agent.run()
    summaries = await summarizer.run()
    critique_result = await critic.run()

    # ── Finalise metrics ──────────────────────────────────────────────────────
    confidence = critique_result.get("confidence_score", 0.0)
    tracker.set_confidence(confidence)
    tracker.set_a2a_count(len(bus.log))
    run_metrics = tracker.finish()

    if store:
        store.save(run_metrics)

    result = PipelineResult(
        query=query,
        plan=plan,
        search_results=search_results,
        summaries=summaries,
        critique=critique_result.get("critique", {}),
        final_report=critique_result.get("final_report", {}),
        confidence_score=confidence,
        metrics=run_metrics.to_dict(),
        a2a_log=bus.log_as_dicts(),
    )

    # Signal the UI that the pipeline is done
    if event_queue is not None:
        from events import PipelineFinished
        event_queue.put(PipelineFinished(success=True))

    return result


def run_pipeline_threaded(
    query: str,
    result_holder: list,          # mutable list; result appended as ("ok"|"error", data)
    store: MetricsStore | None = None,
    event_queue: "EventQueue | None" = None,
) -> threading.Thread:
    """
    Run the pipeline in a background daemon thread.

    The caller passes an empty list as result_holder; when the thread
    finishes it appends either ("ok", PipelineResult) or ("error", str).

    Returns the started Thread so the caller can join or check is_alive().
    """
    def _target():
        try:
            result = asyncio.run(run_pipeline(query, store=store, event_queue=event_queue))
            result_holder.append(("ok", result))
        except Exception as exc:
            result_holder.append(("error", str(exc)))
            if event_queue is not None:
                from events import PipelineFinished
                event_queue.put(PipelineFinished(success=False, error=str(exc)))

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    return thread
