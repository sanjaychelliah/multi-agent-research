"""
Metrics collection via LangChain callbacks + manual timing.

Tracks per-agent:
  - Token usage (prompt / completion / total)
  - Latency (wall-clock per agent invocation)
  - Estimated cost (based on model pricing)
  - Confidence score (provided by the critic agent)

All data flows into RunMetrics, which is persisted by MetricsStore.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult


# ── Pricing table (USD per 1k tokens) ────────────────────────────────────────
COST_PER_1K: dict[str, dict[str, float]] = {
    "gpt-4o": {"prompt": 0.005, "completion": 0.015},
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
    "claude-3-5-sonnet-20241022": {"prompt": 0.003, "completion": 0.015},
    "claude-3-haiku-20240307": {"prompt": 0.00025, "completion": 0.00125},
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = COST_PER_1K.get(model, {"prompt": 0.002, "completion": 0.002})
    return (prompt_tokens / 1000 * pricing["prompt"]) + (completion_tokens / 1000 * pricing["completion"])


@dataclass
class AgentMetrics:
    agent_name: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    llm_calls: int = 0
    _start: float = field(default=0.0, repr=False)

    def start_timer(self) -> None:
        self._start = time.perf_counter()

    def stop_timer(self) -> None:
        if self._start:
            self.latency_ms = (time.perf_counter() - self._start) * 1000

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": round(self.latency_ms, 1),
            "cost_usd": round(self.cost_usd, 6),
            "llm_calls": self.llm_calls,
        }


@dataclass
class RunMetrics:
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    agents: dict[str, AgentMetrics] = field(default_factory=dict)
    confidence_score: float = 0.0
    total_latency_ms: float = 0.0
    a2a_message_count: int = 0
    status: str = "running"   # running | completed | failed
    _run_start: float = field(default=0.0, repr=False)

    def start(self) -> None:
        self._run_start = time.perf_counter()

    def finish(self, status: str = "completed") -> None:
        self.status = status
        if self._run_start:
            self.total_latency_ms = (time.perf_counter() - self._run_start) * 1000

    @property
    def total_tokens(self) -> int:
        return sum(a.total_tokens for a in self.agents.values())

    @property
    def total_cost_usd(self) -> float:
        return sum(a.cost_usd for a in self.agents.values())

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "query": self.query,
            "status": self.status,
            "confidence_score": self.confidence_score,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "a2a_message_count": self.a2a_message_count,
            "agents": {k: v.to_dict() for k, v in self.agents.items()},
        }


class MetricsTracker:
    """Tracks metrics for a single pipeline run."""

    def __init__(self, query: str, model: str) -> None:
        self.run = RunMetrics(query=query)
        self.model = model
        self.run.start()

    def agent(self, name: str) -> AgentMetrics:
        if name not in self.run.agents:
            self.run.agents[name] = AgentMetrics(agent_name=name, model=self.model)
        return self.run.agents[name]

    def make_callback(self, agent_name: str) -> "LangChainMetricsCallback":
        return LangChainMetricsCallback(metrics=self.agent(agent_name), model=self.model)

    def set_confidence(self, score: float) -> None:
        self.run.confidence_score = max(0.0, min(1.0, score))

    def set_a2a_count(self, count: int) -> None:
        self.run.a2a_message_count = count

    def finish(self, status: str = "completed") -> RunMetrics:
        self.run.finish(status)
        return self.run


class LangChainMetricsCallback(BaseCallbackHandler):
    """LangChain callback that populates AgentMetrics on each LLM call."""

    def __init__(self, metrics: AgentMetrics, model: str) -> None:
        super().__init__()
        self.metrics = metrics
        self.model = model

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        self.metrics.llm_calls += 1
        self.metrics.start_timer()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.metrics.stop_timer()
        usage = response.llm_output or {}
        token_usage = usage.get("token_usage", {})
        pt = token_usage.get("prompt_tokens", 0)
        ct = token_usage.get("completion_tokens", 0)
        self.metrics.prompt_tokens += pt
        self.metrics.completion_tokens += ct
        self.metrics.total_tokens += pt + ct
        self.metrics.cost_usd += estimate_cost(self.model, pt, ct)
