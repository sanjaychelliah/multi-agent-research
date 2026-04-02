"""Base class for all research pipeline agents."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from langchain_core.language_models import BaseChatModel

from a2a import AgentCard
from metrics import MetricsTracker


class BaseAgent:
    """
    Shared scaffolding: LLM handle, agent card, metrics, and live event helpers.

    Subclasses get:
      self._emit(event)           — push any event to the live feed
      self._tracked_run(label)    — async context manager that emits
                                    AgentStarted / AgentFinished automatically
    """

    card: AgentCard  # must be set by subclass as a class variable

    def __init__(self, llm: BaseChatModel, tracker: MetricsTracker) -> None:
        self.llm = llm
        self.tracker = tracker
        self._metrics = tracker.agent(self.card.agent_id)

    @property
    def agent_id(self) -> str:
        return self.card.agent_id

    def _emit(self, event: object) -> None:
        """Push an event to the live EventQueue (no-op if none configured)."""
        self.tracker.emit(event)

    @asynccontextmanager
    async def _tracked_run(self, label: str) -> AsyncGenerator[None, None]:
        """
        Async context manager that bookends an agent's run() with
        AgentStarted / AgentFinished events and captures total wall-clock time.

        Usage inside any agent's run():
            async with self._tracked_run("🔍 Search Agent"):
                ...
        """
        from events import AgentStarted, AgentFinished

        self._emit(AgentStarted(agent_name=self.agent_id, agent_label=label))
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            m = self._metrics
            self._emit(AgentFinished(
                agent_name=self.agent_id,
                agent_label=label,
                tokens=m.total_tokens,
                latency_ms=elapsed_ms,
                cost_usd=m.cost_usd,
            ))
