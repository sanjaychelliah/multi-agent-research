"""Base class for all research pipeline agents."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from a2a import AgentCard
from metrics import MetricsTracker


class BaseAgent:
    """Shared scaffolding: LLM handle, agent card, metrics hook."""

    card: AgentCard  # must be set by subclass

    def __init__(self, llm: BaseChatModel, tracker: MetricsTracker) -> None:
        self.llm = llm
        self.tracker = tracker
        self._metrics = tracker.agent(self.card.agent_id)

    @property
    def agent_id(self) -> str:
        return self.card.agent_id
