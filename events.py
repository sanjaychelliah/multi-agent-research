"""
Typed event system for real-time pipeline observability.

Events are pushed from the async pipeline (running in a background thread)
into a thread-safe EventQueue. The Streamlit UI polls the queue on each
rerun to render the live activity feed.
"""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass, field


# ── Event types ───────────────────────────────────────────────────────────────

@dataclass
class BaseEvent:
    event_type: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentStarted(BaseEvent):
    event_type: str = "agent_started"
    agent_name: str = ""
    agent_label: str = ""   # human-readable label with emoji


@dataclass
class AgentFinished(BaseEvent):
    event_type: str = "agent_finished"
    agent_name: str = ""
    agent_label: str = ""
    tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0


@dataclass
class LLMCallStarted(BaseEvent):
    event_type: str = "llm_call_started"
    agent_name: str = ""
    model: str = ""
    prompt_preview: str = ""    # first ~100 chars of the human turn


@dataclass
class LLMCallFinished(BaseEvent):
    event_type: str = "llm_call_finished"
    agent_name: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0


@dataclass
class ToolCallStarted(BaseEvent):
    event_type: str = "tool_call_started"
    agent_name: str = ""
    tool_name: str = ""
    input_preview: str = ""     # e.g. 'query="quantum computing"'


@dataclass
class ToolCallFinished(BaseEvent):
    event_type: str = "tool_call_finished"
    agent_name: str = ""
    tool_name: str = ""
    result_count: int = 0       # number of results returned
    latency_ms: float = 0.0


@dataclass
class A2AMessageSent(BaseEvent):
    event_type: str = "a2a_message"
    sender: str = ""
    receiver: str = ""
    task_type: str = ""


@dataclass
class PipelineFinished(BaseEvent):
    event_type: str = "pipeline_finished"
    success: bool = True
    error: str = ""


# ── Queue ─────────────────────────────────────────────────────────────────────

class EventQueue:
    """Thread-safe, non-blocking queue for pipeline events."""

    def __init__(self) -> None:
        self._q: queue.Queue[BaseEvent] = queue.Queue()

    def put(self, event: BaseEvent) -> None:
        """Push an event (safe to call from any thread or async context)."""
        self._q.put_nowait(event)

    def drain(self) -> list[BaseEvent]:
        """Pull all currently available events without blocking."""
        events: list[BaseEvent] = []
        while True:
            try:
                events.append(self._q.get_nowait())
            except queue.Empty:
                break
        return events
