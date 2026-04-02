"""
A2A (Agent-to-Agent) Protocol definitions.

Lightweight implementation inspired by Google's A2A specification:
https://google.github.io/A2A

Each agent publishes an AgentCard describing its capabilities.
Agents communicate via typed A2AMessage objects tracked in shared state.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentCard(BaseModel):
    """Metadata describing an agent's identity and capabilities."""

    agent_id: str
    name: str
    description: str
    skills: list[str]
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)


class TaskCard(BaseModel):
    """A discrete unit of work passed between agents."""

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str
    input: dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None

    def complete(self, result: dict[str, Any]) -> None:
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def fail(self, error: str) -> None:
        self.status = TaskStatus.FAILED
        self.error = error
        self.completed_at = datetime.now(timezone.utc).isoformat()


class A2AMessage(BaseModel):
    """An envelope wrapping a TaskCard passed from one agent to another."""

    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str
    receiver: str
    task: TaskCard
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def reply(self, sender: str, result: dict[str, Any] | None = None, error: str | None = None) -> "A2AMessage":
        """Create a reply message with updated task status."""
        updated_task = self.task.model_copy(deep=True)
        if result:
            updated_task.complete(result)
        elif error:
            updated_task.fail(error)

        return A2AMessage(
            sender=sender,
            receiver=self.sender,
            task=updated_task,
            metadata={"reply_to": self.message_id},
        )
