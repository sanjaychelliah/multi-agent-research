"""
In-process A2A message bus.

Acts as the communication backbone — agents post messages and the bus
routes them to subscribers. Also maintains a full audit log for the
metrics dashboard and tracing.

When an EventQueue is provided, each published message also emits an
A2AMessageSent event for the live Streamlit activity feed.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import TYPE_CHECKING, Callable, Awaitable

from .protocol import A2AMessage

if TYPE_CHECKING:
    from events import EventQueue


Handler = Callable[[A2AMessage], Awaitable[None]]


class MessageBus:
    """Async pub/sub bus for A2A messages."""

    def __init__(self, event_queue: "EventQueue | None" = None) -> None:
        self._subscribers: dict[str, list[Handler]] = defaultdict(list)
        self._log: list[A2AMessage] = []
        self._event_queue = event_queue

    def subscribe(self, agent_id: str, handler: Handler) -> None:
        """Register a handler to receive messages addressed to agent_id."""
        self._subscribers[agent_id].append(handler)

    async def publish(self, message: A2AMessage) -> None:
        """Dispatch message to all handlers registered for the receiver."""
        self._log.append(message)

        # Emit live event
        if self._event_queue is not None:
            from events import A2AMessageSent
            self._event_queue.put(A2AMessageSent(
                sender=message.sender,
                receiver=message.receiver,
                task_type=message.task.task_type,
            ))

        handlers = self._subscribers.get(message.receiver, [])
        await asyncio.gather(*(h(message) for h in handlers))

    @property
    def log(self) -> list[A2AMessage]:
        """Full ordered log of all messages exchanged."""
        return list(self._log)

    def log_as_dicts(self) -> list[dict]:
        return [
            {
                "message_id": m.message_id,
                "sender": m.sender,
                "receiver": m.receiver,
                "task_type": m.task.task_type,
                "status": m.task.status,
                "timestamp": m.timestamp,
            }
            for m in self._log
        ]

    def clear(self) -> None:
        self._log.clear()
        self._subscribers.clear()
