"""
Search Agent

Receives web_search TaskCards from the orchestrator via the A2A bus.
Uses the MCP search server tool to fetch results, stores them in the
MCP memory server, and dispatches results to the summarizer.

Emits ToolCallStarted / ToolCallFinished events for each MCP tool call.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool

from a2a import AgentCard, A2AMessage, TaskCard, MessageBus
from metrics import MetricsTracker

from .base import BaseAgent


class SearchAgent(BaseAgent):
    card = AgentCard(
        agent_id="search_agent",
        name="Web Search Agent",
        description="Executes web searches using MCP tools and stores results in agent memory.",
        skills=["web search", "MCP tool use", "result storage"],
    )

    def __init__(
        self,
        llm: BaseChatModel,
        tracker: MetricsTracker,
        bus: MessageBus,
        search_tool: StructuredTool,
        memory_write_tool: StructuredTool,
    ) -> None:
        super().__init__(llm, tracker)
        self.bus = bus
        self.search_tool = search_tool
        self.memory_write_tool = memory_write_tool
        self._results: list[dict[str, Any]] = []
        self._pending_tasks: list[A2AMessage] = []

        bus.subscribe(self.agent_id, self._handle_message)

    async def _handle_message(self, message: A2AMessage) -> None:
        """Process an incoming A2A message."""
        if message.task.task_type == "web_search":
            self._pending_tasks.append(message)

    async def run(self) -> list[dict[str, Any]]:
        """Process all queued search tasks, return aggregated results."""
        async with self._tracked_run("🔍 Search Agent"):
            from events import ToolCallStarted, ToolCallFinished

            # Small yield to allow bus dispatches to complete
            await asyncio.sleep(0)

            tasks = list(self._pending_tasks)
            self._pending_tasks.clear()

            for msg in tasks:
                self._metrics.start_timer()
                task_input = msg.task.input
                query = task_input["query"]

                # ── web_search MCP tool call ──────────────────────────────────
                self._emit(ToolCallStarted(
                    agent_name=self.agent_id,
                    tool_name="web_search",
                    input_preview=f'query="{query}"',
                ))
                t0 = time.perf_counter()

                try:
                    raw = await self.search_tool.ainvoke({"query": query})
                    if isinstance(raw, str):
                        parsed = json.loads(raw)
                        results = parsed if isinstance(parsed, list) else []
                    elif isinstance(raw, list):
                        results = raw
                    else:
                        results = []
                except Exception:
                    results = []

                search_ms = (time.perf_counter() - t0) * 1000
                self._metrics.stop_timer()

                self._emit(ToolCallFinished(
                    agent_name=self.agent_id,
                    tool_name="web_search",
                    result_count=len(results),
                    latency_ms=search_ms,
                ))

                # ── memory_write MCP tool call ────────────────────────────────
                mem_key = f"search_results_task_{task_input['subtask_id']}"
                self._emit(ToolCallStarted(
                    agent_name=self.agent_id,
                    tool_name="memory_write",
                    input_preview=f'key="{mem_key}"',
                ))
                t1 = time.perf_counter()
                try:
                    await self.memory_write_tool.ainvoke({"key": mem_key, "value": json.dumps(results)})
                except Exception:
                    pass
                mem_ms = (time.perf_counter() - t1) * 1000
                self._emit(ToolCallFinished(
                    agent_name=self.agent_id,
                    tool_name="memory_write",
                    result_count=1,
                    latency_ms=mem_ms,
                ))

                result_payload = {
                    "subtask_id": task_input["subtask_id"],
                    "query": query,
                    "rationale": task_input.get("rationale", ""),
                    "results": results,
                    "memory_key": mem_key,
                }
                self._results.append(result_payload)

                # Forward to summarizer via A2A
                reply = A2AMessage(
                    sender=self.agent_id,
                    receiver="summarizer_agent",
                    task=TaskCard(
                        task_type="summarize",
                        input=result_payload,
                    ),
                )
                await self.bus.publish(reply)

        return self._results
