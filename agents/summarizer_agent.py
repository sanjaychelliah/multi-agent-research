"""
Summarizer Agent

Receives raw search results from the Search Agent, produces a clean
structured summary per subtask, then forwards all summaries to the
Critic Agent.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from a2a import AgentCard, A2AMessage, TaskCard, MessageBus
from metrics import MetricsTracker

from .base import BaseAgent


SYSTEM_PROMPT = """You are a Research Summarizer. Given a search query and raw
web search results, produce a concise, factual summary.

Return ONLY a valid JSON object:
{{
  "subtask_id": <int>,
  "query": "<original query>",
  "key_findings": ["<finding 1>", "<finding 2>", ...],
  "summary": "<2–3 paragraph synthesis of the results>",
  "sources": ["<url1>", "<url2>", ...]
}}

Rules:
- Be factual. If results are sparse, say so.
- Extract concrete facts, numbers, dates where present.
- Cite source URLs in the sources array.
- Output only JSON.
"""


class SummarizerAgent(BaseAgent):
    card = AgentCard(
        agent_id="summarizer_agent",
        name="Research Summarizer",
        description="Synthesizes raw search results into structured summaries per subtask.",
        skills=["text summarization", "information extraction", "structured output"],
    )

    def __init__(self, llm: BaseChatModel, tracker: MetricsTracker, bus: MessageBus) -> None:
        super().__init__(llm, tracker)
        self.bus = bus
        self._summaries: list[dict[str, Any]] = []
        self._pending: list[A2AMessage] = []

        bus.subscribe(self.agent_id, self._handle_message)

    async def _handle_message(self, message: A2AMessage) -> None:
        if message.task.task_type == "summarize":
            self._pending.append(message)

    async def run(self) -> list[dict[str, Any]]:
        async with self._tracked_run("📝 Summarizer"):
            await asyncio.sleep(0)
            tasks = list(self._pending)
            self._pending.clear()

            for msg in tasks:
                self._metrics.start_timer()
                task_input = msg.task.input

                results_text = json.dumps(task_input.get("results", []), indent=2)
                user_content = (
                    f"Query: {task_input['query']}\n\n"
                    f"Rationale: {task_input.get('rationale', '')}\n\n"
                    f"Search Results:\n{results_text}"
                )

                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=user_content),
                ]

                callback = self.tracker.make_callback("summarizer_agent")
                response = await self.llm.ainvoke(messages, config={"callbacks": [callback]})
                self._metrics.stop_timer()

                try:
                    summary = json.loads(response.content)
                except json.JSONDecodeError:
                    summary = {
                        "subtask_id": task_input.get("subtask_id", 0),
                        "query": task_input["query"],
                        "key_findings": [],
                        "summary": response.content,
                        "sources": [],
                    }

                self._summaries.append(summary)

            # Forward all summaries to critic in one shot
            if self._summaries:
                msg = A2AMessage(
                    sender=self.agent_id,
                    receiver="critic_agent",
                    task=TaskCard(
                        task_type="critique",
                        input={"summaries": self._summaries},
                    ),
                )
                await self.bus.publish(msg)

        return self._summaries
