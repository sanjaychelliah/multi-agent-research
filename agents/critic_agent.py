"""
Critic Agent

Reviews all subtask summaries, identifies gaps and inconsistencies,
assigns a confidence score, and produces the final research report.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from a2a import AgentCard, A2AMessage, MessageBus
from metrics import MetricsTracker

from .base import BaseAgent


SYSTEM_PROMPT = """You are a Research Critic and Report Writer.

You receive a collection of summaries produced by other agents for different
sub-queries about a research topic. Your job is to:
1. Critique the summaries — identify any gaps, contradictions, or weak sources.
2. Synthesize them into a single cohesive final research report.
3. Assign an overall confidence score (0.0–1.0) reflecting completeness and source quality.

Return ONLY a valid JSON object:
{{
  "critique": {{
    "gaps": ["<gap 1>", ...],
    "contradictions": ["<contradiction>", ...],
    "strengths": ["<strength>", ...]
  }},
  "final_report": {{
    "title": "<descriptive title>",
    "executive_summary": "<3–4 sentence overview>",
    "sections": [
      {{"heading": "<heading>", "content": "<detailed paragraphs>"}},
      ...
    ],
    "key_takeaways": ["<takeaway 1>", "<takeaway 2>", ...],
    "all_sources": ["<url>", ...]
  }},
  "confidence_score": <float 0.0–1.0>,
  "confidence_rationale": "<one sentence explaining the score>"
}}

Output only JSON.
"""


class CriticAgent(BaseAgent):
    card = AgentCard(
        agent_id="critic_agent",
        name="Research Critic",
        description="Reviews summaries for quality, synthesizes the final report, and assigns a confidence score.",
        skills=["critical analysis", "report writing", "confidence scoring"],
    )

    def __init__(self, llm: BaseChatModel, tracker: MetricsTracker, bus: MessageBus) -> None:
        super().__init__(llm, tracker)
        self.bus = bus
        self._pending: list[A2AMessage] = []
        self.result: dict[str, Any] | None = None

        bus.subscribe(self.agent_id, self._handle_message)

    async def _handle_message(self, message: A2AMessage) -> None:
        if message.task.task_type == "critique":
            self._pending.append(message)

    async def run(self) -> dict[str, Any]:
        await asyncio.sleep(0)
        tasks = list(self._pending)
        self._pending.clear()

        if not tasks:
            return {"error": "No summaries received for critique."}

        # Merge summaries from all messages
        all_summaries: list[dict] = []
        for msg in tasks:
            all_summaries.extend(msg.task.input.get("summaries", []))

        self._metrics.start_timer()

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Summaries to critique and synthesize:\n\n{json.dumps(all_summaries, indent=2)}"),
        ]

        callback = self.tracker.make_callback("critic_agent")
        response = await self.llm.ainvoke(messages, config={"callbacks": [callback]})
        self._metrics.stop_timer()

        try:
            self.result = json.loads(response.content)
        except json.JSONDecodeError:
            self.result = {
                "critique": {"gaps": [], "contradictions": [], "strengths": []},
                "final_report": {
                    "title": "Research Report",
                    "executive_summary": response.content,
                    "sections": [],
                    "key_takeaways": [],
                    "all_sources": [],
                },
                "confidence_score": 0.5,
                "confidence_rationale": "Could not parse structured output.",
            }

        return self.result
