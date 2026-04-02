"""
Orchestrator Agent

Receives the raw user query and produces a structured research plan:
  - A list of focused subtask queries for the Search Agent
  - A brief description of the overall research goal

Sends an A2A TaskCard to the search agent for each subtask.
"""

from __future__ import annotations

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from a2a import AgentCard, A2AMessage, TaskCard, MessageBus
from metrics import MetricsTracker
from config import cfg

from .base import BaseAgent


SYSTEM_PROMPT = """You are a Research Orchestrator. Your job is to decompose a
user research query into {max_subtasks} focused sub-queries that, taken together,
will produce a comprehensive answer.

Return ONLY a valid JSON object with this exact structure:
{{
  "research_goal": "<one sentence describing the overall aim>",
  "subtasks": [
    {{"id": 1, "query": "<focused search query>", "rationale": "<why this matters>"}},
    ...
  ]
}}

Rules:
- Each sub-query must be independently searchable (no references to other tasks).
- Cover different angles: background, recent developments, challenges, future outlook.
- Max {max_subtasks} subtasks.
- Output only JSON — no markdown fences, no extra text.
"""


class OrchestratorAgent(BaseAgent):
    card = AgentCard(
        agent_id="orchestrator",
        name="Research Orchestrator",
        description="Decomposes a user query into focused subtasks and coordinates the pipeline.",
        skills=["query decomposition", "task planning", "pipeline coordination"],
    )

    def __init__(self, llm: BaseChatModel, tracker: MetricsTracker, bus: MessageBus) -> None:
        super().__init__(llm, tracker)
        self.bus = bus

    async def run(self, query: str) -> dict:
        """Break query into subtasks and dispatch to search agent."""
        self._metrics.start_timer()

        messages = [
            SystemMessage(content=SYSTEM_PROMPT.format(max_subtasks=cfg.MAX_SUBTASKS)),
            HumanMessage(content=f"Research query: {query}"),
        ]

        callback = self.tracker.make_callback("orchestrator")
        response = await self.llm.ainvoke(messages, config={"callbacks": [callback]})

        self._metrics.stop_timer()

        try:
            plan = json.loads(response.content)
        except json.JSONDecodeError:
            # Fallback: treat the whole query as a single subtask
            plan = {
                "research_goal": query,
                "subtasks": [{"id": 1, "query": query, "rationale": "Direct search"}],
            }

        # Dispatch A2A messages to search agent
        for subtask in plan.get("subtasks", []):
            msg = A2AMessage(
                sender=self.agent_id,
                receiver="search_agent",
                task=TaskCard(
                    task_type="web_search",
                    input={
                        "query": subtask["query"],
                        "subtask_id": subtask["id"],
                        "rationale": subtask.get("rationale", ""),
                    },
                ),
            )
            await self.bus.publish(msg)

        return plan
