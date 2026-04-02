"""
Pipeline Evaluator

Runs the research pipeline against a set of benchmark queries and
computes aggregate quality metrics:

  - avg_confidence        Mean confidence score across runs
  - task_completion_rate  % of runs that finished without error
  - avg_total_tokens      Mean token consumption
  - avg_latency_ms        Mean end-to-end latency
  - avg_cost_usd          Mean cost per run

Usage:
    python -m evals.evaluator
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from pipeline import run_pipeline
from metrics import MetricsStore
from config import cfg


BENCHMARK_QUERIES = [
    "What are the latest breakthroughs in quantum computing hardware?",
    "How is generative AI transforming drug discovery?",
    "What is the current state of nuclear fusion energy?",
    "Explain recent advances in CRISPR gene editing technology.",
    "What progress has been made in carbon capture and storage?",
]


@dataclass
class EvalResult:
    query: str
    success: bool
    confidence_score: float = 0.0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    error: str = ""


@dataclass
class EvalReport:
    results: list[EvalResult] = field(default_factory=list)

    @property
    def task_completion_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results)

    @property
    def avg_confidence(self) -> float:
        successful = [r for r in self.results if r.success]
        return sum(r.confidence_score for r in successful) / len(successful) if successful else 0.0

    @property
    def avg_total_tokens(self) -> float:
        successful = [r for r in self.results if r.success]
        return sum(r.total_tokens for r in successful) / len(successful) if successful else 0.0

    @property
    def avg_latency_ms(self) -> float:
        successful = [r for r in self.results if r.success]
        return sum(r.total_latency_ms for r in successful) / len(successful) if successful else 0.0

    @property
    def avg_cost_usd(self) -> float:
        successful = [r for r in self.results if r.success]
        return sum(r.total_cost_usd for r in successful) / len(successful) if successful else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "total_runs": len(self.results),
                "task_completion_rate": round(self.task_completion_rate, 3),
                "avg_confidence": round(self.avg_confidence, 3),
                "avg_total_tokens": round(self.avg_total_tokens, 1),
                "avg_latency_ms": round(self.avg_latency_ms, 1),
                "avg_cost_usd": round(self.avg_cost_usd, 6),
            },
            "runs": [
                {
                    "query": r.query,
                    "success": r.success,
                    "confidence_score": r.confidence_score,
                    "total_tokens": r.total_tokens,
                    "latency_ms": r.total_latency_ms,
                    "cost_usd": r.total_cost_usd,
                    "error": r.error,
                }
                for r in self.results
            ],
        }


class Evaluator:
    def __init__(self, queries: list[str] | None = None) -> None:
        self.queries = queries or BENCHMARK_QUERIES
        self.store = MetricsStore(cfg.METRICS_DB_PATH)

    async def run(self) -> EvalReport:
        report = EvalReport()
        for query in self.queries:
            print(f"  Evaluating: {query[:60]}…")
            try:
                result = await run_pipeline(query, store=self.store)
                m = result.metrics
                report.results.append(EvalResult(
                    query=query,
                    success=True,
                    confidence_score=result.confidence_score,
                    total_tokens=m["total_tokens"],
                    total_latency_ms=m["total_latency_ms"],
                    total_cost_usd=m["total_cost_usd"],
                ))
            except Exception as e:
                report.results.append(EvalResult(query=query, success=False, error=str(e)))
        return report


if __name__ == "__main__":
    cfg.validate()
    print("Running benchmark evaluation…\n")
    report = asyncio.run(Evaluator().run())
    print(json.dumps(report.to_dict(), indent=2))
