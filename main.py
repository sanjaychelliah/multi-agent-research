"""
CLI entry point for the Multi-Agent Research Assistant.

Usage:
    python main.py "What are the latest breakthroughs in quantum computing?"
    python main.py "Explain the current state of fusion energy" --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich import box

from config import cfg
from metrics import MetricsStore
from pipeline import PipelineResult, run_pipeline

console = Console()


def print_plan(plan: dict) -> None:
    console.print(Panel.fit(
        f"[bold cyan]Research Goal:[/bold cyan] {plan.get('research_goal', 'N/A')}",
        title="[bold]Step 1 — Orchestrator[/bold]",
        border_style="cyan",
    ))
    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("Sub-query")
    table.add_column("Rationale", style="dim")
    for st in plan.get("subtasks", []):
        table.add_row(str(st["id"]), st["query"], st.get("rationale", ""))
    console.print(table)


def print_search_results(results: list[dict]) -> None:
    console.print(Panel.fit(
        f"[bold green]Found {sum(len(r.get('results', [])) for r in results)} results across {len(results)} subtasks[/bold green]",
        title="[bold]Step 2 — Search Agent[/bold]",
        border_style="green",
    ))


def print_summaries(summaries: list[dict]) -> None:
    console.print(Panel.fit(
        f"[bold yellow]Summarised {len(summaries)} subtasks[/bold yellow]",
        title="[bold]Step 3 — Summarizer Agent[/bold]",
        border_style="yellow",
    ))
    for s in summaries:
        console.print(f"\n  [bold]Subtask {s.get('subtask_id')}:[/bold] {s.get('query', '')}")
        for finding in s.get("key_findings", [])[:3]:
            console.print(f"    • {finding}")


def print_report(result: PipelineResult) -> None:
    report = result.final_report
    critique = result.critique

    console.print("\n")
    console.print(Panel(
        Markdown(f"# {report.get('title', 'Research Report')}\n\n"
                 f"{report.get('executive_summary', '')}"),
        title="[bold]Step 4 — Final Report (Critic Agent)[/bold]",
        border_style="blue",
        expand=False,
    ))

    for section in report.get("sections", []):
        console.print(f"\n[bold blue]{section.get('heading')}[/bold blue]")
        console.print(section.get("content", ""))

    if report.get("key_takeaways"):
        console.print("\n[bold]Key Takeaways:[/bold]")
        for t in report["key_takeaways"]:
            console.print(f"  ✓ {t}")

    if critique.get("gaps"):
        console.print("\n[bold yellow]Identified Gaps:[/bold yellow]")
        for g in critique["gaps"]:
            console.print(f"  ⚠ {g}")


def print_metrics(result: PipelineResult) -> None:
    m = result.metrics
    console.print("\n")

    table = Table(title="Pipeline Metrics", box=box.ROUNDED, show_lines=True)
    table.add_column("Agent", style="bold")
    table.add_column("Tokens", justify="right")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Cost (USD)", justify="right")
    table.add_column("LLM Calls", justify="right")

    for name, agent in m.get("agents", {}).items():
        table.add_row(
            name,
            str(agent["total_tokens"]),
            f"{agent['latency_ms']:.0f}",
            f"${agent['cost_usd']:.5f}",
            str(agent["llm_calls"]),
        )

    table.add_section()
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{m['total_tokens']}[/bold]",
        f"[bold]{m['total_latency_ms']:.0f}[/bold]",
        f"[bold]${m['total_cost_usd']:.5f}[/bold]",
        "",
    )
    console.print(table)

    score = result.confidence_score
    color = "green" if score >= 0.7 else "yellow" if score >= 0.4 else "red"
    bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
    console.print(
        f"\n  Confidence: [{color}]{bar}[/{color}] "
        f"[bold]{score:.0%}[/bold]   "
        f"A2A Messages: [bold]{m['a2a_message_count']}[/bold]"
    )


def print_a2a_log(log: list[dict]) -> None:
    console.print("\n[dim]─── A2A Message Log ───[/dim]")
    for msg in log:
        console.print(
            f"[dim]  {msg['timestamp'][:19]}  "
            f"[bold]{msg['sender']}[/bold] → [bold]{msg['receiver']}[/bold]  "
            f"[{msg['status']}] {msg['task_type']}[/dim]"
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Agent Research Assistant")
    parser.add_argument("query", help="Research question or topic")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--no-metrics", action="store_true", help="Skip metrics display")
    args = parser.parse_args()

    store = MetricsStore(cfg.METRICS_DB_PATH)

    console.rule("[bold blue]Multi-Agent Research Assistant[/bold blue]")
    console.print(f"\n[bold]Query:[/bold] {args.query}\n")

    try:
        with console.status("[bold green]Running pipeline…[/bold green]"):
            result = await run_pipeline(args.query, store=store)
    except Exception as exc:
        console.print(f"[bold red]Pipeline failed:[/bold red] {exc}")
        raise SystemExit(1)

    if args.json:
        print(json.dumps({
            "query": result.query,
            "plan": result.plan,
            "final_report": result.final_report,
            "confidence_score": result.confidence_score,
            "metrics": result.metrics,
        }, indent=2))
        return

    print_plan(result.plan)
    print_search_results(result.search_results)
    print_summaries(result.summaries)
    print_report(result)

    if not args.no_metrics:
        print_metrics(result)
        print_a2a_log(result.a2a_log)

    console.rule("[bold blue]Done[/bold blue]")


if __name__ == "__main__":
    asyncio.run(main())
