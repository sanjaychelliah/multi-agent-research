# Multi-Agent Research Assistant

A production-style multi-agent pipeline that researches any topic using **LangChain**, **MCP (Model Context Protocol)**, and **A2A (Agent-to-Agent)** communication — with full metrics, tracing, and a Streamlit dashboard.

```
User Query
    │
    ▼
┌─────────────────┐    A2A: web_search tasks
│  Orchestrator   │──────────────────────────►┌──────────────┐
│     Agent       │                           │ Search Agent │ ← MCP: web_search tool
└─────────────────┘                           └──────┬───────┘
                                                     │ A2A: summarize tasks
                                                     ▼
                                              ┌──────────────────┐
                                              │ Summarizer Agent │
                                              └──────┬───────────┘
                                                     │ A2A: critique task
                                                     ▼
                                              ┌──────────────┐
                                              │ Critic Agent │
                                              └──────┬───────┘
                                                     │
                                                     ▼
                                          Final Report + Confidence Score
                                          + Per-Agent Metrics (tokens, latency, cost)
```

## Features

| Feature | Details |
|---|---|
| **LangChain** | ReAct-style agents, LCEL chains, async invocation |
| **MCP Servers** | `memory-server` (key-value store) + `search-server` (web search) |
| **A2A Protocol** | Typed message passing (`AgentCard`, `TaskCard`, `A2AMessage`) |
| **4 Agents** | Orchestrator → Search → Summarizer → Critic |
| **Metrics** | Token usage, latency, cost per agent via LangChain callbacks |
| **Persistence** | SQLite metrics store — survives restarts |
| **Dashboard** | Streamlit UI with Plotly charts (confidence, token usage, latency) |
| **CLI** | Rich-formatted terminal output with `--json` flag |
| **Evals** | Benchmark runner over 5 test queries |

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/multi-agent-research.git
cd multi-agent-research
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
### 2. Configure

```bash
cp .env.example .env
# Edit .env — add OPENAI_API_KEY at minimum
```

### 3. Run (CLI)

```bash
python main.py "What are the latest breakthroughs in quantum computing?"
```

### 4. Launch dashboard

```bash
streamlit run ui/app.py
```

### 5. Run evaluations

```bash
python -m evals.evaluator
```

## Project Structure

```
multi-agent-research/
├── main.py                    # CLI entry point
├── pipeline.py                # End-to-end pipeline orchestration
├── config.py                  # Env-based configuration
│
├── agents/
│   ├── orchestrator.py        # Decomposes query into subtasks
│   ├── search_agent.py        # Fetches web results via MCP
│   ├── summarizer_agent.py    # Synthesises per-subtask summaries
│   └── critic_agent.py        # Reviews, scores, and writes final report
│
├── a2a/
│   ├── protocol.py            # AgentCard, TaskCard, A2AMessage (Pydantic)
│   └── message_bus.py         # Async pub/sub bus + full audit log
│
├── mcp_servers/
│   ├── memory_server.py       # MCP memory tool (read/write/list/clear)
│   └── search_server.py       # MCP search tool (Tavily or DuckDuckGo)
│
├── metrics/
│   ├── tracker.py             # LangChain callback → AgentMetrics/RunMetrics
│   └── store.py               # SQLite persistence
│
├── evals/
│   └── evaluator.py           # Benchmark runner
│
└── ui/
    └── app.py                 # Streamlit dashboard (3 pages)
```

## Architecture Decisions

### Why MCP for tools?
MCP creates a **stable, versioned contract** between agents and tools. Swapping the search provider (DuckDuckGo → Tavily → Perplexity) requires zero agent code changes — only the MCP server changes.

### Why A2A for agent communication?
Direct function calls couple agents together. A2A messaging decouples them — any agent can be replaced, scaled, or moved to a remote process without touching the others. The `MessageBus` audit log also gives free observability.

### Why separate Summarizer and Critic?
Single responsibility: the Summarizer extracts facts, the Critic evaluates quality. This separation is measurable — you can A/B test either agent independently.

## Metrics Tracked

| Metric | Where |
|---|---|
| Tokens (prompt / completion / total) | Per agent, via LangChain callback |
| Latency (ms) | Per agent, wall-clock |
| Cost (USD) | Per agent, model pricing table |
| LLM call count | Per agent |
| Confidence score | Overall run, from Critic |
| A2A message count | Overall run, from MessageBus log |

## Example Output

```
╭─ Step 1 — Orchestrator ───────────────────────────────╮
│ Research Goal: Survey recent quantum computing hardware │
╰────────────────────────────────────────────────────────╯
  #   Sub-query                          Rationale
  1   quantum computing hardware 2024    Current state of hardware
  2   quantum error correction progress  Key technical challenge
  3   IBM Google quantum roadmap 2025    Industry direction

╭─ Final Report ─────────────────────────────────────────╮
│ Quantum Computing Hardware: 2024 Landscape             │
│                                                        │
│ IBM unveiled its 133-qubit Heron processor...          │
╰────────────────────────────────────────────────────────╯

  Agent            Tokens  Latency(ms)  Cost(USD)
  orchestrator       412       1,240    $0.00018
  search_agent         0         890    $0.00000
  summarizer_agent   3,104     4,100    $0.00140
  critic_agent       5,820     6,300    $0.00260
  ─────────────────────────────────────────────
  TOTAL              9,336    12,530    $0.00418

  Confidence: ████████████████░░░░ 82%   A2A Messages: 8
```

## Extending

- **Add an agent**: subclass `BaseAgent`, give it an `AgentCard`, subscribe to the bus.
- **Add a tool**: implement a new `FastMCP` server, connect it in `pipeline.py`.
- **Swap the LLM**: change `LLM_MODEL` in `.env` — no code changes needed.
- **Enable LangSmith**: set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY`.

