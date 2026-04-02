"""
MCP Search Server

Wraps web search providers behind a clean MCP interface so agents
interact with a stable tool contract regardless of which provider is used.

Priority order:
  1. Tavily (if TAVILY_API_KEY is set) — better quality
  2. DuckDuckGo (free, no key required) — fallback

Tools exposed:
  - web_search(query, max_results)  → list of {title, url, snippet}
"""

import json
import os

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("search-server")


def _search_tavily(query: str, max_results: int) -> list[dict]:
    from tavily import TavilyClient
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    resp = client.search(query=query, max_results=max_results)
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
        for r in resp.get("results", [])
    ]


def _search_duckduckgo(query: str, max_results: int) -> list[dict]:
    from duckduckgo_search import DDGS
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(
                {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
            )
    return results


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return structured results.

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default 5)

    Returns:
        JSON array of objects with title, url, and snippet fields.
    """
    try:
        if os.getenv("TAVILY_API_KEY"):
            results = _search_tavily(query, max_results)
        else:
            results = _search_duckduckgo(query, max_results)
        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    mcp.run(transport="stdio")
