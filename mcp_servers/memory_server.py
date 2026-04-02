"""
MCP Memory Server

Provides simple key-value memory that agents can read/write during a run.
Allows agents to share context without coupling directly to each other.

Tools exposed:
  - memory_write(key, value)   → store a note
  - memory_read(key)           → retrieve a note
  - memory_list()              → list all stored keys
  - memory_clear()             → wipe all memory
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("memory-server")

# In-process store (reset per run via memory_clear)
_store: dict[str, str] = {}


@mcp.tool()
def memory_write(key: str, value: str) -> str:
    """Store a value under a key in agent memory.

    Args:
        key: Unique identifier for this memory entry (e.g. 'search_results_q1')
        value: The content to store (text, JSON string, etc.)
    """
    _store[key] = value
    return f"Stored '{key}' ({len(value)} chars)"


@mcp.tool()
def memory_read(key: str) -> str:
    """Retrieve a previously stored value.

    Args:
        key: The key to look up
    """
    return _store.get(key, f"[No memory found for key: '{key}']")


@mcp.tool()
def memory_list() -> str:
    """List all keys currently stored in agent memory."""
    if not _store:
        return "Memory is empty."
    return "\n".join(f"- {k} ({len(v)} chars)" for k, v in _store.items())


@mcp.tool()
def memory_clear() -> str:
    """Clear all stored memory entries."""
    count = len(_store)
    _store.clear()
    return f"Cleared {count} entries."


if __name__ == "__main__":
    mcp.run(transport="stdio")
