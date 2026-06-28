"""
Minimal MCP server with a few example tools.
Run via: python3.11 server.py
(In practice, the client launches it as a subprocess via stdio transport.)
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-server")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@mcp.tool()
def greet(name: str) -> str:
    """Return a greeting message for the given name."""
    return f"Hello, {name}!"


@mcp.tool()
def word_count(text: str) -> int:
    """Count the number of words in a text string."""
    return len(text.split())


if __name__ == "__main__":
    mcp.run(transport="stdio")
