"""
Task 16: Establish an MCP connection and list available tools.

The client launches server.py as a subprocess via the stdio transport —
no network port, no cloud deployment needed.
"""

import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_SCRIPT = Path(__file__).parent / "server.py"


async def list_mcp_tools() -> None:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_SCRIPT)],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.list_tools()

            print(f"Connected to MCP server. {len(result.tools)} tool(s) available:\n")
            for tool in result.tools:
                params = list(tool.inputSchema.get("properties", {}).keys())
                print(f"  {tool.name}({', '.join(params)})")
                print(f"    {tool.description}")


if __name__ == "__main__":
    asyncio.run(list_mcp_tools())
