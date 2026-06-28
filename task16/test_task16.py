"""Tests for task16: MCP connection and tool listing."""

import asyncio
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_SCRIPT = Path(__file__).parent / "server.py"


async def get_tools():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_SCRIPT)],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return result.tools


@pytest.fixture(scope="module")
def tools():
    return asyncio.run(get_tools())


def test_connection_returns_tools(tools):
    assert len(tools) > 0


def test_expected_tools_present(tools):
    names = {t.name for t in tools}
    assert {"add", "greet", "word_count"} == names


def test_tools_have_descriptions(tools):
    for tool in tools:
        assert tool.description, f"Tool '{tool.name}' has no description"


def test_add_has_correct_params(tools):
    add = next(t for t in tools if t.name == "add")
    params = set(add.inputSchema.get("properties", {}).keys())
    assert params == {"a", "b"}
