"""MCP 集成测试。

单元测试（默认）：本地模式 ToolManager（不依赖 MCP Server / API）。
集成测试（integration）：需要先启动 MCP Server（python mcp_server/server.py）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.tools.manager import ToolManager  # noqa: E402

MCP_URL = "http://127.0.0.1:9123/mcp"
EXPECTED_TOOLS = {"query_order", "query_product", "query_logistics", "apply_refund"}


# ---------- 单元测试（本地模式，无需 MCP / API） ----------

def test_tool_manager_local():
    manager = ToolManager(use_mcp=False)
    try:
        names = {d["function"]["name"] for d in manager.tool_definitions}
        assert EXPECTED_TOOLS.issubset(names)

        result = json.loads(manager.execute_tool("query_order", {"order_id": "ORD-20240115-001"}))
        assert result["success"] is True
        assert result["order"]["status"] == "shipped"
    finally:
        manager.close()


def test_tool_manager_unknown_tool():
    manager = ToolManager(use_mcp=False)
    try:
        result = json.loads(manager.execute_tool("no_such_tool", {}))
        assert "未知工具" in result.get("error", "")
    finally:
        manager.close()


# ---------- 集成测试（需要 MCP Server） ----------

@pytest.mark.integration
def test_mcp_connection_live():
    from app.mcp_client import MCPClient

    client = MCPClient(MCP_URL)
    try:
        tools = client.connect()
        assert len(tools) == 4
        assert {t["function"]["name"] for t in tools} == EXPECTED_TOOLS
    finally:
        client.close()


@pytest.mark.integration
def test_mcp_schema_conversion_live():
    from app.mcp_client import MCPClient

    client = MCPClient(MCP_URL)
    try:
        tools = client.connect()
        for tool in tools:
            assert tool.get("type") == "function"
            func = tool["function"]
            assert func.get("name")
            assert func.get("description")
            assert func["parameters"]["type"] == "object"
            assert func["parameters"].get("properties")
    finally:
        client.close()


@pytest.mark.integration
def test_tool_manager_mcp_live():
    manager = ToolManager(use_mcp=True, mcp_server_url=MCP_URL)
    try:
        result = json.loads(manager.execute_tool("query_logistics", {"order_id": "ORD-20240115-001"}))
        assert result["success"] is True
    finally:
        manager.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])