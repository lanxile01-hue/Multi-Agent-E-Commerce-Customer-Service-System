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

from tools.manager import ToolManager  # noqa: E402

MCP_URL = "http://127.0.0.1:9123/mcp"
# 本地注册表与 MCP Server 都提供的工具
COMMON_TOOLS = {"query_order", "query_product", "query_logistics", "apply_refund", "search_knowledge"}
# 只存在于 MCP Server 侧的工具（接入 MCP 才获得的额外能力）
MCP_ONLY_TOOLS = {"query_after_sale_tickets", "create_after_sale_ticket"}
EXPECTED_TOOLS = COMMON_TOOLS | MCP_ONLY_TOOLS


# ---------- 单元测试（本地模式，无需 MCP / API） ----------

def test_tool_manager_local():
    manager = ToolManager(use_mcp=False)
    try:
        names = {d["function"]["name"] for d in manager.tool_definitions}
        assert COMMON_TOOLS.issubset(names)
        # 未启用 MCP 时，MCP 独有的工具不应出现
        assert not (MCP_ONLY_TOOLS & names)

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


def test_tool_manager_mcp_merge_and_dispatch(monkeypatch):
    """MCP 工具的合并规则：同名以 MCP 为准，本地补齐缺失项，按来源分发调用。"""
    from app.mcp_client import MCPClient

    class FakeMCPClient:
        def __init__(self, url):  # noqa: ARG002
            self.called: list[tuple[str, dict]] = []

        def connect(self):
            return [
                {"type": "function", "function": {"name": "query_order", "description": "MCP 版", "parameters": {"type": "object", "properties": {}}}},
                {"type": "function", "function": {"name": "query_after_sale_tickets", "description": "MCP 独有", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}}}},
            ]

        def call_tool(self, name, arguments):
            self.called.append((name, arguments))
            return json.dumps({"success": True, "from": "mcp", "tool": name}, ensure_ascii=False)

        def close(self):
            pass

    monkeypatch.setattr("app.mcp_client.MCPClient", FakeMCPClient)
    manager = ToolManager(use_mcp=True, mcp_server_url=MCP_URL)
    try:
        names = [d["function"]["name"] for d in manager.tool_definitions]
        # 同名工具不会重复出现，且以 MCP 版本为准
        assert names.count("query_order") == 1
        assert "MCP 版" in next(d for d in manager.tool_definitions if d["function"]["name"] == "query_order")["function"]["description"]
        # MCP 独有的工具进来
        assert "query_after_sale_tickets" in names
        # 本地有、MCP 没有的工具由本地补齐
        assert "apply_refund" in names

        # 按来源分发：query_order 走 MCP，list_user_orders（本地独有）走本地
        assert json.loads(manager.execute_tool("query_order", {"order_id": "X"}))["from"] == "mcp"
        assert json.loads(manager.execute_tool("list_user_orders", {}))["success"] is True
    finally:
        manager.close()


def test_tool_manager_mcp_failure_degrades_to_local(monkeypatch):
    """MCP 连不上时安全降级：仍然只有本地工具，且不抛异常。"""
    class BrokenMCPClient:
        def __init__(self, url):  # noqa: ARG002
            pass

        def connect(self):
            raise RuntimeError("connection refused")

        def close(self):
            pass

    monkeypatch.setattr("app.mcp_client.MCPClient", BrokenMCPClient)
    manager = ToolManager(use_mcp=True, mcp_server_url=MCP_URL)
    try:
        names = {d["function"]["name"] for d in manager.tool_definitions}
        assert COMMON_TOOLS.issubset(names)
        assert not (MCP_ONLY_TOOLS & names)
        assert json.loads(manager.execute_tool("query_order", {"order_id": "ORD-20240115-001"}))["success"] is True
    finally:
        manager.close()


def test_new_mcp_tools_are_risk_graded_and_whitelisted():
    """新增 MCP 工具必须纳入风险分级与子图白名单（写操作要人工确认）。"""
    from app.multi_agent.agents import AGENT_CONFIGS
    from tools.risks import requires_confirmation, risk_level

    assert risk_level("query_after_sale_tickets") == "read_only"
    assert requires_confirmation("create_after_sale_ticket") is True

    assert "query_after_sale_tickets" in AGENT_CONFIGS["postsale"]["tools"]
    assert "create_after_sale_ticket" in AGENT_CONFIGS["postsale"]["tools"]
    assert "query_after_sale_tickets" in AGENT_CONFIGS["complaint"]["tools"]
    # 售前不应该拿到售后工单工具
    assert "query_after_sale_tickets" not in AGENT_CONFIGS["presale"]["tools"]


# ---------- 集成测试（需要 MCP Server） ----------

@pytest.mark.integration
def test_mcp_connection_live():
    from app.mcp_client import MCPClient

    client = MCPClient(MCP_URL)
    try:
        tools = client.connect()
        assert len(tools) == len(EXPECTED_TOOLS)
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

        # MCP 独有的售后工单工具：本地 registry 里没有它
        tickets = json.loads(manager.execute_tool("query_after_sale_tickets", {"order_id": "ORD-20240118-004"}))
        assert tickets["success"] is True
        assert tickets["count"] >= 1
        assert tickets["tickets"][0]["status"] == "processing"
    finally:
        manager.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
