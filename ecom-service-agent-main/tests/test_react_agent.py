"""ReAct Agent + Function Calling 测试。

单元测试（默认）：Mock LLM / ToolManager，验证结构与工具执行接线。
集成测试（integration）：真实 API 验证模型主动调用工具与回复质量。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.chat import EcomAgent  # noqa: E402
from app.schemas.response import CustomerServiceResponse, IntentType  # noqa: E402
from tests.fakes import FakeLLM, FakeToolManager  # noqa: E402

TEST_SESSION = str(ROOT / "app" / "sessions" / "test_react_session.json")


def _make_agent(**kwargs) -> EcomAgent:
    kwargs.setdefault("client", FakeLLM())
    kwargs.setdefault("tool_manager", FakeToolManager())
    kwargs.setdefault("memory_enabled", False)
    kwargs.setdefault("skills_enabled", False)
    return EcomAgent(session_path=TEST_SESSION, **kwargs)


@pytest.fixture(autouse=True)
def _clean_session():
    Path(TEST_SESSION).unlink(missing_ok=True)
    yield
    Path(TEST_SESSION).unlink(missing_ok=True)


def _assert_structured(resp: CustomerServiceResponse):
    assert isinstance(resp, CustomerServiceResponse)
    assert isinstance(resp.intent, IntentType)
    assert 0.0 <= resp.confidence <= 1.0
    assert resp.reply


# ---------- 单元测试（Mock） ----------

def test_greeting_no_tool():
    agent = _make_agent(client=FakeLLM(parsed=CustomerServiceResponse(
        intent=IntentType.GREETING, confidence=0.9, reply="您好！", requires_human=False,
    )))
    resp = agent.chat("你好呀~")
    _assert_structured(resp)
    assert agent.tool_manager.calls == []  # 问候不应触发工具


def test_order_query_tool_execution():
    """本地 ToolManager 执行订单查询工具（mock 数据，无 API）。"""
    from tools.manager import ToolManager

    manager = ToolManager(use_mcp=False)
    try:
        result = json.loads(manager.execute_tool(
            "query_order", {"order_id": "ORD-20240115-001"},
        ))
        assert result["success"] is True
        assert result["order"]["status"] == "shipped"
    finally:
        manager.close()


def test_unknown_order_graceful():
    """未知订单返回明确错误，而不是崩溃。"""
    from tools.manager import ToolManager

    manager = ToolManager(use_mcp=False)
    try:
        result = json.loads(manager.execute_tool(
            "query_order", {"order_id": "ORD-99999-000"},
        ))
        assert result["success"] is False
        assert "未找到" in result["error"]
    finally:
        manager.close()


def test_persistence_with_tools():
    """会话持久化：带工具结果的消息也能保存与恢复。"""
    agent = _make_agent()
    agent.chat("帮我查一下订单 ORD-20240120-002")
    assert agent.history_size > 0

    agent.save()
    agent2 = _make_agent()
    assert agent2.history_size == agent.history_size


# ---------- 集成测试（真实 API） ----------

@pytest.mark.integration
def test_react_order_query_live():
    agent = EcomAgent(session_path=TEST_SESSION)
    resp = agent.chat("帮我查一下订单 ORD-20240115-001 的状态")
    _assert_structured(resp)
    keywords = ["发货", "shipped", "Nike", "运动鞋", "899", "顺丰"]
    assert any(kw in resp.reply for kw in keywords)


@pytest.mark.integration
def test_react_unknown_order_live():
    agent = EcomAgent(session_path=TEST_SESSION)
    resp = agent.chat("帮我查订单 ORD-99999-000")
    _assert_structured(resp)
    keywords = ["未找到", "找不到", "不存在", "核实", "确认"]
    assert any(kw in resp.reply for kw in keywords)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])