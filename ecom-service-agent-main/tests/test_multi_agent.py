"""Multi-Agent（旧版 Orchestrator）测试。

单元测试（默认）：工具隔离、Router 关键词预检、Mock 下 chat 接线。
集成测试（integration）：真实 API 验证路由准确性与回复质量。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.manager import ToolManager  # noqa: E402
from app.multi_agent.agents import AGENT_CONFIGS  # noqa: E402
from app.multi_agent.orchestrator import MultiAgentOrchestrator  # noqa: E402
from app.multi_agent.router import Router  # noqa: E402
from app.schemas.response import CustomerServiceResponse, IntentType  # noqa: E402
from tests.fakes import FakeLLM, FakeToolManager  # noqa: E402

TEST_SESSION = str(ROOT / "app" / "sessions" / "test_multi_agent_session.json")


@pytest.fixture(autouse=True)
def _clean():
    Path(TEST_SESSION).unlink(missing_ok=True)
    yield
    Path(TEST_SESSION).unlink(missing_ok=True)


def _orchestrator(**kwargs) -> MultiAgentOrchestrator:
    kwargs.setdefault("client", FakeLLM(parsed=CustomerServiceResponse(
        intent=IntentType.PRODUCT_CONSULT, confidence=0.9,
        reply="推荐这款耳机。", requires_human=False,
    )))
    kwargs.setdefault("memory_enabled", False)
    kwargs.setdefault("skills_enabled", False)
    return MultiAgentOrchestrator(session_path=TEST_SESSION, **kwargs)


# ---------- 单元测试（Mock / 无需 API） ----------

def test_tool_isolation():
    presale = AGENT_CONFIGS["presale"]["tools"]
    postsale = AGENT_CONFIGS["postsale"]["tools"]
    complaint = AGENT_CONFIGS["complaint"]["tools"]

    assert "query_product" in presale and "apply_refund" not in presale
    assert "apply_refund" in postsale and "query_order" in postsale
    assert "apply_refund" not in complaint and "query_order" in complaint


def test_tool_isolation_via_orchestrator():
    tms = {
        key: ToolManager(use_mcp=False, allowed_tools=cfg["tools"])
        for key, cfg in AGENT_CONFIGS.items()
    }
    orch = _orchestrator(tool_managers=tms)

    presale = {d["function"]["name"] for d in orch.agents["presale"].tool_manager.tool_definitions}
    postsale = {d["function"]["name"] for d in orch.agents["postsale"].tool_manager.tool_definitions}
    complaint = {d["function"]["name"] for d in orch.agents["complaint"].tool_manager.tool_definitions}

    assert "query_product" in presale and "apply_refund" not in presale
    assert "apply_refund" in postsale
    assert "query_product" not in complaint and "query_order" in complaint


def test_router_keyword_precheck():
    router = Router(client=FakeLLM(reply="postsale"), model="mock")
    assert router.route("我要投诉！商品太差了") == "complaint"
    assert router.route("有什么耳机推荐吗") == "presale"
    assert router.route("帮我查订单") == "postsale"


def test_chat_mock():
    orch = _orchestrator()
    resp = orch.chat("有什么耳机推荐吗")
    assert isinstance(resp, CustomerServiceResponse)
    assert resp.reply


# ---------- 集成测试（真实 API） ----------

@pytest.mark.integration
def test_routing_live():
    orch = _orchestrator(client=None)
    router = orch.router
    cases = [
        ("有什么耳机推荐吗", "presale"),
        ("帮我查一下订单 ORD-20240115-001", "postsale"),
        ("我要投诉！你们的商品质量太差了", "complaint"),
    ]
    for user_input, expected in cases:
        assert router.route(user_input) == expected


@pytest.mark.integration
def test_complaint_live():
    orch = _orchestrator(client=None)
    resp = orch.chat("我要投诉！订单 ORD-20240110-003 的商品质量太差了！")
    assert isinstance(resp, CustomerServiceResponse)
    assert resp.reply


if __name__ == "__main__":
    pytest.main([__file__, "-v"])