"""客服 Agent 结构化输出测试。

单元测试（默认）：注入 FakeLLM / FakeToolManager，不依赖真实 API。
集成测试（integration）：真实调用 LLM 验证意图识别。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.chat import EcomAgent  # noqa: E402
from app.schemas.response import CustomerServiceResponse, IntentType  # noqa: E402
from tests.fakes import FakeLLM, FakeToolManager  # noqa: E402

TEST_SESSION = str(ROOT / "app" / "sessions" / "test_agent_session.json")


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


# ---------- 单元测试（Mock） ----------

def test_structured_output_mock():
    """单轮对话返回合法 CustomerServiceResponse（mock）。"""
    agent = _make_agent(client=FakeLLM(parsed=CustomerServiceResponse(
        intent=IntentType.GREETING,
        confidence=0.95,
        reply="你好，很高兴为您服务！",
        requires_human=False,
    )))
    resp = agent.chat("你好呀")
    assert isinstance(resp, CustomerServiceResponse)
    assert resp.reply
    assert 0.0 <= resp.confidence <= 1.0
    assert isinstance(resp.requires_human, bool)
    assert resp.intent == IntentType.GREETING


def test_structured_output_fallback_empty_model():
    """模型结构化输出失败且返回空内容时，仍应得到安全兜底回复。"""
    agent = _make_agent(client=FakeLLM(reply=""))
    resp = agent.chat("随便问一句")
    assert isinstance(resp, CustomerServiceResponse)
    assert resp.reply  # 兜底后 reply 不为空
    assert resp.confidence == 0.0


def test_multi_turn_context_mock():
    """多轮对话历史正确累积（每轮 user + react assistant + structured assistant）。"""
    agent = _make_agent()
    agent.chat("我想退掉运动鞋")
    agent.chat("鞋底开胶了")
    # 每轮 3 条（user / react assistant / structured assistant），两轮共 6 条
    assert agent.history_size == 6


def test_reset_mock():
    agent = _make_agent()
    agent.chat("你好")
    assert agent.history_size > 0

    agent.reset()
    assert agent.history_size == 0
    assert agent.summary is None
    assert not Path(TEST_SESSION).exists()


def test_close_is_idempotent():
    agent = _make_agent()
    agent.chat("你好")
    agent.close()
    agent.close()  # 再次 close 不抛异常
    assert agent.tool_manager.closed is True


# ---------- 集成测试（真实 API） ----------

INTENT_CASES = [
    ("你好呀", {IntentType.GREETING}),
    ("我的订单 2024010112345 什么时候发货？", {IntentType.ORDER_QUERY, IntentType.AFTER_SALE}),
    ("这件衣服买了三天了想退货", {IntentType.RETURN_REQUEST, IntentType.AFTER_SALE}),
    ("你们这个商品质量也太差了吧！", {IntentType.COMPLAINT}),
    ("有没有什么优惠活动？", {IntentType.PROMOTION, IntentType.PRODUCT_CONSULT}),
    ("帮我推荐一款适合送女朋友的包包", {IntentType.PRODUCT_CONSULT}),
]


@pytest.mark.integration
def test_structured_output_live():
    """真实调用 LLM：验证结构化输出与意图识别。"""
    agent = EcomAgent(session_path=TEST_SESSION)

    for user_input, expected in INTENT_CASES:
        agent.reset()
        resp = agent.chat(user_input)
        assert isinstance(resp, CustomerServiceResponse)
        assert resp.reply
        assert 0.0 <= resp.confidence <= 1.0
        # 意图允许落在多个合理取值内（模型差异）
        assert resp.intent in expected, f"{user_input!r} 意图 {resp.intent} 不在 {expected}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])