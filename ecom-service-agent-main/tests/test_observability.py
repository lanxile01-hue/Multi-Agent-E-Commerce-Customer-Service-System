"""第六阶段：错误处理与可观测性测试（全部 Mock，不依赖真实 API）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.observability import mask_sensitive, sanitize_record  # noqa: E402
from tools.errors import AgentError  # noqa: E402
from tests.test_supervisor_graph import FakeLLM  # noqa: E402
from tests.test_supervisor_integration import _make_agent  # noqa: E402


def test_mask_sensitive():
    text = "订单 ORD-20240115-001 已发货，运单号 SF1234567890，收货人电话 13812345678"
    masked = mask_sensitive(text)
    assert "ORD-20240115-001" not in masked
    assert "SF1234567890" not in masked
    assert "13812345678" not in masked
    assert "ORD-***" in masked
    assert "138****5678" in masked


def test_sanitize_record_removes_api_key():
    record = {
        "api_key": "sk-secret",
        "OPENAI_API_KEY": "sk-secret2",
        "message": "订单 ORD-20240115-001",
    }
    safe = sanitize_record(record)
    assert "api_key" not in safe
    assert "OPENAI_API_KEY" not in safe
    assert "ORD-20240115-001" not in safe["message"]


def test_agent_error_fields():
    err = AgentError("内部错误", error_type="fatal", retryable=False, user_message="抱歉，请稍后再试")
    assert err.error_type == "fatal"
    assert err.retryable is False
    assert err.user_message == "抱歉，请稍后再试"
    assert isinstance(err, Exception)


def test_supervisor_last_trace_fields():
    fake = FakeLLM(plan_respond=True)
    agent, _, _, _ = _make_agent(client=fake, thread_id="t-trace")
    agent.chat("你好", thread_id="t-trace")

    trace = agent.last_trace
    assert trace
    for key in (
        "trace_id", "thread_id", "user_id", "selected_agent", "node_name",
        "model_name", "model_calls", "tool_calls", "retry_count",
        "total_latency", "requires_human", "checkpoint_backend",
    ):
        assert key in trace, f"缺少字段 {key}"
    # 不记录密钥字段
    assert "api_key" not in trace
    agent.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])