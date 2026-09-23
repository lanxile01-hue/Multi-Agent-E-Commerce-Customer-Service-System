"""第七阶段：服务化 API 测试（Mock SupervisorAgent，不依赖真实 API）。"""

from __future__ import annotations

import http.client
import json
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.api.routes import APIHandler, RateLimiter  # noqa: E402
from app.api.server import create_server  # noqa: E402
from app.multi_agent.supervisor import SupervisorAgent  # noqa: E402
from tests.test_supervisor_graph import FakeLLM  # noqa: E402
from tests.test_supervisor_integration import _make_agent  # noqa: E402


def _handler(thread_id="t-api"):
    fake = FakeLLM(plan_respond=True)
    agent, _, _, _ = _make_agent(client=fake, thread_id=thread_id)
    return agent, APIHandler(agent)


def test_api_handler_chat():
    agent, h = _handler()
    status, body = h.chat("你好", "t-api", "client-1")
    assert status == 200
    assert "reply" in body and "intent" in body and "confidence" in body
    # 不泄漏内部思考 / 工具参数
    assert "tool_calls" not in body
    assert "tool_results" not in body
    agent.close()


def test_api_state_sanitized():
    agent, h = _handler()
    status, body = h.state("t-api")
    assert status == 200
    assert "interrupted" in body
    # 不暴露原始 checkpoint 内部状态
    assert "values" not in body
    assert "tasks" not in body
    assert "writes" not in body
    agent.close()


def test_api_reset():
    agent, h = _handler()
    status, body = h.reset("t-api")
    assert status == 200 and body["ok"] is True
    agent.close()


def test_rate_limiter():
    rl = RateLimiter(max_requests=2, window_seconds=60)
    assert rl.allow("k") is True
    assert rl.allow("k") is True
    assert rl.allow("k") is False  # 超过限制


def test_citation_projection_does_not_leak_full_evidence():
    raw = json.dumps({
        "reliable": True,
        "evidence": [{
            "evidence_id": "doc#00-00",
            "title": "退换货政策",
            "source": "knowledge/退换货政策.md",
            "section": "适用范围",
            "version": "1.0",
            "score": 0.88,
            "text": "不应透传的大段原文",
        }],
    }, ensure_ascii=False)
    citations = SupervisorAgent._extract_citations([
        {"name": "search_knowledge", "result": raw},
    ])
    assert citations == [{
        "evidence_id": "doc#00-00",
        "title": "退换货政策",
        "source": "knowledge/退换货政策.md",
        "section": "适用范围",
        "version": "1.0",
        "score": 0.88,
    }]


def test_server_health_smoke():
    """真实启动 HTTP 服务，验证 /health 入口可访问。"""
    fake = FakeLLM(plan_respond=True)
    agent, _, _, _ = _make_agent(client=fake, thread_id="t-smoke")
    server = create_server(agent, host="127.0.0.1", port=0)  # port=0 随机端口
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        body = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 200
        assert body["status"] == "ok"
        conn.close()
    finally:
        server.shutdown()
        server.server_close()
        agent.close()


def test_server_web_console_smoke():
    fake = FakeLLM(plan_respond=True)
    agent, _, _, _ = _make_agent(client=fake, thread_id="t-web")
    server = create_server(agent, host="127.0.0.1", port=0)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/")
        resp = conn.getresponse()
        body = resp.read().decode("utf-8")
        assert resp.status == 200
        assert "电商客服 Agent 控制台" in body
        conn.close()
    finally:
        server.shutdown()
        server.server_close()
        agent.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
