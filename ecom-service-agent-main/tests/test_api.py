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


def test_knowledge_catalog(tmp_path, monkeypatch):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "制度.md").write_text(
        "---\n"
        "title: 报销制度\nversion: 1.0\ndepartment: finance\n---\n"
        "# 报销制度\n\n## 申请\n需要提交发票。\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("app.api.routes.settings.kb_dir", str(knowledge_dir))
    agent, h = _handler()
    status, body = h.knowledge_catalog()
    assert status == 200
    assert body["total_documents"] == 1
    assert body["documents"][0]["title"] == "报销制度"
    assert body["documents"][0]["department"] == "finance"
    agent.close()


def test_citation_projection_does_not_leak_full_evidence():
    raw = json.dumps({
        "reliable": True,
        "evidence": [{
            "evidence_id": "doc#00-00",
            "title": "报销制度",
            "source": "finance/expense-policy.md",
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
        "title": "报销制度",
        "source": "finance/expense-policy.md",
        "section": "适用范围",
        "version": "1.0",
        "score": 0.88,
    }]


def test_upsert_knowledge_document_starts_index_task(tmp_path, monkeypatch):
    knowledge_dir = tmp_path / "knowledge"
    monkeypatch.setattr("app.api.routes.settings.kb_dir", str(knowledge_dir))

    class _Task:
        task_id = "task-1"
        status = "pending"

    monkeypatch.setattr("app.api.routes.index_manager.submit", lambda: _Task())
    agent, h = _handler()
    status, body = h.upsert_knowledge_document(
        "expense.md",
        "# 报销制度\n\n## 范围\n适用于员工。",
        {"title": "报销制度", "department": "finance", "tags": ["报销", "财务"]},
    )
    assert status == 202
    assert body["task_id"] == "task-1"
    saved = (knowledge_dir / "expense.md").read_text(encoding="utf-8")
    assert saved.startswith("---\n")
    assert "department: finance" in saved
    agent.close()


def test_document_version_and_recoverable_delete(tmp_path, monkeypatch):
    knowledge_dir = tmp_path / "knowledge"
    monkeypatch.setattr("app.api.routes.settings.kb_dir", str(knowledge_dir))

    class _Task:
        task_id = "task-version"
        status = "pending"

    monkeypatch.setattr("app.api.routes.index_manager.submit", lambda: _Task())
    agent, h = _handler()
    h.upsert_knowledge_document("policy.md", "# v1", {"version": "1.0"})
    h.upsert_knowledge_document("policy.md", "# v2", {"version": "2.0"})
    status, body = h.list_document_versions("policy.md")
    assert status == 200 and "1.0" in body["versions"]
    status, body = h.rollback_document("policy.md", "1.0")
    assert status == 202 and body["version"] == "1.0"
    restored = (knowledge_dir / "policy.md").read_text(encoding="utf-8")
    assert "version: 1.0" in restored and "# v1" in restored
    status, body = h.delete_document("policy.md")
    assert status == 202 and not (knowledge_dir / "policy.md").exists()
    assert Path(body["recoverable_path"]).exists()
    agent.close()


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
        assert "企业知识库 Agent 控制台" in body
        conn.close()
    finally:
        server.shutdown()
        server.server_close()
        agent.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
