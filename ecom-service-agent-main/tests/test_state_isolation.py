"""第四阶段：父图 / 子图状态隔离测试（全部 Mock，不依赖真实 API）。

验证：
- 子图入口 agent_enter 重置所有临时字段
- 多 Agent 切换时 tool_results 不交叉污染
- 多线程 / 多 thread 并发互不影响
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.multi_agent.state import empty_state  # noqa: E402
from tests.test_supervisor_graph import (  # noqa: E402
    FakeLLM,
    _build,
    _invoke,
)


def test_agent_enter_resets_subagent_temp_fields():
    """子图入口重置所有子图临时字段（旧请求状态不残留）。"""
    graph, fake, _ = _build("presale")
    fake.plan_respond = True

    st = empty_state()
    st["user_input"] = "推荐耳机"
    st["conversation"] = [{"role": "user", "content": "推荐耳机"}]
    st["metadata"] = {}
    # 注入脏状态（模拟上次请求遗留）
    st["tool_results"] = [{"name": "query_product", "arguments": {}, "result": "STALE", "ok": True, "error": "ok"}]
    st["pending_tool_calls"] = [{"name": "query_product", "arguments": {"keyword": "耳机"}}]
    st["iteration"] = 99
    st["confirmation_result"] = "confirmed"
    st["is_complete"] = True

    result = graph.invoke(st)
    assert result.get("tool_results") == []
    assert result.get("pending_tool_calls") == []
    assert result.get("confirmation_result") == ""
    assert result.get("iteration") == 1  # 从入口重置后重新计数


def test_multi_agent_switch_isolation():
    """presale → postsale 切换：postsale 不残留 presale 的 tool_results。"""
    g1, f1, _ = _build("presale")
    f1.plan_tools = [{"name": "query_product", "arguments": {"keyword": "耳机"}}]
    r1 = _invoke(g1, "推荐耳机")
    assert [tr["name"] for tr in r1.get("tool_results", [])] == ["query_product"]

    g2, f2, _ = _build("postsale")
    f2.plan_tools = [{"name": "query_order", "arguments": {"order_id": "ORD-1"}}]
    r2 = _invoke(g2, "查订单")

    names2 = [tr["name"] for tr in r2.get("tool_results", [])]
    assert "query_order" in names2
    assert "query_product" not in names2  # 不交叉污染


def test_concurrent_threads_independent():
    """两个线程并发运行不同子图，各自 tool_results 互不污染。"""
    results: dict[str, list[str]] = {}

    def run(key: str, agent_key: str, tool: dict):
        fake = FakeLLM()
        fake.plan_tools = [tool]
        graph, _, _ = _build(agent_key, fake=fake)
        result = _invoke(graph, f"test-{key}")
        results[key] = [tr["name"] for tr in result.get("tool_results", [])]

    t1 = threading.Thread(target=run, args=(
        "presale", "presale", {"name": "query_product", "arguments": {"keyword": "耳机"}},
    ))
    t2 = threading.Thread(target=run, args=(
        "postsale", "postsale", {"name": "query_order", "arguments": {"order_id": "ORD-1"}},
    ))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results["presale"] == ["query_product"]
    assert results["postsale"] == ["query_order"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])