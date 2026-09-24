"""第三阶段：高风险业务操作安全与幂等测试（全部 Mock，不依赖真实 API）。

覆盖：
- 首次确认执行退款
- 重复确认不重复退款
- 取消不执行退款
- 模糊输入不执行退款
- 退款参数错误不执行
- 订单不存在不执行
- 退款超时不重试（返回状态未知）
- 同一个幂等键重复调用只执行一次
- 服务重启后重复确认不重复执行
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.manager import ToolManager  # noqa: E402
from tools.registry import TOOL_DEFINITIONS  # noqa: E402
from tools.errors import validate_tool_arguments  # noqa: E402
from tools.refund import (  # noqa: E402
    apply_refund,
    clear_refund_idempotency,
    get_refund_audit_log,
)
from app.multi_agent.subgraphs import build_postsale_graph  # noqa: E402
from tests.test_supervisor_graph import (  # noqa: E402
    ALLOWED,
    FakeLLM,
    FakeToolManager,
    _invoke,
)
from tests.test_supervisor_integration import _make_agent  # noqa: E402


def _refund_plan() -> list[dict]:
    return [{
        "name": "apply_refund",
        "arguments": {"order_id": "ORD-20240115-001", "reason": "尺码不合适"},
    }]


def _refund_tool_managers() -> dict:
    """postsale 使用真实 ToolManager（执行真实 apply_refund），其余用 Fake。"""
    return {
        "presale": FakeToolManager(ALLOWED["presale"]),
        "postsale": ToolManager(use_mcp=False, allowed_tools=ALLOWED["postsale"]),
        "complaint": FakeToolManager(ALLOWED["complaint"]),
    }


@pytest.fixture(autouse=True)
def _clean_refund_state():
    clear_refund_idempotency()
    yield
    clear_refund_idempotency()


# ---------- 工具函数级（直接测 apply_refund） ----------

def test_refund_idempotency_same_key():
    r1 = apply_refund("ORD-20240115-001", "尺码不合适", idempotency_key="k-dup")
    assert r1["status"] == "success"

    r2 = apply_refund("ORD-20240115-001", "尺码不合适", idempotency_key="k-dup")
    assert r2.get("duplicate") is True
    assert r2["status"] == "success"
    assert len(get_refund_audit_log()) == 1  # 只真正执行一次


def test_refund_order_not_found():
    r = apply_refund("ORD-99999-000", "不想要了", idempotency_key="k-notfound")
    assert r["status"] == "error"
    assert r["error_type"] == "not_found"
    assert len(get_refund_audit_log()) == 0  # 未执行，无审计


def test_refund_eligibility_refund_processing():
    # ORD-20240118-004 已是 refund_processing
    r = apply_refund("ORD-20240118-004", "不想要了", idempotency_key="k-proc")
    assert r["status"] == "error"
    assert r["error_type"] == "forbidden"
    assert len(get_refund_audit_log()) == 0


def test_refund_param_error_no_execution():
    ok, err = validate_tool_arguments(
        "apply_refund", {"order_id": "ORD-20240115-001"}, TOOL_DEFINITIONS,
    )
    assert not ok
    assert "reason" in err


# ---------- 图流程级（SupervisorAgent + 真实 apply_refund） ----------

def test_first_confirm_executes_refund():
    fake = FakeLLM(router_result="postsale", plan_tools=_refund_plan())
    agent, _, _, _ = _make_agent(
        client=fake, thread_id="t-first", tool_managers=_refund_tool_managers(),
    )
    agent.chat("我要退款订单 ORD-20240115-001", thread_id="t-first")
    assert agent.get_state("t-first").next  # 已 interrupt

    agent.resume("t-first", "确认")
    assert len(get_refund_audit_log()) == 1
    agent.close()


def test_duplicate_confirm_no_double_refund():
    fake = FakeLLM(router_result="postsale", plan_tools=_refund_plan())
    agent, _, _, _ = _make_agent(
        client=fake, thread_id="t-dup", tool_managers=_refund_tool_managers(),
    )
    agent.chat("我要退款订单 ORD-20240115-001", thread_id="t-dup")
    agent.resume("t-dup", "确认")
    assert len(get_refund_audit_log()) == 1

    fake.plan_respond = True  # 后续"确认"不再规划退款
    agent.chat("确认", thread_id="t-dup")
    assert len(get_refund_audit_log()) == 1
    agent.close()


def test_cancel_no_refund():
    fake = FakeLLM(router_result="postsale", plan_tools=_refund_plan())
    agent, _, _, _ = _make_agent(
        client=fake, thread_id="t-cancel", tool_managers=_refund_tool_managers(),
    )
    agent.chat("我要退款订单 ORD-20240115-001", thread_id="t-cancel")
    agent.resume("t-cancel", "取消")
    assert len(get_refund_audit_log()) == 0
    agent.close()


def test_vague_input_no_refund():
    fake = FakeLLM(router_result="postsale", plan_tools=_refund_plan())
    agent, _, _, _ = _make_agent(
        client=fake, thread_id="t-vague", tool_managers=_refund_tool_managers(),
    )
    agent.chat("我要退款订单 ORD-20240115-001", thread_id="t-vague")
    fake.plan_respond = True
    agent.resume("t-vague", "随便说说")  # 模糊 → re_explain，不执行退款
    assert len(get_refund_audit_log()) == 0
    agent.close()


def test_restart_duplicate_confirm_no_reexecution():
    fake1 = FakeLLM(router_result="postsale", plan_tools=_refund_plan())
    agent1, session_dir, _, _ = _make_agent(
        client=fake1, backend="sqlite", thread_id="t-restart",
        tool_managers=_refund_tool_managers(),
    )
    agent1.chat("我要退款订单 ORD-20240115-001", thread_id="t-restart")
    assert agent1.get_state("t-restart").next
    agent1.close()  # 模拟重启

    fake2 = FakeLLM(router_result="postsale", plan_tools=_refund_plan())
    agent2, _, _, _ = _make_agent(
        client=fake2, backend="sqlite", thread_id="t-restart",
        session_dir=session_dir, tool_managers=_refund_tool_managers(),
    )
    agent2.resume("t-restart", "确认")  # 跨进程恢复，执行退款一次
    assert len(get_refund_audit_log()) == 1

    fake2.plan_respond = True
    agent2.chat("确认", thread_id="t-restart")  # 再次确认不重复执行
    assert len(get_refund_audit_log()) == 1
    agent2.close()


# ---------- 子图层（超时 → 状态未知，不重试） ----------

class _BlockingToolManager(FakeToolManager):
    def execute_tool(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        time.sleep(0.5)
        return json.dumps({"success": True}, ensure_ascii=False)


def test_refund_timeout_unknown_no_retry():
    fake = FakeLLM()
    fake.plan_tools = _refund_plan()
    tm = _BlockingToolManager(ALLOWED["postsale"])
    graph = build_postsale_graph(
        tool_manager=tm, client=fake, model="mock", temperature=0.0,
        max_steps=5, human_confirmation_enabled=False, tool_timeout=0.05,
    )
    result = _invoke(graph, "我要退款")

    tr = result.get("tool_results", [])
    assert tr
    payload = json.loads(tr[0]["result"])
    assert payload["status"] == "unknown"  # 状态未知，而不是 success/error
    assert "人工核验" in payload["message"]
    # 非幂等写操作不自动重试：apply_refund 只被调用一次
    assert len([c for c in tm.calls if c[0] == "apply_refund"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])