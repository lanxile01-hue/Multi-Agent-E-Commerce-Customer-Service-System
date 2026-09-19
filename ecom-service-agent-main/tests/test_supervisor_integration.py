"""真实 SupervisorAgent 集成测试（Mock，不依赖真实 API Key）。

覆盖：
1. SupervisorAgent 初始化
2. chat 路由 presale / postsale / complaint
3. chat 返回 CustomerServiceResponse
4. 父图确实调用对应子图
5. tool_results 回填到最终状态
6. session 保存
7. reset 清理会话 + checkpoint
8. close 关闭所有 ToolManager
9. Guardrails 越界阻断（Router/ToolManager 不被调用）
10. checkpoint 持久化（sqlite 文件）
11. interrupt 跨进程恢复（重建 SupervisorAgent 后确认/取消）

用法：python tests/test_supervisor_integration.py
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from app.agent.storage import load_session  # noqa: E402
from app.multi_agent.checkpoint import CheckpointerHandle  # noqa: E402
from app.multi_agent.supervisor import SupervisorAgent  # noqa: E402
from app.schemas.response import CustomerServiceResponse, IntentType  # noqa: E402
from tests.test_supervisor_graph import (  # noqa: E402
    ALLOWED,
    FakeLLM,
    FakeToolManager,
    _flatten,
)


def _make_agent(
    client=None,
    *,
    backend: str = "memory",
    thread_id: str = "it-tid",
    tool_managers=None,
    guardrails_enabled: bool = False,
    checkpoint_enabled: bool = True,
    session_dir=None,
):
    """构建真实 SupervisorAgent，注入 Mock client / tool_managers / checkpointer。"""
    session_dir = session_dir or tempfile.mkdtemp()
    session_path = os.path.join(session_dir, "session.json")
    client = client or FakeLLM()
    if tool_managers is None:
        tool_managers = {key: FakeToolManager(ALLOWED[key]) for key in ALLOWED}

    db_path = None
    if checkpoint_enabled:
        if backend == "sqlite":
            from langgraph.checkpoint.sqlite import SqliteSaver

            db_path = os.path.join(session_dir, "checkpoints.db")
            conn = sqlite3.connect(db_path, check_same_thread=False)
            handle = CheckpointerHandle(saver=SqliteSaver(conn), kind="sqlite")
        else:
            handle = CheckpointerHandle(saver=MemorySaver(), kind="memory")
    else:
        handle = None

    agent = SupervisorAgent(
        session_path=session_path,
        client=client,
        tool_managers=tool_managers,
        checkpointer=handle,
        checkpoint_enabled=checkpoint_enabled,
        thread_id=thread_id,
        memory_enabled=False,
        skills_enabled=False,
        guardrails_enabled=guardrails_enabled,
    )
    return agent, session_dir, tool_managers, db_path


def _ok(msg): print(f"  [OK] {msg}")


# ---------- 1：初始化 ----------
def test_agent_init():
    print("\n[1] SupervisorAgent 初始化")
    agent, _, _, _ = _make_agent()
    assert len(agent.subgraphs) == 3
    assert agent.checkpointer is not None
    _ok("初始化成功，包含 3 个子图 + checkpointer")
    agent.close()


# ---------- 2/3/4/5：路由 + 结构化输出 + 子图调用 ----------
def test_chat_routing():
    print("\n[2] chat 路由 presale/postsale/complaint")
    print("[3] chat 返回 CustomerServiceResponse")
    print("[4] 父图确实调用对应子图")

    cases = [
        ("有什么耳机推荐吗", IntentType.PRODUCT_CONSULT, "presale", "售前咨询专家"),
        ("帮我查订单", IntentType.AFTER_SALE, "postsale", "售后服务专家"),
        ("我要投诉", IntentType.COMPLAINT, "complaint", "投诉处理专家"),
    ]
    for user_input, intent, agent_key, prompt_marker in cases:
        fake = FakeLLM(router_result="postsale", plan_respond=True)
        agent, _, _, _ = _make_agent(client=fake, thread_id=f"r-{agent_key}")
        resp = agent.chat(user_input, thread_id=f"r-{agent_key}")

        assert isinstance(resp, CustomerServiceResponse)
        assert resp.intent == intent, f"{agent_key} 意图错误: {resp.intent}"
        # 子图被调用：FakeLLM 捕获到对应子 Agent 的 system prompt
        all_text = "\n".join(_flatten(m) for m in fake.captured)
        assert prompt_marker in all_text, f"{agent_key} 未调用对应子图"
        _ok(f"{user_input} → {agent_key}（intent={resp.intent.value}）")
        agent.close()


# ---------- 5：tool_results 回填 ----------
def test_tool_results_backfill():
    print("\n[5] tool_results 回填到最终状态")
    fake = FakeLLM()
    fake.plan_tools = [{"name": "query_product", "arguments": {"keyword": "耳机"}}]
    agent, _, tms, _ = _make_agent(client=fake, thread_id="t-backfill")
    agent.chat("推荐耳机", thread_id="t-backfill")

    # 工具调用轨迹回填到 raw_messages（tool 消息 + assistant tool_calls）
    roles = [m.get("role") for m in agent.raw_messages]
    assert "tool" in roles, "raw_messages 缺少 tool 消息"
    assert any(m.get("tool_calls") for m in agent.raw_messages), "raw_messages 缺少 tool_calls"
    assert tms["presale"].calls, "presale 工具未被调用"
    _ok("tool_results 已回填到 raw_messages")
    agent.close()


# ---------- 6：session 保存 ----------
def test_session_save():
    print("\n[6] session 能保存")
    fake = FakeLLM(plan_respond=True)
    agent, session_dir, _, _ = _make_agent(client=fake, thread_id="t-save")
    agent.chat("你好", thread_id="t-save")
    session_path = os.path.join(session_dir, "session.json")
    assert os.path.exists(session_path), "session.json 未创建"
    loaded = load_session(session_path)
    assert loaded and len(loaded["messages"]) >= 2
    _ok("session.json 已保存")
    agent.close()


# ---------- 7/8：reset + close ----------
def test_reset_and_close():
    print("\n[7] reset 清理会话")
    print("[8] close 关闭所有 ToolManager")
    fake = FakeLLM(plan_respond=True)
    agent, session_dir, tms, _ = _make_agent(client=fake, thread_id="t-reset")
    agent.chat("你好", thread_id="t-reset")
    assert agent.history_size > 0

    agent.reset()
    assert agent.history_size == 0
    assert not os.path.exists(os.path.join(session_dir, "session.json"))

    agent.close()
    assert all(tm.closed for tm in tms.values()), "存在未关闭的 ToolManager"
    _ok("reset 清空会话 + close 关闭全部 ToolManager")


# ---------- 9：Guardrails 越界阻断 ----------
def test_guardrails_block():
    print("\n[9] Guardrails 越界阻断（Router/ToolManager 不被调用）")
    fake = FakeLLM(boundary_in_scope=False, plan_tools=[{"name": "query_order", "arguments": {"order_id": "ORD-1"}}])
    agent, _, tms, _ = _make_agent(client=fake, thread_id="t-guard", guardrails_enabled=True)
    resp = agent.chat("帮我写一篇论文", thread_id="t-guard")

    assert resp.requires_human is True, "越界请求应转人工"
    all_text = "\n".join(_flatten(m) for m in fake.captured)
    assert "意图分类器" not in all_text, "Router 不应被调用"
    assert "任务规划器" not in all_text, "子图不应被调用"
    assert all(len(tm.calls) == 0 for tm in tms.values()), "ToolManager 不应被调用"
    _ok("越界请求被阻断，未进入 Router / 子图 / 工具")
    agent.close()


# ---------- 10/11：checkpoint 持久化 + 跨进程恢复 ----------
def test_checkpoint_persistence_and_cross_process():
    print("\n[10] checkpoint sqlite 文件创建 + 同进程恢复")
    print("[11] interrupt 跨进程恢复（重建 SupervisorAgent）")

    # 用 sqlite 后端，使 checkpoint 真正落盘
    fake1 = FakeLLM(router_result="postsale",
                    plan_tools=[{"name": "apply_refund", "arguments": {"order_id": "ORD-20240115-001", "reason": "尺码不合适"}}])
    agent1, session_dir, tms1, db_path = _make_agent(client=fake1, backend="sqlite", thread_id="t-cross")
    agent1.chat("我要退款订单 ORD-20240115-001", thread_id="t-cross")

    assert db_path and os.path.exists(db_path), "sqlite checkpoint 文件未创建"

    # 暂停：get_state().next 非空
    snap = agent1.get_state("t-cross")
    assert snap is not None and snap.next, "退款流程应处于 interrupt 暂停状态"
    _ok("sqlite checkpoint 文件已创建，退款流程已暂停")

    # 模拟进程重启：关闭 agent1，用同一 sqlite 文件 + 同一 thread_id 重建 agent2
    agent1.close()

    # 进程重建后「确认」→ 恢复执行退款
    agent2, _, tms2, _ = _make_agent(client=FakeLLM(router_result="postsale"),
                                     backend="sqlite", thread_id="t-cross",
                                     session_dir=session_dir)
    resp = agent2.chat("确认", thread_id="t-cross")
    assert "apply_refund" in [c[0] for c in tms2["postsale"].calls], "跨进程确认后未执行退款"
    _ok("跨进程确认后继续执行退款")
    agent2.close()

    # 进程重建后「取消」→ 不执行退款
    fake3 = FakeLLM(router_result="postsale",
                    plan_tools=[{"name": "apply_refund", "arguments": {"order_id": "ORD-20240115-001", "reason": "尺码不合适"}}])
    agent3, _, tms3, _ = _make_agent(client=fake3, backend="sqlite", thread_id="t-cross-cancel")
    agent3.chat("我要退款订单 ORD-20240115-001", thread_id="t-cross-cancel")
    agent3.close()

    agent4, _, tms4, _ = _make_agent(client=FakeLLM(router_result="postsale"),
                                     backend="sqlite", thread_id="t-cross-cancel",
                                     session_dir=session_dir)
    agent4.chat("取消", thread_id="t-cross-cancel")
    assert "apply_refund" not in [c[0] for c in tms4["postsale"].calls], "取消后不应执行退款"
    _ok("跨进程取消后不执行退款")
    agent4.close()

    # 模糊文本 → 不执行退款（normalize_confirmation 默认 re_explain，不会直接确认）
    fake5 = FakeLLM(router_result="postsale",
                    plan_tools=[{"name": "apply_refund", "arguments": {"order_id": "ORD-20240115-001", "reason": "尺码不合适"}}])
    agent5, _, tms5, _ = _make_agent(client=fake5, backend="sqlite", thread_id="t-cross-vague")
    agent5.chat("我要退款订单 ORD-20240115-001", thread_id="t-cross-vague")
    agent5.close()

    agent6, _, tms6, _ = _make_agent(client=FakeLLM(router_result="postsale"),
                                     backend="sqlite", thread_id="t-cross-vague",
                                     session_dir=session_dir)
    agent6.chat("随便说说", thread_id="t-cross-vague")
    assert "apply_refund" not in [c[0] for c in tms6["postsale"].calls], "模糊输入不应执行退款"
    _ok("模糊文本不执行退款")
    agent6.close()


# ---------- 12：checkpoint disabled ----------
def test_checkpoint_disabled():
    print("\n[12] checkpoint disabled 时图可运行，但不支持 interrupt")
    fake = FakeLLM(router_result="postsale",
                   plan_tools=[{"name": "apply_refund", "arguments": {"order_id": "ORD-20240115-001", "reason": "尺码不合适"}}])
    agent, _, tms, _ = _make_agent(client=fake, thread_id="t-disabled", checkpoint_enabled=False)
    assert agent.checkpointer is None
    assert agent.human_confirmation_enabled is False

    resp = agent.chat("我要退款订单 ORD-20240115-001", thread_id="t-disabled")
    assert isinstance(resp, CustomerServiceResponse)
    # 无 checkpointer → 不做 interrupt，直接执行退款
    assert "apply_refund" in [c[0] for c in tms["postsale"].calls], "无 checkpoint 时应直接执行退款"
    assert agent.get_state("t-disabled") is None
    _ok("checkpoint 关闭时图正常执行，无 interrupt 恢复能力")
    agent.close()


# ---------- 13：reset 清理 checkpoint ----------
def test_reset_clears_checkpoint():
    print("\n[13] reset 清理 checkpoint（不恢复历史 interrupt/工具调用）")
    fake = FakeLLM(router_result="postsale",
                   plan_tools=[{"name": "apply_refund", "arguments": {"order_id": "ORD-20240115-001", "reason": "尺码不合适"}}])
    agent, session_dir, tms, db_path = _make_agent(client=fake, backend="sqlite", thread_id="t-reset-checkpoint")
    agent.chat("我要退款订单 ORD-20240115-001", thread_id="t-reset-checkpoint")
    snap = agent.get_state("t-reset-checkpoint")
    assert snap is not None and snap.next, "退款流程应处于暂停状态"

    agent.reset()
    snap2 = agent.get_state("t-reset-checkpoint")
    assert snap2 is None or not snap2.next, "reset 后不应恢复 interrupt 状态"
    agent.close()

    # 重建 agent（同一 sqlite 文件 + 同一 thread_id），不应恢复旧 interrupt/工具调用
    agent2, _, tms2, _ = _make_agent(client=FakeLLM(router_result="presale", plan_respond=True),
                                     backend="sqlite", thread_id="t-reset-checkpoint",
                                     session_dir=session_dir)
    resp = agent2.chat("推荐耳机", thread_id="t-reset-checkpoint")
    assert not tms2["postsale"].calls, "reset 后不应执行历史退款工具"
    assert isinstance(resp, CustomerServiceResponse)
    _ok("reset 后不恢复历史 interrupt 与工具调用")
    agent2.close()


def main():
    print("=" * 60)
    print("  SupervisorAgent 集成测试 · Mock")
    print("=" * 60)
    test_agent_init()
    test_chat_routing()
    test_tool_results_backfill()
    test_session_save()
    test_reset_and_close()
    test_guardrails_block()
    test_checkpoint_persistence_and_cross_process()
    test_checkpoint_disabled()
    test_reset_clears_checkpoint()
    print("\n" + "=" * 60)
    print("  ALL INTEGRATION TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()