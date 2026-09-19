"""Checkpoint / Session / 并发安全 测试（全部 Mock，不依赖真实 API）。

覆盖第二阶段要求：
- MemorySaver 同进程恢复
- SQLite 跨进程恢复
- 自定义 thread_id reset
- 多个 thread 互不影响
- 同一个 thread 并发请求（SessionBusyError）
- reset 后不恢复旧 interrupt
- 重启后不重复执行旧工具
- checkpoint 文件损坏时安全降级
- SQLite 连接关闭后不影响普通 session 保存
"""

from __future__ import annotations

import json
import os
import threading
import time
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.storage import load_session, save_session  # noqa: E402
from app.multi_agent.checkpoint import build_checkpointer  # noqa: E402
from app.multi_agent.supervisor import SessionBusyError  # noqa: E402
from app.schemas.response import CustomerServiceResponse  # noqa: E402
from tests.test_supervisor_graph import ALLOWED, FakeLLM, FakeToolManager  # noqa: E402
from tests.test_supervisor_integration import _make_agent  # noqa: E402


def _refund_plan() -> list[dict]:
    return [{
        "name": "apply_refund",
        "arguments": {"order_id": "ORD-20240115-001", "reason": "尺码不合适"},
    }]


# ---------- 1. MemorySaver 同进程恢复 ----------

def test_memory_saver_same_process_recovery():
    fake = FakeLLM(router_result="postsale", plan_tools=_refund_plan())
    agent, _, tms, _ = _make_agent(client=fake, thread_id="t-mem")

    agent.chat("我要退款订单 ORD-20240115-001", thread_id="t-mem")
    assert agent.get_state("t-mem").next  # 已 interrupt 暂停

    agent.resume("t-mem", "确认")
    assert "apply_refund" in [c[0] for c in tms["postsale"].calls]
    agent.close()


# ---------- 2. SQLite 跨进程恢复 ----------

def test_sqlite_cross_process_recovery():
    fake1 = FakeLLM(router_result="postsale", plan_tools=_refund_plan())
    agent1, session_dir, _, _ = _make_agent(client=fake1, backend="sqlite", thread_id="t-x")
    agent1.chat("我要退款订单 ORD-20240115-001", thread_id="t-x")
    assert agent1.get_state("t-x").next  # 暂停
    agent1.close()  # 模拟进程退出

    # 进程重建：同一 sqlite 文件 + 同一 thread_id
    agent2, _, tms2, _ = _make_agent(
        client=FakeLLM(router_result="postsale"),
        backend="sqlite",
        thread_id="t-x",
        session_dir=session_dir,
    )
    agent2.resume("t-x", "确认")
    assert "apply_refund" in [c[0] for c in tms2["postsale"].calls]
    agent2.close()


# ---------- 3. 自定义 thread_id reset ----------

def test_custom_thread_id_reset():
    fake = FakeLLM(router_result="postsale", plan_tools=_refund_plan())
    agent, _, _, _ = _make_agent(client=fake, thread_id="t-primary")

    agent.chat("我要退款订单 ORD-20240115-001", thread_id="custom-thread")
    assert agent.get_state("custom-thread").next  # 自定义 thread 已 interrupt

    agent.reset(thread_id="custom-thread")
    snap = agent.get_state("custom-thread")
    assert snap.values == {} and not snap.next  # checkpoint + interrupt 已清理
    agent.close()


# ---------- 4. 多个 thread 互不影响 ----------

def test_multiple_threads_independent():
    fake = FakeLLM(router_result="postsale")
    fake.plan_tools = _refund_plan()
    agent, _, _, _ = _make_agent(client=fake, thread_id="t-base")

    # t1 触发退款 interrupt
    agent.chat("我要退款订单 ORD-20240115-001", thread_id="t1")
    assert agent.get_state("t1").next

    # t2 正常完成，不受 t1 影响
    fake.plan_tools = []
    fake.plan_respond = True
    resp = agent.chat("你好", thread_id="t2")
    assert isinstance(resp, CustomerServiceResponse)
    assert not agent.get_state("t2").next

    # reset t1 不影响 t2 已有状态
    agent.reset(thread_id="t1")
    assert agent.get_state("t1").values == {} and not agent.get_state("t1").next
    assert agent.get_state("t2").values != {}  # t2 未被误删
    agent.close()


# ---------- 5. 同一个 thread 并发请求 ----------

def test_concurrent_same_thread_rejected():
    release = threading.Event()

    def blocking_apply_refund(args):
        release.wait(timeout=5)
        return json.dumps({"success": True, "message": "退款已提交"}, ensure_ascii=False)

    tms = {key: FakeToolManager(ALLOWED[key]) for key in ALLOWED}
    tms["postsale"] = FakeToolManager(ALLOWED["postsale"], results={"apply_refund": blocking_apply_refund})

    fake = FakeLLM(router_result="postsale", plan_tools=_refund_plan())
    agent, _, _, _ = _make_agent(client=fake, thread_id="t-race", tool_managers=tms)

    agent.chat("我要退款订单 ORD-20240115-001", thread_id="t-race")  # interrupt

    def confirm():
        try:
            agent.resume("t-race", "确认")
        except Exception as e:  # noqa: BLE001
            return e
        return None

    ta = threading.Thread(target=confirm)
    ta.start()

    # 等待线程 A 真正进入 apply_refund（此时它持有 thread 锁）
    for _ in range(100):
        if any(c[0] == "apply_refund" for c in tms["postsale"].calls):
            break
        time.sleep(0.02)

    # 主线程对同一 thread 再发起请求 → 安全拒绝
    with pytest.raises(SessionBusyError):
        agent.chat("确认", thread_id="t-race")

    release.set()
    ta.join(timeout=5)
    # 退款只执行了一次，没有被并发重复执行
    assert sum(1 for c in tms["postsale"].calls if c[0] == "apply_refund") == 1
    agent.close()


# ---------- 6. reset 后不恢复旧 interrupt ----------

def test_reset_clears_old_interrupt():
    fake = FakeLLM(router_result="postsale", plan_tools=_refund_plan())
    agent, _, tms, _ = _make_agent(client=fake, backend="sqlite", thread_id="t-reset-int")

    agent.chat("我要退款订单 ORD-20240115-001", thread_id="t-reset-int")
    assert agent.get_state("t-reset-int").next

    agent.reset(thread_id="t-reset-int")
    snap = agent.get_state("t-reset-int")
    assert snap.values == {} and not snap.next

    # reset 后重新对话，不再执行旧的退款工具
    fake.plan_tools = []
    fake.plan_respond = True
    agent.chat("你好", thread_id="t-reset-int")
    assert "apply_refund" not in [c[0] for c in tms["postsale"].calls]
    agent.close()


# ---------- 7. 重启后不重复执行旧工具 ----------

def test_restart_does_not_reexecute_old_tools():
    fake1 = FakeLLM(router_result="presale", plan_tools=[{"name": "query_product", "arguments": {"keyword": "耳机"}}])
    agent1, session_dir, tms1, _ = _make_agent(client=fake1, backend="sqlite", thread_id="t-re")
    agent1.chat("推荐耳机", thread_id="t-re")
    assert "query_product" in [c[0] for c in tms1["presale"].calls]
    agent1.close()  # 重启

    # 新进程：同一 db + 同一 thread，新请求不再重跑旧工具
    agent2, _, tms2, _ = _make_agent(
        client=FakeLLM(router_result="presale", plan_respond=True),
        backend="sqlite",
        thread_id="t-re",
        session_dir=session_dir,
    )
    agent2.chat("再推荐一个", thread_id="t-re")
    assert "query_product" not in [c[0] for c in tms2["presale"].calls]
    agent2.close()


# ---------- 8. checkpoint 文件损坏时安全降级 ----------

def test_checkpoint_corruption_safe_degradation(tmp_path):
    db = tmp_path / "corrupt.db"
    db.write_bytes(b"this is not a sqlite database at all")

    handle = build_checkpointer(enabled=True, backend="sqlite", path=str(db))
    assert handle is not None
    assert handle.kind == "memory"  # 损坏 → 安全降级为 memory
    handle.close()


def test_checkpoint_path_sqlite_effective(tmp_path):
    db = tmp_path / "custom" / "cp.db"
    handle = build_checkpointer(enabled=True, backend="sqlite", path=str(db))
    assert handle is not None
    assert handle.kind == "sqlite"
    assert db.exists()  # CHECKPOINT_PATH 真正生效
    handle.close()


# ---------- 9. SQLite 连接关闭后不影响普通 session 保存 ----------

def test_sqlite_close_does_not_affect_session_save(tmp_path):
    agent, session_dir, _, _ = _make_agent(
        client=FakeLLM(plan_respond=True), backend="sqlite", thread_id="t-close-save",
    )
    agent.chat("你好", thread_id="t-close-save")
    agent.close()  # 关闭 sqlite 连接 + checkpoint handle

    # session JSON 保存独立于 checkpoint DB
    sp = os.path.join(session_dir, "another_session.json")
    save_session(sp, [{"role": "user", "content": "hi"}], None)
    loaded = load_session(sp)
    assert loaded["messages"][0]["content"] == "hi"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])