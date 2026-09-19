"""生产级健壮性异常测试（全部 Mock，不依赖真实 API）。

覆盖清单（第一阶段）：
- 空输入 / 非法 JSON / 模型返回空内容 / 模型调用超时
- 工具调用超时 / 工具参数缺失 / 未知工具 / 订单不存在 / RAG 无召回
- Checkpoint 不存在 / thread_id 不存在 / interrupt 恢复输入模糊
- 多次重复确认 / 重复 reset / close 后再次调用
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

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from app.agent.json_utils import extract_json_object  # noqa: E402
from app.agent.tools import knowledge as knowledge_tool  # noqa: E402
from app.agent.tools.errors import (  # noqa: E402
    TOOL_NOT_FOUND,
    TOOL_PARAM_ERROR,
    TOOL_RETRYABLE,
    TOOL_UNKNOWN,
    classify_tool_error,
    validate_tool_arguments,
)
from app.agent.tools.order import query_order  # noqa: E402
from app.agent.tools.registry import TOOL_DEFINITIONS, execute_tool  # noqa: E402
from app.multi_agent.state import normalize_confirmation  # noqa: E402
from app.multi_agent.structured import (  # noqa: E402
    build_structured_response,
    coerce_structured,
    deserialize_response,
)
from app.multi_agent.subgraphs import build_postsale_graph  # noqa: E402
from app.schemas.response import CustomerServiceResponse  # noqa: E402
from tests.fakes import EmptyVectorBackend, FakeEmbedder  # noqa: E402
from tests.test_supervisor_graph import (  # noqa: E402
    ALLOWED,
    FakeLLM,
    FakeToolManager,
    _build,
    _invoke,
)
from tests.test_supervisor_integration import _make_agent  # noqa: E402


# ---------- 子类化替身 ----------

class _Msg:
    def __init__(self, content: str = "", parsed=None):
        self.content = content
        self.parsed = parsed


class _Choice:
    def __init__(self, message):
        self.message = message


class _Resp:
    def __init__(self, message):
        self.choices = [_Choice(message)]


class _EmptyLLM(FakeLLM):
    """模型始终返回空内容。"""

    def _create(self, kwargs):
        self.model_calls += 1
        self.captured.append(kwargs.get("messages", []))
        return _Resp(_Msg(content=""))


class _TimeoutLLM(FakeLLM):
    """模型调用始终超时。"""

    def _create(self, kwargs):
        self.model_calls += 1
        self.captured.append(kwargs.get("messages", []))
        raise TimeoutError("mock model timeout")


class _SlowToolManager(FakeToolManager):
    """工具执行会阻塞，用于触发工具调用超时。"""

    def execute_tool(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        time.sleep(0.5)
        return json.dumps({"success": True}, ensure_ascii=False)


# ---------- 纯函数层 ----------

def test_empty_input():
    assert normalize_confirmation("") == "re_explain"
    resp = build_structured_response("", "presale")
    assert resp.reply  # 兜底 reply 不为空
    assert knowledge_tool.search_knowledge("")["success"] is False


def test_invalid_json():
    with pytest.raises(json.JSONDecodeError):
        extract_json_object("this is not json")
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}


def test_coerce_invalid_structured():
    resp = coerce_structured({"reply": ""}, "presale", fallback_reply="兜底回复")
    assert resp.reply == "兜底回复"


def test_normalize_confirmation_vague():
    assert normalize_confirmation("随便说说") == "re_explain"
    assert normalize_confirmation("确认") == "confirmed"
    assert normalize_confirmation("取消") == "cancelled"


def test_validate_tool_arguments_missing():
    ok, err = validate_tool_arguments("query_order", {}, TOOL_DEFINITIONS)
    assert not ok and "缺少" in err


def test_validate_tool_arguments_unknown():
    ok, err = validate_tool_arguments("no_such_tool", {}, TOOL_DEFINITIONS)
    assert not ok and "未知工具" in err


def test_validate_tool_arguments_valid():
    ok, _ = validate_tool_arguments("query_order", {"order_id": "ORD-1"}, TOOL_DEFINITIONS)
    assert ok


def test_classify_tool_error():
    assert classify_tool_error(json.dumps({"success": False, "error": "未找到订单 ORD-1"})) == TOOL_NOT_FOUND
    assert classify_tool_error(json.dumps({"success": False, "error": "网络超时"})) == TOOL_RETRYABLE
    assert classify_tool_error(json.dumps({"success": False, "error": "缺少必填参数 order_id"})) == TOOL_PARAM_ERROR
    assert classify_tool_error(json.dumps({"success": False, "error": "未知工具: x"})) == TOOL_UNKNOWN


def test_order_not_found():
    assert query_order("ORD-99999-000")["success"] is False
    assert "未找到" in query_order("ORD-99999-000")["error"]


def test_registry_unknown_tool():
    result = json.loads(execute_tool("no_such_tool", {}))
    assert "未知工具" in result.get("error", "")


def test_rag_no_results():
    from app.agent.rag.retriever import KnowledgeRetriever

    retriever = KnowledgeRetriever(embedder=FakeEmbedder(), backend=EmptyVectorBackend())
    retriever.load()
    assert retriever.search("不存在的问题") == []


def test_search_knowledge_no_results(monkeypatch):
    from app.agent.rag.retriever import KnowledgeRetriever

    retriever = KnowledgeRetriever(embedder=FakeEmbedder(), backend=EmptyVectorBackend())
    monkeypatch.setattr(knowledge_tool, "_get_retriever", lambda: retriever)
    monkeypatch.setattr(knowledge_tool, "_adaptive_mode", False)
    result = knowledge_tool.search_knowledge("随便问")
    assert result["success"] is True
    assert result["results"] == []


# ---------- 子图层 ----------

def test_model_returns_empty_content():
    graph, _, _ = _build("presale", fake=_EmptyLLM())
    result = _invoke(graph, "推荐耳机")
    assert result.get("final_response")
    resp = deserialize_response(result.get("structured_response") or {})
    assert resp.reply


def test_model_call_timeout():
    graph, _, _ = _build("presale", fake=_TimeoutLLM())
    result = _invoke(graph, "推荐耳机")
    assert result.get("final_response")
    resp = deserialize_response(result.get("structured_response") or {})
    assert resp.reply


def test_tool_call_timeout():
    fake = FakeLLM()
    fake.plan_tools = [{"name": "query_product", "arguments": {"keyword": "耳机"}}]
    tm = _SlowToolManager(ALLOWED["presale"])
    graph, _, _ = _build("presale", fake=fake, tm=tm, tool_timeout=0.05)
    result = _invoke(graph, "推荐耳机")

    tr = result.get("tool_results", [])
    assert tr and tr[0]["error"] == TOOL_RETRYABLE
    assert "超时" in tr[0]["result"]
    assert len([c for c in tm.calls if c[0] == "query_product"]) == 3  # 1 + 2 重试


def test_tool_param_missing_route():
    fake = FakeLLM()
    fake.plan_tools = [{"name": "query_order", "arguments": {}}]  # 缺少 order_id
    tm = FakeToolManager(ALLOWED["postsale"])
    graph, _, _ = _build("postsale", fake=fake, tm=tm)
    result = _invoke(graph, "查订单")
    assert result.get("final_response")
    assert tm.calls == []  # 参数缺失被拒绝，未执行任何工具


def test_unknown_tool_route():
    fake = FakeLLM()
    fake.plan_tools = [{"name": "hack_tool", "arguments": {}}]
    tm = FakeToolManager(ALLOWED["postsale"])
    graph, _, _ = _build("postsale", fake=fake, tm=tm)
    result = _invoke(graph, "随便问")
    assert result.get("final_response")
    assert tm.calls == []


def test_interrupt_vague_resume():
    fake = FakeLLM()
    fake.plan_tools = [{
        "name": "apply_refund",
        "arguments": {"order_id": "ORD-20240115-001", "reason": "尺码不合适"},
    }]
    tm = FakeToolManager(ALLOWED["postsale"], results={
        "apply_refund": json.dumps({"success": True, "message": "退款已提交"}, ensure_ascii=False),
    })
    graph = build_postsale_graph(
        tool_manager=tm, client=fake, model="mock", temperature=0.0,
        max_steps=5, human_confirmation_enabled=True, checkpointer=MemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "t-vague"}}
    _invoke(graph, "我要退款", config=cfg)  # interrupt 暂停
    fake.plan_respond = True  # 模糊恢复后不再规划退款
    graph.invoke(Command(resume="随便说说"), config=cfg)
    assert "apply_refund" not in [c[0] for c in tm.calls]


def test_repeated_confirmation():
    fake = FakeLLM(router_result="postsale", plan_tools=[{
        "name": "apply_refund",
        "arguments": {"order_id": "ORD-20240115-001", "reason": "尺码不合适"},
    }])
    agent, _, tms, _ = _make_agent(client=fake, thread_id="t-repeat")
    agent.chat("我要退款订单 ORD-20240115-001", thread_id="t-repeat")  # interrupt
    agent.resume("t-repeat", "确认")  # 确认 → 执行退款

    assert sum(1 for c in tms["postsale"].calls if c[0] == "apply_refund") == 1

    # 第二次"确认"作为新消息，不应再次执行退款
    fake.plan_respond = True
    agent.chat("确认", thread_id="t-repeat")
    assert sum(1 for c in tms["postsale"].calls if c[0] == "apply_refund") == 1
    agent.close()


# ---------- SupervisorAgent 层 ----------

def test_checkpoint_missing():
    agent, _, _, _ = _make_agent(thread_id="t-missing")
    snap = agent.get_state("nonexistent-thread")
    # 不存在的 thread：返回空快照，无状态、无待恢复的 interrupt
    assert snap is not None
    assert snap.values == {}
    assert not snap.next
    agent.close()


def test_thread_id_missing():
    fake = FakeLLM(plan_respond=True)
    agent, _, _, _ = _make_agent(client=fake, thread_id="t-fresh")
    resp = agent.chat("你好", thread_id="t-fresh")
    assert isinstance(resp, CustomerServiceResponse)
    snap = agent.get_state("totally-unknown")
    assert snap is not None
    assert snap.values == {}
    assert not snap.next
    agent.close()


def test_repeated_reset():
    fake = FakeLLM(plan_respond=True)
    agent, _, _, _ = _make_agent(client=fake, thread_id="t-reset2")
    agent.chat("你好", thread_id="t-reset2")
    assert agent.history_size > 0

    agent.reset()
    agent.reset()  # 重复 reset 不抛异常
    assert agent.history_size == 0
    agent.close()


def test_close_then_call():
    agent, _, _, _ = _make_agent(thread_id="t-close")
    agent.close()
    agent.close()  # 幂等
    with pytest.raises(RuntimeError):
        agent.chat("你好", thread_id="t-close")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])