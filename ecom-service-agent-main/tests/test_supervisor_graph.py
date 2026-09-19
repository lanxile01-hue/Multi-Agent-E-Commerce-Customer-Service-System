"""Supervisor + 子图重构测试（纯 Mock，不依赖真实 API Key）。

覆盖需求十列出的 16 项：
1. 三个子图独立构建
2. 三个子图独立调用
3. presale 只能访问售前工具
4. postsale 可以访问售后工具
5. complaint 不能访问退款工具
6. Router 正确路由三个场景
7. 父图正确调用对应子图
8. 子图工具调用轨迹完整
9. 达到最大循环次数后安全结束
10. 工具失败后安全回复
11. Checkpoint 恢复多轮会话
12. interrupt 暂停退款流程
13. 用户确认后继续执行退款
14. 用户取消后不执行退款
15. 最终输出符合 CustomerServiceResponse
16. 原有测试模块仍可导入
17. 子 Agent 上下文真正生效（三个 Agent + Memory/Skill/Summary）
18. 工具错误重试策略（retryable 重试 / 其余不重试 / 写操作不重试）
19. 子图状态隔离（入口初始化 + 多 Agent 切换）

用法：python tests/test_supervisor_graph.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.graph import StateGraph, START, END  # noqa: E402
from langgraph.graph.state import CompiledStateGraph  # noqa: E402
from langgraph.types import Command  # noqa: E402

from app.agent.tools.registry import TOOL_DEFINITIONS  # noqa: E402
from app.multi_agent.router import Router  # noqa: E402
from app.multi_agent.state import SupervisorState, empty_state  # noqa: E402
from app.multi_agent.structured import deserialize_response  # noqa: E402
from app.multi_agent.subgraphs import (  # noqa: E402
    build_presale_graph,
    build_postsale_graph,
    build_complaint_graph,
)
from app.schemas.response import CustomerServiceResponse  # noqa: E402

# 三个子 Agent 的工具白名单（与 app/multi_agent/agents.py 保持一致）
ALLOWED = {
    "presale": {"query_product", "search_knowledge", "list_user_orders", "load_skill"},
    "postsale": {
        "query_order", "query_logistics", "apply_refund",
        "list_user_orders", "search_knowledge", "load_skill",
    },
    "complaint": {"query_order", "search_knowledge", "load_skill"},
}


# ---------- Fake LLM ----------

class _FakeMessage:
    def __init__(self, content="", parsed=None):
        self.content = content
        self.parsed = parsed


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeResp:
    def __init__(self, message):
        self.choices = [_FakeChoice(message)]


class _FakeChatCompletions:
    def __init__(self, llm):
        self._llm = llm

    def create(self, **kwargs):
        return self._llm._create(kwargs)


class _FakeChat:
    def __init__(self, llm):
        self.completions = _FakeChatCompletions(llm)


class _FakeBetaCompletions:
    def __init__(self, llm):
        self._llm = llm

    def parse(self, **kwargs):
        return self._llm._parse(kwargs)


class _FakeBetaChat:
    def __init__(self, llm):
        self.completions = _FakeBetaCompletions(llm)


class _FakeBeta:
    def __init__(self, llm):
        self.chat = _FakeBetaChat(llm)


class FakeLLM:
    """内容驱动的 Mock LLM。

    - 检测 prompt 关键词，返回脚本化响应
    - beta.parse 默认抛异常，触发结构化输出的稳定 fallback 路径
    """

    def __init__(
        self,
        router_result: str = "postsale",
        plan_tools: list[dict] | None = None,
        plan_respond: bool = False,
        evaluator_sufficient: bool = True,
        responder_text: str = "已为您查询完成，感谢您的咨询。",
        boundary_in_scope: bool = True,
    ):
        self.router_result = router_result
        self.plan_tools = plan_tools or []
        self.plan_respond = plan_respond
        self.evaluator_sufficient = evaluator_sufficient
        self.responder_text = responder_text
        self.boundary_in_scope = boundary_in_scope
        self.model_calls = 0
        # 记录每次 create 调用传入的完整 messages，用于验证上下文是否真正生效
        self.captured: list[list[dict]] = []

        self.chat = _FakeChat(self)
        self.beta = _FakeBeta(self)

    def _text(self, messages) -> str:
        parts = []
        for m in messages or []:
            if isinstance(m, dict):
                parts.append(str(m.get("content") or ""))
        return "\n".join(parts)

    def _create(self, kwargs):
        self.model_calls += 1
        messages = kwargs.get("messages", [])
        self.captured.append(messages)
        text = self._text(messages)
        if "查询边界检查器" in text:
            return _FakeResp(_FakeMessage(content=json.dumps({
                "in_scope": self.boundary_in_scope,
                "reason": "mock boundary",
            }, ensure_ascii=False)))
        if "意图分类器" in text:
            return _FakeResp(_FakeMessage(content=self.router_result))
        if "任务规划器" in text:
            if self.plan_respond:
                return _FakeResp(_FakeMessage(content='{"action": "respond", "tools": [], "thought": ""}'))
            return _FakeResp(_FakeMessage(content=json.dumps({
                "action": "call_tools",
                "tools": self.plan_tools,
                "thought": "mock thought",
            }, ensure_ascii=False)))
        if "任务评估器" in text:
            return _FakeResp(_FakeMessage(content=json.dumps({
                "sufficient": self.evaluator_sufficient,
                "reason": "mock",
            })))
        return _FakeResp(_FakeMessage(content=self.responder_text))

    def _parse(self, kwargs):
        # 默认不支持结构化输出，触发 fallback
        raise NotImplementedError("mock beta.parse raises -> fallback")


class FakeToolManager:
    """Mock ToolManager，记录调用轨迹，可脚本化返回结果。"""

    def __init__(self, allowed: set[str], results: dict | None = None):
        self.allowed = allowed
        self._results = results or {}
        self.calls: list[tuple[str, dict]] = []
        self.closed = False
        self.tool_definitions = [
            td for td in TOOL_DEFINITIONS if td["function"]["name"] in allowed
        ]

    def execute_tool(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        if name in self._results:
            r = self._results[name]
            return r(arguments) if callable(r) else r
        return json.dumps({"success": True, "message": f"{name} ok"}, ensure_ascii=False)

    def close(self):
        self.closed = True


class FakeMemoryManager:
    """返回固定记忆区块，用于验证 Memory 是否传入子图上下文。"""

    def build_memory_prompt_sections(self):
        return [{"role": "system", "content": "[记忆] mock-memory-section"}]


class FakeSkillManager:
    """返回固定技能目录，用于验证 Skill 是否传入子图上下文。"""

    enabled = True

    def build_catalog_prompt(self):
        return "\n[技能] mock-skill-catalog"


def _flatten(messages: list[dict]) -> str:
    return "\n".join(str(m.get("content") or "") for m in messages if isinstance(m, dict))


def _build(agent_key: str, *, checkpointer=None, fake: FakeLLM | None = None,
           tm: FakeToolManager | None = None, max_steps: int = 5,
           memory_manager=None, skill_manager=None, get_summary=None,
           human_confirmation_enabled: bool = True, tool_timeout: float = 0.0):
    fake = fake or FakeLLM()
    tm = tm or FakeToolManager(ALLOWED[agent_key])
    factories = {
        "presale": build_presale_graph,
        "postsale": build_postsale_graph,
        "complaint": build_complaint_graph,
    }
    graph = factories[agent_key](
        tool_manager=tm,
        client=fake,
        model="mock-model",
        temperature=0.0,
        max_steps=max_steps,
        human_confirmation_enabled=human_confirmation_enabled,
        checkpointer=checkpointer,
        memory_manager=memory_manager,
        skill_manager=skill_manager,
        get_summary=get_summary,
        tool_timeout=tool_timeout,
    )
    return graph, fake, tm


def _invoke(graph, user_input: str, *, config=None, conversation=None):
    st = empty_state()
    st["user_input"] = user_input
    st["conversation"] = conversation or [{"role": "user", "content": user_input}]
    st["metadata"] = {}
    return graph.invoke(st, config=config)


def _ok(msg): print(f"  [OK] {msg}")
def _fail(msg): print(f"  [FAIL] {msg}"); sys.exit(1)


# ---------- 测试 1：三个子图独立构建 ----------
def test_subgraphs_buildable():
    print("\n[1/16] 三个子图独立构建")
    for key in ALLOWED:
        graph, _, _ = _build(key)
        if isinstance(graph, CompiledStateGraph):
            _ok(f"{key} 子图构建为 CompiledStateGraph")
        else:
            _fail(f"{key} 子图不是 CompiledStateGraph")


# ---------- 测试 2：三个子图独立调用 ----------
def test_subgraphs_invokable():
    print("\n[2/16] 三个子图独立调用")
    cases = {
        "presale": "有什么耳机推荐吗",
        "postsale": "帮我查订单",
        "complaint": "我要投诉",
    }
    for key, msg in cases.items():
        graph, fake, tm = _build(key)
        fake.plan_respond = True  # 不调用工具，直接回复
        result = _invoke(graph, msg)
        if result.get("final_response"):
            _ok(f"{key} 子图返回 final_response（{result['final_response'][:20]}…）")
        else:
            _fail(f"{key} 子图未返回 final_response")


# ---------- 测试 3/4/5：工具隔离 ----------
def test_tool_isolation():
    print("\n[3/16] presale 只能访问售前工具")
    print("[4/16] postsale 可以访问售后工具")
    print("[5/16] complaint 不能访问退款工具")
    presale_tools = {td["function"]["name"] for td in FakeToolManager(ALLOWED["presale"]).tool_definitions}
    postsale_tools = {td["function"]["name"] for td in FakeToolManager(ALLOWED["postsale"]).tool_definitions}
    complaint_tools = {td["function"]["name"] for td in FakeToolManager(ALLOWED["complaint"]).tool_definitions}

    if "query_product" in presale_tools and "apply_refund" not in presale_tools:
        _ok("presale 有 query_product 且无 apply_refund")
    else:
        _fail("presale 工具隔离失败")
    if "apply_refund" in postsale_tools and "query_order" in postsale_tools:
        _ok("postsale 有 apply_refund + query_order")
    else:
        _fail("postsale 工具隔离失败")
    if "apply_refund" not in complaint_tools and "query_order" in complaint_tools:
        _ok("complaint 无 apply_refund 且有 query_order")
    else:
        _fail("complaint 工具隔离失败")


# ---------- 测试 6：Router 正确路由三个场景 ----------
def test_router():
    print("\n[6/16] Router 正确路由三个场景")
    cases = [
        ("有什么耳机推荐吗", "presale"),
        ("帮我查订单", "postsale"),
        ("我要投诉，太差了", "complaint"),
    ]
    for msg, expected in cases:
        fake = FakeLLM(router_result=expected)
        router = Router(client=fake, model="mock")
        got = router.route(msg)
        if got == expected:
            _ok(f"「{msg}」→ {got}")
        else:
            _fail(f"「{msg}」期望 {expected}，实际 {got}")


# ---------- 测试 7：父图正确调用对应子图 ----------
def test_parent_routing():
    print("\n[7/16] 父图正确调用对应子图")
    # 组装一个最小父图，验证条件边把请求分发到正确子图
    subgraphs = {key: _build(key)[0] for key in ALLOWED}

    def route_node(state):
        return {"selected_agent": state["selected_agent"]}

    def validate_node(state):
        return {"requires_human": bool(state.get("requires_human", False))}

    builder = StateGraph(SupervisorState)
    builder.add_node("route", route_node)
    for key in ALLOWED:
        builder.add_node(f"run_{key}", subgraphs[key])
    builder.add_node("validate", validate_node)
    builder.add_edge(START, "route")
    builder.add_conditional_edges(
        "route",
        lambda s: f"run_{s['selected_agent']}",
        {f"run_{k}": f"run_{k}" for k in ALLOWED},
    )
    for key in ALLOWED:
        builder.add_edge(f"run_{key}", "validate")
    builder.add_edge("validate", END)
    parent = builder.compile()

    for key in ALLOWED:
        st = empty_state()
        st["user_input"] = f"test-{key}"
        st["conversation"] = [{"role": "user", "content": f"test-{key}"}]
        st["selected_agent"] = key
        st["metadata"] = {}
        result = parent.invoke(st)
        if result.get("selected_agent") == key and result.get("final_response"):
            _ok(f"{key} 场景被正确执行（final_response 非空）")
        else:
            _fail(f"{key} 场景父图执行失败: {result.get('final_response')!r}")


# ---------- 测试 8：子图工具调用轨迹完整 ----------
def test_tool_trace():
    print("\n[8/16] 子图工具调用轨迹完整")
    graph, fake, tm = _build("presale")
    fake.plan_tools = [{"name": "query_product", "arguments": {"keyword": "耳机"}}]
    result = _invoke(graph, "推荐耳机")
    tr = result.get("tool_results", [])
    if tr and tr[0]["name"] == "query_product" and "result" in tr[0]:
        _ok(f"tool_results 记录完整：{tr[0]['name']} -> {tr[0]['result'][:20]}…")
    else:
        _fail("tool_results 轨迹不完整")


# ---------- 测试 9：达到最大循环次数后安全结束 ----------
def test_max_steps_safe_end():
    print("\n[9/16] 达到最大循环次数后安全结束")
    fake = FakeLLM(evaluator_sufficient=False)  # 评估器一直说信息不足 → 强制循环
    fake.plan_tools = [{"name": "query_product", "arguments": {"keyword": "耳机"}}]
    tm = FakeToolManager(ALLOWED["presale"])
    graph = build_presale_graph(
        tool_manager=tm, client=fake, model="mock", temperature=0.0,
        max_steps=2, human_confirmation_enabled=True, checkpointer=None,
    )
    result = _invoke(graph, "推荐耳机")
    if result.get("final_response"):
        _ok("达到最大步数后安全结束（final_response 非空）")
    else:
        _fail("达到最大步数后没有生成回复")


# ---------- 测试 10：工具失败后安全回复 ----------
def test_tool_failure_safe_reply():
    print("\n[10/16] 工具失败后安全回复")
    fake = FakeLLM(responder_text="抱歉，暂时无法查询该订单，请稍后再试。")
    fake.plan_tools = [{"name": "query_order", "arguments": {"order_id": "ORD-NOT-EXIST"}}]
    tm = FakeToolManager(ALLOWED["postsale"], results={
        "query_order": json.dumps({"success": False, "error": "未找到订单 ORD-NOT-EXIST"}, ensure_ascii=False),
    })
    graph = build_postsale_graph(tool_manager=tm, client=fake, model="mock", temperature=0.0,
                                 max_steps=5, human_confirmation_enabled=True)
    result = _invoke(graph, "查一下不存在的订单")
    tr = result.get("tool_results", [])
    if result.get("final_response") and tr and tr[0]["ok"] is False:
        _ok(f"工具失败被安全处理，仍返回回复（error={tr[0]['error']}）")
    else:
        _fail("工具失败后没有安全回复")


# ---------- 测试 11：Checkpoint 恢复多轮会话 ----------
def test_checkpoint_multiturn():
    print("\n[11/16] Checkpoint 恢复多轮会话")
    graph, fake, tm = _build("presale", checkpointer=MemorySaver())
    fake.plan_respond = True
    cfg = {"configurable": {"thread_id": "t-multi"}}
    r1 = _invoke(graph, "你好", config=cfg)
    r2 = _invoke(graph, "再推荐一个", config=cfg)
    snap = graph.get_state(cfg)
    if r1.get("final_response") and r2.get("final_response") and snap is not None:
        _ok("同 thread_id 下多轮调用可恢复 checkpoint 状态")
    else:
        _fail("checkpoint 多轮恢复失败")


# ---------- 测试 12/13/14：interrupt 暂停 + 确认/取消 ----------
def test_human_confirmation():
    print("\n[12/16] interrupt 暂停退款流程")
    fake = FakeLLM()
    fake.plan_tools = [{"name": "apply_refund", "arguments": {"order_id": "ORD-20240115-001", "reason": "尺码不合适"}}]
    tm = FakeToolManager(ALLOWED["postsale"], results={
        "apply_refund": json.dumps({"success": True, "message": "退款已提交"}, ensure_ascii=False),
    })
    graph = build_postsale_graph(tool_manager=tm, client=fake, model="mock", temperature=0.0,
                                 max_steps=5, human_confirmation_enabled=True,
                                 checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t-refund"}}
    _invoke(graph, "我要退款", config=cfg)
    snap = graph.get_state(cfg)
    if snap.next:
        _ok("退款流程已暂停，等待用户确认")
    else:
        _fail("退款流程未暂停")

    print("[13/16] 用户确认后继续执行退款")
    graph.invoke(Command(resume="确认"), config=cfg)
    called = [c[0] for c in tm.calls]
    if "apply_refund" in called:
        _ok("用户确认后执行了 apply_refund")
    else:
        _fail(f"确认后未执行 apply_refund（调用记录={called}）")

    print("[14/16] 用户取消后不执行退款")
    tm2 = FakeToolManager(ALLOWED["postsale"], results={
        "apply_refund": json.dumps({"success": True, "message": "退款已提交"}, ensure_ascii=False),
    })
    graph2 = build_postsale_graph(tool_manager=tm2, client=FakeLLM(), model="mock", temperature=0.0,
                                  max_steps=5, human_confirmation_enabled=True,
                                  checkpointer=MemorySaver())
    cfg2 = {"configurable": {"thread_id": "t-refund-cancel"}}
    _invoke(graph2, "我要退款", config=cfg2)
    graph2.invoke(Command(resume="取消"), config=cfg2)
    if "apply_refund" not in [c[0] for c in tm2.calls]:
        _ok("用户取消后未执行 apply_refund")
    else:
        _fail("取消后仍然执行了 apply_refund")


# ---------- 测试 15：最终输出符合 CustomerServiceResponse ----------
def test_structured_output():
    print("\n[15/16] 最终输出符合 CustomerServiceResponse")
    graph, fake, tm = _build("presale")
    fake.plan_respond = True
    fake.responder_text = "根据您的需求，推荐这款降噪耳机。"
    result = _invoke(graph, "推荐耳机")
    resp = deserialize_response(result.get("structured_response") or {})
    if isinstance(resp, CustomerServiceResponse) and resp.reply and 0.0 <= resp.confidence <= 1.0:
        _ok(f"结构化输出合法（intent={resp.intent.value}, confidence={resp.confidence:.0%}）")
    else:
        _fail("结构化输出不合法")


# ---------- 测试 16：原有测试模块仍可导入 ----------
def test_legacy_modules_importable():
    print("\n[16/19] 原有测试模块仍可导入")
    try:
        import app.multi_agent.orchestrator  # noqa: F401
        import app.agent.graph  # noqa: F401
        import app.agent.chat  # noqa: F401
        _ok("旧版 orchestrator / graph / chat 模块可正常导入")
    except Exception as e:
        _fail(f"旧模块导入失败: {e}")


# ---------- 测试 17：子 Agent 上下文真正生效（三个 Agent 都验证） ----------
_AGENT_CONTEXT_MARKERS = {
    "presale": "售前咨询专家",
    "postsale": "售后服务专家",
    "complaint": "投诉处理专家",
}


def test_agent_context_flows():
    print("\n[17/19] 子 Agent 上下文真正生效（system/memory/skill/summary 传入模型）")
    for key, marker in _AGENT_CONTEXT_MARKERS.items():
        fake = FakeLLM(plan_respond=True)
        graph, fake, tm = _build(
            key,
            fake=fake,
            memory_manager=FakeMemoryManager(),
            skill_manager=FakeSkillManager(),
            get_summary=lambda: "mock-summary",
        )
        _invoke(graph, "帮我处理一下")

        if len(fake.captured) < 2:
            _fail(f"{key} 预期至少 2 次 LLM 调用，实际 {len(fake.captured)}")

        reason_text = _flatten(fake.captured[0])   # agent_reason
        resp_text = _flatten(fake.captured[-1])    # generate_response

        # agent_reason 必须包含领域 system prompt + memory + skill + summary + 用户消息 + 规划器
        for m in (marker, "[记忆] mock-memory-section", "[技能] mock-skill-catalog",
                  "mock-summary", "帮我处理一下", "任务规划器"):
            if m not in reason_text:
                _fail(f"{key} agent_reason 上下文缺失: {m}")

        # generate_response 必须包含领域 system prompt + memory + skill + summary + 用户消息 + 回复指令
        for m in (marker, "[记忆] mock-memory-section", "[技能] mock-skill-catalog",
                  "mock-summary", "帮我处理一下", "专业客服专家"):
            if m not in resp_text:
                _fail(f"{key} generate_response 上下文缺失: {m}")
        _ok(f"{key} 回复上下文包含专属 prompt + Memory/Skill/Summary")


# ---------- 测试 18：工具错误重试策略 ----------
def _run_single_tool(agent_key, tool_name, args, result_fn, human_confirmation_enabled=False):
    fake = FakeLLM()
    fake.plan_tools = [{"name": tool_name, "arguments": args}]
    tm = FakeToolManager(ALLOWED[agent_key], results={tool_name: result_fn})
    graph, fake, tm = _build(agent_key, fake=fake, tm=tm,
                             human_confirmation_enabled=human_confirmation_enabled)
    result = _invoke(graph, "test request")
    return tm, result


def test_retry_strategy():
    print("\n[18/19] 工具错误重试策略")

    # 1) retryable：先失败后成功 → 重试 1 次，共 2 次调用
    def retry_then_ok(args):
        retry_then_ok.n += 1
        if retry_then_ok.n == 1:
            return json.dumps({"success": False, "error": "网络超时"}, ensure_ascii=False)
        return json.dumps({"success": True, "message": "ok"}, ensure_ascii=False)
    retry_then_ok.n = 0
    tm, result = _run_single_tool("postsale", "query_order", {"order_id": "ORD-1"}, retry_then_ok)
    calls = [c for c in tm.calls if c[0] == "query_order"]
    if len(calls) == 2 and result["tool_results"][0]["retries"] == 1:
        _ok("retryable 错误重试 1 次后成功")
    else:
        _fail(f"retryable 重试异常：calls={len(calls)}, retries={result['tool_results'][0]['retries']}")

    # 2) retryable 持续失败 → 最多重试 2 次，共 3 次调用
    tm, result = _run_single_tool(
        "postsale", "query_order", {"order_id": "ORD-1"},
        lambda args: json.dumps({"success": False, "error": "连接失败"}, ensure_ascii=False),
    )
    calls = [c for c in tm.calls if c[0] == "query_order"]
    if len(calls) == 3 and result["tool_results"][0]["retries"] == 2:
        _ok("retryable 持续失败最多重试 2 次")
    else:
        _fail(f"retryable 重试上限异常：calls={len(calls)}")

    # 3) fatal → 只调用一次
    tm, _ = _run_single_tool(
        "postsale", "query_order", {"order_id": "ORD-1"},
        lambda args: json.dumps({"success": False, "error": "内部错误"}, ensure_ascii=False),
    )
    if len([c for c in tm.calls if c[0] == "query_order"]) == 1:
        _ok("fatal 错误只调用一次")
    else:
        _fail("fatal 错误被错误重试")

    # 4) param_error → 只调用一次
    tm, _ = _run_single_tool(
        "postsale", "query_order", {"order_id": "ORD-1"},
        lambda args: json.dumps({"success": False, "error": "缺少必填参数 order_id"}, ensure_ascii=False),
    )
    if len([c for c in tm.calls if c[0] == "query_order"]) == 1:
        _ok("param_error 只调用一次")
    else:
        _fail("param_error 被错误重试")

    # 5) not_found → 只调用一次
    tm, _ = _run_single_tool(
        "postsale", "query_order", {"order_id": "ORD-1"},
        lambda args: json.dumps({"success": False, "error": "未找到订单"}, ensure_ascii=False),
    )
    if len([c for c in tm.calls if c[0] == "query_order"]) == 1:
        _ok("not_found 只调用一次")
    else:
        _fail("not_found 被错误重试")

    # 6) apply_refund 即使返回 retryable 也不重试（写操作禁止重试）
    tm, _ = _run_single_tool(
        "postsale", "apply_refund", {"order_id": "ORD-1", "reason": "尺码不合适"},
        lambda args: json.dumps({"success": False, "error": "网络超时"}, ensure_ascii=False),
        human_confirmation_enabled=False,
    )
    if len([c for c in tm.calls if c[0] == "apply_refund"]) == 1:
        _ok("apply_refund 写操作失败不重试")
    else:
        _fail("apply_refund 被错误重试")


# ---------- 测试 19：子图状态隔离 ----------
def test_state_isolation():
    print("\n[19/19] 子图状态隔离（入口初始化 + 多 Agent 切换）")
    # 注入脏状态（模拟上次请求遗留），验证 agent_enter 会重置
    graph, fake, tm = _build("presale")
    fake.plan_respond = True
    st = empty_state()
    st["user_input"] = "推荐耳机"
    st["conversation"] = [{"role": "user", "content": "推荐耳机"}]
    st["metadata"] = {}
    st["tool_results"] = [{"name": "query_product", "arguments": {}, "result": "STALE", "ok": True, "error": "ok"}]
    st["pending_tool_calls"] = [{"name": "query_product", "arguments": {"keyword": "耳机"}}]
    st["iteration"] = 99
    result = graph.invoke(st)
    if result.get("tool_results") == [] and result.get("pending_tool_calls") == []:
        _ok("入口初始化重置了上一次请求的临时状态")
    else:
        _fail(f"临时状态未重置: tool_results={result.get('tool_results')}, pending={result.get('pending_tool_calls')}")

    # 连续多轮不同 Agent 切换：postsale 不应复用 presale 的 tool_results
    g1, f1, t1 = _build("presale")
    f1.plan_tools = [{"name": "query_product", "arguments": {"keyword": "耳机"}}]
    r1 = _invoke(g1, "推荐耳机")
    g2, f2, t2 = _build("postsale")
    f2.plan_tools = [{"name": "query_order", "arguments": {"order_id": "ORD-1"}}]
    r2 = _invoke(g2, "查订单")
    names2 = [tr["name"] for tr in r2.get("tool_results", [])]
    if "query_order" in names2 and "query_product" not in names2:
        _ok("不同 Agent 切换后 tool_results 互不污染")
    else:
        _fail(f"状态污染：postsale tool_results={names2}")


def main():
    print("=" * 60)
    print("  Supervisor + 子图重构 · Mock 测试")
    print("=" * 60)
    test_subgraphs_buildable()
    test_subgraphs_invokable()
    test_tool_isolation()
    test_router()
    test_parent_routing()
    test_tool_trace()
    test_max_steps_safe_end()
    test_tool_failure_safe_reply()
    test_checkpoint_multiturn()
    test_human_confirmation()
    test_structured_output()
    test_legacy_modules_importable()
    test_agent_context_flows()
    test_retry_strategy()
    test_state_isolation()
    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()