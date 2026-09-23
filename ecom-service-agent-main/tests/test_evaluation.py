"""Agent 评估体系测试（沙箱 + 双层测评）。

单元测试（默认）：数据集加载、规则指标、LLM judge（Mock client）。
集成测试（integration）：沙箱采集、Evaluator 双层评分（真实 API）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config.settings import settings  # noqa: E402
from app.evaluation import metrics  # noqa: E402
from app.evaluation.dataset import EvalCase, load_dataset  # noqa: E402
from app.evaluation.evaluator import Evaluator  # noqa: E402
from app.evaluation.sandbox import Sandbox  # noqa: E402
from app.evaluation.trace import RunTrace, ToolObservation  # noqa: E402
from app.schemas.response import CustomerServiceResponse, IntentType  # noqa: E402
from tests.fakes import FakeLLM  # noqa: E402

DATASET = ROOT / "app" / "evaluation" / "cases.json"


# ---------- 单元测试（无需 API） ----------

def test_dataset_load():
    cases = load_dataset(DATASET)
    assert len(cases) >= 8
    assert isinstance(cases[0], EvalCase)
    assert cases[0].id and cases[0].turns
    assert any(c.min_tool_calls is not None for c in cases)
    assert any(c.max_tokens is not None for c in cases)
    # 黄金集同时标注单 Agent 意图与（多 Agent / Supervisor）路由期望
    assert all(c.expected_route for c in cases)


def test_metrics_rule_based():
    assert metrics.tool_accuracy(["query_order"], ["query_order", "load_skill"]) == 1.0
    assert metrics.tool_accuracy(["query_order", "query_logistics"], ["query_order"]) == 0.5
    assert metrics.tool_accuracy([], ["query_order"]) is None
    assert metrics.tool_efficiency(1, 2) == 0.5
    assert metrics.tool_efficiency(2, 2) == 1.0
    assert metrics.token_cost_pass(5000, 6000) is True
    assert metrics.token_cost_pass(7000, 6000) is False
    assert metrics.keyword_coverage(["Nike", "899"], "您的 Nike 鞋 899 元") == 1.0
    assert metrics.intent_match("order_query", "order_query") == 1.0
    assert metrics.intent_match("order_query", "complaint") == 0.0
    assert metrics.requires_human_match(None, True) is None


def test_judge_answer_quality_mock():
    client = FakeLLM(responder=lambda kw: json.dumps({"score": 4, "reason": "清晰准确"}))
    score, _ = metrics.judge_answer_quality(client, "m", "问题", "回复", ["顺丰"])
    assert score == 4.0


def test_judge_faithfulness_mock():
    client = FakeLLM(responder=lambda kw: json.dumps({"faithful": True, "reason": "无幻觉"}))
    obs = [ToolObservation(name="query_logistics", arguments={}, result="...")]
    faithful, _ = metrics.judge_faithfulness(client, "m", "回复", obs)
    assert faithful == 1.0


def test_judge_process_soundness_mock():
    client = FakeLLM(responder=lambda kw: json.dumps({"score": 5, "reason": "顺序合理"}))
    score, _ = metrics.judge_process_soundness(client, "m", "问题", ["query_order"])
    assert score == 5.0


def test_judge_parse_failure_fallback():
    client = FakeLLM(reply="not a json")
    score, reason = metrics.judge_answer_quality(client, "m", "问题", "回复")
    assert score == 0.0
    assert "解析失败" in reason


def test_sandbox_isolation():
    sandbox = Sandbox(mode="single")
    assert sandbox.session_path_for("case_a") != sandbox.session_path_for("case_b")
    agent = sandbox._build_agent(sandbox.session_path_for("case_a"))
    try:
        assert agent.memory_manager.memory_enabled is False
    finally:
        sandbox._close_tool_managers(agent)


def test_sandbox_supervisor_mode_builds_supervisor_agent():
    """supervisor 模式构建主线架构，且三个子 Agent 各自独立 ToolManager。"""
    sandbox = Sandbox(mode="supervisor")
    agent = sandbox._build_agent(sandbox.session_path_for("case_sup"))
    try:
        assert type(agent).__name__ == "SupervisorAgent"
        assert set(agent.tool_managers) == {"presale", "postsale", "complaint"}
        assert agent.memory_manager.memory_enabled is False
        assert len(sandbox._tool_managers(agent)) == 3
    finally:
        sandbox._close_tool_managers(agent)


def _fake_trace(route: str | None, intent: IntentType) -> RunTrace:
    trace = RunTrace(case_id="c1", turns=["问题"])
    trace.route = route
    trace.final_response = CustomerServiceResponse(
        intent=intent, confidence=0.9, reply="七天无理由退货：Nike 899", requires_human=False,
    )
    return trace


class _StubSandbox:
    """只提供 run() 的假沙箱，用于验证评分口径。"""

    def __init__(self, mode: str, trace: RunTrace):
        self.mode = mode
        self._trace = trace

    def run(self, case: EvalCase) -> RunTrace:
        return self._trace


def test_routed_mode_scores_route_not_intent():
    """多 Agent / Supervisor 模式下意图词表不同，应改用 expected_route 校验路由。"""
    evaluator = Evaluator(
        sandbox=_StubSandbox("supervisor", _fake_trace("postsale", IntentType.AFTER_SALE)),
        client=None, model="m", use_judge=False,
    )
    case = EvalCase(
        id="c1", description="查订单", turns=["查订单"],
        expected_intent="order_query", expected_route="postsale", expected_tools=["query_order"],
    )
    res = evaluator.run_case(case)
    assert res.route_match == 1.0
    assert res.intent_match is None  # 不再拿 presale/postsale 去比 order_query


def test_single_mode_still_scores_intent():
    """单 Agent 模式仍按 IntentType 校验意图。"""
    evaluator = Evaluator(
        sandbox=_StubSandbox("single", _fake_trace(None, IntentType.ORDER_QUERY)),
        client=None, model="m", use_judge=False,
    )
    case = EvalCase(id="c1", description="查订单", turns=["查订单"], expected_intent="order_query")
    res = evaluator.run_case(case)
    assert res.intent_match == 1.0
    assert res.route_match is None


# ---------- 集成测试（真实 API） ----------

@pytest.mark.integration
def test_sandbox_trace_live():
    sandbox = Sandbox(mode="single")
    case = EvalCase(
        id="trace_probe",
        description="探针：查询订单",
        turns=["帮我查一下订单 ORD-20240115-001"],
        expected_tools=["query_order"],
    )
    trace = sandbox.run(case)
    assert not trace.error
    assert trace.total_tokens > 0
    assert trace.num_llm_calls >= 2
    assert trace.final_response is not None


@pytest.mark.integration
def test_evaluator_two_tier_live():
    from openai import OpenAI

    from app.evaluation.evaluator import Evaluator

    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    cases = [
        EvalCase(
            id="eval_order",
            description="查询订单",
            turns=["帮我查一下订单 ORD-20240115-001"],
            expected_intent="order_query",
            expected_tools=["query_order"],
            min_tool_calls=1,
            max_tokens=8000,
        ),
        EvalCase(
            id="eval_greeting",
            description="问候",
            turns=["你好"],
            expected_intent="greeting",
            expected_tools=[],
            min_tool_calls=0,
            max_tokens=4000,
        ),
    ]

    sandbox = Sandbox(mode="single")
    evaluator = Evaluator(sandbox=sandbox, client=client, model=settings.model_name, use_judge=False)
    report = evaluator.run_all(cases)

    assert report["summary"]["total"] == 2
    c0 = report["cases"][0]
    assert not c0["error"]
    assert c0["result"]["answer_quality"] is None  # no-judge 时 judge 维度为 None
    assert c0["result"]["intent_match"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
