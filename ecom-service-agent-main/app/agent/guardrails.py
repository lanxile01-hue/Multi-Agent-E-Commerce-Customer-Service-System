"""
护栏与兜底机制：幻觉检测 + 自纠错重试 + 双向 Fallback + 查询边界。

核心能力:
1. 幻觉检测：LLM-as-Judge 比对回复与工具返回，发现虚构信息
2. 自纠错重试：检测到幻觉后追加纠错 prompt 重新生成（最多 2 次）
3. 任务分解兜底：规划节点失败时回退到默认工具列表
4. 工具执行兜底：工具调用失败时尝试备用策略
5. 查询边界：检查用户问题是否超出 Agent 能力范围
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from openai import OpenAI


# ---------- Prompts ----------

HALLUCINATION_CHECK_PROMPT = """你是一个幻觉检测器。对照下面的工具返回数据，检查客服回复中是否存在虚构信息。

工具返回数据:
{tool_results}

客服回复:
{reply}

判断规则:
1. 回复中的订单号、金额、状态、商品名等具体数值必须与工具返回一致
2. 回复中的政策引用（如"根据XX政策"）必须可在工具返回中找到依据
3. 回复中的流程描述（如"1-3个工作日"）必须与工具返回一致，不能自己推断
4. 回复中可以适当礼貌用语和组织语言，但不能添加不存在的事实
5. 如果工具返回为空，回复中不应出现具体的订单号、金额、物流状态等数据，但礼貌性问候和通用客服话术不算幻觉

输出 JSON:
{{"has_hallucination": true/false, "hallucinations": ["具体虚构内容1", ...], "confidence": 0.0-1.0}}"""


SELF_CORRECT_PROMPT = """你之前的回复中存在以下虚构信息，需要修正:

{hallucinations}

请基于工具返回的真实数据重新生成回复。只使用工具返回中存在的具体数据，不确定的地方请用户自行确认。

工具返回数据:
{tool_results}

用户问题: {user_input}

请生成修正后的回复:"""


QUERY_BOUNDARY_PROMPT = """你是一个查询边界检查器。判断用户问题是否在电商客服 Agent 的能力范围内。

能力范围:
- 订单查询、物流跟踪、退换货、退款
- 商品推荐、商品信息查询
- 平台政策咨询（会员、配送、售后等）
- 账户基本问题、优惠活动

超出范围:
- 医疗建议、法律咨询、投资理财
- 政治敏感话题、违法内容
- 与电商完全无关的问题

用户问题: {user_input}

输出 JSON:
{{"in_scope": true/false, "reason": "简短说明"}}"""


# ---------- Data Structures ----------

@dataclass
class HallucinationResult:
    has_hallucination: bool
    hallucinations: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class BoundaryResult:
    in_scope: bool
    reason: str = ""


# ---------- Guardrails ----------

class Guardrails:
    """护栏系统：幻觉检测 + 自纠错 + 双向兜底 + 查询边界。"""

    def __init__(self, client: OpenAI, model: str, enabled: bool = True):
        self.client = client
        self.model = model
        self.enabled = enabled
        self.max_correction_attempts = 2

    # ---- 幻觉检测 ----

    def detect_hallucination(
        self, reply: str, tool_results: list[dict],
    ) -> HallucinationResult:
        """检测回复中是否存在幻觉。无工具调用时跳过检测。"""
        if not self.enabled:
            return HallucinationResult(has_hallucination=False)

        if not tool_results:
            return HallucinationResult(has_hallucination=False)

        if not reply or not reply.strip():
            return HallucinationResult(has_hallucination=False)

        results_text = "\n".join(
            f"[{r['name']}({json.dumps(r.get('arguments', {}), ensure_ascii=False)})] {r.get('result', '')}"
            for r in tool_results
        ) if tool_results else "（本次未调用任何工具）"

        prompt = HALLUCINATION_CHECK_PROMPT.format(
            tool_results=results_text,
            reply=reply,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
            )
            raw = (response.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            data = json.loads(raw)
            return HallucinationResult(
                has_hallucination=data.get("has_hallucination", False),
                hallucinations=data.get("hallucinations", []),
                confidence=data.get("confidence", 0.0),
            )
        except Exception as e:
            # 检测器异常时打日志，不再静默吞掉；回退为保守策略
            print(f"  [护栏警告] 幻觉检测器异常 ({type(e).__name__}: {e})，回退为 has_hallucination=False")
            return HallucinationResult(has_hallucination=False)

    # ---- 自纠错 ----

    def self_correct(
        self,
        original_reply: str,
        user_input: str,
        tool_results: list[dict],
        hallucinations: list[str],
    ) -> str:
        """检测到幻觉后，重新生成修正回复。"""
        results_text = "\n".join(
            f"[{r['name']}] {r.get('result', '')}" for r in tool_results
        ) if tool_results else "（无工具数据）"

        prompt = SELF_CORRECT_PROMPT.format(
            hallucinations="\n".join(f"- {h}" for h in hallucinations),
            tool_results=results_text,
            user_input=user_input,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return response.choices[0].message.content or original_reply
        except Exception:
            return original_reply

    def generate_with_guard(
        self,
        user_input: str,
        tool_results: list[dict],
        generate_fn,
    ) -> tuple[str, int]:
        """带护栏的回复生成：生成 → 幻觉检测 → 自纠错（最多 2 次）。

        Returns (reply, correction_attempts)。
        """
        if not self.enabled:
            return generate_fn(), 0

        reply = generate_fn()
        for attempt in range(self.max_correction_attempts):
            check = self.detect_hallucination(reply, tool_results)
            if not check.has_hallucination:
                return reply, attempt

            print(f"  [护栏] 检测到幻觉 (置信度={check.confidence:.0%})，自纠错第{attempt+1}次")
            reply = self.self_correct(
                original_reply=reply,
                user_input=user_input,
                tool_results=tool_results,
                hallucinations=check.hallucinations,
            )

        return reply, self.max_correction_attempts

    # ---- 查询边界 ----

    def check_boundary(self, user_input: str) -> BoundaryResult:
        """检查用户问题是否在 Agent 能力范围内。"""
        if not self.enabled:
            return BoundaryResult(in_scope=True)

        prompt = QUERY_BOUNDARY_PROMPT.format(user_input=user_input)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            raw = (response.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            data = json.loads(raw)
            return BoundaryResult(
                in_scope=data.get("in_scope", True),
                reason=data.get("reason", ""),
            )
        except Exception:
            return BoundaryResult(in_scope=True)

    # ---- 任务分解兜底 ----

    @staticmethod
    def task_fallback(intent: str) -> list[dict]:
        """规划节点失败时，回退到默认工具列表。"""
        fallback_map = {
            "presale": [{"name": "query_product", "arguments": {}}],
            "postsale": [{"name": "list_user_orders", "arguments": {}}],
            "complaint": [{"name": "query_order", "arguments": {"order_id": ""}}],
        }
        return fallback_map.get(intent, [{"name": "list_user_orders", "arguments": {}}])

    # ---- 工具执行兜底 ----

    @staticmethod
    def tool_fallback_chain(failed_tool: str) -> list[str]:
        """工具调用失败时的备用工具链。"""
        fallback_chains = {
            "query_order": ["list_user_orders"],
            "query_logistics": ["query_order"],
            "apply_refund": ["query_order", "search_knowledge"],
            "query_product": ["search_knowledge"],
        }
        return fallback_chains.get(failed_tool, ["list_user_orders"])