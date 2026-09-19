"""统一结构化输出：把子图最终回复收敛为 CustomerServiceResponse。

设计目标（对应需求七）：
1. 优先直接输出 CustomerServiceResponse（response_format / beta.parse）。
2. 不支持的模型走稳定 fallback：纯文本生成 + 确定性字段映射，**不再二次调用 LLM 提取**。
3. 任何路径都校验 JSON / Pydantic schema，缺失字段给安全默认值。
4. 内部思考与工具调用信息绝不返回给用户。
"""

from __future__ import annotations

from typing import Any, Optional

from app.schemas.response import CustomerServiceResponse, IntentType

# 子 Agent 标识 → 客服意图类型（确定性映射，供 fallback 使用）
AGENT_TO_INTENT: dict[str, IntentType] = {
    "presale": IntentType.PRODUCT_CONSULT,
    "postsale": IntentType.AFTER_SALE,
    "complaint": IntentType.COMPLAINT,
}

# 转人工的启发式：投诉场景默认倾向转人工；售后/售前默认不转
AGENT_REQUIRES_HUMAN: dict[str, bool] = {
    "presale": False,
    "postsale": False,
    "complaint": True,
}


def normalize_intent(value: Any, selected_agent: str = "") -> IntentType:
    """把任意输入安全地归一化为 IntentType。"""
    if isinstance(value, IntentType):
        return value
    if isinstance(value, str):
        # 先尝试按枚举值匹配
        try:
            return IntentType(value)
        except ValueError:
            pass
        # 再尝试按子 Agent 标识匹配
        if value in AGENT_TO_INTENT:
            return AGENT_TO_INTENT[value]
    if selected_agent in AGENT_TO_INTENT:
        return AGENT_TO_INTENT[selected_agent]
    return IntentType.OTHER


def coerce_confidence(value: Any) -> float:
    """把 confidence 安全地收敛到 [0, 1]。"""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, f))


def build_structured_response(
    reply: str,
    selected_agent: str,
    intent: Any = None,
    confidence: float = 0.85,
    requires_human: Optional[bool] = None,
    follow_up_question: Optional[str] = None,
) -> CustomerServiceResponse:
    """确定性构造 CustomerServiceResponse（不依赖第二次 LLM 调用）。"""
    safe_reply = reply or "抱歉，我暂时无法处理您的请求。"
    if requires_human is None:
        requires_human = AGENT_REQUIRES_HUMAN.get(selected_agent, False)
    return CustomerServiceResponse(
        intent=normalize_intent(intent, selected_agent),
        confidence=coerce_confidence(confidence),
        reply=safe_reply,
        requires_human=bool(requires_human),
        follow_up_question=follow_up_question,
    )


def safe_default_response(
    reply: str,
    selected_agent: str = "",
    requires_human: bool = True,
) -> CustomerServiceResponse:
    """最终兜底：任何解析失败都落到这里，保证不抛异常。"""
    return build_structured_response(
        reply=reply or "抱歉，我暂时无法处理您的请求。",
        selected_agent=selected_agent,
        confidence=0.0,
        requires_human=requires_human,
    )


def coerce_structured(
    data: Any,
    selected_agent: str,
    fallback_reply: str,
) -> CustomerServiceResponse:
    """把任意来源的解析结果（dict / CustomerServiceResponse / None）安全转换为标准结构。

    - 校验 JSON / Pydantic schema
    - 缺失字段给安全默认值
    - reply 为空时回退到 fallback_reply，绝不为空
    """
    if isinstance(data, CustomerServiceResponse):
        obj = data
    elif isinstance(data, dict):
        try:
            obj = CustomerServiceResponse.model_validate(data)
        except Exception:
            obj = None
    else:
        obj = None

    if obj is None:
        return build_structured_response(
            reply=fallback_reply,
            selected_agent=selected_agent,
        )

    reply = obj.reply if obj.reply and obj.reply.strip() else fallback_reply
    if reply is None:
        reply = fallback_reply

    return CustomerServiceResponse(
        intent=obj.intent if isinstance(obj.intent, IntentType) else normalize_intent(obj.intent, selected_agent),
        confidence=coerce_confidence(obj.confidence),
        reply=reply,
        requires_human=bool(obj.requires_human),
        follow_up_question=obj.follow_up_question,
        citations=obj.citations,
    )


def serialize_response(resp: CustomerServiceResponse) -> dict:
    """转为可序列化的 dict，供写入图状态（MemorySaver 需要 JSON 可序列化）。"""
    return {
        "intent": resp.intent.value,
        "confidence": resp.confidence,
        "reply": resp.reply,
        "requires_human": resp.requires_human,
        "follow_up_question": resp.follow_up_question,
        "citations": resp.citations,
    }


def deserialize_response(data: dict) -> CustomerServiceResponse:
    """从图状态中的 dict 还原 CustomerServiceResponse。"""
    return coerce_structured(data, selected_agent="", fallback_reply="")
