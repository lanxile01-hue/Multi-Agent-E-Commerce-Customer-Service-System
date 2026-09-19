"""Supervisor / SubGraph 共享状态模型。

字段归属（第三阶段澄清，用于隔离父图与子图状态）：

父图专属字段（由父图节点读写，子图只读）：
- user_input / selected_agent / intent / conversation / messages / metadata
- final_response / structured_response / requires_human / blocked / error
- confirmation_payload / confirmation_result / confirmation_time（人工确认记录）

子图专属字段（子图入口 agent_enter 会重置，避免跨请求污染）：
- iteration / pending_tool_calls / tool_results / is_complete
- waiting_for_confirmation

说明：当前 LangGraph 版本下三个子图作为父图节点运行时共享 `SupervisorState`，
未做独立的子图 schema 映射（会显著增加复杂度）。为明确边界，这里定义
`SubAgentState` 作为子图私有状态的目标形态（文档用），子图入口 `agent_enter`
已显式重置所有子图临时字段，保证：
1. presale 的 tool_results 不污染 postsale；
2. 多轮调用不复用上一次 pending_tool_calls；
3. 中断恢复时新请求不会覆盖旧请求的临时状态；
4. 同一父图并行调用多个子图时字段不冲突（父图/子图字段已分离）。
"""

from __future__ import annotations

from typing import TypedDict


class SubAgentState(TypedDict, total=False):
    """子图私有状态的目标形态（文档约定）。

    三个子图共用同一套节点逻辑，理论上各自应持有独立状态；当前实现为
    避免复杂的子图 schema 映射，仍复用 SupervisorState 并靠入口重置隔离。
    """

    user_input: str
    conversation: list[dict]
    iteration: int
    pending_tool_calls: list[dict]
    tool_results: list[dict]
    final_response: str
    structured_response: dict
    confirmation_result: str
    is_complete: bool
    error: str


class SupervisorState(TypedDict, total=False):
    """父图与子图共享的全局状态。

    字段说明（标注归属）：
    - messages            : [父图] 构建的完整上下文（system + memory + summary + conversation）
    - conversation        : [父图] 仅含 user/assistant/tool 的原始对话历史（子图据此重建领域上下文）
    - user_input          : [父图] 当前轮用户输入
    - intent              : [父图] 细粒度意图（可选，路由主要产出 selected_agent）
    - selected_agent      : [父图] 路由选中的子 Agent（presale / postsale / complaint）
    - iteration           : [子图] 内部循环计数
    - pending_tool_calls  : [子图] 待执行的工具调用（[{"name": str, "arguments": dict}, ...]）
    - tool_results        : [子图] 已执行的工具结果（[{"name","arguments","result","ok","error"}, ...]）
    - final_response      : [父图] 最终回复文本
    - structured_response : [父图] CustomerServiceResponse 序列化 dict
    - requires_human      : [父图] 是否转人工
    - waiting_for_confirmation : [子图] 是否处于等待用户确认状态
    - confirmation_payload    : [父图] interrupt 暂停时的上下文（含确认结果/时间/工具名/参数）
    - confirmation_result     : [父图] 用户确认结果（confirmed / cancelled / re_explain）
    - confirmation_time       : [父图] 用户做出确认决定的时间戳
    - is_complete         : [子图] evaluate_result 写入，判断信息是否足够
    - error               : [父图] 顶层错误信息（非致命错误兜底）
    - blocked             : [父图] preprocess 判定越界，直接跳过 Router/子图
    - metadata            : [父图] 统计信息（trace_id/thread_id/工具调用次数/模型调用次数/耗时等）
    """

    messages: list[dict]
    conversation: list[dict]
    user_input: str
    intent: str
    selected_agent: str
    iteration: int
    pending_tool_calls: list[dict]
    tool_results: list[dict]
    final_response: str
    structured_response: dict
    requires_human: bool
    waiting_for_confirmation: bool
    confirmation_payload: dict
    confirmation_result: str
    confirmation_time: float
    is_complete: bool
    error: str
    blocked: bool
    metadata: dict


def empty_state() -> SupervisorState:
    """返回一个字段齐全的初始状态，避免 TypedDict 缺 key 触发序列化/访问问题。"""
    return {
        "messages": [],
        "conversation": [],
        "user_input": "",
        "intent": "",
        "selected_agent": "",
        "iteration": 0,
        "pending_tool_calls": [],
        "tool_results": [],
        "final_response": "",
        "structured_response": {},
        "requires_human": False,
        "waiting_for_confirmation": False,
        "confirmation_payload": {},
        "confirmation_result": "",
        "confirmation_time": 0.0,
        "is_complete": False,
        "error": "",
        "blocked": False,
        "metadata": {},
    }


# 确认结果归一化：用户可能输入中文/英文/拼音等
_CONFIRM_WORDS = {
    "确认": "confirmed", "确定": "confirmed", "同意": "confirmed", "好的": "confirmed",
    "可以": "confirmed", "confirm": "confirmed", "ok": "confirmed", "yes": "confirmed",
    "y": "confirmed", "是": "confirmed", "行": "confirmed", "好": "confirmed",
    "提交": "confirmed", "申请": "confirmed",
}
_CANCEL_WORDS = {
    "取消": "cancelled", "不要": "cancelled", "不用": "cancelled", "算了": "cancelled",
    "cancel": "cancelled", "no": "cancelled", "n": "cancelled", "否": "cancelled",
    "拒绝": "cancelled", "不用了": "cancelled",
}
_RE_EXPLAIN_WORDS = {
    "重新说明": "re_explain", "重新输入": "re_explain", "再说一次": "re_explain",
    "重新描述": "re_explain", "改一下": "re_explain", "不是这个": "re_explain",
}


def normalize_confirmation(user_input: str) -> str:
    """把用户的确认输入归一化为 confirmed / cancelled / re_explain。

    无法识别时默认视为 re_explain（把新输入当成补充说明重新进入推理），
    而不是默认确认，避免误触发退款等敏感操作。
    """
    text = (user_input or "").strip()
    low = text.lower()
    for k, v in _CONFIRM_WORDS.items():
        if low == k or low.startswith(k):
            return v
    for k, v in _CANCEL_WORDS.items():
        if low == k or low.startswith(k):
            return v
    for k, v in _RE_EXPLAIN_WORDS.items():
        if k in low:
            return v
    return "re_explain"