"""工具风险分级：区分只读工具与需要人工确认的高风险写操作。

风险级别：
- read_only            : 只读查询，无副作用，失败可安全重试。
- requires_confirmation: 高风险写操作，非幂等，禁止自动重试，必须先经人工确认。

当前实现边界（重要）：
- 项目只实现了 `apply_refund` 这一个高风险写工具。
- `compensation`、`complaint_escalation` 尚未实现（不在 registry 中），
  此处仅为未来工具预留风险策略，**不虚构实现**。当它们被实现并注册时，
  应把它们加入 TOOL_RISK_LEVELS 并同样走 requires_confirmation 流程。
"""

READ_ONLY = "read_only"
REQUIRES_CONFIRMATION = "requires_confirmation"

TOOL_RISK_LEVELS: dict[str, str] = {
    # 只读查询（无副作用）
    "query_product": READ_ONLY,
    "query_order": READ_ONLY,
    "query_logistics": READ_ONLY,
    "search_knowledge": READ_ONLY,
    "list_user_orders": READ_ONLY,
    "recall_user_memory": READ_ONLY,
    "load_skill": READ_ONLY,
    # 高风险写操作（需人工确认 + 幂等键，禁止自动重试）
    "apply_refund": REQUIRES_CONFIRMATION,
    # 未来预留（当前未实现）：
    # "compensation": REQUIRES_CONFIRMATION,
    # "complaint_escalation": REQUIRES_CONFIRMATION,
}


def risk_level(tool_name: str) -> str:
    """返回工具的风险级别；未知工具默认按只读处理（会先被参数校验拒绝）。"""
    return TOOL_RISK_LEVELS.get(tool_name, READ_ONLY)


def requires_confirmation(tool_name: str) -> bool:
    """是否需要人工确认。"""
    return risk_level(tool_name) == REQUIRES_CONFIRMATION


def is_read_only(tool_name: str) -> bool:
    return risk_level(tool_name) == READ_ONLY


def is_non_idempotent(tool_name: str) -> bool:
    """是否为非幂等写操作（禁止自动重试）。"""
    return risk_level(tool_name) == REQUIRES_CONFIRMATION