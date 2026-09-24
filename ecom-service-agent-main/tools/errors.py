"""工具错误分类、统一结果格式与参数校验。

供 LangGraph 单 Agent（app/agent/graph.py）与 Multi-Agent 子图（app/multi_agent/）
复用，避免两处各自维护一套脆弱的重试/失败判断逻辑。

错误分类（classify_tool_error）：
- ok          : 调用成功
- retryable   : 可重试（网络/超时/临时故障）
- fatal       : 不可重试（业务失败/未知错误/未知状态）
- param_error : 参数错误（缺参、类型错误、非法取值）
- not_found   : 目标资源不存在（如订单号查不到）
- unknown_tool: 未知工具（安全拒绝，不执行）

统一工具结果格式（standard_result，推荐给写工具使用）：
{
    "ok": True/False,
    "status": "success" | "error" | "unknown",
    "error_type": None | "not_found" | "param_error" | "retryable" | "timeout" |
                  "forbidden" | "fatal" | "unknown" | "unknown_tool",
    "message": "...",
    "request_id": "...",
    "idempotency_key": "...",
    "retryable": False,
}
"""

from __future__ import annotations

import json
from typing import Any

# 分类常量
TOOL_OK = "ok"
TOOL_RETRYABLE = "retryable"
TOOL_FATAL = "fatal"
TOOL_PARAM_ERROR = "param_error"
TOOL_NOT_FOUND = "not_found"
TOOL_UNKNOWN = "unknown_tool"

# 可重试错误关键词（网络/超时/限流/服务暂不可用）
_RETRYABLE_KEYWORDS = (
    "timeout", "timed out", "超时", "连接失败", "connection",
    "network", "网络", "rate limit", "限流", "too many",
    "temporarily", "临时", "service unavailable", "服务不可用",
    "retry", "重试", "500", "502", "503", "504",
)

# 参数错误关键词
_PARAM_KEYWORDS = (
    "缺少", "缺失", "必填", "required", "参数", "argument",
    "invalid", "非法", "类型错误", "must be", "expect",
)

# 资源不存在关键词
_NOT_FOUND_KEYWORDS = (
    "未找到", "不存在", "not found", "no such", "没有找到",
    "无法找到", "查不到",
)

# 未知工具关键词
_UNKNOWN_KEYWORDS = ("未知工具", "unknown tool", "no such tool", "不存在该工具")

# 写操作工具（非幂等，禁止自动重试，避免重复执行退款等）。
# 注意：风险分级请以 tools/risks.py 为准，这里保留一份便于旧代码兼容。
WRITE_TOOLS = {"apply_refund"}


def is_write_tool(name: str) -> bool:
    """是否为写操作工具（如退款），写操作默认禁止自动重试。"""
    return name in WRITE_TOOLS


def _lower(text: str) -> str:
    return (text or "").lower()


def standard_result(
    status: str,
    message: str,
    *,
    error_type: str | None = None,
    request_id: str | None = None,
    idempotency_key: str | None = None,
    retryable: bool = False,
    **extra: Any,
) -> dict:
    """构造统一工具结果。status ∈ {"success", "error", "unknown"}。"""
    return {
        "ok": status == "success",
        "status": status,
        "error_type": error_type,
        "message": message,
        "request_id": request_id,
        "idempotency_key": idempotency_key,
        "retryable": retryable,
        **extra,
    }


def classify_tool_error(result_str: str) -> str:
    """根据工具返回的 JSON 字符串，分类执行结果。

    兼容两种格式：
    1. 统一格式（含 status / error_type / retryable 字段）。
    2. 旧格式（含 success / error 字段）。

    返回 TOOL_OK / TOOL_RETRYABLE / TOOL_FATAL / TOOL_PARAM_ERROR /
    TOOL_NOT_FOUND / TOOL_UNKNOWN 之一。
    """
    text = result_str or ""
    lower = _lower(text)

    data: dict[str, Any] | None = None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            data = parsed
    except (json.JSONDecodeError, TypeError):
        data = None

    if data is not None:
        # 统一格式
        if "status" in data:
            if data.get("status") == "success":
                return TOOL_OK
            err_type = _lower(str(data.get("error_type") or ""))
            if err_type:
                return _classify_error_type(err_type)
            return _classify_message(_lower(str(data.get("message") or "")))

        # 旧格式
        if data.get("success") is True:
            return TOOL_OK
        err = _lower(str(data.get("error", "")))
        if err:
            return _classify_message(err)

    # 非结构化输出：直接对全文做关键词分类
    return _classify_message(lower)


def _classify_error_type(error_type: str) -> str:
    """把统一格式的 error_type 映射到 TOOL_* 分类。"""
    mapping = {
        "not_found": TOOL_NOT_FOUND,
        "param_error": TOOL_PARAM_ERROR,
        "unknown_tool": TOOL_UNKNOWN,
        "retryable": TOOL_RETRYABLE,
        "timeout": TOOL_RETRYABLE,  # 超时可重试，但非幂等工具由调用方禁止重试
        "forbidden": TOOL_FATAL,
        "fatal": TOOL_FATAL,
        "unknown": TOOL_FATAL,      # 状态未知，不自动重试
    }
    if error_type in mapping:
        return mapping[error_type]
    return _classify_message(error_type)


def _classify_message(lower: str) -> str:
    if any(k in lower for k in _UNKNOWN_KEYWORDS):
        return TOOL_UNKNOWN
    if any(k in lower for k in _NOT_FOUND_KEYWORDS):
        return TOOL_NOT_FOUND
    if any(k in lower for k in _PARAM_KEYWORDS):
        return TOOL_PARAM_ERROR
    if any(k in lower for k in _RETRYABLE_KEYWORDS):
        return TOOL_RETRYABLE
    return TOOL_FATAL


def validate_tool_arguments(
    name: str,
    arguments: dict,
    tool_definitions: list[dict],
) -> tuple[bool, str]:
    """校验工具参数。

    返回 (是否合法, 错误信息)。错误信息为空表示合法。
    - 未知工具直接拒绝。
    - 缺少 required 参数拒绝。
    - 参数非 dict 拒绝。
    """
    definition = None
    for td in tool_definitions:
        if td.get("function", {}).get("name") == name:
            definition = td
            break

    if definition is None:
        return False, f"未知工具: {name}"

    if not isinstance(arguments, dict):
        return False, f"工具 {name} 的参数必须是 JSON 对象"

    params = definition.get("function", {}).get("parameters", {})
    required = set(params.get("required", []) or [])

    missing = [p for p in required if p not in arguments or arguments[p] in (None, "")]
    if missing:
        return False, f"工具 {name} 缺少必填参数: {', '.join(missing)}"

    return True, ""


def is_retryable(result_str: str) -> bool:
    """是否为可重试错误。"""
    return classify_tool_error(result_str) == TOOL_RETRYABLE


class AgentError(Exception):
    """统一 Agent 异常：携带 error_type / retryable / 对用户的安全话术。

    - error_type  : 与工具结果统一格式的 error_type 对齐（not_found / retryable /
                    param_error / fatal / timeout / unknown 等）
    - retryable   : 是否可安全重试（非幂等写操作恒为 False）
    - user_message: 可直接展示给用户的脱敏话术
    """

    def __init__(
        self,
        message: str,
        *,
        error_type: str = "fatal",
        retryable: bool = False,
        user_message: str = "抱歉，服务暂时不可用，请稍后再试。",
    ):
        super().__init__(message)
        self.error_type = error_type
        self.retryable = retryable
        self.user_message = user_message