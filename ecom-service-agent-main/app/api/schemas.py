"""服务化 API 的请求/响应 schema（复用 Pydantic，不引入新依赖）。"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel

from app.schemas.response import CustomerServiceResponse


class ChatRequest(BaseModel):
    user_input: str
    thread_id: Optional[str] = None


class ResumeRequest(BaseModel):
    decision: str


class ErrorResponse(BaseModel):
    error: str
    error_type: str = "internal"
    trace_id: Optional[str] = None


def response_to_dict(resp: CustomerServiceResponse) -> dict:
    """把 CustomerServiceResponse 转成对外 JSON（不含内部思考/工具参数）。"""
    return {
        "intent": resp.intent.value,
        "confidence": resp.confidence,
        "reply": resp.reply,
        "requires_human": resp.requires_human,
        "follow_up_question": resp.follow_up_question,
        "citations": resp.citations,
    }


def error_to_dict(status: int, error_type: str, message: str, trace_id: Optional[str] = None) -> tuple[int, dict]:
    """统一错误 JSON 格式。"""
    body: dict[str, Any] = {"error": message, "error_type": error_type}
    if trace_id:
        body["trace_id"] = trace_id
    return status, body
