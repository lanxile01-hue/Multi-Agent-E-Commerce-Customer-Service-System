"""可观测性辅助：敏感信息脱敏与日志字段安全化。

规则（第六阶段）：
1. 不记录 API Key。
2. 不记录完整敏感个人信息。
3. 订单号、手机号、地址等敏感字段脱敏。
"""

from __future__ import annotations

import re

# 订单号：ORD-20240115-001 / ORD-XXXX → 保留前缀 + 掩码
_ORDER_PATTERN = re.compile(r"\b(ORD-[A-Za-z0-9]{2,})[A-Za-z0-9-]*\b")

# 手机号：11 位，保留前 3 后 4
_PHONE_PATTERN = re.compile(r"\b(1[3-9]\d)\d{4}(\d{4})\b")

# 运单号/快递单号（顺丰 SF / 京东 JD / 圆通 YT 等大写字母+数字）
_TRACKING_PATTERN = re.compile(r"\b([A-Z]{2}\d{8,})\b")


def mask_order_id(text: str) -> str:
    """订单号脱敏：ORD-20240115-001 → ORD-***。"""
    return _ORDER_PATTERN.sub(lambda m: f"{m.group(1).split('-')[0]}-***", text)


def mask_phone(text: str) -> str:
    """手机号脱敏：13812345678 → 138****5678。"""
    return _PHONE_PATTERN.sub(r"\1****\2", text)


def mask_tracking_number(text: str) -> str:
    """运单号脱敏：SF1234567890 → SF***。"""
    return _TRACKING_PATTERN.sub(lambda m: f"{m.group(1)[:2]}***", text)


def mask_sensitive(text: str) -> str:
    """对文本中的订单号、手机号、运单号统一脱敏。"""
    if not text:
        return text
    text = mask_order_id(text)
    text = mask_phone(text)
    text = mask_tracking_number(text)
    return text


def sanitize_record(record: dict) -> dict:
    """把日志记录中的敏感字段脱敏，返回新 dict（不修改原对象）。

    - 明确不记录 API Key（若出现则剔除）。
    - 对 message / reply / user_input / tool result 等字符串做脱敏。
    """
    safe = {}
    for key, value in record.items():
        if key.lower() in {"api_key", "openai_api_key", "authorization", "token", "langchain_api_key"}:
            continue  # 不记录任何密钥
        if isinstance(value, str):
            safe[key] = mask_sensitive(value)
        else:
            safe[key] = value
    return safe