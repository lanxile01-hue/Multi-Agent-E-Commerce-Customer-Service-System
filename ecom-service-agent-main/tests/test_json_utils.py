"""app/agent/json_utils 的稳健 JSON 提取专项测试（纯函数，无 Mock）。

extract_json_object / strip_non_json_syntax 被 Guardrails、Router、Rewriter、
Evaluator 等所有「让模型输出 JSON」的链路共享，解析失败会静默退化成兜底
策略，因此这里对四类常见噪音逐一锁定行为：markdown 代码块、// 注释、
尾随逗号、字符串字面量内的 URL（不能被当成注释误删）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.json_utils import extract_json_object, strip_non_json_syntax  # noqa: E402


def test_extract_from_markdown_fence():
    """模型爱用 ```json 代码块包裹输出，且前后常有解释文字。"""
    raw = '好的，结果如下：\n```json\n{"strategy": "hybrid"}\n```\n以上。'
    assert extract_json_object(raw) == {"strategy": "hybrid"}


def test_extract_ignores_line_comments_and_trailing_commas():
    """// 注释与尾随逗号都不是合法 JSON，但模型常模仿 prompt 示例输出。"""
    raw = """
    {
      // 先看订单号
      "in_scope": true,
      "reason": "订单查询", // 尾随注释
    }
    """
    assert extract_json_object(raw) == {"in_scope": True, "reason": "订单查询"}


def test_strip_keeps_url_inside_string_literal():
    """https:// 里的 // 不是注释，误删会把 URL 拦腰截断。"""
    text = '{"base": "https://api.deepseek.com/v1"}'
    assert strip_non_json_syntax(text) == text


def test_extract_raises_when_no_json_object():
    """完全不含 JSON 对象时必须抛错，让调用方走兜底而不是静默成功。"""
    with pytest.raises(ValueError):  # json.JSONDecodeError 是 ValueError 子类
        extract_json_object("this is not json at all")
