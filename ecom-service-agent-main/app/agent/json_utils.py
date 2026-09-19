"""LLM 输出的 JSON 提取工具。

LLM 的输出经常不是合法 JSON（markdown 代码块包裹、// 注释、解释文字、尾随逗号），
本模块提供稳健的提取函数，供各 Agent 与评测脚本复用。
"""

from __future__ import annotations

import json
import re


def strip_non_json_syntax(text: str) -> str:
    """删除 JSON 标准不允许、但 LLM 常输出的语法：// 行注释、尾随逗号。

    只处理字符串字面量之外的位置，避免误伤 URL（https://...）。
    """
    out: list[str] = []
    in_string = False
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:      # 转义字符原样保留
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":   # 跳过 // 到行尾
                i += 1
            continue
        out.append(ch)
        i += 1
    cleaned = "".join(out)
    return re.sub(r",(\s*[}\]])", r"\1", cleaned)   # 删除尾随逗号


def extract_json_object(raw: str) -> dict:
    """从 LLM 输出中稳健提取 JSON 对象，容忍三类常见噪音：

    1. markdown 代码块包裹（前后可有解释文字）
    2. // 注释（模型模仿 prompt 示例输出的非法语法）
    3. 正文中混入的解释文字
    """
    text = raw.strip()
    # 剥掉 markdown 代码块（```json ... ``` 或 ``` ... ```）
    fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    # 删除注释和尾随逗号后，截取第一个 { 到最后一个 } 的块
    cleaned = strip_non_json_syntax(text)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        return json.loads(cleaned[start : end + 1])
    raise json.JSONDecodeError("未找到 JSON 对象", cleaned, 0)