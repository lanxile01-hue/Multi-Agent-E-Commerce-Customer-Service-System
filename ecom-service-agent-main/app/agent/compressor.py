"""
Token 预算管理与 3 层渐进式压缩模块。

策略:
- 128k Token 预算（可配置）
- 三层压缩流水线：
  L1 (50%-70%): 轻量截断 — 工具结果压缩到 500 字
  L2 (70%-85%): 中度压缩 — 工具结果压缩到 200 字，移除旧 tool 消息
  L3 (>85%):    自动压缩 — LLM 摘要最老的 N 条消息（增量）
- 滑动窗口: 始终保留最近 `keep_recent` 条消息原文
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import tiktoken


class CompressionLevel(Enum):
    NONE = 0       # < 50%
    LIGHT = 1      # 50-70%: 工具结果截断
    MEDIUM = 2     # 70-85%: 工具结果深度截断 + 移除旧 tool 消息
    AGGRESSIVE = 3 # >85%: LLM 摘要压缩


@dataclass
class CompressionResult:
    level: CompressionLevel
    messages: list[dict]
    summary: str | None
    tokens_before: int
    tokens_after: int
    actions: list[str]


class TokenCounter:
    """Token 计数器，基于 tiktoken。"""

    _enc: tiktoken.Encoding | None = None

    @classmethod
    def _get_enc(cls) -> tiktoken.Encoding:
        if cls._enc is None:
            cls._enc = tiktoken.get_encoding("cl100k_base")
        return cls._enc

    @classmethod
    def count(cls, text: str) -> int:
        return len(cls._get_enc().encode(text))

    @classmethod
    def count_messages(cls, messages: list[dict]) -> int:
        """估算消息列表的 token 数。"""
        total = 0
        for msg in messages:
            content = msg.get("content", "") or ""
            total += cls.count(content)
            # tool_calls 额外开销
            for tc in msg.get("tool_calls", []):
                args = tc.get("function", {}).get("arguments", "")
                total += cls.count(args) if args else 0
        return total


class ContextCompressor:
    """上下文压缩器：3 层渐进式压缩。"""

    def __init__(
        self,
        token_budget: int = 128_000,
        keep_recent: int = 6,
        summarizer_fn: Callable | None = None,
    ):
        self.token_budget = token_budget
        self.keep_recent = keep_recent
        self._summarizer = summarizer_fn

    def compress(
        self,
        messages: list[dict],
        current_summary: str | None = None,
    ) -> CompressionResult:
        """对消息列表应用渐进式压缩，返回压缩结果。"""
        tokens = TokenCounter.count_messages(messages)
        utilization = tokens / self.token_budget if self.token_budget else 0

        level = self._determine_level(utilization)
        actions: list[str] = []

        if level == CompressionLevel.NONE:
            return CompressionResult(
                level=level, messages=messages, summary=current_summary,
                tokens_before=tokens, tokens_after=tokens, actions=["无需压缩"],
            )

        compressed = list(messages)
        new_summary = current_summary

        # L1: 轻量截断工具结果
        if level.value >= CompressionLevel.LIGHT.value:
            compressed, l1_actions = self._light_compress(compressed)
            actions.extend(l1_actions)

        # L2: 深度截断 + 移除旧 tool 消息
        if level.value >= CompressionLevel.MEDIUM.value:
            compressed, l2_actions = self._medium_compress(compressed)
            actions.extend(l2_actions)

        # L3: LLM 摘要压缩
        if level.value >= CompressionLevel.AGGRESSIVE.value:
            compressed, new_summary, l3_actions = self._aggressive_compress(
                compressed, current_summary,
            )
            actions.extend(l3_actions)

        tokens_after = TokenCounter.count_messages(compressed)
        return CompressionResult(
            level=level, messages=compressed, summary=new_summary,
            tokens_before=tokens, tokens_after=tokens_after, actions=actions,
        )

    def _determine_level(self, utilization: float) -> CompressionLevel:
        if utilization < 0.50:
            return CompressionLevel.NONE
        if utilization < 0.70:
            return CompressionLevel.LIGHT
        if utilization < 0.85:
            return CompressionLevel.MEDIUM
        return CompressionLevel.AGGRESSIVE

    def _light_compress(self, messages: list[dict]) -> tuple[list[dict], list[str]]:
        """L1: 工具结果截断到 500 字。"""
        actions = []
        compressed = []
        for msg in messages:
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if len(content) > 500:
                    compressed.append({**msg, "content": content[:500] + "..."})
                    actions.append("工具结果截断(>500字)")
                else:
                    compressed.append(msg)
            else:
                compressed.append(msg)
        return compressed, actions

    def _medium_compress(self, messages: list[dict]) -> tuple[list[dict], list[str]]:
        """L2: 工具结果截断到 200 字，移除旧 tool 消息（保留最近 6 条 tool）。"""
        actions = []
        # 找到 tool 消息索引
        tool_indices = [
            i for i, m in enumerate(messages) if m.get("role") == "tool"
        ]
        keep_tool_count = 6
        remove_tool_indices = set(tool_indices[:-keep_tool_count]) if len(tool_indices) > keep_tool_count else set()

        compressed = []
        for i, msg in enumerate(messages):
            if i in remove_tool_indices:
                actions.append("移除旧tool消息")
                continue
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if len(content) > 200:
                    compressed.append({**msg, "content": content[:200] + "..."})
                    actions.append("工具结果深度截断(>200字)")
                else:
                    compressed.append(msg)
            else:
                compressed.append(msg)
        return compressed, actions

    def _aggressive_compress(
        self,
        messages: list[dict],
        prev_summary: str | None,
    ) -> tuple[list[dict], str | None, list[str]]:
        """L3: LLM 摘要压缩最老的 N 条消息。"""
        if not self._summarizer:
            return messages, prev_summary, ["无摘要器，跳过L3"]

        split = len(messages) - self.keep_recent
        while split > 0 and messages[split].get("role") in ("tool",):
            split -= 1
        if split <= 0:
            return messages, prev_summary, ["消息不足，跳过L3"]

        old_messages = messages[:split]
        recent = messages[split:]

        new_summary = self._summarizer(
            old_messages=old_messages,
            prev_summary=prev_summary,
        )
        return recent, new_summary, [f"LLM摘要压缩 {len(old_messages)} 条消息"]


def create_compressor(client, model) -> ContextCompressor:
    """创建带 LLM 摘要能力的压缩器。"""
    from app.agent.summarizer import summarize

    def summarizer_fn(old_messages, prev_summary):
        return summarize(
            client=client,
            model=model,
            old_messages=old_messages,
            prev_summary=prev_summary,
        )

    return ContextCompressor(
        token_budget=128_000,
        keep_recent=6,
        summarizer_fn=summarizer_fn,
    )