"""
Query Rewriter：用 LLM 改写多轮对话中的省略和指代。

在用户说 "那个政策是什么" 时，结合对话历史改写为 "退换货政策是什么"，
提升后续检索的命中率。
"""

from __future__ import annotations

import json

from openai import OpenAI


REWRITE_PROMPT = """你是一个查询改写助手。基于对话历史，将用户问题改写为独立、完整的检索查询。

对话历史:
{history}

用户当前问题: {query}

规则:
1. 如果用户使用了代词（"那个"、"它"、"这个"），替换为具体指代
2. 如果用户省略了上下文信息，从历史中补充
3. 如果用户问题已经完整独立，原样返回
4. 只输出一个 JSON:
{{"rewritten": "改写后的查询", "changed": true/false, "reason": "简短说明"}}"""


class QueryRewriter:
    """用 LLM 改写多轮对话中的查询。"""

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def rewrite(
        self, query: str, history: list[dict] | None = None,
        max_history: int = 4,
    ) -> tuple[str, bool]:
        """改写查询，返回 (rewritten_query, was_changed)。"""
        history_text = self._format_history(history, max_history) if history else ""

        if not history_text:
            return query, False

        prompt = REWRITE_PROMPT.format(history=history_text, query=query)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            raw = (response.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            data = json.loads(raw)
            rewritten = data.get("rewritten", query)
            changed = data.get("changed", False)
            return rewritten, changed
        except Exception:
            return query, False

    @staticmethod
    def _format_history(history: list[dict], max_turns: int) -> str:
        """格式化最近 N 轮对话。"""
        recent = [m for m in history[-max_turns * 2:]
                  if m.get("role") in ("user", "assistant")]
        lines = []
        for m in recent:
            role = "用户" if m["role"] == "user" else "客服"
            content = m.get("content", "")
            if content and len(content) < 300:
                lines.append(f"{role}: {content}")
        return "\n".join(lines) if lines else ""