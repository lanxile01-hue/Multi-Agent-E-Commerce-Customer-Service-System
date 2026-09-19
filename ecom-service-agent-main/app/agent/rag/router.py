"""
Retrieval Router：用 LLM 判断查询适合哪种检索策略。

策略:
- vector: 适合语义理解类问题（"怎么退货"、"会员有什么权益"）
- keyword: 适合精确匹配类问题（"ORD-20240115-001"、"七天无理由"）
- hybrid:  混合检索，兼顾两者（默认策略）
"""

from __future__ import annotations

import json

from openai import OpenAI


ROUTER_PROMPT = """你是一个检索策略路由器。根据用户问题，判断应该使用哪种检索策略。

策略说明:
- vector: 语义向量检索。适合需要理解含义的问题（政策说明、流程指引、概念解释）
- keyword: 关键词精确匹配。适合包含具体编号、代码、专有名词的问题
- hybrid: 混合检索（向量 + 关键词）。适合大多数通用问题

用户问题: {query}

输出 JSON:
{{"strategy": "vector" | "keyword" | "hybrid", "reason": "简述理由"}}

判断规则:
1. 问题包含订单号、具体数字、SKU 编号时 → keyword
2. 问题是"怎么"、"如何"、"什么是"类型的语义问题 → vector
3. 一般性问题 → hybrid"""


class RetrievalRouter:
    """用 LLM 判断检索策略。"""

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def route(self, query: str) -> tuple[str, str]:
        """返回 (strategy, reason)。
        strategy ∈ {"vector", "keyword", "hybrid"}
        """
        prompt = ROUTER_PROMPT.format(query=query)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            raw = (response.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            data = json.loads(raw)
            strategy = data.get("strategy", "hybrid")
            reason = data.get("reason", "")
            if strategy not in ("vector", "keyword", "hybrid"):
                strategy = "hybrid"
            return strategy, reason
        except Exception:
            return "hybrid", "默认策略"