"""意图路由器：分析用户消息，决定分发给哪个子 Agent。"""

from typing import List, Optional

from openai import OpenAI

from app.prompts.agents import ROUTER_PROMPT

VALID_AGENTS = {"presale", "postsale", "complaint"}
CHINESE_MAP = {
    "售前": "presale", "商品": "presale", "推荐": "presale", "咨询": "presale",
    "售后": "postsale", "订单": "postsale", "物流": "postsale", "退款": "postsale", "退货": "postsale",
    "投诉": "complaint", "不满": "complaint", "愤怒": "complaint", "维权": "complaint",
}
DEFAULT_AGENT = "postsale"

# 投诉关键词预检：强信号词直接路由到 complaint，减少 LLM 调用
_COMPLAINT_KEYWORDS = [
    "投诉", "举报", "差评", "假货", "骗子", "诈骗", "欺骗",
    "12315", "消费者协会", "工商", "报警", "维权",
    "空包裹", "货不对板", "太差了", "太坑", "太过分",
    "一直不回复", "一直拖着", "再也不买", "赔偿", "道歉",
]


class Router:
    """使用 LLM 对用户意图分类，路由到对应的子 Agent。"""

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def route(self, user_input: str, history=None) -> str:
        """返回子 Agent 标识。关键词预检 → LLM 分类。"""
        # 关键词预检：投诉类强信号
        lower = user_input.lower()
        for kw in _COMPLAINT_KEYWORDS:
            if kw in lower:
                return "complaint"

        # 售前关键词预检
        presale_kw = ["推荐", "有没有", "介绍", "多少钱", "评价", "比价", "哪个好"]
        for kw in presale_kw:
            if kw in lower:
                return "presale"
        recent_context = ""
        if history:
            recent = [
                m for m in history[-4:]
                if m.get("role") in ("user", "assistant")
            ]
            if recent:
                lines = []
                for m in recent:
                    role = "用户" if m["role"] == "user" else "客服"
                    content = m.get("content", "")
                    if content and len(content) < 200:
                        lines.append(f"{role}: {content}")
                if lines:
                    recent_context = "\n最近对话：\n" + "\n".join(lines) + "\n"

        prompt = ROUTER_PROMPT.format(user_input=user_input)
        if recent_context:
            prompt = recent_context + "\n" + prompt

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        raw = (response.choices[0].message.content or "").strip().lower()

        # 中英文关键词映射
        chinese_map = {"售前": "presale", "售后": "postsale", "投诉": "complaint"}
        for cn, en in chinese_map.items():
            if cn in raw:
                return en

        for agent_key in VALID_AGENTS:
            if agent_key in raw:
                return agent_key

        # 中文关键词兜底
        for cn_word, agent_key in CHINESE_MAP.items():
            if cn_word in raw:
                return agent_key

        return DEFAULT_AGENT
