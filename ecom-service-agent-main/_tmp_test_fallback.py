# -*- coding: utf-8 -*-
"""临时回归测试：模型返回空串/非法 JSON 时，结构化提取兜底不再抛错。"""
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

from app.agent.chat import EcomAgent
from app.agent.graph import LangGraphAgent
from app.schemas.response import IntentType


class FakeResp:
    def __init__(self, content):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]


class FakeCompletions:
    def __init__(self, content):
        self._content = content

    def create(self, **kwargs):
        return FakeResp(self._content)


class FakeChat:
    def __init__(self, content):
        self.completions = FakeCompletions(content)


class FakeClient:
    def __init__(self, content):
        self.chat = FakeChat(content)


def make_agent(cls, content):
    agent = cls.__new__(cls)
    agent.client = FakeClient(content)
    agent.model = "fake"
    return agent


TEXT = "你好呀，很高兴为您服务，有什么可以帮您？"

# 场景1: 模型返回空字符串（本次 bug 的复现场景）
g = make_agent(LangGraphAgent, "")
r = g._extract_structured_fallback(TEXT)
assert r.reply == TEXT, r.reply
print("1 graph 空串 -> OK, reply 用原文, intent =", r.intent)

# 场景2: 非法 JSON
g2 = make_agent(LangGraphAgent, "抱歉，我无法提取")
r2 = g2._extract_structured_fallback(TEXT)
assert r2.reply == TEXT
print("2 graph 非法JSON -> OK")

# 场景3: 带 markdown 代码块的合法 JSON
g3 = make_agent(LangGraphAgent, '```json\n{"intent": "greeting", "confidence": 0.9, "reply": "' + TEXT + '", "requires_human": false}\n```')
r3 = g3._extract_structured_fallback(TEXT)
assert r3.intent == IntentType.GREETING and r3.reply == TEXT
print("3 graph 代码块JSON -> OK, intent =", r3.intent)

# 场景4: 纯合法 JSON（无代码块）
g4 = make_agent(LangGraphAgent, '{"intent": "greeting", "confidence": 0.95, "reply": "' + TEXT + '", "requires_human": false}')
r4 = g4._extract_structured_fallback(TEXT)
assert r4.confidence == 0.95
print("4 graph 纯JSON -> OK, confidence =", r4.confidence)

# 场景5: chat.py 同样的空串场景
c = make_agent(EcomAgent, "")
r5 = c._extract_structured_fallback(TEXT)
assert r5.reply == TEXT
print("5 chat 空串 -> OK, reply 用原文")

print("全部通过")