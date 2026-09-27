"""共享测试替身：FakeLLM / FakeToolManager / FakeEmbedder / 空向量后端。

面向 Supervisor 链路与 Memory / RAG / Skills 等模块的单元测试注入，
避免真实 OpenAI API 调用。

注意：Supervisor 子图级别的 FakeLLM / FakeToolManager 在
`tests/test_supervisor_graph.py` 里另有一份（与既有专项测试保持一致）。
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from app.agent.rag.backends.base import RetrievedChunk, VectorBackend
from app.schemas.response import CustomerServiceResponse


# ---------- 最小 openai SDK 返回结构 ----------

class FakeMessage:
    def __init__(self, content: str = "", parsed: Any = None, tool_calls: Any = None):
        self.content = content
        self.parsed = parsed
        self.tool_calls = tool_calls


class FakeChoice:
    def __init__(self, message: FakeMessage):
        self.message = message


class FakeResponse:
    def __init__(self, message: FakeMessage):
        self.choices = [FakeChoice(message)]


class FakeChatCompletions:
    def __init__(self, llm):
        self._llm = llm

    def create(self, **kwargs):
        return self._llm._create(kwargs)


class FakeChat:
    def __init__(self, llm):
        self.completions = FakeChatCompletions(llm)


class FakeBetaCompletions:
    def __init__(self, llm):
        self._llm = llm

    def parse(self, **kwargs):
        return self._llm._parse(kwargs)


class FakeBetaChat:
    def __init__(self, llm):
        self.completions = FakeBetaCompletions(llm)


class FakeBeta:
    def __init__(self, llm):
        self.chat = FakeBetaChat(llm)


class FakeEmbeddings:
    """最小 Embeddings 客户端，返回固定维度向量（供 RAG 单元测试）。"""

    def __init__(self, dim: int = 8):
        self._dim = dim
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        inputs = kwargs.get("input", [])
        if isinstance(inputs, str):
            inputs = [inputs]
        data = [
            type("_Emb", (), {"embedding": [0.1] * self._dim})()
            for _ in inputs
        ]
        return type("_Resp", (), {"data": data})()


class FakeLLM:
    """内容驱动的 Mock LLM。

    - `_create`：默认返回 `reply` 文本（无 tool_calls）；可用 `responder` 回调按
      请求内容脚本化返回；`create_error` 非空时模拟模型调用超时/异常。
    - `_parse`：`parsed_response` 非空时返回结构化结果；否则抛异常触发 fallback。
    """

    def __init__(
        self,
        reply: str = "这是模拟回复。",
        parsed: Optional[CustomerServiceResponse] = None,
        create_error: Optional[BaseException] = None,
        responder: Optional[Callable[[dict], str]] = None,
    ):
        self.reply = reply
        self.parsed_response = parsed
        self.create_error = create_error
        self._responder = responder
        self.create_calls: list[dict] = []
        self.parse_calls: list[dict] = []
        self.chat = FakeChat(self)
        self.beta = FakeBeta(self)
        self.embeddings = FakeEmbeddings()

    def _create(self, kwargs: dict):
        self.create_calls.append(kwargs)
        if self.create_error is not None:
            raise self.create_error
        if self._responder is not None:
            content = self._responder(kwargs)
        else:
            content = self.reply
        return FakeResponse(FakeMessage(content=content))

    def _parse(self, kwargs: dict):
        self.parse_calls.append(kwargs)
        if self.parsed_response is not None:
            return FakeResponse(FakeMessage(parsed=self.parsed_response))
        raise NotImplementedError("fake beta.parse -> 触发结构化输出 fallback")


class FakeToolManager:
    """Mock ToolManager：记录调用轨迹，可脚本化返回结果。"""

    def __init__(self, results: Optional[dict] = None, tool_definitions: Optional[list[dict]] = None):
        self._results = results or {}
        self.tool_definitions = tool_definitions or []
        self.calls: list[tuple[str, dict]] = []
        self.closed = False

    def execute_tool(self, name: str, arguments: dict) -> str:
        self.calls.append((name, dict(arguments)))
        r = self._results.get(name)
        if callable(r):
            return r(arguments)
        if r is not None:
            return r
        return json.dumps({"success": True, "message": f"{name} ok"}, ensure_ascii=False)

    def close(self) -> None:
        self.closed = True


class FakeEmbedder:
    """Mock Embedder：不调用真实 embedding 接口，返回固定维度向量。"""

    def __init__(self, model: str = "text-embedding-3-small", dim: int = 8):
        self._model = model
        self._dim = dim

    @property
    def model(self) -> str:
        return self._model

    def encode(self, texts) -> list[list[float]]:
        return [[0.1] * self._dim for _ in list(texts)]

    def encode_one(self, text: str) -> list[float]:
        return [0.1] * self._dim


class EmptyVectorBackend(VectorBackend):
    """始终返回空召回的向量后端，用于「RAG 没有召回结果」异常测试。"""

    def upsert(self, chunks, vectors, embedding_model) -> None:
        pass

    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
        return []

    def size(self) -> int:
        return 0

    def load(self) -> None:
        pass

    def expected_embedding_model(self) -> str:
        return ""
