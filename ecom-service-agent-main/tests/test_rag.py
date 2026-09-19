"""RAG 知识库检索测试（两套后端）。

单元测试（默认）：Markdown 切分。
集成测试（integration）：索引构建 / Top-K 召回 / 工具结构 / Agent 端到端，
需要真实 Embedding / LLM API。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config.settings import settings  # noqa: E402
from app.agent.rag.backends import create_backend  # noqa: E402
from app.agent.rag.chunker import chunk_markdown_dir  # noqa: E402
from app.agent.rag.embedder import Embedder  # noqa: E402
from app.agent.rag.retriever import KnowledgeRetriever  # noqa: E402
from app.agent.tools import knowledge as knowledge_tool  # noqa: E402

TEST_SESSION = str(ROOT / "app" / "sessions" / "test_rag_session.json")
EXPECTED_DOCS = {"退换货政策", "配送说明", "会员权益", "常见问题FAQ"}


def _make_embedder() -> Embedder:
    return Embedder(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.embedding_model,
    )


def _make_backend(backend_name: str):
    if backend_name == "numpy":
        return create_backend("numpy", index_path=ROOT / settings.kb_index_path)
    if backend_name == "chroma":
        return create_backend(
            "chroma",
            persist_dir=ROOT / settings.chroma_persist_dir,
            collection_name=settings.chroma_collection,
        )
    raise ValueError(backend_name)


# ---------- 单元测试（无需 API） ----------

def test_chunker():
    chunks = chunk_markdown_dir(ROOT / settings.kb_dir)
    assert chunks
    docs = {c.doc for c in chunks}
    assert EXPECTED_DOCS.issubset(docs)
    for c in chunks:
        assert c.text.startswith("【")


# ---------- 集成测试（真实 API） ----------

@pytest.mark.integration
@pytest.mark.parametrize("backend_name", ["numpy", "chroma"])
def test_build_index_live(backend_name):
    embedder = _make_embedder()
    backend = _make_backend(backend_name)
    try:
        backend.load()
        if backend.expected_embedding_model() == settings.embedding_model and backend.size() > 0:
            return
    except FileNotFoundError:
        pass
    chunks = chunk_markdown_dir(ROOT / settings.kb_dir)
    vectors = embedder.encode([c.text for c in chunks])
    backend.upsert(chunks=chunks, vectors=vectors, embedding_model=settings.embedding_model)
    assert backend.size() > 0


@pytest.mark.integration
@pytest.mark.parametrize("backend_name", ["numpy", "chroma"])
def test_retriever_top_k_live(backend_name):
    retriever = KnowledgeRetriever(embedder=_make_embedder(), backend=_make_backend(backend_name))
    retriever.load()
    cases = [
        ("七天无理由退货可以吗", "退换货政策"),
        ("钻石会员有哪些权益", "会员权益"),
    ]
    for query, expected_doc in cases:
        hits = retriever.search(query, top_k=3)
        assert hits
        assert any(h.chunk.doc == expected_doc for h in hits)


@pytest.mark.integration
@pytest.mark.parametrize("backend_name", ["numpy", "chroma"])
def test_tool_shape_live(backend_name):
    settings.rag_backend = backend_name
    knowledge_tool.reset_retriever()
    try:
        result = knowledge_tool.search_knowledge("退款多久到账", top_k=2)
        assert result.get("success") is True
        assert result.get("backend") == backend_name
        assert result.get("results")
        assert len(result["results"]) <= 2
        for key in ("doc", "section", "score", "text"):
            assert key in result["results"][0]
    finally:
        knowledge_tool.reset_retriever()


@pytest.mark.integration
@pytest.mark.parametrize("backend_name", ["numpy", "chroma"])
def test_agent_end_to_end_live(backend_name):
    from app.agent.chat import EcomAgent
    from app.schemas.response import CustomerServiceResponse

    settings.rag_backend = backend_name
    knowledge_tool.reset_retriever()
    Path(TEST_SESSION).unlink(missing_ok=True)
    try:
        agent = EcomAgent(session_path=TEST_SESSION, memory_enabled=False)
        resp = agent.chat("我想了解一下你们的七天无理由退货是怎么算运费的？")
        assert isinstance(resp, CustomerServiceResponse)
        assert resp.reply
        assert any(kw in resp.reply for kw in ["运费", "上门取件", "12 元", "12元", "顾客"])
    finally:
        knowledge_tool.reset_retriever()
        Path(TEST_SESSION).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])