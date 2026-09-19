"""第五阶段：RAG 证据链与检索评估指标测试（纯函数 + Mock，不依赖真实 API）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation.rag_metrics import (  # noqa: E402
    empty_result_accuracy,
    mrr,
    ndcg,
    recall_at_k,
    source_hit_rate,
)
from app.agent.tools import knowledge as knowledge_tool  # noqa: E402
from app.agent.rag.retriever import KnowledgeRetriever  # noqa: E402
from app.agent.rag.chunker import _chunk_one_file  # noqa: E402
from app.knowledge.access import AccessContext, reset_access_context, set_access_context  # noqa: E402
from tests.fakes import EmptyVectorBackend, FakeEmbedder  # noqa: E402


# ---------- 指标函数 ----------

def test_recall_at_k():
    assert recall_at_k(["a", "b"], ["a", "c", "b"]) == 1.0
    assert recall_at_k(["a", "b"], ["a", "c"]) == 0.5
    assert recall_at_k([], ["a"]) == 1.0


def test_mrr():
    assert mrr(["a"], ["x", "a", "y"]) == 0.5
    assert mrr(["a"], ["a", "x"]) == 1.0
    assert mrr(["a"], ["x", "y"]) == 0.0


def test_ndcg():
    # 完美排序：DCG == IDCG → 1.0
    assert ndcg(["a", "b"], ["a", "b"]) == 1.0
    # 无关文档排前面 → 低于 1.0
    assert 0.0 < ndcg(["a", "b"], ["x", "a", "b"]) < 1.0
    # 完全无关 → 0.0
    assert ndcg(["a"], ["x", "y"]) == 0.0


def test_source_hit_rate():
    assert source_hit_rate(["退换货政策"], ["退换货政策", "配送说明"]) == 1.0
    assert source_hit_rate(["退换货政策", "会员权益"], ["退换货政策"]) == 0.5
    assert source_hit_rate([], ["x"]) == 1.0


def test_empty_result_accuracy():
    # 真值：[应空, 应非空, 应空]，实际：[空, 空, 空] → 命中 2/3
    assert empty_result_accuracy([True, False, True], [True, True, True]) == pytest.approx(2 / 3)
    assert empty_result_accuracy([], []) == 0.0


# ---------- 证据链 / 空召回安全回复 ----------

def test_search_knowledge_no_recall_reliable_false(monkeypatch):
    retriever = KnowledgeRetriever(embedder=FakeEmbedder(), backend=EmptyVectorBackend())
    retriever.load()
    monkeypatch.setattr(knowledge_tool, "_get_retriever", lambda: retriever)
    monkeypatch.setattr(knowledge_tool, "_adaptive_mode", False)

    result = knowledge_tool.search_knowledge("完全无关的问题")
    assert result["success"] is True
    assert result["results"] == []
    assert result["reliable"] is False
    assert result["message"]  # 给出安全回复提示，而不是编造政策


def test_search_knowledge_evidence_shape(monkeypatch):
    """有召回时，结果带证据链字段（source/section/score/retrieval_method）。"""
    from app.agent.rag.backends.base import RetrievedChunk
    from app.agent.rag.chunker import Chunk

    class _FakeBackend(EmptyVectorBackend):
        def search(self, query_vector, top_k):
            return [RetrievedChunk(
                chunk=Chunk(chunk_id="退换货政策#00-00", doc="退换货政策", section="七天无理由", text="..."),
                score=0.91,
            )]

    retriever = KnowledgeRetriever(embedder=FakeEmbedder(), backend=_FakeBackend())
    retriever.load()
    monkeypatch.setattr(knowledge_tool, "_get_retriever", lambda: retriever)
    monkeypatch.setattr(knowledge_tool, "_adaptive_mode", False)

    result = knowledge_tool.search_knowledge("七天无理由退货")
    assert result["reliable"] is True
    r = result["results"][0]
    assert r["source"] == "退换货政策"
    assert r["section"] == "七天无理由"
    assert r["retrieval_method"] == "vector"
    assert "score" in r


def test_chunk_frontmatter_preserves_enterprise_metadata(tmp_path):
    doc = tmp_path / "报销制度.md"
    doc.write_text(
        "---\n"
        "title: 报销制度\n"
        "source: finance/expense-policy.md\n"
        "version: 2026.1\n"
        "effective_time: 2026-01-01\n"
        "department: finance\n"
        "access_level: internal\n"
        "tags: 报销,财务\n"
        "---\n\n"
        "# 报销制度\n\n## 适用范围\n员工差旅报销规则。\n",
        encoding="utf-8",
    )
    chunks = _chunk_one_file(doc)
    assert len(chunks) == 2
    chunk = chunks[-1]
    assert chunk.title == "报销制度"
    assert chunk.source == "finance/expense-policy.md"
    assert chunk.version == "2026.1"
    assert chunk.department == "finance"
    assert chunk.tags == ["报销", "财务"]


def test_search_knowledge_filters_by_department(monkeypatch):
    from app.agent.rag.backends.base import RetrievedChunk
    from app.agent.rag.chunker import Chunk

    class _FilteredBackend(EmptyVectorBackend):
        def search(self, query_vector, top_k):
            return [
                RetrievedChunk(
                    chunk=Chunk(
                        chunk_id="eng#00-00", doc="eng", section="runbook",
                        text="engineering", department="engineering",
                    ), score=0.95,
                ),
                RetrievedChunk(
                    chunk=Chunk(
                        chunk_id="finance#00-00", doc="finance", section="policy",
                        text="finance", department="finance",
                    ), score=0.90,
                ),
            ]

    retriever = KnowledgeRetriever(embedder=FakeEmbedder(), backend=_FilteredBackend())
    retriever.load()
    monkeypatch.setattr(knowledge_tool, "_get_retriever", lambda: retriever)
    monkeypatch.setattr(knowledge_tool, "_adaptive_mode", False)
    result = knowledge_tool.search_knowledge("报销制度", department="finance")
    assert result["reliable"] is True
    assert [item["department"] for item in result["results"]] == ["finance"]


def test_search_knowledge_enforces_tenant_and_role(monkeypatch):
    from app.agent.rag.backends.base import RetrievedChunk
    from app.agent.rag.chunker import Chunk

    class _AccessBackend(EmptyVectorBackend):
        def search(self, query_vector, top_k):
            return [
                RetrievedChunk(
                    chunk=Chunk(
                        chunk_id="public#00-00", doc="public", section="faq",
                        text="public", access_level="public", tenant_id="tenant-a",
                    ), score=0.95,
                ),
                RetrievedChunk(
                    chunk=Chunk(
                        chunk_id="other#00-00", doc="other", section="secret",
                        text="other", access_level="internal", tenant_id="tenant-b",
                    ), score=0.94,
                ),
            ]

    retriever = KnowledgeRetriever(embedder=FakeEmbedder(), backend=_AccessBackend())
    retriever.load()
    monkeypatch.setattr(knowledge_tool, "_get_retriever", lambda: retriever)
    monkeypatch.setattr(knowledge_tool, "_adaptive_mode", False)
    token = set_access_context(AccessContext(tenant_id="tenant-a", role="guest", departments=()))
    try:
        result = knowledge_tool.search_knowledge("帮助")
    finally:
        reset_access_context(token)
    assert result["reliable"] is True
    assert [item["tenant_id"] for item in result["results"]] == ["tenant-a"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
