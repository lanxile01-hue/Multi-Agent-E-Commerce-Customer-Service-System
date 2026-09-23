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


def test_chunk_frontmatter_preserves_evidence_metadata(tmp_path):
    """带 frontmatter 的文档仍能解析出证据链预留字段（source/version/生效时间）。"""
    doc = tmp_path / "退换货政策.md"
    doc.write_text(
        "---\n"
        "title: 退换货政策\n"
        "source: knowledge/退换货政策.md\n"
        "version: 2026.1\n"
        "effective_time: 2026-01-01\n"
        "updated_at: 2026-01-10\n"
        "tags: 退换货,售后\n"
        "---\n\n"
        "# 退换货政策\n\n## 适用范围\n自签收之日起七天内可无理由退货。\n",
        encoding="utf-8",
    )
    chunks = _chunk_one_file(doc)
    assert len(chunks) == 2
    chunk = chunks[-1]
    assert chunk.title == "退换货政策"
    assert chunk.source == "knowledge/退换货政策.md"
    assert chunk.version == "2026.1"
    assert chunk.effective_time == "2026-01-01"
    assert chunk.updated_at == "2026-01-10"
    assert chunk.tags == ["退换货", "售后"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
