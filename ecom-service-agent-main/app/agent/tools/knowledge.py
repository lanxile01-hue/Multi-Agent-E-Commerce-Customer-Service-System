"""知识库检索工具：支持标准 RAG 和 Adaptive-RAG 双模式。

标准模式：向量检索（NumpyBackend / ChromaBackend）
Adaptive-RAG：Query Rewrite → Strategy Route → Hybrid Search → RRF → Rerank
"""

from pathlib import Path
from typing import Optional

from openai import OpenAI

from app.config.settings import settings
from app.agent.rag.backends import create_backend
from app.agent.rag.backends.bm25_backend import BM25Backend
from app.agent.rag.embedder import Embedder
from app.agent.rag.retriever import KnowledgeRetriever

_retriever = None  # KnowledgeRetriever | HybridRetriever
_adaptive_mode: bool = False


def _embedding_creds() -> tuple[str, str]:
    """获取 Embedding API 凭证（优先用独立配置，为空则回退到 LLM 配置）。"""
    key = settings.embedding_api_key or settings.openai_api_key
    base = settings.embedding_api_base or settings.openai_base_url
    return key, base


def _create_standard_retriever():
    """标准 RAG 检索器（向量）。"""
    api_key, api_base = _embedding_creds()
    embedder = Embedder(
        api_key=api_key,
        base_url=api_base,
        model=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )
    name = settings.rag_backend.lower()
    if name == "numpy":
        backend = create_backend("numpy", index_path=Path(settings.kb_index_path))
    elif name == "chroma":
        backend = create_backend(
            "chroma",
            persist_dir=Path(settings.chroma_persist_dir),
            collection_name=settings.chroma_collection,
        )
    else:
        raise ValueError(f"未知 RAG 后端: {name}")
    ret = KnowledgeRetriever(embedder=embedder, backend=backend)
    ret.load()
    return ret


def _create_adaptive_retriever():
    """Adaptive-RAG 混合检索器。"""
    from app.agent.rag.hybrid_retriever import HybridRetriever

    api_key, api_base = _embedding_creds()
    embedder = Embedder(
        api_key=api_key,
        base_url=api_base,
        model=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )
    name = settings.rag_backend.lower()
    if name == "numpy":
        vector_backend = create_backend("numpy", index_path=Path(settings.kb_index_path))
    elif name == "chroma":
        vector_backend = create_backend(
            "chroma",
            persist_dir=Path(settings.chroma_persist_dir),
            collection_name=settings.chroma_collection,
        )
    else:
        raise ValueError(f"未知 RAG 后端: {name}")

    bm25 = BM25Backend(index_path=Path(settings.bm25_index_path))

    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )

    ret = HybridRetriever(
        embedder=embedder,
        vector_backend=vector_backend,
        bm25=bm25,
        client=client,
        model=settings.model_name,
        cohere_api_key=settings.cohere_api_key,
    )
    ret.load()
    return ret


def _get_retriever():
    global _retriever, _adaptive_mode
    if _retriever is not None:
        return _retriever

    if settings.adaptive_rag_enabled:
        _retriever = _create_adaptive_retriever()
        _adaptive_mode = True
    else:
        _retriever = _create_standard_retriever()
        _adaptive_mode = False
    return _retriever


def reset_retriever() -> None:
    global _retriever
    _retriever = None


def search_knowledge(
    query: str,
    top_k: int = 3,
    history: Optional[list[dict]] = None,
) -> dict:
    """检索知识库。Adaptive-RAG 模式下 history 参数用于 Query Rewrite。

    Returns:
        {
          "success": bool,
          "backend": "numpy" | "chroma" | "adaptive_rag",
          "mode": "standard" | "adaptive",
          "query": str,
          "results": [{"doc": "...", "section": "...", "score": 0.83, "text": "..."}, ...],
          "error": "..."  // only on failure
        }
    """
    if not query or not query.strip():
        return {"success": False, "error": "query 不能为空", "query": query, "results": []}

    try:
        retriever = _get_retriever()
    except FileNotFoundError as e:
        return {
            "success": False,
            "error": str(e),
            "backend": settings.rag_backend,
            "query": query,
            "results": [],
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"知识库初始化失败: {e}",
            "backend": settings.rag_backend,
            "query": query,
            "results": [],
        }

    top_k = max(1, min(int(top_k or 3), 10))

    if _adaptive_mode:
        from app.agent.rag.hybrid_retriever import HybridRetriever
        assert isinstance(retriever, HybridRetriever)
        hits = retriever.search(query, top_k=top_k, history=history)
        mode_label = "adaptive"
        retrieval_method = "bm25+vector+rrf"
    else:
        assert isinstance(retriever, KnowledgeRetriever)
        hits = retriever.search(query, top_k=top_k)
        mode_label = "standard"
        retrieval_method = "vector"

    results = [
        {
            "evidence_id": h.chunk.chunk_id,
            "doc": h.chunk.doc,
            "title": h.chunk.title or h.chunk.doc,
            "section": h.chunk.section,
            "score": round(h.score, 4),
            "text": h.chunk.text,
            "source": h.chunk.source or h.chunk.doc,
            "version": h.chunk.version,
            "effective_time": h.chunk.effective_time,
            "updated_at": h.chunk.updated_at,
            "tags": h.chunk.tags or [],
            "retrieval_method": retrieval_method,
        }
        for h in hits
    ]
    results = results[:top_k]

    # 可靠性判定：无召回或 top1 低于阈值 → 不视为可靠政策依据
    threshold_name = "rag_adaptive_score_threshold" if _adaptive_mode else "rag_score_threshold"
    threshold = getattr(settings, threshold_name, 0.0)
    reliable = bool(results) and (threshold <= 0 or results[0]["score"] >= threshold)

    # 低置信度结果仍保留给调试和评估，但明确禁止上层把它当作事实依据。
    if not reliable:
        results = []

    return {
        "success": True,
        "reliable": reliable,
        "backend": settings.rag_backend if not _adaptive_mode else "adaptive_rag",
        "mode": mode_label,
        "query": query,
        "results": results,
        "evidence": results,  # 结构化证据链（来源/版本/章节/分数/检索方式）
        "message": "" if reliable else "未检索到可靠的政策依据，请转人工核实或直接查询",
    }
