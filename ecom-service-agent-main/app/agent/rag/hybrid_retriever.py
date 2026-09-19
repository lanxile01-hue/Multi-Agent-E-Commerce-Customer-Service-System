"""
HybridRetriever：混合检索引擎，整合向量检索 + BM25 关键词检索。

核心能力:
1. 策略路由：根据查询类型选择 vector / keyword / hybrid
2. RRF 融合：将多源结果按倒数排名融合去重
3. Cohere 重排序（可选）：调用 Cohere Rerank API 精排
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import httpx

from app.agent.rag.backends.base import RetrievedChunk, VectorBackend
from app.agent.rag.backends.bm25_backend import BM25Backend
from app.agent.rag.chunker import Chunk
from app.agent.rag.embedder import Embedder
from app.agent.rag.rewriter import QueryRewriter
from app.agent.rag.router import RetrievalRouter
from openai import OpenAI


def _rrf_fusion(
    ranked_lists: list[list[tuple[Chunk, float]]],
    k: int = 60,
    top_k: int = 5,
) -> list[tuple[Chunk, float]]:
    """RRF (Reciprocal Rank Fusion) 多源融合去重。

    score(chunk) = sum( 1 / (k + rank_i) )  across all lists
    """
    chunk_scores: dict[str, float] = {}
    chunk_map: dict[str, Chunk] = {}

    for ranked in ranked_lists:
        for rank, (chunk, _score) in enumerate(ranked):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            chunk_scores[cid] = chunk_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

    sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
    return [(chunk_map[cid], score) for cid, score in sorted_chunks[:top_k]]


class HybridRetriever:
    """混合检索器：vector + BM25 + RRF + optional rerank。"""

    def __init__(
        self,
        embedder: Embedder,
        vector_backend: VectorBackend,
        bm25: BM25Backend,
        client: OpenAI,
        model: str,
        cohere_api_key: str = "",
        cohere_model: str = "rerank-multilingual-v3.0",
    ):
        self._embedder = embedder
        self._vector = vector_backend
        self._bm25 = bm25
        self._rewriter = QueryRewriter(client, model)
        self._router = RetrievalRouter(client, model)
        self._cohere_key = cohere_api_key
        self._cohere_model = cohere_model
        self._loaded = False

    @property
    def size(self) -> int:
        return self._vector.size()

    def load(self) -> None:
        if self._loaded:
            return
        self._vector.load()
        self._bm25.load()
        self._loaded = True

    def search(
        self,
        query: str,
        top_k: int = 5,
        history: Optional[list[dict]] = None,
    ) -> list[RetrievedChunk]:
        """主检索入口：rewrite → route → search → fuse → rerank。"""
        if not self._loaded:
            self.load()

        # 1) 查询改写
        rewritten, was_rewritten = self._rewriter.rewrite(query, history)
        search_query = rewritten if was_rewritten else query
        if was_rewritten:
            print(f"  [Query Rewriter] '{query}' -> '{rewritten}'")

        # 2) 策略路由
        strategy, reason = self._router.route(search_query)
        print(f"  [Router] strategy={strategy} ({reason})")

        # 3) 按策略检索
        vector_results: list[tuple[Chunk, float]] = []
        keyword_results: list[tuple[Chunk, float]] = []

        if strategy in ("vector", "hybrid"):
            q_vec = self._embedder.encode_one(search_query)
            vector_results = [
                (r.chunk, r.score) for r in self._vector.search(q_vec, top_k=top_k * 2)
            ]

        if strategy in ("keyword", "hybrid"):
            keyword_results = self._bm25.search(search_query, top_k=top_k * 2)

        if not vector_results and not keyword_results:
            return []

        # 4) RRF 融合
        if strategy == "vector":
            ranked = vector_results
        elif strategy == "keyword":
            ranked = keyword_results
        else:
            ranked = _rrf_fusion([vector_results, keyword_results], top_k=top_k)

        # 5) Cohere 重排序（可选）
        if self._cohere_key and len(ranked) > 1:
            ranked = self._cohere_rerank(search_query, ranked, top_k)

        return [
            RetrievedChunk(chunk=chunk, score=round(score, 4))
            for chunk, score in ranked[:top_k]
        ]

    def _cohere_rerank(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
        top_k: int,
    ) -> list[tuple[Chunk, float]]:
        """调用 Cohere Rerank API 重排序。"""
        try:
            docs = [c.text for c, _ in candidates]
            resp = httpx.post(
                "https://api.cohere.ai/v2/rerank",
                headers={
                    "Authorization": f"Bearer {self._cohere_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._cohere_model,
                    "query": query,
                    "documents": docs,
                    "top_n": min(top_k, len(docs)),
                },
                timeout=15,
            )
            data = resp.json()
            results = data.get("results", [])
            reranked = []
            for r in results:
                idx = r.get("index", 0)
                score = r.get("relevance_score", candidates[idx][1])
                if idx < len(candidates):
                    reranked.append((candidates[idx][0], score))
            return reranked if reranked else candidates[:top_k]
        except Exception:
            return candidates[:top_k]