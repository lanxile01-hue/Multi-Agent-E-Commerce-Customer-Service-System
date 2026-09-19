"""
BM25 关键词检索后端：对 chunks 建立倒排索引，实现稀疏检索。

与 VectorBackend 不同，BM25 不依赖向量 Embedding，而是基于词频-逆文档频率
对查询和文档做关键词匹配。作为 Adaptive-RAG 中混合检索的稀疏侧。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.agent.rag.chunker import Chunk


def _tokenize(text: str) -> list[str]:
    """中文+英文混合分词：按中文单字和英文单词切分。"""
    tokens = []
    for part in re.findall(r"[\u4e00-\u9fff]|[a-zA-Z]+|\d+", text.lower()):
        if re.match(r"[\u4e00-\u9fff]", part):
            tokens.extend(part)
        else:
            tokens.append(part)
    return tokens


class BM25Backend:
    """BM25 关键词检索，轻量无依赖。"""

    def __init__(self, index_path: str | Path = ""):
        self._index_path = Path(index_path) if index_path else None
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._tokenized: list[list[str]] = []

    def upsert(self, chunks: list[Chunk]) -> None:
        """构建 BM25 索引（覆盖式）。"""
        self._chunks = list(chunks)
        self._tokenized = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized)

        if self._index_path:
            self._save()

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """BM25 检索，返回 [(chunk, normalized_score), ...]"""
        if not self._chunks:
            self._load()

        if not self._bm25:
            return []

        query_tokens = _tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        # 归一化到 [0, 1]
        max_score = max(scores) if max(scores) > 0 else 1.0

        scored = sorted(
            [(self._chunks[i], scores[i] / max_score) for i in range(len(scores))],
            key=lambda x: x[1],
            reverse=True,
        )
        return scored[:top_k]

    def size(self) -> int:
        return len(self._chunks)

    def load(self) -> None:
        """从 JSON 加载索引。"""
        if not self._index_path or not self._index_path.exists():
            return
        data = json.loads(self._index_path.read_text(encoding="utf-8"))
        self._chunks = [Chunk(**c) for c in data["chunks"]]
        self._tokenized = [_tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(self._tokenized)

    def _save(self) -> None:
        if not self._index_path:
            return
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"chunks": [c.to_dict() for c in self._chunks]}
        self._index_path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    def reset(self) -> None:
        self._chunks = []
        self._bm25 = None
        self._tokenized = []