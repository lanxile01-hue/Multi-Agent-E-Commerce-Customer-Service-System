"""RAG 检索评估指标（纯函数，不依赖真实 API，可直接单测）。

提供：Recall@k / MRR / nDCG / 空结果准确率 / 来源命中率。
注意：这些函数只负责「计算」，不负责「跑评估」。真实评估结果请运行
`python -m app.scripts.run_rag_eval` 后得到，不要把手工推测的数字写入报告。
"""

from __future__ import annotations

import math


def _topk(items: list[str], k: int | None) -> list[str]:
    return items[:k] if k else items


def recall_at_k(expected: list[str], retrieved: list[str], k: int | None = None) -> float:
    """Recall@k：期望文档在 Top-K 中被命中的比例。expected 为空返回 1.0。"""
    if not expected:
        return 1.0
    top = set(_topk(retrieved, k))
    return len(set(expected) & top) / len(expected)


def mrr(expected: list[str], retrieved: list[str]) -> float:
    """MRR：第一个命中期望文档的位置的倒数。未命中返回 0.0。"""
    expected_set = set(expected)
    for rank, doc in enumerate(retrieved, 1):
        if doc in expected_set:
            return 1.0 / rank
    return 0.0


def ndcg(expected: list[str], retrieved: list[str], k: int | None = None) -> float:
    """nDCG（二元相关性）：命中期望文档记 1，否则 0。"""
    top = _topk(retrieved, k)
    expected_set = set(expected)
    dcg = sum(
        (1.0 if doc in expected_set else 0.0) / math.log2(i + 2)
        for i, doc in enumerate(top)
    )
    ideal_count = min(len(expected), len(top))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))
    return dcg / idcg if idcg > 0 else 0.0


def source_hit_rate(expected_sources: list[str], retrieved_sources: list[str]) -> float:
    """来源命中率：期望来源被检索结果命中的比例。expected 为空返回 1.0。"""
    if not expected_sources:
        return 1.0
    return len(set(expected_sources) & set(retrieved_sources)) / len(expected_sources)


def empty_result_accuracy(expected_empty: list[bool], actual_empty: list[bool]) -> float:
    """空结果准确率：对「是否应为空」的二分类准确率。

    - expected_empty[i]：第 i 个 query 是否「本应无召回结果」（如无关问题）
    - actual_empty[i]  ：第 i 个 query 实际是否无召回
    返回预测正确的比例。
    """
    if not expected_empty:
        return 0.0
    correct = sum(1 for e, a in zip(expected_empty, actual_empty) if bool(e) == bool(a))
    return correct / len(expected_empty)