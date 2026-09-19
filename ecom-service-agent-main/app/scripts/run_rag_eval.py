"""
RAG 对比评测 v2：标准模式 vs Adaptive-RAG，按问题类型分类对比。

问题分类: keyword（编号/代码类）, semantic（语义/描述类）, hybrid（混合类）

用法: python -m app.scripts.run_rag_eval
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.config.settings import settings  # noqa: E402
from app.agent.tools.knowledge import (  # noqa: E402
    _create_adaptive_retriever,
    _create_standard_retriever,
    reset_retriever,
)

DATASET_PATH = ROOT / "app/evaluation/rag_cases.json"
TOP_K = 10


def load_cases() -> list[dict]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def precision(case: dict, hits: list) -> float:
    expected = set(case["expected_docs"])
    found = set(h["doc"] for h in hits)
    return len(found & expected) / len(found) if found else 0.0


def recall(case: dict, hits: list) -> float:
    expected = set(case["expected_docs"])
    found = set(h["doc"] for h in hits)
    return len(found & expected) / len(expected) if expected else 1.0


def mrr(case: dict, hits: list) -> float:
    expected = set(case["expected_docs"])
    for rank, h in enumerate(hits, 1):
        if h["doc"] in expected:
            return 1.0 / rank
    return 0.0


def evaluate(cases: list[dict], use_adaptive: bool) -> dict:
    """跑一轮评测。"""
    original = settings.adaptive_rag_enabled
    settings.adaptive_rag_enabled = use_adaptive
    reset_retriever()

    try:
        if use_adaptive:
            retriever = _create_adaptive_retriever()
        else:
            retriever = _create_standard_retriever()

        results = []
        for case in cases:
            hits = retriever.search(case["query"], top_k=TOP_K)
            formatted = [
                {"doc": h.chunk.doc, "section": h.chunk.section, "score": round(h.score, 4)}
                for h in hits
            ]
            results.append({
                "query": case["query"],
                "category": case.get("category", "hybrid"),
                "precision": precision(case, formatted),
                "recall": recall(case, formatted),
                "mrr": mrr(case, formatted),
                "hits": formatted,
            })

        precisions = [r["precision"] for r in results]
        recalls = [r["recall"] for r in results]
        mrrs = [r["mrr"] for r in results]

    finally:
        settings.adaptive_rag_enabled = original
        reset_retriever()

    return {
        "avg_precision": sum(precisions) / len(precisions),
        "avg_recall": sum(recalls) / len(recalls),
        "avg_mrr": sum(mrrs) / len(mrrs),
        "total_queries": len(results),
        "details": results,
    }


def category_breakdown(details: list[dict]) -> dict:
    """按 category 分组统计。"""
    groups: dict[str, list] = {}
    for r in details:
        cat = r.get("category", "hybrid")
        groups.setdefault(cat, []).append(r)

    result = {}
    for cat, items in groups.items():
        result[cat] = {
            "count": len(items),
            "precision": sum(r["precision"] for r in items) / len(items),
            "recall": sum(r["recall"] for r in items) / len(items),
            "mrr": sum(r["mrr"] for r in items) / len(items),
        }
    return result


def main():
    cases = load_cases()
    print(f"\n  RAG 对比评测 v2: {len(cases)} 条用例, Top-K={TOP_K}")
    print("=" * 70)

    print("\n  [1/2] 标准模式（纯向量检索）...")
    std = evaluate(cases, use_adaptive=False)
    print(f"    Precision: {std['avg_precision']:.1%}")
    print(f"    Recall:    {std['avg_recall']:.1%}")
    print(f"    MRR:       {std['avg_mrr']:.3f}")

    print("\n  [2/2] Adaptive-RAG（Rewrite + Router + Hybrid + RRF）...")
    adapt = evaluate(cases, use_adaptive=True)
    print(f"    Precision: {adapt['avg_precision']:.1%}")
    print(f"    Recall:    {adapt['avg_recall']:.1%}")
    print(f"    MRR:       {adapt['avg_mrr']:.3f}")

    # 总体对比
    print("\n" + "=" * 70)
    print("  总体对比")
    print("=" * 70)
    print(f"{'指标':<15} {'标准模式':>12} {'Adaptive-RAG':>12} {'提升':>10}")
    print("-" * 52)
    p_std, p_ad = std["avg_precision"], adapt["avg_precision"]
    r_std, r_ad = std["avg_recall"], adapt["avg_recall"]
    m_std, m_ad = std["avg_mrr"], adapt["avg_mrr"]

    def delta(new, old):
        if old == 0: return "+inf"
        return f"{(new - old) / old:+.0%}"

    print(f"{'Precision':<15} {p_std:>11.1%} {p_ad:>12.1%} {delta(p_ad, p_std):>10}")
    print(f"{'Recall':<15} {r_std:>11.1%} {r_ad:>12.1%} {delta(r_ad, r_std):>10}")
    print(f"{'MRR':<15} {m_std:>11.3f} {m_ad:>12.3f} {delta(m_ad, m_std):>10}")

    # 按类别分组对比
    std_cat = category_breakdown(std["details"])
    adapt_cat = category_breakdown(adapt["details"])

    category_labels = {"keyword": "编号/代码类", "semantic": "语义/描述类", "hybrid": "混合类"}

    print("\n" + "=" * 70)
    print("  按问题类型分组对比")
    print("=" * 70)

    for cat_key in ["keyword", "semantic", "hybrid"]:
        label = category_labels.get(cat_key, cat_key)
        s = std_cat.get(cat_key, {})
        a = adapt_cat.get(cat_key, {})
        n = s.get("count", 0)

        if not s or not a:
            continue

        print(f"\n── {label}（{n} 条）──")
        print(f"{'指标':<15} {'标准模式':>12} {'Adaptive-RAG':>12} {'提升':>10}")
        print("-" * 52)

        for metric_name, key in [("Precision", "precision"), ("Recall", "recall"), ("MRR", "mrr")]:
            sv = s.get(key, 0)
            av = a.get(key, 0)
            if key == "mrr":
                print(f"{metric_name:<15} {sv:>11.3f} {av:>12.3f} {delta(av, sv):>10}")
            else:
                print(f"{metric_name:<15} {sv:>11.1%} {av:>12.1%} {delta(av, sv):>10}")

    print("\n评测完成。")


if __name__ == "__main__":
    main()