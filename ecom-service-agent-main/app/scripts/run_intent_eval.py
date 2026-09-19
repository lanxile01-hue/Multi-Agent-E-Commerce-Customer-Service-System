"""
意图路由评测：测试 Router 对 presale/postsale/complaint 三路分类的准确率。

用法: python -m app.scripts.run_intent_eval
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from openai import OpenAI
from app.config.settings import settings
from app.multi_agent.router import Router

CASES_PATH = ROOT / "app/evaluation/intent_cases.json"


def main():
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    router = Router(client, settings.model_name)

    correct = 0
    results = []

    print(f"意图路由评测: {len(cases)} 条\n" + "=" * 60)

    for i, case in enumerate(cases):
        predicted = router.route(case["query"])
        expected = case["expected"]
        is_correct = predicted == expected
        if is_correct:
            correct += 1

        results.append({
            "query": case["query"],
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
        })

        status = "+" if is_correct else "x"
        if not is_correct or i % 20 == 0:
            print(f"  [{status}] [{expected}] -> [{predicted}] {case['query'][:40]}")

    accuracy = correct / len(cases)
    print("\n" + "=" * 60)
    print(f"准确率: {correct}/{len(cases)} = {accuracy:.1%}")

    # 按类别统计
    from collections import Counter
    by_cat = Counter()
    by_cat_correct = Counter()
    for r in results:
        by_cat[r["expected"]] += 1
        if r["correct"]:
            by_cat_correct[r["expected"]] += 1

    labels = {"presale": "售前", "postsale": "售后", "complaint": "投诉"}
    print("\n按类别:")
    for cat in ["presale", "postsale", "complaint"]:
        total = by_cat.get(cat, 0)
        ok = by_cat_correct.get(cat, 0)
        acc = ok / total if total else 0
        print(f"  {labels.get(cat, cat)}: {ok}/{total} = {acc:.1%}")

    print("\n评测完成。")
    return accuracy


if __name__ == "__main__":
    main()