"""多轮对话管理（压缩 / 持久化 / 重置 / 损坏降级）测试。

全部使用 Mock LLM / ToolManager，不依赖真实 API。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.chat import EcomAgent  # noqa: E402
from app.agent.storage import load_session  # noqa: E402
from tests.fakes import FakeLLM, FakeToolManager  # noqa: E402

TEST_SESSION = str(ROOT / "app" / "sessions" / "test_conversation_session.json")


def _make_agent(threshold: int = 20, keep: int = 6) -> EcomAgent:
    agent = EcomAgent(
        session_path=TEST_SESSION,
        client=FakeLLM(),
        tool_manager=FakeToolManager(),
        memory_enabled=False,
        skills_enabled=False,
    )
    agent.history_threshold = threshold
    agent.history_keep_recent = keep
    return agent


@pytest.fixture(autouse=True)
def _clean():
    Path(TEST_SESSION).unlink(missing_ok=True)
    yield
    Path(TEST_SESSION).unlink(missing_ok=True)


def test_summary_triggered():
    agent = _make_agent(threshold=4, keep=2)
    prompts = [
        "你好，我的订单 A123456 还没发货，怎么回事？",
        "我下单两天了，明天要出差用",
        "那可以帮我催一下吗？",
        "另外这个订单能不能改地址？",
    ]
    for p in prompts:
        agent.chat(p)

    assert agent.summary is not None  # 压缩被触发，summary 已生成
    # 压缩后 raw_messages 被截断到 keep 附近（≤ keep + 每轮 3 条的余量）
    assert len(agent.raw_messages) <= 6


def test_persistence_and_recovery():
    agent = _make_agent()
    agent.chat("我的订单号是 B999888，想退货")
    agent.chat("商品有质量问题，收到就是坏的")

    assert Path(TEST_SESSION).exists()

    raw = json.loads(Path(TEST_SESSION).read_text(encoding="utf-8"))
    assert raw.get("version") == 1
    assert len(raw["messages"]) == agent.history_size

    # 模拟重启：新实例加载同一文件
    agent2 = _make_agent()
    assert agent2.history_size == agent.history_size


def test_reset():
    agent = _make_agent()
    agent.chat("随便说一句话")
    assert Path(TEST_SESSION).exists()

    agent.reset()
    assert not Path(TEST_SESSION).exists()
    assert agent.history_size == 0
    assert agent.summary is None


def test_corrupted_json():
    Path(TEST_SESSION).parent.mkdir(parents=True, exist_ok=True)
    Path(TEST_SESSION).write_text("{ this is not valid json", encoding="utf-8")

    agent = _make_agent()  # 不应崩溃
    assert agent.history_size == 0
    assert agent.summary is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])