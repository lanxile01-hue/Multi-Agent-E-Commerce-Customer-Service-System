"""Memory（短期记忆 / 长期记忆）测试。

单元测试（默认）：直接测 MemoryManager / ShortTermMemory / LongTermMemory，
用 FakeLLM 模拟事实提取，不依赖真实 API。
集成测试（integration）：走 EcomAgent 真实链路。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.memory import MemoryManager, ShortTermMemory  # noqa: E402
from tools.memory_tool import recall_user_memory, set_memory_manager  # noqa: E402
from tests.fakes import FakeLLM  # noqa: E402


def _stm_llm() -> FakeLLM:
    return FakeLLM(responder=lambda kw: "- 张三\n- 钻石会员\n- 红色运动鞋")


def _ltm_llm() -> FakeLLM:
    return FakeLLM(responder=lambda kw: json.dumps({
        "facts": [{"content": "用户是张三", "category": "identity"}],
        "interaction_summary": "咨询红色运动鞋",
    }, ensure_ascii=False))


def _make_manager(memory_dir: str, *, enabled: bool = True) -> MemoryManager:
    return MemoryManager(
        client=_stm_llm(),
        model="mock-model",
        user_id="test_user",
        memory_dir=memory_dir,
        memory_enabled=enabled,
        max_ltm_facts=50,
    )


# ---------- 短期记忆 ----------

def test_stm_extraction(tmp_path):
    mgr = _make_manager(str(tmp_path / "memory"))
    mgr.update_short_term([{"role": "user", "content": "我是张三，钻石会员"}])
    assert mgr.stm.facts == ["张三", "钻石会员", "红色运动鞋"]


def test_stm_prompt_section(tmp_path):
    mgr = _make_manager(str(tmp_path / "memory"))
    mgr.update_short_term([{"role": "user", "content": "hi"}])
    sections = mgr.build_memory_prompt_sections()
    assert sections
    assert "短期记忆" in sections[-1]["content"]


def test_stm_roundtrip(tmp_path):
    mgr = _make_manager(str(tmp_path / "memory"))
    mgr.update_short_term([{"role": "user", "content": "hi"}])
    restored = ShortTermMemory.from_dict(mgr.stm_to_dict())
    assert restored.facts == mgr.stm.facts


def test_stm_reset(tmp_path):
    mgr = _make_manager(str(tmp_path / "memory"))
    mgr.update_short_term([{"role": "user", "content": "hi"}])
    assert mgr.stm.facts
    mgr.reset_short_term()
    assert mgr.stm.facts == []


# ---------- 长期记忆 ----------

def test_ltm_extraction(tmp_path):
    mgr = _make_manager(str(tmp_path / "memory"))
    mgr.client = _ltm_llm()
    mgr.consolidate_to_long_term(
        [{"role": "user", "content": "我是张三"}], summary=None,
    )
    memory_path = Path(mgr.ltm.memory_path)
    assert memory_path.exists()
    data = json.loads(memory_path.read_text(encoding="utf-8"))
    assert len(data["facts"]) == 1
    assert data["facts"][0]["content"] == "用户是张三"


def test_ltm_cross_session(tmp_path):
    memory_dir = str(tmp_path / "memory")
    mgr1 = _make_manager(memory_dir)
    mgr1.client = _ltm_llm()
    mgr1.consolidate_to_long_term([{"role": "user", "content": "hi"}], None)

    mgr2 = _make_manager(memory_dir)
    assert len(mgr2.ltm.facts) == 1


# ---------- 开关与工具 ----------

def test_memory_disabled(tmp_path):
    mgr = _make_manager(str(tmp_path / "memory"), enabled=False)
    mgr.update_short_term([{"role": "user", "content": "hi"}])
    assert mgr.stm.facts == []
    assert mgr.build_memory_prompt_sections() == []


def test_recall_memory_tool(tmp_path):
    mgr = _make_manager(str(tmp_path / "memory"))
    mgr.update_short_term([{"role": "user", "content": "hi"}])
    set_memory_manager(mgr)
    result = recall_user_memory()
    assert result.get("success") is True
    assert result["short_term_facts"] == ["张三", "钻石会员", "红色运动鞋"]
    set_memory_manager(None)


# ---------- 集成测试（真实 API） ----------

@pytest.mark.integration
def test_memory_live(tmp_path):
    from app.agent.chat import EcomAgent
    from app.agent.memory import MemoryManager as MM

    agent = EcomAgent(
        session_path=str(tmp_path / "session.json"),
        memory_enabled=False,
        skills_enabled=False,
    )
    agent.memory_manager = MM(
        client=agent.client,
        model=agent.model,
        user_id="live_test_user",
        memory_dir=str(tmp_path / "memory"),
        memory_enabled=True,
    )
    set_memory_manager(agent.memory_manager)
    agent.chat("你好，我是李四，钻石会员，喜欢蓝色商品")
    assert len(agent.memory_manager.stm.facts) > 0
    set_memory_manager(None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])