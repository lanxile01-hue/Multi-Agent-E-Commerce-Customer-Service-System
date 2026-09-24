"""Skill 可复用能力模块测试。

单元测试（默认）：SkillManager 发现 / catalog / load_skill / 关闭开关 / 延迟加载。
集成测试（integration）：Agent 集成，验证 catalog 注入 prompt。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.skills import SkillManager  # noqa: E402
from app.agent.tools.skill_tool import load_skill, set_skill_manager  # noqa: E402

SKILLS_DIR = str(ROOT / "app" / "agent" / "skills" / "definitions")


def _manager(enabled: bool = True) -> SkillManager:
    return SkillManager(skills_dir=SKILLS_DIR, enabled=enabled)


def test_skill_discovery():
    sm = _manager()
    assert sm.skill_count >= 4
    expected = {
        "process-return", "track-order", "product-recommend", "complaint-escalation",
    }
    assert expected.issubset(set(sm.skill_names))


def test_catalog_generation():
    sm = _manager()
    catalog = sm.get_catalog()
    assert len(catalog) >= 3
    for entry in catalog:
        assert entry["name"] and entry["description"]
    prompt = sm.build_catalog_prompt()
    assert "可用技能" in prompt and "load_skill" in prompt


def test_load_skill_success():
    sm = _manager()
    set_skill_manager(sm)
    result = load_skill("process-return")
    assert result["success"] is True
    assert "退货" in result["instructions"]
    assert "query_order" in result["instructions"]
    set_skill_manager(None)


def test_load_complaint_escalation_skill():
    """投诉升级技能：目录可发现、加载后包含分级处理与转人工判据。"""
    sm = _manager()
    set_skill_manager(sm)
    result = load_skill("complaint-escalation")
    assert result["success"] is True
    instructions = result["instructions"]
    assert "query_order" in instructions
    assert "search_knowledge" in instructions
    assert "L2 · 需要人工介入" in instructions
    assert "投诉" in instructions
    # 技能里不应引导模型自己执行退款（投诉子图也没有退款工具）
    assert "apply_refund" not in instructions
    set_skill_manager(None)


def test_load_skill_unknown():
    sm = _manager()
    set_skill_manager(sm)
    result = load_skill("nonexistent-skill")
    assert result["success"] is False
    assert "未找到" in result["error"]
    set_skill_manager(None)


def test_skills_disabled():
    sm = _manager(enabled=False)
    assert sm.skill_count == 0
    assert sm.build_catalog_prompt() == ""

    set_skill_manager(sm)
    result = load_skill("process-return")
    assert result["success"] is False
    assert "未启用" in result["error"]
    set_skill_manager(None)


def test_lazy_loading():
    sm = _manager()
    skill = sm._skills["process-return"]
    assert skill._body_loaded is False
    body = skill.load_body()
    assert skill._body_loaded is True
    assert len(body) > 100


@pytest.mark.integration
def test_agent_integration_live(tmp_path):
    from app.agent.chat import EcomAgent

    agent = EcomAgent(
        session_path=str(tmp_path / "session.json"),
        memory_enabled=False,
        skills_enabled=True,
    )
    assert agent.skill_manager.enabled
    messages = agent._build_messages()
    assert "可用技能" in messages[0]["content"]
    assert "process-return" in messages[0]["content"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
