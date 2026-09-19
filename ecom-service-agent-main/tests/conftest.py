"""pytest 全局配置：注册 `integration` 标记，并默认跳过需要真实 API / 外部服务的测试。

约定（第一阶段测试体系）：
- 默认 `pytest -q` 只跑纯 Mock 的单元测试，不依赖 OPENAI_API_KEY / MCP Server。
- 需要真实 API 的测试打上 `@pytest.mark.integration`。
- 显式启用方式（二选一）：
    1. `pytest -m integration`       —— 只跑 integration 测试
    2. `RUN_LIVE_TESTS=true pytest`  —— 单元 + integration 一起跑
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# 保证 `import app` / `import tests` 在 pytest 收集阶段可用。
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: 需要真实 API / MCP Server 等外部依赖，需显式启用"
        "（-m integration 或 RUN_LIVE_TESTS=true）",
    )


def _live_tests_requested(config) -> bool:
    """判断用户是否显式要求跑 integration 测试。"""
    if os.environ.get("RUN_LIVE_TESTS", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    marker_expr = config.getoption("markexpr") or ""
    return "integration" in marker_expr


def pytest_collection_modifyitems(config, items) -> None:
    """未显式启用时，把 integration 测试整体标记为 skip，避免默认跑真实 API。"""
    if _live_tests_requested(config):
        return

    skip_integration = pytest.mark.skip(
        reason="integration 测试未启用：请使用 -m integration 或 RUN_LIVE_TESTS=true"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)