"""Multi-Agent 协作模块。

实现客服路由与多 Agent 调度：
- Router：意图分类，按用户意图路由到专业子 Agent
- SubAgent：专业子 Agent（售前咨询、售后处理、投诉升级）—— 旧版 ReAct 实现
- MultiAgentOrchestrator：旧版编排器（Router + SubAgent）
- SupervisorAgent：新版 LangGraph 父图 + 三个 CompiledStateGraph 子图
- build_presale_graph / build_postsale_graph / build_complaint_graph：子图工厂
"""

from app.multi_agent.agents import SubAgent
from app.multi_agent.orchestrator import MultiAgentOrchestrator
from app.multi_agent.router import Router
from app.multi_agent.state import SupervisorState
from app.multi_agent.subgraphs import (
    build_presale_graph,
    build_postsale_graph,
    build_complaint_graph,
)
from app.multi_agent.supervisor import SupervisorAgent

__all__ = [
    "MultiAgentOrchestrator",
    "Router",
    "SubAgent",
    "SupervisorAgent",
    "SupervisorState",
    "build_presale_graph",
    "build_postsale_graph",
    "build_complaint_graph",
]