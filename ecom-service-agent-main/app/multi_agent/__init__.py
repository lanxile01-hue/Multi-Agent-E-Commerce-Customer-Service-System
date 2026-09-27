"""Multi-Agent 模块（唯一主线：LangGraph Supervisor）。

- Router：意图分类，输出 presale / postsale / complaint
- SupervisorAgent：父图（preprocess → route → run_* → validate_response → persist）
- build_presale_graph / build_postsale_graph / build_complaint_graph：三个子图工厂
- SupervisorState：父图与子图共享的状态模型
"""

from app.multi_agent.router import Router
from app.multi_agent.state import SupervisorState
from app.multi_agent.subgraphs import (
    build_presale_graph,
    build_postsale_graph,
    build_complaint_graph,
)
from app.multi_agent.supervisor import SupervisorAgent

__all__ = [
    "Router",
    "SupervisorAgent",
    "SupervisorState",
    "build_presale_graph",
    "build_postsale_graph",
    "build_complaint_graph",
]
