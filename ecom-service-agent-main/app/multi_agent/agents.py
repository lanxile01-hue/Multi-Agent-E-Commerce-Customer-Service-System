"""子 Agent 配置：每个领域子图专属的 system prompt 与工具白名单。

这里只保留"配置"，不含执行逻辑：真正的子图执行在 `subgraphs.py`
（`_build_subagent_graph` 通用工厂）。工具白名单由 `ToolManager(allowed_tools=...)`
在工具定义层生效——模型看不到越权工具，也就不会调用。
"""

from app.prompts.agents import COMPLAINT_PROMPT, POSTSALE_PROMPT, PRESALE_PROMPT

AGENT_CONFIGS = {
    "presale": {
        "name": "小乐-售前",
        "prompt": PRESALE_PROMPT,
        "tools": {"query_product", "search_knowledge", "list_user_orders", "load_skill"},
    },
    "postsale": {
        "name": "小乐-售后",
        "prompt": POSTSALE_PROMPT,
        "tools": {
            "query_order", "query_logistics", "apply_refund",
            "list_user_orders", "search_knowledge", "load_skill",
        },
    },
    "complaint": {
        "name": "小乐-投诉",
        "prompt": COMPLAINT_PROMPT,
        "tools": {"query_order", "search_knowledge", "load_skill"},
    },
}
