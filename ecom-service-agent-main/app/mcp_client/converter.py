"""MCP 工具 schema → OpenAI function calling 格式转换。"""


def mcp_tools_to_openai(mcp_tools: list) -> list[dict]:
    """将 MCP Tool 列表转换为 OpenAI function calling 所需的 tools 格式。

    MCP Tool 的 inputSchema 本身就是 JSON Schema，和 OpenAI parameters 字段格式一致，
    只需要调整外层信封结构。
    """
    result = []
    for tool in mcp_tools:
        # 1.x 字段名是 inputSchema，2.x 改成 input_schema
        schema = (
            getattr(tool, "input_schema", None)
            or getattr(tool, "inputSchema", None)
            or {"type": "object", "properties": {}}
        )
        result.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": schema,
            },
        })
    return result
