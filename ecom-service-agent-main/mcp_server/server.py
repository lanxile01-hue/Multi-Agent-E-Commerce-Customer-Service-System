"""MCP Server：通过 Streamable HTTP 暴露电商工具服务。

启动方式：python mcp_server/server.py
默认监听：http://127.0.0.1:9123/mcp

兼容 mcp SDK 1.x 与 2.x：
- 1.x：`mcp.server.fastmcp.FastMCP`，host/port 传给构造函数；
- 2.x：`FastMCP` 改名为 `mcp.server.mcpserver.MCPServer`，host/port 传给 `run()`。
两边的 `@server.tool()` 装饰器用法一致，所以只在这里做一次适配。
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HOST = "127.0.0.1"
PORT = 9123
PATH = "/mcp"

try:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as ServerClass

    _MCP_V1 = True
except ModuleNotFoundError:  # mcp 2.x：FastMCP 重命名为 MCPServer
    from mcp.server.mcpserver import MCPServer as ServerClass

    _MCP_V1 = False

from mcp_server.after_sale_store import create_after_sale_ticket as _create_after_sale_ticket
from mcp_server.after_sale_store import query_after_sale_tickets as _query_after_sale_tickets
from tools.order import query_order as _query_order
from tools.product import query_product as _query_product
from tools.logistics import query_logistics as _query_logistics
from tools.refund import apply_refund as _apply_refund
from tools.knowledge import search_knowledge as _search_knowledge

server = (
    ServerClass("ecom-tools", host=HOST, port=PORT)
    if _MCP_V1
    else ServerClass("ecom-tools")
)


@server.tool()
def query_order(order_id: str) -> str:
    """根据订单号查询订单详情，包括订单状态、商品信息、金额、物流单号等"""
    result = _query_order(order_id)
    return json.dumps(result, ensure_ascii=False)


@server.tool()
def query_product(keyword: str) -> str:
    """根据商品名称关键词或商品ID查询商品信息，包括价格、库存、规格等。支持模糊搜索"""
    result = _query_product(keyword)
    return json.dumps(result, ensure_ascii=False)


@server.tool()
def query_logistics(order_id: str) -> str:
    """根据订单号查询物流轨迹信息，包括快递公司、运单号、运输状态和轨迹事件"""
    result = _query_logistics(order_id)
    return json.dumps(result, ensure_ascii=False)


@server.tool()
def apply_refund(order_id: str, reason: str) -> str:
    """为指定订单申请退款。注意：这是一个敏感操作，调用前应先与用户确认"""
    result = _apply_refund(order_id, reason)
    return json.dumps(result, ensure_ascii=False)


@server.tool()
def search_knowledge(query: str, top_k: int = 3) -> str:
    """检索帅团的政策与帮助文档（退换货政策、配送说明、会员权益、FAQ）。
    当顾客询问规则、流程、时效、是否支持等政策类问题时使用。
    返回 Top-K 命中片段及来源文档，请基于检索结果回答，不要编造政策。
    """
    result = _search_knowledge(query, top_k=top_k)
    return json.dumps(result, ensure_ascii=False)


# ---------- 以下两个工具只存在于 MCP 侧（本地 registry 没有），
# ---------- 用于演示"接入 MCP 后 Agent 获得本地不具备的能力"。

@server.tool()
def query_after_sale_tickets(order_id: str) -> str:
    """查询指定订单的售后工单进度（工单号、问题类型、状态、处理人、承诺时限、最近进展）。
    当用户询问"我的售后/投诉处理到哪一步了""有没有在处理"时使用。
    状态取值：pending 待受理 / processing 处理中 / waiting_user 等待用户补充 /
    resolved 已解决 / rejected 不受理。"""
    result = _query_after_sale_tickets(order_id)
    return json.dumps(result, ensure_ascii=False)


@server.tool()
def create_after_sale_ticket(order_id: str, issue_type: str, description: str) -> str:
    """为指定订单创建售后工单（写操作，调用前必须与用户确认订单号、问题类型和描述）。
    issue_type 取值：quality 质量问题 / logistics 物流异常 / refund 退款争议 / service 服务态度。"""
    result = _create_after_sale_ticket(order_id, issue_type, description)
    return json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":
    run_kwargs: dict = {"transport": "streamable-http"}
    if not _MCP_V1:  # 2.x 的 host/port/path 在 run() 上
        run_kwargs.update(host=HOST, port=PORT, streamable_http_path=PATH)
    print(f"🚀 MCP Server 已启动: http://{HOST}:{PORT}{PATH}（mcp SDK {'1.x' if _MCP_V1 else '2.x'}）")
    server.run(**run_kwargs)
