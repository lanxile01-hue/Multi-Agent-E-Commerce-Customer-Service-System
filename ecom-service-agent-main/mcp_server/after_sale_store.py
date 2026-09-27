"""售后工单服务（MCP Server 侧维护）。

真实场景下这里是"调用外部工单系统"的适配层（HTTP / RPC / 数据库），
本地用内存字典模拟，保证 MCP 工具链路可以端到端验证、且不依赖外部服务。

状态取值：pending（待受理）/ processing（处理中）/ waiting_user（等待用户补充）
        / resolved（已解决）/ rejected（不受理）
"""

from __future__ import annotations

import time
from typing import Any

VALID_ISSUE_TYPES = {"quality", "logistics", "refund", "service"}

_TICKET_SEQ = 1000

# 预置历史工单：模拟"这个订单之前已经开过工单"的场景
_TICKETS: dict[str, list[dict[str, Any]]] = {
    "ORD-20240118-004": [
        {
            "ticket_id": "AS-20240120-1001",
            "order_id": "ORD-20240118-004",
            "issue_type": "quality",
            "status": "processing",
            "handler": "售后-张敏",
            "description": "用户反馈牛仔裤尺码不合适，要求退款",
            "last_update": "2026-09-25 10:20 已受理，等待商家举证",
            "sla_deadline": "2026-09-29 18:00",
        }
    ],
    "ORD-20240110-003": [
        {
            "ticket_id": "AS-20240112-0887",
            "order_id": "ORD-20240110-003",
            "issue_type": "logistics",
            "status": "resolved",
            "handler": "售后-李强",
            "description": "快递在转运中心滞留 3 天",
            "last_update": "2026-01-14 09:30 已催促网点，包裹恢复正常派送",
            "sla_deadline": "2026-01-15 12:00",
        }
    ],
}


def _next_ticket_id() -> str:
    global _TICKET_SEQ
    _TICKET_SEQ += 1
    stamp = time.strftime("%Y%m%d")
    return f"AS-{stamp}-{_TICKET_SEQ}"


def query_after_sale_tickets(order_id: str) -> dict:
    """按订单号查询售后工单列表。"""
    order_id = (order_id or "").strip()
    if not order_id:
        return {"success": False, "error": "order_id 不能为空", "tickets": []}

    tickets = _TICKETS.get(order_id, [])
    return {
        "success": True,
        "order_id": order_id,
        "count": len(tickets),
        "tickets": tickets,
        "note": "" if tickets else "该订单暂无售后工单",
    }


def create_after_sale_ticket(order_id: str, issue_type: str, description: str) -> dict:
    """为订单创建售后工单（写操作，调用前需人工确认）。"""
    order_id = (order_id or "").strip()
    issue_type = (issue_type or "").strip().lower()
    description = (description or "").strip()

    if not order_id:
        return {"success": False, "error": "order_id 不能为空"}
    if issue_type not in VALID_ISSUE_TYPES:
        return {
            "success": False,
            "error": f"issue_type 非法，可选：{', '.join(sorted(VALID_ISSUE_TYPES))}",
        }
    if len(description) < 5:
        return {"success": False, "error": "description 太短，请描述具体问题"}

    ticket = {
        "ticket_id": _next_ticket_id(),
        "order_id": order_id,
        "issue_type": issue_type,
        "status": "pending",
        "handler": "待分配",
        "description": description,
        "last_update": time.strftime("%Y-%m-%d %H:%M") + " 用户侧提交，等待受理",
        "sla_deadline": time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + 48 * 3600)),
    }
    _TICKETS.setdefault(order_id, []).append(ticket)
    return {"success": True, "ticket": ticket}
