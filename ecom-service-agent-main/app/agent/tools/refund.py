"""退款工具：高风险写操作，带确认、持久化幂等和审计日志。

上层子图负责 interrupt/用户确认；本模块负责订单校验、幂等落库和审计。
幂等与审计使用 SQLite，因此同一数据库下的重启和多进程调用也能共享结果。
``clear_refund_idempotency`` 仅供测试隔离使用。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional

from app.agent.tools.errors import standard_result
from app.agent.tools.mock_data import ORDERS
from app.config.settings import settings

_REFUND_PROCESSING = "refund_processing"
_refund_lock = threading.RLock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _db_path() -> str:
    path = settings.refund_store_path or "app/sessions/refunds.db"
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    return path


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(), timeout=10.0)
    conn.execute("PRAGMA busy_timeout = 10000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS refund_idempotency (
            idempotency_key TEXT PRIMARY KEY,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS refund_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event TEXT NOT NULL,
            time TEXT NOT NULL,
            order_id TEXT NOT NULL,
            reason TEXT NOT NULL,
            request_id TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            order_status TEXT
        )
        """
    )
    return conn


def _load_idempotent(conn: sqlite3.Connection, key: str) -> dict | None:
    if not key:
        return None
    row = conn.execute(
        "SELECT result_json FROM refund_idempotency WHERE idempotency_key = ?",
        (key,),
    ).fetchone()
    return json.loads(row[0]) if row else None


def _eligible(order: dict) -> tuple[bool, str]:
    """退款资格校验。返回 (是否可退, 拒绝原因)。"""
    if order.get("status") == _REFUND_PROCESSING:
        return False, "该订单已有退款申请正在处理中，请勿重复提交"
    return True, ""


def apply_refund(
    order_id: str,
    reason: str,
    idempotency_key: Optional[str] = None,
) -> dict:
    """为指定订单申请退款（同一幂等键只提交一次）。"""
    key = idempotency_key or ""

    # 进程内锁 + SQLite 事务双重保护：避免同进程竞态，并让多进程共享唯一键。
    with _refund_lock:
        conn = _connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            previous = _load_idempotent(conn, key)
            if previous is not None:
                conn.commit()
                return {**previous, "duplicate": True, "message": "退款已提交，请勿重复提交"}

            order = ORDERS.get(order_id)
            if not order:
                conn.rollback()
                return standard_result(
                    status="error", error_type="not_found",
                    message=f"未找到订单 {order_id}，请核实订单号",
                    idempotency_key=key,
                )

            eligible, reject_reason = _eligible(order)
            if not eligible:
                conn.rollback()
                return standard_result(
                    status="error", error_type="forbidden",
                    message=reject_reason, idempotency_key=key,
                )

            if order.get("status") == "pending":
                message = (
                    f"订单 {order_id} 尚未发货，已直接取消并发起退款。"
                    f"退款原因：{reason}。退款将在 1-3 个工作日内原路退回。"
                )
            else:
                message = (
                    f"退款申请已提交。订单 {order_id}，退款原因：{reason}。"
                    f"预计 1-3 个工作日内审核完成，届时会通知您退货地址。"
                )

            request_id = uuid.uuid4().hex
            result = standard_result(
                status="success", message=message,
                request_id=request_id, idempotency_key=key,
            )

            # 同一事务写入幂等结果与审计记录，避免“已执行但没记账”。
            if key:
                conn.execute(
                    "INSERT INTO refund_idempotency "
                    "(idempotency_key, result_json, created_at) VALUES (?, ?, ?)",
                    (key, json.dumps(result, ensure_ascii=False), _now()),
                )
            conn.execute(
                "INSERT INTO refund_audit "
                "(event, time, order_id, reason, request_id, idempotency_key, order_status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "refund_submitted", _now(), order_id, reason,
                    request_id, key, order.get("status"),
                ),
            )
            conn.commit()
            return result
        except sqlite3.IntegrityError:
            # 另一个进程刚写入同一 key；读取已提交结果，不重复执行。
            conn.rollback()
            previous = _load_idempotent(conn, key)
            if previous is not None:
                return {**previous, "duplicate": True, "message": "退款已提交，请勿重复提交"}
            raise
        finally:
            conn.close()


def get_refund_audit_log() -> list[dict]:
    """返回持久化审计日志副本，按写入顺序排列。"""
    with _refund_lock:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT event, time, order_id, reason, request_id, "
                "idempotency_key, order_status FROM refund_audit ORDER BY id"
            ).fetchall()
            columns = [
                "event", "time", "order_id", "reason", "request_id",
                "idempotency_key", "order_status",
            ]
            return [dict(zip(columns, row)) for row in rows]
        finally:
            conn.close()


def clear_refund_idempotency() -> None:
    """清空退款幂等与审计数据，仅供测试隔离使用。"""
    with _refund_lock:
        conn = _connect()
        try:
            conn.execute("DELETE FROM refund_idempotency")
            conn.execute("DELETE FROM refund_audit")
            conn.commit()
        finally:
            conn.close()
