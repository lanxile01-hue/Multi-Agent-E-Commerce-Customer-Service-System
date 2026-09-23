"""服务化 API 的路由处理器（与传输层解耦，便于单测）。

统一使用一个 SupervisorAgent 实例；对外只暴露结构化 CustomerServiceResponse，
绝不暴露内部思考、工具参数或原始 checkpoint 状态。
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from app.multi_agent.supervisor import SessionBusyError, SupervisorAgent

from .schemas import response_to_dict


class RateLimiter:
    """进程内固定窗口限流（按客户端 key）。"""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > self.window_seconds:
                q.popleft()
            if len(q) >= self.max_requests:
                return False
            q.append(now)
            return True


class APIHandler:
    """把 HTTP 语义与 SupervisorAgent 解耦。"""

    def __init__(self, agent: SupervisorAgent, rate_limiter: RateLimiter | None = None):
        self.agent = agent
        self.rate_limiter = rate_limiter or RateLimiter()

    # ---------- 各端点 ----------

    def health(self) -> tuple[int, dict]:
        return 200, {"status": "ok"}

    def chat(self, user_input: str, thread_id: str | None, client_key: str) -> tuple[int, dict]:
        if not self.rate_limiter.allow(client_key):
            return 429, {"error": "请求过于频繁，请稍后再试", "error_type": "rate_limited"}
        try:
            resp = self.agent.chat(user_input, thread_id=thread_id)
            return 200, response_to_dict(resp)
        except SessionBusyError as e:
            return 409, {"error": str(e), "error_type": "session_busy"}
        except Exception:  # noqa: BLE001
            return 500, {"error": "服务内部错误", "error_type": "internal", "trace_id": self._trace_id()}

    def resume(self, thread_id: str, decision: str, client_key: str) -> tuple[int, dict]:
        if not self.rate_limiter.allow(client_key):
            return 429, {"error": "请求过于频繁，请稍后再试", "error_type": "rate_limited"}
        try:
            resp = self.agent.resume(thread_id, decision)
            return 200, response_to_dict(resp)
        except SessionBusyError as e:
            return 409, {"error": str(e), "error_type": "session_busy"}
        except Exception:  # noqa: BLE001
            return 500, {"error": "服务内部错误", "error_type": "internal", "trace_id": self._trace_id()}

    def reset(self, thread_id: str | None) -> tuple[int, dict]:
        self.agent.reset(thread_id=thread_id)
        return 200, {"ok": True, "thread_id": thread_id or self.agent.thread_id}

    def state(self, thread_id: str) -> tuple[int, dict]:
        """返回脱敏后的会话状态（不暴露原始 checkpoint 内部结构）。"""
        snap = self.agent.get_state(thread_id)
        if snap is None:
            return 200, {"thread_id": thread_id, "interrupted": False, "has_checkpoint": False}
        return 200, {
            "thread_id": thread_id,
            "interrupted": bool(snap.next),
            "has_checkpoint": bool(snap.values),
        }

    # ---------- 内部 ----------

    def _trace_id(self) -> str:
        return self.agent.last_trace.get("trace_id", "")


def to_response_body(payload: dict) -> bytes:
    import json

    return json.dumps(payload, ensure_ascii=False).encode("utf-8")
