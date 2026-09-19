"""Checkpointer 工厂与清理工具。

后端语义（务必区分，不要混用）：
- memory  : MemorySaver，**仅进程内**。进程重启即丢失，**不能跨进程恢复**，
            不能持久化 interrupt 状态。
- sqlite  : SqliteSaver，写入磁盘（CHECKPOINT_PATH），**可跨进程恢复**，
            interrupt 状态随 checkpoint 落盘，进程重启后可继续 resume。
- postgres: 未来扩展的后端（需额外安装 langgraph-checkpoint-postgres），
            用于多实例共享持久化；当前未安装时安全回退。

Checkpointer 协作约定（避免重复嵌套）：
- 只有父图 compile 时传入 checkpointer。
- 三个子图作为父图节点运行，不单独携带 checkpointer，继承父图 thread_id。
- 子图内部 interrupt() 冒泡到父图，父图整体暂停/恢复。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

# 支持的后端标识
BACKEND_MEMORY = "memory"
BACKEND_SQLITE = "sqlite"
BACKEND_POSTGRES = "postgres"
BACKEND_AUTO = "auto"


@dataclass
class CheckpointerHandle:
    """checkpointer 句柄：持有 saver、后端类型与底层连接，便于 close / reset 清理。"""

    saver: Any
    kind: str  # "memory" | "sqlite" | "postgres"
    conn: Any = None           # sqlite 连接（memory 为 None）
    db_path: str = ""          # sqlite 文件路径 / postgres 连接串
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def is_persistent(self) -> bool:
        """是否磁盘持久化（可跨进程）。memory 返回 False。"""
        return self.kind in (BACKEND_SQLITE, BACKEND_POSTGRES)

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """幂等关闭底层连接。关闭后普通 session JSON 保存不受影响。"""
        if self._closed:
            return
        self._closed = True
        conn = self.conn or getattr(self.saver, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        self.conn = None


def build_checkpointer(
    enabled: bool = True,
    backend: str = BACKEND_MEMORY,
    path: str = "",
) -> Optional[CheckpointerHandle]:
    """构建 checkpointer。

    - enabled=False 返回 None（图不带 checkpointer，interrupt 不可用）
    - backend="memory"  ：MemorySaver（仅进程内）
    - backend="sqlite"  ：SqliteSaver（磁盘持久化，可跨进程）
    - backend="postgres"：PostgresSaver（未来扩展，未安装则回退）
    - backend="auto"    ：优先 sqlite，再 postgres，最后回退 memory

    CHECKPOINT_PATH 只在 sqlite（文件路径）与 postgres（连接串）下生效。
    初始化失败一律回退到 memory，并打印告警，不抛异常保证可运行。
    """
    if not enabled:
        return None

    backend = (backend or BACKEND_MEMORY).lower()

    if backend == BACKEND_MEMORY:
        return _build_memory()

    if backend == BACKEND_SQLITE:
        handle = _build_sqlite(path)
        if handle is not None:
            return handle
        print("⚠️  [Checkpoint] SQLite 初始化失败，回退到 MemorySaver（仅进程内）")
        return _build_memory()

    if backend == BACKEND_POSTGRES:
        handle = _build_postgres(path)
        if handle is not None:
            return handle
        print("⚠️  [Checkpoint] Postgres 初始化失败，回退到 MemorySaver（仅进程内）")
        return _build_memory()

    if backend == BACKEND_AUTO:
        for handle in (_build_sqlite(path), _build_postgres(path)):
            if handle is not None:
                return handle
        print("⚠️  [Checkpoint] 无可用持久化后端，回退到 MemorySaver（仅进程内）")
        return _build_memory()

    print(f"⚠️  [Checkpoint] 未知后端 {backend!r}，回退到 MemorySaver（仅进程内）")
    return _build_memory()


def _build_sqlite(path: str) -> Optional[CheckpointerHandle]:
    """构建 SqliteSaver。主动 setup() 以便尽早发现损坏的 db 文件。"""
    conn = None
    try:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        db_path = path or _default_sqlite_path()
        parent = os.path.dirname(os.path.abspath(db_path))
        os.makedirs(parent, exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        saver = SqliteSaver(conn)
        # connect 是惰性的，损坏文件要到真正读写时才会报错；提前建表可尽早暴露。
        saver.setup()
        return CheckpointerHandle(saver=saver, kind=BACKEND_SQLITE, conn=conn, db_path=db_path)
    except Exception as e:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        print(f"⚠️  [Checkpoint] SqliteSaver 初始化异常（可能文件损坏）({e})")
        return None


def _build_postgres(conn_string: str) -> Optional[CheckpointerHandle]:
    """Postgres 后端（未来扩展，最小接入）。

    未安装 langgraph-checkpoint-postgres 或未提供连接串时返回 None，由调用方回退。
    """
    if not conn_string:
        print("⚠️  [Checkpoint] Postgres 后端需要连接串（CHECKPOINT_PATH），未配置则忽略")
        return None
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError:
        print("⚠️  [Checkpoint] 未安装 langgraph-checkpoint-postgres，无法使用 Postgres 后端")
        return None
    try:
        saver = PostgresSaver.from_conn_string(conn_string)
        saver.setup()
        return CheckpointerHandle(saver=saver, kind=BACKEND_POSTGRES, db_path=conn_string)
    except Exception as e:
        print(f"⚠️  [Checkpoint] PostgresSaver 初始化异常 ({e})")
        return None


def _build_memory() -> CheckpointerHandle:
    from langgraph.checkpoint.memory import MemorySaver

    return CheckpointerHandle(saver=MemorySaver(), kind=BACKEND_MEMORY)


def _default_sqlite_path() -> str:
    return os.path.join(
        os.path.dirname(__file__), "..", "sessions", "checkpoints.db"
    )


def delete_thread_checkpoint(checkpointer, thread_id: str) -> bool:
    """安全删除单个 thread 的 checkpoint 及其 interrupt 状态（reset 用）。

    返回是否成功。checkpointer 为 None 视为成功；不支持 delete_thread 返回 False。
    """
    if checkpointer is None:
        return True
    delete = getattr(checkpointer, "delete_thread", None)
    if delete is None:
        return False
    try:
        delete(thread_id)
        return True
    except Exception:
        return False


def thread_id_from_session(session_path: str, user_id: str = "default") -> str:
    """从 session 路径与用户 ID 派生稳定 thread_id。

    保证同一用户同一会话跨进程重启后 thread_id 不变，从而复用 checkpoint。
    """
    import hashlib

    raw = f"{user_id}:{session_path}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]
    return f"{user_id}-{digest}"