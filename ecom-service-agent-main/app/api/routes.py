"""服务化 API 的路由处理器（与传输层解耦，便于单测）。

统一使用一个 SupervisorAgent 实例；对外只暴露结构化 CustomerServiceResponse，
绝不暴露内部思考、工具参数或原始 checkpoint 状态。
"""

from __future__ import annotations

import threading
import time
import shutil
from collections import defaultdict, deque
from pathlib import Path

from app.multi_agent.supervisor import SessionBusyError, SupervisorAgent
from app.agent.rag.chunker import _parse_frontmatter, chunk_markdown_dir
from app.agent.tools.knowledge import reset_retriever
from app.config.settings import settings
from app.knowledge.index_manager import index_manager
from app.knowledge.access import AccessContext, reset_access_context, set_access_context

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

    @staticmethod
    def _knowledge_dir() -> Path:
        path = Path(settings.kb_dir)
        return path if path.is_absolute() else Path(__file__).resolve().parents[2] / path

    @classmethod
    def _document_path(cls, filename: str) -> Path | None:
        if not filename or Path(filename).name != filename or not filename.lower().endswith(".md"):
            return None
        return cls._knowledge_dir() / filename

    @classmethod
    def _archive_dir(cls, filename: str) -> Path:
        return cls._knowledge_dir() / ".versions" / filename

    # ---------- 各端点 ----------

    def health(self) -> tuple[int, dict]:
        return 200, {"status": "ok"}

    def knowledge_catalog(self) -> tuple[int, dict]:
        """返回知识文档目录和可用于引用的版本元数据。"""
        kb_dir = self._knowledge_dir()
        if not kb_dir.exists():
            return 200, {"documents": [], "total_documents": 0, "total_chunks": 0}

        chunks = chunk_markdown_dir(kb_dir)
        docs: dict[str, dict] = {}
        for chunk in chunks:
            item = docs.setdefault(chunk.doc, {
                "doc": chunk.doc,
                "title": chunk.title or chunk.doc,
                "source": chunk.source or chunk.doc,
                "version": chunk.version,
                "effective_time": chunk.effective_time,
                "updated_at": chunk.updated_at,
                "department": chunk.department,
                "access_level": chunk.access_level,
                "tags": chunk.tags or [],
                "chunk_count": 0,
            })
            item["chunk_count"] += 1
        return 200, {
            "documents": list(docs.values()),
            "total_documents": len(docs),
            "total_chunks": len(chunks),
        }

    def reload_knowledge(self) -> tuple[int, dict]:
        """让下一次问答重新加载索引；索引构建仍由离线命令负责。"""
        reset_retriever()
        status, body = self.knowledge_catalog()
        body["reloaded"] = True
        return status, body

    def upsert_knowledge_document(
        self,
        filename: str,
        content: str,
        metadata: dict | None = None,
    ) -> tuple[int, dict]:
        """写入一个 Markdown 知识文档，并异步提交索引任务。"""
        filename = (filename or "").strip()
        if not filename or Path(filename).name != filename or not filename.lower().endswith(".md"):
            return 400, {"error": "filename 必须是当前知识库目录下的 .md 文件名", "error_type": "invalid_filename"}
        if not isinstance(content, str) or not content.strip():
            return 400, {"error": "content 不能为空", "error_type": "invalid_content"}
        kb_dir = Path(settings.kb_dir)
        if not kb_dir.is_absolute():
            kb_dir = Path(__file__).resolve().parents[2] / kb_dir
        kb_dir.mkdir(parents=True, exist_ok=True)

        metadata = metadata or {}
        allowed = {
            "title", "source", "version", "effective_time", "updated_at",
            "department", "access_level", "tags", "tenant_id",
        }
        frontmatter: list[str] = []
        for key in allowed:
            value = metadata.get(key)
            if value is None or value == "":
                continue
            if isinstance(value, list):
                value = ",".join(str(item).strip() for item in value if str(item).strip())
            value = str(value).replace("\n", " ").strip()
            frontmatter.append(f"{key}: {value}")
        body = content
        if frontmatter:
            body = "---\n" + "\n".join(frontmatter) + "\n---\n\n" + content.lstrip()
        target = self._document_path(filename)
        assert target is not None
        if target.exists():
            old_meta, _ = _parse_frontmatter(target.read_text(encoding="utf-8"))
            archive_key = str(old_meta.get("version") or f"archived-{int(target.stat().st_mtime)}")
            archive_key = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in archive_key)
            archive = self._archive_dir(filename) / f"{archive_key}.md"
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, archive)
        target.write_text(body, encoding="utf-8")
        task = index_manager.submit()
        return 202, {
            "accepted": True,
            "filename": filename,
            "task_id": task.task_id,
            "status": task.status,
        }

    def list_document_versions(self, filename: str) -> tuple[int, dict]:
        target = self._document_path(filename)
        if target is None:
            return 400, {"error": "filename 必须是 .md 文件名", "error_type": "invalid_filename"}
        archive_dir = self._archive_dir(filename)
        versions = sorted(path.stem for path in archive_dir.glob("*.md")) if archive_dir.exists() else []
        return 200, {"filename": filename, "current": target.exists(), "versions": versions}

    def rollback_document(self, filename: str, version: str) -> tuple[int, dict]:
        target = self._document_path(filename)
        if target is None or not version or Path(version).name != version:
            return 400, {"error": "filename 或 version 不合法", "error_type": "invalid_version"}
        archive = self._archive_dir(filename) / f"{version}.md"
        if not archive.exists():
            return 404, {"error": "文档版本不存在", "error_type": "version_not_found"}
        if target.exists():
            backup = self._archive_dir(filename) / f"rollback-backup-{int(time.time())}.md"
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
        shutil.copy2(archive, target)
        task = index_manager.submit()
        return 202, {"accepted": True, "filename": filename, "version": version, "task_id": task.task_id}

    def delete_document(self, filename: str) -> tuple[int, dict]:
        target = self._document_path(filename)
        if target is None:
            return 400, {"error": "filename 必须是 .md 文件名", "error_type": "invalid_filename"}
        if not target.exists():
            return 404, {"error": "文档不存在", "error_type": "document_not_found"}
        trash = self._knowledge_dir() / ".trash" / f"{int(time.time())}-{filename}"
        trash.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target), str(trash))
        task = index_manager.submit()
        return 202, {"accepted": True, "filename": filename, "task_id": task.task_id, "recoverable_path": str(trash)}

    def create_index_task(self, backend: str | None = None) -> tuple[int, dict]:
        try:
            task = index_manager.submit(backend)
        except ValueError as exc:
            return 400, {"error": str(exc), "error_type": "invalid_backend"}
        return 202, task.to_dict()

    def index_task(self, task_id: str) -> tuple[int, dict]:
        task = index_manager.get(task_id)
        if task is None:
            return 404, {"error": "索引任务不存在", "error_type": "task_not_found"}
        return 200, task.to_dict()

    def chat(
        self,
        user_input: str,
        thread_id: str | None,
        client_key: str,
        tenant_id: str = "default",
        user_id: str = "anonymous",
        role: str = "member",
        departments: list[str] | None = None,
    ) -> tuple[int, dict]:
        if not self.rate_limiter.allow(client_key):
            return 429, {"error": "请求过于频繁，请稍后再试", "error_type": "rate_limited"}
        try:
            token = set_access_context(AccessContext(
                tenant_id=tenant_id or "default",
                user_id=user_id or "anonymous",
                role=role or "member",
                departments=tuple(departments or ["*"]),
            ))
            try:
                resp = self.agent.chat(user_input, thread_id=thread_id)
            finally:
                reset_access_context(token)
            return 200, response_to_dict(resp)
        except SessionBusyError as e:
            return 409, {"error": str(e), "error_type": "session_busy"}
        except Exception as e:  # noqa: BLE001
            return 500, {"error": "服务内部错误", "error_type": "internal", "trace_id": self._trace_id()}

    def resume(self, thread_id: str, decision: str, client_key: str) -> tuple[int, dict]:
        if not self.rate_limiter.allow(client_key):
            return 429, {"error": "请求过于频繁，请稍后再试", "error_type": "rate_limited"}
        try:
            resp = self.agent.resume(thread_id, decision)
            return 200, response_to_dict(resp)
        except SessionBusyError as e:
            return 409, {"error": str(e), "error_type": "session_busy"}
        except Exception as e:  # noqa: BLE001
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
