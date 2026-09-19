"""知识库文档写入与异步索引任务管理。

第一阶段只做单进程任务管理：任务状态保存在内存，索引文件由当前配置的后端持久化。
后续可以把 task store 替换为数据库或消息队列，不改变 API 契约。
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.agent.rag.backends import create_backend
from app.agent.rag.backends.bm25_backend import BM25Backend
from app.agent.rag.chunker import chunk_markdown_dir
from app.agent.rag.embedder import Embedder
from app.agent.tools.knowledge import reset_retriever
from app.config.settings import settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class IndexTask:
    task_id: str
    status: str = "pending"
    backend: str = ""
    created_at: str = ""
    started_at: str = ""
    finished_at: str = ""
    total_documents: int = 0
    total_chunks: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class KnowledgeIndexManager:
    """单进程、单 worker 的索引任务管理器。"""

    def __init__(self, max_workers: int = 1):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="kb-index")
        self._tasks: dict[str, IndexTask] = {}
        self._lock = threading.Lock()

    def submit(self, backend: str | None = None) -> IndexTask:
        name = (backend or settings.rag_backend).lower()
        if name not in {"numpy", "chroma"}:
            raise ValueError("backend 只支持 numpy 或 chroma")
        task = IndexTask(task_id=uuid.uuid4().hex, backend=name, created_at=_now())
        with self._lock:
            self._tasks[task.task_id] = task
        self._executor.submit(self._run, task.task_id)
        return task

    def get(self, task_id: str) -> IndexTask | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return IndexTask(**asdict(task)) if task else None

    def _update(self, task_id: str, **changes: Any) -> None:
        with self._lock:
            task = self._tasks[task_id]
            for key, value in changes.items():
                setattr(task, key, value)

    def _path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else Path(__file__).resolve().parents[2] / path

    def _run(self, task_id: str) -> None:
        task = self.get(task_id)
        if task is None:
            return
        self._update(task_id, status="running", started_at=_now())
        try:
            kb_dir = self._path(settings.kb_dir)
            chunks = chunk_markdown_dir(kb_dir)
            if not chunks:
                raise ValueError("知识库目录中没有可索引的 Markdown 文档")

            embedder = Embedder(
                api_key=settings.embedding_api_key or settings.openai_api_key,
                base_url=settings.embedding_api_base or settings.openai_base_url,
                model=settings.embedding_model,
                batch_size=settings.embedding_batch_size,
            )
            vectors = embedder.encode([chunk.text for chunk in chunks])
            if task.backend == "numpy":
                vector_backend = create_backend("numpy", index_path=self._path(settings.kb_index_path))
            else:
                vector_backend = create_backend(
                    "chroma",
                    persist_dir=self._path(settings.chroma_persist_dir),
                    collection_name=settings.chroma_collection,
                )
            vector_backend.upsert(chunks, vectors, settings.embedding_model)

            bm25 = BM25Backend(index_path=self._path(settings.bm25_index_path))
            bm25.upsert(chunks)
            reset_retriever()
            self._update(
                task_id,
                status="success",
                finished_at=_now(),
                total_documents=len({chunk.doc for chunk in chunks}),
                total_chunks=len(chunks),
            )
        except Exception as exc:  # noqa: BLE001 - 任务状态必须记录失败原因
            self._update(task_id, status="failed", finished_at=_now(), error=str(exc))


index_manager = KnowledgeIndexManager()
