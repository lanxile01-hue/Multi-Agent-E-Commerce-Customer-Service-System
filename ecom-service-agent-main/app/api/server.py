"""服务化 HTTP API（基于标准库 http.server，零额外依赖）。

启动：python -m app.api.server
不破坏 CLI（main.py 不受影响），API 复用同一个 SupervisorAgent。
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from app.multi_agent.supervisor import SupervisorAgent

from .routes import APIHandler

WEB_INDEX = Path(__file__).resolve().parents[1] / "web" / "index.html"


class _APIRequestHandler(BaseHTTPRequestHandler):
    handler: APIHandler = None  # 由 server 注入

    server_version = "EcomServiceAgent/1.0"

    def _client_key(self) -> str:
        return self.client_address[0] if self.client_address else "default"

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _send(self, status: int, body: dict) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_bytes(self, status: int, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _route(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        method = self.command
        h = self.handler

        # 本地单页控制台：与 API 使用同一个 localhost 服务，零额外前端依赖。
        if method == "GET" and path in {"", "/", "/index.html"}:
            try:
                self._send_bytes(200, WEB_INDEX.read_bytes(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send(404, {"error": "web console not found", "error_type": "not_found"})
            return

        # GET /health
        if method == "GET" and path == "/health":
            status, body = h.health()
            self._send(status, body)
            return

        # GET /knowledge：知识文档目录与版本元数据
        if method == "GET" and path == "/knowledge":
            status, body = h.knowledge_catalog()
            self._send(status, body)
            return

        # POST /knowledge/reload：清理进程内检索器缓存
        if method == "POST" and path == "/knowledge/reload":
            status, body = h.reload_knowledge()
            self._send(status, body)
            return

        # POST /knowledge/documents：写入 Markdown 文档并异步索引
        if method == "POST" and path == "/knowledge/documents":
            data = self._read_json()
            status, body = h.upsert_knowledge_document(
                data.get("filename", ""), data.get("content", ""), data.get("metadata"),
            )
            self._send(status, body)
            return

        # POST /knowledge/index：提交全量索引任务
        if method == "POST" and path == "/knowledge/index":
            data = self._read_json()
            status, body = h.create_index_task(data.get("backend"))
            self._send(status, body)
            return

        # GET /knowledge/documents/{filename}/versions
        if method == "GET" and path.startswith("/knowledge/documents/") and path.endswith("/versions"):
            filename = unquote(path.split("/")[3])
            status, body = h.list_document_versions(filename)
            self._send(status, body)
            return

        # POST /knowledge/documents/{filename}/rollback
        if method == "POST" and path.startswith("/knowledge/documents/") and path.endswith("/rollback"):
            filename = unquote(path.split("/")[3])
            data = self._read_json()
            status, body = h.rollback_document(filename, data.get("version", ""))
            self._send(status, body)
            return

        # DELETE /knowledge/documents/{filename}：移动到可恢复回收站
        if method == "DELETE" and path.startswith("/knowledge/documents/"):
            filename = unquote(path.split("/")[3])
            status, body = h.delete_document(filename)
            self._send(status, body)
            return

        # GET /knowledge/index/{task_id}：查询异步索引任务
        if method == "GET" and path.startswith("/knowledge/index/"):
            task_id = unquote(path.rsplit("/", 1)[-1])
            status, body = h.index_task(task_id)
            self._send(status, body)
            return

        # POST /chat
        if method == "POST" and path == "/chat":
            data = self._read_json()
            status, body = h.chat(
                data.get("user_input", ""), data.get("thread_id"), self._client_key(),
                data.get("tenant_id", "default"), data.get("user_id", "anonymous"),
                data.get("role", "member"), data.get("departments", []),
            )
            self._send(status, body)
            return

        # /chat/{thread_id}/resume | reset | state
        parts = path.split("/")
        # 期望形如 ["", "chat", "{thread_id}", "resume"]
        if len(parts) >= 4 and parts[1] == "chat":
            thread_id = parts[2]
            action = parts[3]
            if method == "POST" and action == "resume":
                data = self._read_json()
                status, body = h.resume(thread_id, data.get("decision", ""), self._client_key())
                self._send(status, body)
                return
            if method == "POST" and action == "reset":
                status, body = h.reset(thread_id)
                self._send(status, body)
                return
            if method == "GET" and action == "state":
                status, body = h.state(thread_id)
                self._send(status, body)
                return

        self._send(404, {"error": "not found", "error_type": "not_found"})

    def do_GET(self) -> None:
        self._route()

    def do_POST(self) -> None:
        self._route()

    def do_DELETE(self) -> None:
        self._route()

    def log_message(self, format, *args):  # noqa: A002 - 静默默认访问日志，避免泄漏
        pass


def create_server(agent: SupervisorAgent, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    """构建带统一 SupervisorAgent 的 HTTP 服务。"""
    handler_cls = type("Handler", (_APIRequestHandler,), {"handler": APIHandler(agent)})
    server = ThreadingHTTPServer((host, port), handler_cls)
    server.timeout = 30  # 请求超时（socket poll 间隔）
    return server


def main() -> None:
    from app.config.settings import settings

    agent = SupervisorAgent()
    host = "127.0.0.1"
    port = 8000
    server = create_server(agent, host=host, port=port)
    print(f"✅ Ecom-Service-Agent API 已启动: http://{host}:{port} (backend={settings.checkpoint_backend})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在关闭...")
    finally:
        server.server_close()
        agent.close()


if __name__ == "__main__":
    main()
