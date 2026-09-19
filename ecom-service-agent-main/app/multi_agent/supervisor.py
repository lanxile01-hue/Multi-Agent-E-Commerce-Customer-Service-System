"""父级 Supervisor Graph：调度三个领域子图，统一校验与持久化。

流程：
    START → preprocess
        ├── blocked → validate_response → persist → END
        └── allowed → route → (run_presale | run_postsale | run_complaint)
                    → validate_response → persist → END

职责边界：
- 父图只负责调度、路由、校验、持久化，不参与领域工具调用。
- 子图只负责领域任务（推理/工具/回复）。
- 只有父图携带 checkpointer；子图作为节点运行并继承父图 thread_id。
- 子图内部 interrupt()（人工确认）冒泡到父图，父图整体暂停/恢复。
- 是否等待人工确认，依据 checkpointer snapshot 的 next/interrupt 判断，不依赖内存字典，
  因此进程重启后仍能恢复确认流程（前提：使用 sqlite 持久化 checkpointer）。

对外接口与 EcomAgent / MultiAgentOrchestrator 一致：
    chat(user_input, thread_id=...) / resume(thread_id, decision)
    reset(thread_id=...) / save() / close()
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.agent.guardrails import Guardrails
from app.agent.memory import MemoryManager
from app.agent.observability import sanitize_record
from app.agent.skills import SkillManager
from app.agent.storage import delete_session, load_session, save_session
from app.agent.tools.manager import ToolManager
from app.config.settings import settings
from app.multi_agent.agents import AGENT_CONFIGS
from app.multi_agent.checkpoint import (
    CheckpointerHandle,
    build_checkpointer,
    delete_thread_checkpoint,
    thread_id_from_session,
)
from app.multi_agent.router import Router
from app.multi_agent.state import SupervisorState, empty_state
from app.multi_agent.structured import (
    build_structured_response,
    deserialize_response,
    safe_default_response,
    serialize_response,
)
from app.multi_agent.subgraphs import (
    build_complaint_graph,
    build_postsale_graph,
    build_presale_graph,
)
from app.schemas.response import CustomerServiceResponse


class SessionBusyError(Exception):
    """同一 thread_id 正在被并发处理，拒绝重复请求以避免竞态（如重复退款）。"""


# 进程内 thread 级锁注册表（key=thread_id）。用于防止同一个 thread 被并发执行。
# 说明：threading.Lock 是进程内锁，能拦住同一进程的多线程并发；跨进程强一致需要
# 依赖 SQLite 事务锁或外部分布式锁，这里先提供轻量级进程内保障。
_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _thread_lock(thread_id: str) -> threading.Lock:
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(thread_id)
        if lock is None:
            lock = threading.Lock()
            _THREAD_LOCKS[thread_id] = lock
        return lock


@contextmanager
def _acquire_thread(thread_id: str):
    """对指定 thread_id 加锁；已被占用则立即抛 SessionBusyError（非阻塞拒绝）。"""
    lock = _thread_lock(thread_id)
    if not lock.acquire(blocking=False):
        raise SessionBusyError(f"会话 {thread_id} 正在处理中，请稍后再试")
    try:
        yield
    finally:
        lock.release()


class SupervisorAgent:
    """LangGraph Supervisor：父图 + 三个子图。

    支持依赖注入（client / tool_managers / checkpointer 等），便于单元测试 Mock，
    生产代码不包含任何测试逻辑。
    """

    def __init__(
        self,
        session_path: Optional[str] = None,
        client: Any = None,
        tool_managers: Optional[dict[str, ToolManager]] = None,
        checkpointer: Optional[CheckpointerHandle] = None,
        checkpoint_backend: Optional[str] = None,
        checkpoint_enabled: Optional[bool] = None,
        thread_id: Optional[str] = None,
        memory_enabled: Optional[bool] = None,
        skills_enabled: Optional[bool] = None,
        guardrails_enabled: Optional[bool] = None,
        tool_timeout: Optional[float] = None,
    ):
        self.client = client or _default_client()
        self.model = settings.model_name
        self.temperature = settings.temperature
        self.session_path = session_path or settings.session_path
        self.history_threshold = settings.history_threshold
        self.history_keep_recent = settings.history_keep_recent
        self.subgraph_max_steps = settings.subgraph_max_steps
        self.tool_timeout = settings.tool_timeout if tool_timeout is None else tool_timeout

        _memory_enabled = settings.memory_enabled if memory_enabled is None else memory_enabled
        _skills_enabled = settings.skills_enabled if skills_enabled is None else skills_enabled
        _guardrails_enabled = settings.guardrails_enabled if guardrails_enabled is None else guardrails_enabled

        self.router = Router(self.client, self.model)

        # 三个子图：各自独立 ToolManager，工具隔离由 allowed_tools 保证
        if tool_managers is not None:
            self.tool_managers = tool_managers
        else:
            self.tool_managers = {
                key: ToolManager(
                    use_mcp=settings.mcp_enabled,
                    mcp_server_url=settings.mcp_server_url,
                    allowed_tools=cfg["tools"],
                )
                for key, cfg in AGENT_CONFIGS.items()
            }

        self.memory_manager = MemoryManager(
            client=self.client,
            model=self.model,
            user_id=settings.memory_user_id,
            memory_dir=settings.memory_dir,
            memory_enabled=_memory_enabled,
            max_ltm_facts=settings.max_ltm_facts,
        )
        if _memory_enabled:
            from app.agent.tools.memory_tool import set_memory_manager
            set_memory_manager(self.memory_manager)

        self.skill_manager = SkillManager(
            skills_dir=settings.skills_dir,
            enabled=_skills_enabled,
        )
        if _skills_enabled:
            from app.agent.tools.skill_tool import set_skill_manager
            set_skill_manager(self.skill_manager)

        # Checkpointer：父图持有，子图继承
        if checkpointer is not None:
            self.checkpoint_handle = checkpointer
        else:
            _enabled = settings.checkpoint_enabled if checkpoint_enabled is None else checkpoint_enabled
            _backend = settings.checkpoint_backend if checkpoint_backend is None else checkpoint_backend
            self.checkpoint_handle = build_checkpointer(
                enabled=_enabled,
                backend=_backend,
                path=settings.checkpoint_path,
            )
        self.checkpointer = self.checkpoint_handle.saver if self.checkpoint_handle else None

        # 人工确认依赖 interrupt()，而 interrupt() 依赖 checkpointer；无 checkpointer 时自动降级关闭
        self.human_confirmation_enabled = (
            settings.human_confirmation_enabled and self.checkpointer is not None
        )
        self.thread_id = thread_id or thread_id_from_session(
            self.session_path, settings.memory_user_id,
        )

        self.raw_messages: list[dict] = []
        self.summary: Optional[str] = None

        loaded = load_session(self.session_path)
        if loaded:
            self.summary = loaded["summary"]
            self.raw_messages = loaded["messages"]
            if loaded.get("short_term_memory"):
                self.memory_manager.restore_stm(loaded["short_term_memory"])

        # 构建三个子图（不携带 checkpointer，由父图统一管理）
        subgraph_kwargs = dict(
            client=self.client,
            model=self.model,
            temperature=self.temperature,
            memory_manager=self.memory_manager,
            skill_manager=self.skill_manager,
            get_summary=lambda: self.summary,
            max_steps=self.subgraph_max_steps,
            human_confirmation_enabled=self.human_confirmation_enabled,
            tool_timeout=self.tool_timeout,
        )
        self.subgraphs: dict[str, CompiledStateGraph] = {
            "presale": build_presale_graph(
                tool_manager=self.tool_managers["presale"], **subgraph_kwargs,
            ),
            "postsale": build_postsale_graph(
                tool_manager=self.tool_managers["postsale"], **subgraph_kwargs,
            ),
            "complaint": build_complaint_graph(
                tool_manager=self.tool_managers["complaint"], **subgraph_kwargs,
            ),
        }

        self._guardrails = Guardrails(
            self.client, self.model, enabled=_guardrails_enabled,
        )
        self._closed = False
        self.last_trace: dict = {}
        self._graph = self._build_graph()

    @property
    def history_size(self) -> int:
        return len(self.raw_messages)

    # ---------- Public API ----------

    def chat(
        self,
        user_input: str,
        thread_id: Optional[str] = None,
        callbacks: Optional[list] = None,
    ) -> CustomerServiceResponse:
        """处理用户输入；若该 thread 处于 interrupt 暂停中，则视为确认决策。"""
        if self._closed:
            raise RuntimeError("SupervisorAgent 已关闭，无法继续对话")
        tid = thread_id or self.thread_id
        with _acquire_thread(tid):
            return self._chat_locked(user_input, tid, callbacks)

    def _chat_locked(
        self,
        user_input: str,
        tid: str,
        callbacks: Optional[list],
    ) -> CustomerServiceResponse:
        config = self._config(tid, callbacks)

        # 依据 checkpointer snapshot 判断该 thread 是否处于暂停（不依赖内存字典）
        if self.checkpointer is not None and self._is_interrupted(config):
            return self._resume_locked(tid, user_input, callbacks)

        self.raw_messages.append({"role": "user", "content": user_input})
        # 先落盘用户消息，保证进程重启后恢复确认流程时历史不丢失
        self.save()

        initial = self._initial_state(user_input, tid)
        result = self._graph.invoke(initial, config)

        if self.checkpointer is not None and self._is_interrupted(config):
            payload = self._interrupt_payload(self._graph.get_state(config))
            return self._waiting_response(payload)

        return self._finalize(result)

    def resume(
        self,
        thread_id: str,
        decision: str,
        callbacks: Optional[list] = None,
    ) -> CustomerServiceResponse:
        """在人工确认暂停后恢复执行。decision 为用户确认输入（确认/取消/重新说明）。"""
        if self._closed:
            raise RuntimeError("SupervisorAgent 已关闭，无法继续恢复")
        tid = thread_id or self.thread_id
        with _acquire_thread(tid):
            return self._resume_locked(tid, decision, callbacks)

    def _resume_locked(
        self,
        thread_id: str,
        decision: str,
        callbacks: Optional[list],
    ) -> CustomerServiceResponse:
        config = self._config(thread_id, callbacks)
        result = self._graph.invoke(Command(resume=decision), config)

        if self.checkpointer is not None and self._is_interrupted(config):
            payload = self._interrupt_payload(self._graph.get_state(config))
            return self._waiting_response(payload)

        return self._finalize(result)

    def reset(self, thread_id: Optional[str] = None):
        """清理指定 thread 的会话与 checkpoint；不带参数时重置主 thread。

        - 清理该 thread 的 checkpoint 与 interrupt 状态（含自定义 thread_id）。
        - 仅当目标是主 thread（tid == self.thread_id）时，同时清空 raw_messages /
          summary / 短期记忆 / session JSON，保持与旧版无参 reset() 行为一致。
        """
        tid = thread_id or self.thread_id
        with _acquire_thread(tid):
            self._reset_locked(tid)

    def _reset_locked(self, tid: str):
        if tid == self.thread_id:
            self.raw_messages = []
            self.summary = None
            self.memory_manager.reset_short_term()
            delete_session(self.session_path)
        # 无论是否主 thread，都清理该 thread 的 checkpoint（含 interrupt 状态）
        if self.checkpointer is not None:
            delete_thread_checkpoint(self.checkpointer, tid)

    def save(self) -> None:
        save_session(
            self.session_path, self.raw_messages, self.summary,
            short_term_memory=self.memory_manager.stm_to_dict(),
        )

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.memory_manager.consolidate_to_long_term(self.raw_messages, self.summary)
        for tm in self.tool_managers.values():
            tm.close()
        if self.checkpoint_handle is not None:
            self.checkpoint_handle.close()

    def get_state(self, thread_id: Optional[str] = None):
        """暴露 checkpoint 状态供测试/调试。"""
        if self.checkpointer is None:
            return None
        return self._graph.get_state(self._config(thread_id or self.thread_id))

    # ---------- Graph Construction ----------

    def _build_graph(self) -> CompiledStateGraph:
        builder = StateGraph(SupervisorState)

        builder.add_node("preprocess", self._node_preprocess)
        builder.add_node("route", self._node_route)
        builder.add_node("run_presale", self.subgraphs["presale"])
        builder.add_node("run_postsale", self.subgraphs["postsale"])
        builder.add_node("run_complaint", self.subgraphs["complaint"])
        builder.add_node("validate_response", self._node_validate_response)
        builder.add_node("persist", self._node_persist)

        builder.add_edge(START, "preprocess")
        # 越界请求直接进入校验/结束，不进入 Router，也不调用工具
        builder.add_conditional_edges(
            "preprocess",
            self._route_after_preprocess,
            {"route": "route", "validate_response": "validate_response"},
        )
        builder.add_conditional_edges(
            "route",
            self._route_after_route,
            {
                "run_presale": "run_presale",
                "run_postsale": "run_postsale",
                "run_complaint": "run_complaint",
            },
        )
        for key in ("presale", "postsale", "complaint"):
            builder.add_edge(f"run_{key}", "validate_response")
        builder.add_edge("validate_response", "persist")
        builder.add_edge("persist", END)

        return builder.compile(checkpointer=self.checkpointer)

    # ---------- Parent Nodes ----------

    def _node_preprocess(self, state: SupervisorState) -> dict:
        """输入安全检查与预处理：越界请求直接阻断。"""
        user_input = (state.get("user_input") or "").strip()
        meta = dict(state.get("metadata", {}))
        meta.setdefault("start_time", time.time())
        meta.setdefault("model_calls", 0)
        meta.setdefault("tool_calls", 0)
        meta.setdefault("retry_count", 0)

        # 可选边界检查（默认关闭，避免额外 LLM 调用）
        if self._guardrails.enabled:
            boundary = self._guardrails.check_boundary(user_input)
            if not boundary.in_scope:
                safe = build_structured_response(
                    "抱歉，这个问题超出了我能处理的范围，已为您转接人工客服。",
                    selected_agent="",
                    requires_human=True,
                )
                return {
                    "user_input": user_input,
                    "blocked": True,
                    "requires_human": True,
                    "final_response": safe.reply,
                    "structured_response": serialize_response(safe),
                    "metadata": meta,
                }

        return {"user_input": user_input, "blocked": False, "metadata": meta}

    def _node_route(self, state: SupervisorState) -> dict:
        """调用 Router 选择子 Agent。"""
        selected = self.router.route(state["user_input"], state.get("conversation", []))
        print(f"\n[Supervisor·路由] → {selected}")
        meta = dict(state.get("metadata", {}))
        meta["selected_agent"] = selected
        return {"selected_agent": selected, "intent": selected, "metadata": meta}

    def _node_validate_response(self, state: SupervisorState) -> dict:
        """统一校验结构化响应，缺失时兜底，绝不把内部信息返回给用户。"""
        final_text = state.get("final_response", "")
        structured = state.get("structured_response", {}) or {}

        resp = deserialize_response(structured) if structured else None
        if resp is None or not resp.reply:
            resp = safe_default_response(final_text, state.get("selected_agent", ""))

        # 越界阻断时强制转人工
        if state.get("blocked", False):
            resp.requires_human = True

        # 确保 reply 不包含内部标记（工具/思考等）
        reply = resp.reply
        for marker in ("[调用工具", "[思考", "load_skill", "tool_calls"):
            if marker in reply:
                reply = "抱歉，我暂时无法处理您的请求。"
                break

        validated = CustomerServiceResponse(
            intent=resp.intent,
            confidence=resp.confidence,
            reply=reply,
            requires_human=resp.requires_human,
            follow_up_question=resp.follow_up_question,
        )
        return {
            "final_response": reply,
            "requires_human": validated.requires_human,
            "structured_response": serialize_response(validated),
        }

    def _node_persist(self, state: SupervisorState) -> dict:
        """记录持久化标记与总耗时（真实 JSON 落盘在 chat() 中完成，避免节点改外部对象）。"""
        meta = dict(state.get("metadata", {}))
        if "start_time" in meta:
            meta["elapsed_seconds"] = round(time.time() - meta["start_time"], 3)
        meta["persisted"] = True
        meta["node_name"] = "persist"
        if state.get("error"):
            meta["error"] = state["error"]
        return {"metadata": meta}

    # ---------- Router helpers ----------

    @staticmethod
    def _route_after_preprocess(state: SupervisorState) -> str:
        return "validate_response" if state.get("blocked", False) else "route"

    @staticmethod
    def _route_after_route(state: SupervisorState) -> str:
        selected = state.get("selected_agent", "")
        if selected in ("presale", "postsale", "complaint"):
            return f"run_{selected}"
        return "run_postsale"

    # ---------- Internal helpers ----------

    def _initial_state(self, user_input: str, thread_id: str) -> SupervisorState:
        state = empty_state()
        state["user_input"] = user_input
        state["conversation"] = list(self.raw_messages)  # 复制，避免节点改动外部对象
        state["messages"] = self._build_messages()
        state["metadata"] = {
            "trace_id": uuid.uuid4().hex,
            "thread_id": thread_id,
            "user_id": settings.memory_user_id,
            "model_name": self.model,
            "checkpoint_backend": self.checkpoint_handle.kind if self.checkpoint_handle else "none",
            "start_time": time.time(),
            "model_calls": 0,
            "tool_calls": 0,
            "retry_count": 0,
            "model_latency": 0.0,
            "tool_latency": 0.0,
        }
        return state

    def _config(self, thread_id: str, callbacks: Optional[list] = None) -> dict:
        cfg: dict = {"configurable": {"thread_id": thread_id}}
        if callbacks is not None:
            cfg["callbacks"] = callbacks
        return cfg

    def _is_interrupted(self, config: dict) -> bool:
        """根据 checkpoint snapshot 的 next 节点判断该 thread 是否处于 interrupt 暂停。"""
        try:
            snapshot = self._graph.get_state(config)
            return bool(snapshot.next)
        except Exception:
            return False

    def _interrupt_payload(self, snapshot) -> dict:
        try:
            for task in snapshot.tasks:
                if task.interrupts:
                    return task.interrupts[0].value
        except Exception:
            pass
        return {}

    def _build_messages(self) -> list[dict]:
        """构建通用上下文（父图保留；子图内部会按领域 prompt 重建）。"""
        from app.prompts.customer_service import SYSTEM_PROMPT

        content = SYSTEM_PROMPT
        if self.skill_manager and self.skill_manager.enabled:
            content += self.skill_manager.build_catalog_prompt()
        messages: list[dict] = [{"role": "system", "content": content}]
        messages.extend(self.memory_manager.build_memory_prompt_sections())
        if self.summary:
            messages.append({
                "role": "system",
                "content": f"以下是此前对话的摘要：\n{self.summary}",
            })
        messages.extend(self.raw_messages)
        return messages

    def _waiting_response(self, payload: dict) -> CustomerServiceResponse:
        question = payload.get("question", "请确认是否继续？")
        from app.multi_agent.structured import normalize_intent
        intent = normalize_intent(payload.get("agent", ""))
        return CustomerServiceResponse(
            intent=intent,
            confidence=0.9,
            reply=question,
            requires_human=False,
            follow_up_question=question,
        )

    def _finalize(self, result: dict) -> CustomerServiceResponse:
        """图执行完成后：同步 raw_messages、记忆、持久化，返回结构化响应。"""
        structured = result.get("structured_response") or {}
        resp = deserialize_response(structured) if structured else None
        if resp is None or not resp.reply:
            resp = safe_default_response(
                result.get("final_response", ""),
                result.get("selected_agent", ""),
            )

        # 只把知识库证据的摘要暴露给调用方，不把完整工具结果透传出去。
        if not resp.citations:
            resp.citations = self._extract_citations(result.get("tool_results", []))

        # 把工具调用轨迹回填到 raw_messages（节点不直接修改外部对象）
        tool_results = result.get("tool_results", [])
        for i, tr in enumerate(tool_results):
            call_id = f"call_{i + 1}"
            self.raw_messages.append({
                "role": "assistant",
                "content": f"[调用工具 {tr['name']}]",
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tr["name"],
                        "arguments": json.dumps(tr.get("arguments", {}), ensure_ascii=False),
                    },
                }],
            })
            self.raw_messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": tr.get("result", ""),
            })

        self.memory_manager.update_short_term(self.raw_messages[-6:])
        self.raw_messages.append({
            "role": "assistant",
            "content": resp.model_dump_json(ensure_ascii=False),
        })

        self._maybe_compress()
        self.save()

        # 记录可观测性 trace（脱敏后，不记录密钥/完整敏感信息）
        meta = result.get("metadata", {}) or {}
        self.last_trace = sanitize_record({
            "trace_id": meta.get("trace_id", ""),
            "thread_id": meta.get("thread_id", ""),
            "user_id": settings.memory_user_id,
            "selected_agent": result.get("selected_agent", ""),
            "node_name": "persist",
            "model_name": self.model,
            "model_calls": meta.get("model_calls", 0),
            "tool_calls": meta.get("tool_calls", 0),
            "retry_count": meta.get("retry_count", 0),
            "model_latency": round(meta.get("model_latency", 0.0), 4),
            "tool_latency": round(meta.get("tool_latency", 0.0), 4),
            "total_latency": round(meta.get("elapsed_seconds", 0.0), 4),
            "error_type": result.get("error") or None,
            "requires_human": resp.requires_human,
            "checkpoint_backend": self.checkpoint_handle.kind if self.checkpoint_handle else "none",
        })
        return resp

    @staticmethod
    def _extract_citations(tool_results: list[dict]) -> list[dict]:
        citations: list[dict] = []
        seen: set[str] = set()
        for item in tool_results or []:
            if item.get("name") != "search_knowledge":
                continue
            raw = item.get("result", "")
            try:
                payload = json.loads(raw) if isinstance(raw, str) else raw
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict) or not payload.get("reliable"):
                continue
            for evidence in payload.get("evidence", []):
                evidence_id = str(evidence.get("evidence_id") or "")
                if not evidence_id or evidence_id in seen:
                    continue
                seen.add(evidence_id)
                citations.append({
                    "evidence_id": evidence_id,
                    "title": evidence.get("title") or evidence.get("doc", ""),
                    "source": evidence.get("source", ""),
                    "section": evidence.get("section", ""),
                    "version": evidence.get("version", ""),
                    "score": evidence.get("score", 0.0),
                })
        return citations[:5]

    def _maybe_compress(self) -> None:
        """基于消息数量做简单截断，避免历史无限膨胀（复用 summarize）。"""
        if len(self.raw_messages) <= self.history_threshold:
            return
        from app.agent.summarizer import summarize

        keep = self.history_keep_recent
        split = len(self.raw_messages) - keep
        while split > 0 and self.raw_messages[split].get("role") in ("tool",):
            split -= 1
        if split <= 0:
            return
        old_messages = self.raw_messages[:split]
        recent = self.raw_messages[split:]
        self.summary = summarize(
            client=self.client,
            model=self.model,
            old_messages=old_messages,
            prev_summary=self.summary,
        )
        self.raw_messages = recent
        print(f"\n[Supervisor·压缩] 已压缩 {len(old_messages)} 条历史")


def _default_client():
    from openai import OpenAI

    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
