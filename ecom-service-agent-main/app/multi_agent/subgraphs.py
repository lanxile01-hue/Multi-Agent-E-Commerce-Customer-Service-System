"""三个领域子图工厂。

把原 SubAgent.handle() 的 ReAct 循环拆分为可复用、可独立构建/测试的
LangGraph CompiledStateGraph：

    agent_reason → agent_decide_tool → [human_confirmation] → execute_tools
                → evaluate_result → generate_response

三个子图共享同一套节点逻辑（通过 `_build_subagent_graph` 复用），
仅在 system prompt 与 allowed_tools 上不同，满足「不复制三份代码」。

子图作为父图节点运行时不携带自己的 checkpointer，interrupt() 会冒泡到父图；
单独测试时可通过 checkpointer 参数注入 MemorySaver。
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from app.agent.json_utils import extract_json_object
from app.agent.tools.errors import (
    TOOL_OK,
    TOOL_RETRYABLE,
    classify_tool_error,
    standard_result,
    validate_tool_arguments,
)
from app.agent.tools.risks import is_non_idempotent, requires_confirmation
from app.multi_agent.state import SupervisorState, normalize_confirmation
from app.multi_agent.structured import (
    build_structured_response,
    coerce_structured,
    serialize_response,
)
from app.prompts.agents import COMPLAINT_PROMPT, POSTSALE_PROMPT, PRESALE_PROMPT
from app.schemas.response import CustomerServiceResponse

# 需要人工确认的敏感工具：以 app/agent/tools/risks.py 的 requires_confirmation 为准。
# （apply_refund 属于 requires_confirmation；compensation / complaint_escalation 尚未实现）

# 各子 Agent 的工具白名单（与 agents.AGENT_CONFIGS 保持一致，作为兜底）
AGENT_ALLOWED_TOOLS = {
    "presale": {"query_product", "search_knowledge", "list_user_orders", "load_skill"},
    "postsale": {
        "query_order", "query_logistics", "apply_refund",
        "list_user_orders", "search_knowledge", "load_skill",
    },
    "complaint": {"query_order", "search_knowledge", "load_skill"},
}


def _make_idempotency_key(thread_id: str, tool_name: str, arguments: dict) -> str:
    """为敏感工具生成确定性的幂等键。

    基于 thread_id + tool_name + order_id + reason 计算，保证：
    - interrupt 暂停 → resume 恢复 两次执行 human_confirmation 节点时键值一致；
    - 同一 thread 对同一订单的重复确认不会生成新键。
    """
    order_id = str(arguments.get("order_id", ""))
    reason = str(arguments.get("reason", ""))
    raw = f"{thread_id}:{tool_name}:{order_id}:{reason}"
    return f"{tool_name}-{hashlib.md5(raw.encode('utf-8')).hexdigest()[:16]}"


# ---------- Prompts ----------

_PLANNER_PROMPT = """你是一个企业知识库任务规划器，是一名「{agent_label}」领域的专家。

当前领域: {selected_agent}

基于用户的最新消息，决定下一步动作：
1. 调用工具检索制度、技术文档或加载审核过的流程技能
2. 直接回复用户（当你已经有足够信息时）

可用工具:
{tool_list}

用户消息: {user_input}

已有工具结果:
{tool_results}

请输出 JSON 格式（只输出 JSON 对象本身，不要 markdown 代码块、不要注释、不要解释文字）:
{{
  "action": "call_tools" | "respond",
  "tools": [
    {{"name": "工具名", "arguments": {{"参数名": "参数值"}}}}
  ],
  "thought": "你的推理过程"
}}

规则:
1. 如果已有工具结果足以回答问题，action 设为 "respond"
2. 每次最多调用 3 个工具，优先调用最相关的
3. 工具参数名必须严格使用上面「可用工具」中列出的名称，不要自己编造
4. 涉及制度、规则、流程和技术事实时，优先调用 search_knowledge
5. 涉及权限、生产、安全、数据删除等敏感操作时，必须说明人工审批要求
6. action 为 "call_tools" 时 tools 必填，每项 name 和 arguments 都要完整给出
"""

_EVALUATOR_PROMPT = """你是一个任务评估器。

用户问题: {user_input}
已调用工具: {tools_called}
工具结果摘要: {tool_results_summary}

判断当前信息是否足够回答用户问题。输出 JSON:
{{"sufficient": true | false, "reason": "判断理由"}}

规则:
- 如果工具返回了用户需要的具体数据（订单状态、物流、政策），sufficient = true
- 如果工具返回空结果或错误，但还有其他工具可尝试，sufficient = false
- 如果已经循环 {max_steps} 次以上，sufficient = true（避免死循环）
"""

_RESPONDER_PROMPT = """你是企业知识库与流程协同平台的专业助手。

## 当前上下文
领域: {agent_label}
用户问题: {user_input}
查询到的数据:
{context}

请生成一段友好、准确的中文回复，直接回答用户。

重要:
- 只输出最终给用户看的回复文本
- 不要输出 [调用工具]、[思考]、load_skill 等内部标记
- 不要引用"根据工具返回"等内部描述
- 像可靠的企业内部助手一样说话，清楚区分知识依据、建议步骤和待确认事项
- 如果证据不足，明确说明当前知识库无法确认，不得编造制度或操作结果
"""


# ---------- 工具列表格式化 ----------

def _format_tool_list(tool_definitions: list[dict]) -> str:
    lines = []
    for td in tool_definitions:
        func = td["function"]
        params = func.get("parameters", {}).get("properties", {})
        required = set(func.get("parameters", {}).get("required", []))
        param_strs = []
        for pname, pinfo in params.items():
            mark = "(必填)" if pname in required else "(可选)"
            param_strs.append(f"    {pname} {mark}: {pinfo.get('description', '')}")
        param_text = "\n".join(param_strs) if param_strs else "    无参数"
        lines.append(f"- {func['name']}: {func['description']}\n  参数:\n{param_text}")
    return "\n".join(lines)


def _format_tool_results(tool_results: list[dict]) -> str:
    if not tool_results:
        return "（暂无）"
    return "\n".join(
        f"[{r['name']}] {r['result'][:300]}" for r in tool_results
    )


# ---------- 子图构建 ----------

def _build_subagent_graph(
    *,
    agent_key: str,
    agent_label: str,
    system_prompt: str,
    tool_manager,
    client,
    model: str,
    temperature: float,
    memory_manager=None,
    skill_manager=None,
    get_summary: Optional[Callable[[], Optional[str]]] = None,
    max_steps: int = 5,
    human_confirmation_enabled: bool = True,
    checkpointer=None,
    tool_timeout: float = 0.0,
) -> CompiledStateGraph:
    """构建一个领域子图（通用工厂，三个子图共用）。"""

    def _agent_messages(state: SupervisorState) -> list[dict]:
        """用领域 system prompt 重建子图上下文（不被父图通用 messages 污染）。"""
        content = system_prompt
        if skill_manager is not None and getattr(skill_manager, "enabled", False):
            content += skill_manager.build_catalog_prompt()

        messages: list[dict] = [{"role": "system", "content": content}]
        if memory_manager is not None:
            messages.extend(memory_manager.build_memory_prompt_sections())
        summary = get_summary() if get_summary else None
        if summary:
            messages.append({
                "role": "system",
                "content": f"以下是此前对话的摘要：\n{summary}",
            })
        messages.extend(state.get("conversation", []))
        return messages

    # ---- 节点 0：agent_enter（子图入口，显式重置临时状态，实现状态隔离）----
    def agent_enter(state: SupervisorState) -> dict:
        """子图入口初始化节点。

        三个子图共享 SupervisorState，为避免：
        1. presale 的临时状态污染 postsale
        2. 多次调用同一子图复用上一次请求的 tool_results / pending_tool_calls

        在进入子图时显式重置所有临时字段，但保留父图注入的 user_input /
        selected_agent / intent / conversation / metadata。
        """
        return {
            "iteration": 0,
            "pending_tool_calls": [],
            "tool_results": [],
            "is_complete": False,
            "confirmation_result": "",
            "waiting_for_confirmation": False,
            "error": "",
        }

    # ---- 节点 1：agent_reason（分析任务 + 产出工具计划）----
    def agent_reason(state: SupervisorState) -> dict:
        iteration = state.get("iteration", 0)
        if iteration >= max_steps:
            # 达到最大步数，强制进入回复，避免死循环
            return {"pending_tool_calls": [], "is_complete": True, "iteration": iteration}

        tool_list = _format_tool_list(tool_manager.tool_definitions)
        prompt = _PLANNER_PROMPT.format(
            agent_label=agent_label,
            selected_agent=agent_key,
            tool_list=tool_list,
            user_input=state["user_input"],
            tool_results=_format_tool_results(state.get("tool_results", [])),
        )
        meta = dict(state.get("metadata", {}))
        meta["model_calls"] = int(meta.get("model_calls", 0)) + 1

        try:
            # 使用领域 Agent 的完整上下文（专属 system prompt + memory + skill + summary + conversation）
            working_messages = _agent_messages(state)
            working_messages.append({"role": "user", "content": prompt})
            t0 = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=working_messages,
                temperature=0.0,
            )
            meta["model_latency"] = round(meta.get("model_latency", 0.0) + (time.time() - t0), 4)
            raw = (response.choices[0].message.content or "").strip()
            plan_data = extract_json_object(raw)
        except Exception as e:
            # 规划失败：安全兜底为直接回复，不中断流程
            return {
                "pending_tool_calls": [],
                "is_complete": True,
                "iteration": iteration + 1,
                "error": f"规划失败: {e}",
                "metadata": meta,
            }

        thought = plan_data.get("thought", "")
        if thought:
            print(f"  [{agent_label}·思考] {thought}")

        tools = plan_data.get("tools", []) if plan_data.get("action") == "call_tools" else []
        # 规范化 tool call 结构
        normalized = []
        for tc in tools:
            if isinstance(tc, dict) and tc.get("name"):
                normalized.append({
                    "name": str(tc["name"]),
                    "arguments": tc.get("arguments", {}) or {},
                })
        return {
            "pending_tool_calls": normalized,
            "iteration": iteration + 1,
            "metadata": meta,
        }

    # ---- 节点 2：agent_decide_tool（校验工具调用 + 敏感工具标记）----
    def agent_decide_tool(state: SupervisorState) -> dict:
        pending = state.get("pending_tool_calls", [])
        if not pending:
            return {}

        valid_calls: list[dict] = []
        errors: list[str] = []
        for tc in pending:
            name = tc.get("name", "")
            ok, err = validate_tool_arguments(name, tc.get("arguments", {}), tool_manager.tool_definitions)
            if not ok:
                errors.append(err)
                print(f"  [{agent_label}·工具校验] 拒绝 {name}: {err}")
                continue
            valid_calls.append(tc)

        result: dict = {"pending_tool_calls": valid_calls}
        if errors:
            result["error"] = "; ".join(errors)
        return result

    # ---- 节点 3：human_confirmation（敏感工具人工确认）----
    def human_confirmation(state: SupervisorState) -> dict:
        pending = state.get("pending_tool_calls", [])
        sensitive = [tc for tc in pending if requires_confirmation(tc.get("name", ""))]

        # 无敏感工具或未启用确认 → 直接放行执行
        if not sensitive or not human_confirmation_enabled:
            return {
                "waiting_for_confirmation": False,
                "confirmation_result": "",
            }

        thread_id = str(state.get("metadata", {}).get("thread_id", ""))
        # 给每个敏感工具注入确定性幂等键（interrupt 前注入，恢复时键值一致）
        updated_pending: list[dict] = []
        for tc in pending:
            if requires_confirmation(tc.get("name", "")):
                args = dict(tc.get("arguments") or {})
                args["idempotency_key"] = _make_idempotency_key(
                    thread_id, tc["name"], args,
                )
                updated_pending.append({**tc, "arguments": args})
            else:
                updated_pending.append(tc)

        # 需要确认：调用 interrupt 暂停，等待用户确认
        payload = {
            "kind": "human_confirmation",
            "agent": agent_key,
            "thread_id": thread_id,
            "sensitive_tools": [tc["name"] for tc in sensitive],
            "tool_calls": [
                {**tc, "arguments": {k: v for k, v in tc.get("arguments", {}).items() if k != "idempotency_key"}}
                for tc in sensitive
            ],
            "question": "是否确认执行以下敏感操作？请回复「确认」「取消」或「重新说明」。",
        }
        decision = interrupt(payload)

        normalized = normalize_confirmation(decision)
        # 记录人工确认恢复信息（第三阶段要求）
        confirmation_payload = {
            **payload,
            "confirmation_result": normalized,
            "confirmation_time": time.time(),
            "tool_name": [tc["name"] for tc in sensitive],
            "tool_arguments": [tc.get("arguments") for tc in sensitive],
            "idempotency_keys": [
                tc.get("arguments", {}).get("idempotency_key") for tc in updated_pending
                if requires_confirmation(tc.get("name", ""))
            ],
        }
        result: dict = {
            "waiting_for_confirmation": False,
            "confirmation_result": normalized,
            "confirmation_payload": confirmation_payload,
            "confirmation_time": time.time(),
            "pending_tool_calls": updated_pending,
        }
        if normalized == "re_explain":
            # 用户重新说明：把新输入当作本轮 user_input 重新推理
            result["user_input"] = decision
            result["pending_tool_calls"] = []
        return result

    # ---- 节点 4：execute_tools（执行工具，含错误分类 + 有限重试 + 超时保护）----
    def _call_once(name: str, arguments: dict) -> str:
        """单次工具调用，带超时保护。"""
        if tool_timeout <= 0:
            return tool_manager.execute_tool(name, arguments)
        ex = ThreadPoolExecutor(max_workers=1)
        fut = ex.submit(tool_manager.execute_tool, name, arguments)
        try:
            return fut.result(timeout=tool_timeout)
        except TimeoutError:
            if is_non_idempotent(name):
                # 高风险写操作超时：结果未知，不能当成功也不能当失败，禁止自动重试
                return json.dumps(standard_result(
                    status="unknown",
                    message=f"操作 {name} 执行超时，结果状态未知，请人工核验",
                    error_type="unknown",
                    retryable=False,
                ), ensure_ascii=False)
            return json.dumps(standard_result(
                status="error",
                message=f"工具 {name} 执行超时（>{tool_timeout}s）",
                error_type="retryable",
                retryable=True,
            ), ensure_ascii=False)
        except Exception as e:
            return json.dumps(standard_result(
                status="error",
                message=f"工具 {name} 执行异常: {e}",
                error_type="fatal",
                retryable=False,
            ), ensure_ascii=False)
        finally:
            ex.shutdown(wait=False)

    def _execute_one(name: str, arguments: dict) -> tuple[str, int]:
        """执行单个工具，返回 (result_str, retries)。

        重试策略：
        - 只有 retryable 错误才有限重试
        - fatal / param_error / not_found / unknown_tool / 成功 均不重试
        - 非幂等写操作（apply_refund）禁止自动重试，避免重复执行退款
        """
        result_str = _call_once(name, arguments)
        category = classify_tool_error(result_str)
        retries = 0

        if category == TOOL_RETRYABLE and not is_non_idempotent(name):
            for attempt in range(2):
                time.sleep(0.1 * (2 ** attempt))
                result_str = _call_once(name, arguments)
                retries += 1
                if classify_tool_error(result_str) == TOOL_OK:
                    break
        return result_str, retries

    def execute_tools(state: SupervisorState) -> dict:
        pending = state.get("pending_tool_calls", [])
        new_results = list(state.get("tool_results", []))
        meta = dict(state.get("metadata", {}))

        for tc in pending:
            name = tc["name"]
            args = tc.get("arguments", {})
            meta["tool_calls"] = int(meta.get("tool_calls", 0)) + 1
            print(f"  [{agent_label}·调用工具] {name}({json.dumps(args, ensure_ascii=False)})")
            t0 = time.time()
            result_str, retries = _execute_one(name, args)
            meta["tool_latency"] = round(meta.get("tool_latency", 0.0) + (time.time() - t0), 4)
            meta["retry_count"] = int(meta.get("retry_count", 0)) + retries
            category = classify_tool_error(result_str)
            display = result_str if len(result_str) <= 300 else result_str[:300] + "..."
            print(f"  [{agent_label}·工具结果] ({category}, 重试{retries}次) {display}")
            new_results.append({
                "name": name,
                "arguments": args,
                "result": result_str,
                "ok": category == TOOL_OK,
                "error": category,
                "retries": retries,
            })

        return {
            "tool_results": new_results,
            "pending_tool_calls": [],
            "metadata": meta,
        }

    # ---- 节点 5：evaluate_result（判断信息是否足够）----
    def evaluate_result(state: SupervisorState) -> dict:
        iteration = state.get("iteration", 0)
        meta = dict(state.get("metadata", {}))

        # 达到最大步数 → 强制结束
        if iteration >= max_steps:
            return {"is_complete": True}

        tool_results = state.get("tool_results", [])
        if not tool_results:
            return {"is_complete": True}

        tools_called = [r["name"] for r in tool_results]
        results_summary = "\n".join(f"[{r['name']}] {r['result'][:200]}" for r in tool_results)

        prompt = _EVALUATOR_PROMPT.format(
            user_input=state["user_input"],
            tools_called=", ".join(tools_called) if tools_called else "无",
            tool_results_summary=results_summary or "无",
            max_steps=max_steps,
        )
        meta["model_calls"] = int(meta.get("model_calls", 0)) + 1

        sufficient = True
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            raw = (response.choices[0].message.content or "").strip()
            eval_data = extract_json_object(raw)
            sufficient = bool(eval_data.get("sufficient", True))
        except Exception:
            # 评估失败保守处理：默认信息足够，避免死循环
            sufficient = True

        if sufficient:
            print(f"  [{agent_label}·评估] 信息充足，生成回复")
        else:
            print(f"  [{agent_label}·评估] 信息不足，继续规划")

        return {"is_complete": sufficient, "metadata": meta}

    # ---- 节点 6：generate_response（生成最终回复 + 结构化输出）----
    def generate_response(state: SupervisorState) -> dict:
        # 取消场景：直接给固定回复，不再调用 LLM
        if state.get("confirmation_result") == "cancelled":
            reply = "已为您取消本次操作（如退款申请）。如需继续，请重新说明您的需求。"
            resp = build_structured_response(reply, agent_key)
            return {
                "final_response": reply,
                "structured_response": serialize_response(resp),
                "requires_human": resp.requires_human,
            }

        context = "\n".join(
            f"- [{r['name']}({json.dumps(r['arguments'], ensure_ascii=False)})] {r['result']}"
            for r in state.get("tool_results", [])
        ) if state.get("tool_results") else "（未查询额外数据）"

        responder_prompt = _RESPONDER_PROMPT.format(
            agent_label=agent_label,
            user_input=state["user_input"],
            context=context,
        )
        meta = dict(state.get("metadata", {}))
        meta["model_calls"] = int(meta.get("model_calls", 0)) + 1

        # 统一使用领域 Agent 完整上下文，回复指令作为最后一条 user 消息追加
        working_messages = _agent_messages(state)
        working_messages.append({"role": "user", "content": responder_prompt})

        # 优先尝试直接结构化输出（response_format），失败回退到纯文本 + 确定性映射
        parsed = None
        try:
            response = client.beta.chat.completions.parse(
                model=model,
                messages=working_messages,
                temperature=temperature,
                response_format=CustomerServiceResponse,
            )
            parsed = response.choices[0].message.parsed
        except Exception:
            parsed = None

        if parsed is not None and getattr(parsed, "reply", None):
            resp = coerce_structured(parsed, agent_key, fallback_reply="")
        else:
            # 稳定 fallback：纯文本生成 + 确定性映射（不再二次调用 LLM 提取）
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=working_messages,
                    temperature=temperature,
                )
                reply = (response.choices[0].message.content or "").strip()
            except Exception:
                reply = ""
            reply = reply or "抱歉，我暂时无法处理您的请求。"
            resp = build_structured_response(reply, agent_key)

        print(f"\n[{agent_label}·生成回复] 已生成 ({len(resp.reply)} 字)")
        return {
            "final_response": resp.reply,
            "structured_response": serialize_response(resp),
            "requires_human": resp.requires_human,
            "metadata": meta,
        }

    # ---- 条件路由函数 ----
    def route_after_decide(state: SupervisorState) -> str:
        if state.get("pending_tool_calls"):
            return "human_confirmation"
        return "generate_response"

    def route_after_confirmation(state: SupervisorState) -> str:
        result = state.get("confirmation_result", "")
        if result == "cancelled":
            return "generate_response"
        if result == "re_explain":
            return "agent_reason"
        return "execute_tools"  # confirmed 或无需确认

    def route_after_evaluate(state: SupervisorState) -> str:
        return "generate_response" if state.get("is_complete", True) else "agent_reason"

    # ---- 组装子图 ----
    builder = StateGraph(SupervisorState)
    builder.add_node("agent_enter", agent_enter)
    builder.add_node("agent_reason", agent_reason)
    builder.add_node("agent_decide_tool", agent_decide_tool)
    builder.add_node("human_confirmation", human_confirmation)
    builder.add_node("execute_tools", execute_tools)
    builder.add_node("evaluate_result", evaluate_result)
    builder.add_node("generate_response", generate_response)

    builder.add_edge(START, "agent_enter")
    builder.add_edge("agent_enter", "agent_reason")
    builder.add_edge("agent_reason", "agent_decide_tool")
    builder.add_conditional_edges(
        "agent_decide_tool",
        route_after_decide,
        {"human_confirmation": "human_confirmation", "generate_response": "generate_response"},
    )
    builder.add_conditional_edges(
        "human_confirmation",
        route_after_confirmation,
        {
            "execute_tools": "execute_tools",
            "generate_response": "generate_response",
            "agent_reason": "agent_reason",
        },
    )
    builder.add_edge("execute_tools", "evaluate_result")
    builder.add_conditional_edges(
        "evaluate_result",
        route_after_evaluate,
        {"generate_response": "generate_response", "agent_reason": "agent_reason"},
    )
    builder.add_edge("generate_response", END)

    return builder.compile(checkpointer=checkpointer)


# ---------- 三个公共工厂函数 ----------

def build_presale_graph(
    tool_manager,
    client,
    model: str,
    temperature: float,
    *,
    memory_manager=None,
    skill_manager=None,
    get_summary: Optional[Callable[[], Optional[str]]] = None,
    max_steps: int = 5,
    human_confirmation_enabled: bool = True,
    checkpointer=None,
    tool_timeout: float = 0.0,
) -> CompiledStateGraph:
    """构建售前子图（商品咨询）。"""
    return _build_subagent_graph(
        agent_key="presale",
        agent_label="售前-商品咨询",
        system_prompt=PRESALE_PROMPT,
        tool_manager=tool_manager,
        client=client,
        model=model,
        temperature=temperature,
        memory_manager=memory_manager,
        skill_manager=skill_manager,
        get_summary=get_summary,
        max_steps=max_steps,
        human_confirmation_enabled=human_confirmation_enabled,
        checkpointer=checkpointer,
        tool_timeout=tool_timeout,
    )


def build_postsale_graph(
    tool_manager,
    client,
    model: str,
    temperature: float,
    *,
    memory_manager=None,
    skill_manager=None,
    get_summary: Optional[Callable[[], Optional[str]]] = None,
    max_steps: int = 5,
    human_confirmation_enabled: bool = True,
    checkpointer=None,
    tool_timeout: float = 0.0,
) -> CompiledStateGraph:
    """构建售后子图（订单/物流/退款）。"""
    return _build_subagent_graph(
        agent_key="postsale",
        agent_label="售后-订单服务",
        system_prompt=POSTSALE_PROMPT,
        tool_manager=tool_manager,
        client=client,
        model=model,
        temperature=temperature,
        memory_manager=memory_manager,
        skill_manager=skill_manager,
        get_summary=get_summary,
        max_steps=max_steps,
        human_confirmation_enabled=human_confirmation_enabled,
        checkpointer=checkpointer,
        tool_timeout=tool_timeout,
    )


def build_complaint_graph(
    tool_manager,
    client,
    model: str,
    temperature: float,
    *,
    memory_manager=None,
    skill_manager=None,
    get_summary: Optional[Callable[[], Optional[str]]] = None,
    max_steps: int = 5,
    human_confirmation_enabled: bool = True,
    checkpointer=None,
    tool_timeout: float = 0.0,
) -> CompiledStateGraph:
    """构建投诉子图（投诉/维权）。"""
    return _build_subagent_graph(
        agent_key="complaint",
        agent_label="投诉处理",
        system_prompt=COMPLAINT_PROMPT,
        tool_manager=tool_manager,
        client=client,
        model=model,
        temperature=temperature,
        memory_manager=memory_manager,
        skill_manager=skill_manager,
        get_summary=get_summary,
        max_steps=max_steps,
        human_confirmation_enabled=human_confirmation_enabled,
        checkpointer=checkpointer,
        tool_timeout=tool_timeout,
    )
