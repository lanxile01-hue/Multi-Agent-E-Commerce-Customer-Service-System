"""
LangGraph Agent: 基于状态图的多智能体架构，替换原有 ReAct 循环。

架构: classify_intent → plan → execute → evaluate → respond
- classify_intent: LLM 分类用户意图（presale/postsale/complaint）
- plan: LLM 分解子任务并决定调用哪些工具
- execute: 调用工具，含指数退避+抖动重试
- evaluate: 条件路由 → 继续 plan（loop）或进入 respond（done）
- respond: LLM 生成最终回复

对外接口与 EcomAgent 完全一致（chat/reset/save/close），可在 main.py 无缝切换。
"""

from __future__ import annotations

import json
import time
from typing import Literal, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from openai import OpenAI

from app.agent.storage import delete_session, load_session, save_session
from app.agent.compressor import TokenCounter, create_compressor
from app.agent.guardrails import Guardrails
from app.agent.json_utils import extract_json_object
from app.agent.tools.errors import (
    TOOL_OK,
    TOOL_RETRYABLE,
    classify_tool_error,
    validate_tool_arguments,
)
from app.agent.tools.manager import ToolManager
from app.config.settings import settings
from app.prompts.agents import ROUTER_PROMPT
from app.prompts.customer_service import SYSTEM_PROMPT
from app.schemas.response import CustomerServiceResponse, IntentType


# ---------- State ----------

class AgentState(TypedDict):
    messages: list[dict]          # 完整对话历史
    user_input: str               # 当前用户输入
    intent: str                   # 分类后的意图
    iteration: int                # 当前循环次数
    pending_tool_calls: list[dict]  # plan 节点产出的工具调用列表
    tool_results: list[dict]      # execute 节点产出的工具结果
    final_response: str           # respond 节点产出的最终文本
    is_complete: bool             # evaluate 节点写入：信息是否足够
    metadata: dict                # 工具调用/模型调用/耗时统计


# ---------- Prompts ----------

PLANNER_PROMPT = """你是一个任务规划器，工作在电商客服 Agent 系统中。

当前用户意图: {intent}

基于用户的最新消息，决定下一步动作。你可以：
1. 调用工具查询数据（如查订单、查物流、搜政策等）
2. 直接回复用户（当你已经有足够信息时）

可用工具:
{tool_list}

用户消息: {user_input}

已有工具结果:
{tool_results}

请输出 JSON 格式（只输出 JSON 对象本身，不要 markdown 代码块、不要注释、不要任何解释文字）:
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
4. 如果用户没有提供订单号，先用 list_user_orders
5. 退货前必须先查订单状态
6. action 为 "call_tools" 时 tools 必填，tools 数组中每项的 name 和 arguments 都要完整给出
"""


EVALUATOR_PROMPT = """你是一个任务评估器。

用户问题: {user_input}
已调用工具: {tools_called}
工具结果摘要: {tool_results_summary}

判断当前信息是否足够回答用户问题。输出 JSON:
{{
  "sufficient": true | false,
  "reason": "判断理由"
}}

规则:
- 如果工具返回了用户需要的具体数据（订单状态、物流信息、政策内容），sufficient = true
- 如果工具返回空结果或错误，但还有其他工具可以尝试，sufficient = false
- 如果已经循环 3 次以上，sufficient = true（避免死循环）
"""


RESPONDER_PROMPT = """你是「帅团」电商平台的智能客服助手，名字叫「小乐」。

## 角色定位
你是一名专业、耐心、有温度的电商客服，负责为顾客提供优质的购物服务体验。

## 回复规范
1. 语气友好，始终保持礼貌和专业
2. 基于工具返回的真实数据回复，不要编造
3. 如果涉及订单号、金额等关键信息，明确列出
4. 主动提供后续选项（如"需要我帮您申请退款吗？"）

## 当前上下文
用户意图: {intent}
用户问题: {user_input}
查询到的数据:
{context}

请生成一段友好、准确的中文回复。

重要:
- 只输出最终给用户看的回复文本，不要包含任何内部标记
- 不要输出 [调用工具]、[思考]、load_skill 等内部指令
- 不要引用"根据工具返回"等内部描述
- 像正常客服一样说话即可"""


# ---------- Graph Agent ----------

class LangGraphAgent:
    """基于 LangGraph 的电商客服 Agent。"""

    def __init__(self, session_path: Optional[str] = None):
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self.model = settings.model_name
        self.temperature = settings.temperature
        self.session_path = session_path or settings.session_path
        self.history_threshold = settings.history_threshold
        self.history_keep_recent = settings.history_keep_recent
        self.max_iterations = settings.max_react_steps

        self.tool_manager = ToolManager(
            use_mcp=settings.mcp_enabled,
            mcp_server_url=settings.mcp_server_url,
        )

        from app.agent.memory import MemoryManager
        self.memory_manager = MemoryManager(
            client=self.client,
            model=self.model,
            user_id=settings.memory_user_id,
            memory_dir=settings.memory_dir,
            memory_enabled=settings.memory_enabled,
            max_ltm_facts=settings.max_ltm_facts,
        )
        if settings.memory_enabled:
            from app.agent.tools.memory_tool import set_memory_manager
            set_memory_manager(self.memory_manager)

        from app.agent.skills import SkillManager
        self.skill_manager = SkillManager(
            skills_dir=settings.skills_dir,
            enabled=settings.skills_enabled,
        )
        if settings.skills_enabled:
            from app.agent.tools.skill_tool import set_skill_manager
            set_skill_manager(self.skill_manager)

        self.raw_messages: list[dict] = []
        self.summary: Optional[str] = None

        self._compressor = create_compressor(self.client, self.model)
        self._guardrails = Guardrails(self.client, self.model, enabled=settings.guardrails_enabled)

        loaded = load_session(self.session_path)
        if loaded:
            self.summary = loaded["summary"]
            self.raw_messages = loaded["messages"]
            if loaded.get("short_term_memory"):
                self.memory_manager.restore_stm(loaded["short_term_memory"])

        self._graph = self._build_graph()

    @property
    def history_size(self) -> int:
        return len(self.raw_messages)

    # ---------- Public API (same as EcomAgent) ----------

    def chat(self, user_input: str) -> CustomerServiceResponse:
        """处理用户输入，通过 LangGraph 状态图执行。"""
        self.raw_messages.append({"role": "user", "content": user_input})

        initial_state: AgentState = {
            "messages": self._build_messages(),
            "user_input": user_input,
            "intent": "",
            "iteration": 0,
            "pending_tool_calls": [],
            "tool_results": [],
            "final_response": "",
            "is_complete": False,
            "metadata": {
                "start_time": time.time(),
                "model_calls": 0,
                "tool_calls": 0,
            },
        }

        result = self._graph.invoke(initial_state)
        final_text = result["final_response"]

        # 把工具调用轨迹回填到 raw_messages（节点不再直接修改外部对象）
        for i, tr in enumerate(result.get("tool_results", [])):
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

        structured = self._extract_structured_response(final_text)

        self.memory_manager.update_short_term(self.raw_messages[-6:])

        self.raw_messages.append(
            {"role": "assistant", "content": structured.model_dump_json(ensure_ascii=False)}
        )

        self._maybe_compress()

        meta = result.get("metadata", {})
        if "start_time" in meta:
            meta["elapsed_seconds"] = round(time.time() - meta["start_time"], 3)
            print(
                f"[LangGraph·统计] 工具调用={meta.get('tool_calls', 0)} "
                f"模型调用={meta.get('model_calls', 0)} 耗时={meta['elapsed_seconds']}s"
            )

        save_session(
            self.session_path, self.raw_messages, self.summary,
            short_term_memory=self.memory_manager.stm_to_dict(),
        )
        return structured

    def reset(self):
        self.raw_messages = []
        self.summary = None
        self.memory_manager.reset_short_term()
        delete_session(self.session_path)

    def save(self) -> None:
        save_session(
            self.session_path, self.raw_messages, self.summary,
            short_term_memory=self.memory_manager.stm_to_dict(),
        )

    def close(self):
        self.memory_manager.consolidate_to_long_term(self.raw_messages, self.summary)
        self.tool_manager.close()

    # ---------- Graph Construction ----------

    def _build_graph(self) -> CompiledStateGraph:
        builder = StateGraph(AgentState)

        builder.add_node("classify_intent", self._node_classify_intent)
        builder.add_node("plan", self._node_plan)
        builder.add_node("execute", self._node_execute)
        builder.add_node("evaluate", self._node_evaluate)
        builder.add_node("respond", self._node_respond)

        builder.add_edge(START, "classify_intent")
        builder.add_edge("classify_intent", "plan")
        # plan 判断 action=respond 时直接进入回复，不再执行无意义的 execute/evaluate
        builder.add_conditional_edges(
            "plan",
            self._route_after_plan,
            {"execute": "execute", "respond": "respond"},
        )
        builder.add_edge("execute", "evaluate")
        builder.add_conditional_edges(
            "evaluate",
            self._route_after_evaluate,
            {"plan": "plan", "respond": "respond"},
        )
        builder.add_edge("respond", END)

        return builder.compile()

    # ---------- Nodes ----------

    def _node_classify_intent(self, state: AgentState) -> dict:
        """意图路由节点：关键词预筛 + LLM 分类。"""
        user_input = state["user_input"]

        # 关键词快速预筛（LLM 不可靠时的兜底）
        presale_keywords = [
            "推荐", "有没有", "什么好", "哪个好", "介绍一下", "多少钱",
            "有卖", "优惠", "活动", "打折", "促销", "便宜", "性价比",
            "款式", "颜色", "尺码", "材质", "质量怎么样", "好不好用",
        ]
        complaint_keywords = [
            "投诉", "差评", "骗子", "太差", "垃圾", "坑", "骗",
            "维权", "举报", "曝光", "赔偿",
        ]

        for kw in presale_keywords:
            if kw in user_input:
                print(f"\n[LangGraph·路由] -> 售前-商品咨询 (关键词: {kw})")
                return {"intent": "presale"}

        for kw in complaint_keywords:
            if kw in user_input:
                print(f"\n[LangGraph·路由] -> 投诉处理 (关键词: {kw})")
                return {"intent": "complaint"}

        # 关键词没命中 → LLM 分类
        prompt = ROUTER_PROMPT.format(user_input=user_input)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        raw = (response.choices[0].message.content or "").strip().lower()

        valid = {"presale", "postsale", "complaint"}
        intent = next((w for w in raw.split() if w in valid), "postsale")

        intent_labels = {"presale": "售前-商品咨询", "postsale": "售后-订单服务", "complaint": "投诉处理"}
        label = intent_labels.get(intent, intent)
        print(f"\n[LangGraph·路由] -> {label}")

        return {"intent": intent}

    def _node_plan(self, state: AgentState) -> dict:
        """规划节点：LLM 决定调用哪些工具或直接回复。"""
        iteration = state["iteration"]
        if iteration >= self.max_iterations:
            return {"pending_tool_calls": [], "iteration": iteration, "is_complete": True}

        tool_list_lines = []
        for td in self.tool_manager.tool_definitions:
            func = td["function"]
            params = func.get("parameters", {}).get("properties", {})
            param_strs = []
            for pname, pinfo in params.items():
                required = pname in func.get("parameters", {}).get("required", [])
                req_mark = "(必填)" if required else "(可选)"
                param_strs.append(f"    {pname} {req_mark}: {pinfo.get('description', '')}")
            param_text = "\n".join(param_strs) if param_strs else "    无参数"
            tool_list_lines.append(f"- {func['name']}: {func['description']}\n  参数:\n{param_text}")
        tool_list = "\n".join(tool_list_lines)
        results_text = "\n".join(
            f"[{r['name']}] {r['result'][:300]}" for r in state["tool_results"]
        ) if state["tool_results"] else "（暂无）"

        prompt = PLANNER_PROMPT.format(
            intent=state["intent"],
            tool_list=tool_list,
            user_input=state["user_input"],
            tool_results=results_text,
        )

        meta = dict(state.get("metadata", {}))
        meta["model_calls"] = int(meta.get("model_calls", 0)) + 1

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        raw = (response.choices[0].message.content or "").strip()

        try:
            plan_data = extract_json_object(raw)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  [规划器] JSON 解析失败，直接回复（{e}）")
            return {"pending_tool_calls": [], "iteration": iteration + 1, "metadata": meta}

        thought = plan_data.get("thought", "")
        if thought:
            print(f"  [规划器] {thought}")

        if plan_data.get("action") == "respond":
            return {"pending_tool_calls": [], "iteration": iteration + 1, "metadata": meta}

        # 规范化工具调用结构，过滤掉结构非法的条目
        tools = []
        for tc in plan_data.get("tools", []) or []:
            if isinstance(tc, dict) and tc.get("name"):
                tools.append({
                    "name": str(tc["name"]),
                    "arguments": tc.get("arguments", {}) or {},
                })
        return {"pending_tool_calls": tools, "iteration": iteration + 1, "metadata": meta}

    def _node_execute(self, state: AgentState) -> dict:
        """执行节点：校验参数 → 调用工具 → 错误分类（含有限重试）。"""
        pending = state.get("pending_tool_calls", [])
        if not pending:
            return {"tool_results": state["tool_results"], "pending_tool_calls": []}

        meta = dict(state.get("metadata", {}))
        new_results = list(state["tool_results"])
        for tc in pending:
            name = tc.get("name", "")
            args = tc.get("arguments", {}) or {}

            # 未知工具 / 参数非法 → 安全拒绝，不执行
            ok, err = validate_tool_arguments(name, args, self.tool_manager.tool_definitions)
            if not ok:
                print(f"  [执行器] 拒绝 {name}: {err}")
                new_results.append({
                    "name": name, "arguments": args,
                    "result": json.dumps({"error": err}, ensure_ascii=False),
                    "ok": False, "error": "unknown_tool" if name not in {
                        td["function"]["name"] for td in self.tool_manager.tool_definitions
                    } else "param_error",
                })
                continue

            print(f"  [执行器] {name}({json.dumps(args, ensure_ascii=False)})")
            meta["tool_calls"] = int(meta.get("tool_calls", 0)) + 1

            result_str = self.tool_manager.execute_tool(name, args)
            category = classify_tool_error(result_str)
            # 可重试错误 → 有限次退避重试
            if category == TOOL_RETRYABLE:
                for attempt in range(2):
                    time.sleep(0.1 * (2 ** attempt))
                    result_str = self.tool_manager.execute_tool(name, args)
                    if classify_tool_error(result_str) == TOOL_OK:
                        break
                category = classify_tool_error(result_str)

            display = result_str if len(result_str) <= 300 else result_str[:300] + "..."
            print(f"  [工具结果] ({category}) {display}")

            new_results.append({
                "name": name,
                "arguments": args,
                "result": result_str,
                "ok": category == TOOL_OK,
                "error": category,
            })

        return {"tool_results": new_results, "pending_tool_calls": [], "metadata": meta}

    def _node_evaluate(self, state: AgentState) -> dict:
        """评估节点：判断信息是否足够，把 is_complete 写入 State。"""
        if state["iteration"] >= self.max_iterations:
            return {"is_complete": True}

        if not state["tool_results"] and not state["pending_tool_calls"]:
            return {"is_complete": True}

        # 有工具结果时，用 LLM 评估是否足够
        tools_called = [r["name"] for r in state["tool_results"]]
        results_summary = "\n".join(
            f"[{r['name']}] {r['result'][:200]}" for r in state["tool_results"]
        )

        prompt = EVALUATOR_PROMPT.format(
            user_input=state["user_input"],
            tools_called=", ".join(tools_called) if tools_called else "无",
            tool_results_summary=results_summary if results_summary else "无",
        )

        meta = dict(state.get("metadata", {}))
        meta["model_calls"] = int(meta.get("model_calls", 0)) + 1

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            raw = (response.choices[0].message.content or "").strip()
            eval_data = extract_json_object(raw)
            sufficient = eval_data.get("sufficient", True)
        except Exception:
            sufficient = True

        if sufficient:
            print(f"  [评估器] 信息充足，生成回复")
        else:
            print(f"  [评估器] 信息不足，继续规划")

        return {"is_complete": sufficient, "metadata": meta}

    def _node_respond(self, state: AgentState) -> dict:
        """回复节点：LLM 生成最终客服回复。"""
        context = "\n".join(
            f"- [{r['name']}({json.dumps(r['arguments'], ensure_ascii=False)})] {r['result']}"
            for r in state["tool_results"]
        ) if state["tool_results"] else "（未查询额外数据）"

        prompt = RESPONDER_PROMPT.format(
            intent=state["intent"],
            user_input=state["user_input"],
            context=context,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                *state["messages"],
            ],
            temperature=self.temperature,
        )
        final_text = response.choices[0].message.content or "抱歉，我暂时无法处理您的请求。"

        # 护栏：幻觉检测 + 自纠错
        final_text, corrections = self._guardrails.generate_with_guard(
            user_input=state["user_input"],
            tool_results=state.get("tool_results", []),
            generate_fn=lambda: final_text,
        )
        if corrections > 0:
            print(f"  [护栏] 完成 {corrections} 次自纠错")

        meta = dict(state.get("metadata", {}))
        meta["model_calls"] = int(meta.get("model_calls", 0)) + 1

        print(f"\n[小乐·LangGraph] 已生成回复 ({len(final_text)} 字)")
        return {"final_response": final_text, "metadata": meta}

    # ---------- Router ----------

    @staticmethod
    def _route_after_plan(state: AgentState) -> Literal["execute", "respond"]:
        """plan 之后：有工具调用则执行，否则直接回复。"""
        if state.get("pending_tool_calls"):
            return "execute"
        return "respond"

    def _route_after_evaluate(self, state: AgentState) -> Literal["plan", "respond"]:
        """条件路由：判断是回到 plan 继续循环，还是进入 respond 生成回复。"""
        # 超过最大循环次数 → 强制回复
        if state["iteration"] >= self.max_iterations:
            return "respond"

        # evaluate 已判定信息充足 → 回复
        if state.get("is_complete", False):
            return "respond"

        # 信息不足 → 回到 plan 继续规划
        return "plan"

    # ---------- Helpers (reuse from EcomAgent pattern) ----------

    def _build_messages(self) -> list[dict]:
        system_content = SYSTEM_PROMPT
        if self.skill_manager and self.skill_manager.enabled:
            system_content += self.skill_manager.build_catalog_prompt()

        messages: list[dict] = [
            {"role": "system", "content": system_content}
        ]
        messages.extend(self.memory_manager.build_memory_prompt_sections())
        if self.summary:
            messages.append({
                "role": "system",
                "content": f"以下是此前对话的摘要，用于延续上下文记忆：\n{self.summary}",
            })
        messages.extend(self.raw_messages)
        return messages

    def _extract_structured_response(self, text: str) -> CustomerServiceResponse:
        # 兜底：如果 text 为空或看起来不像客服回复，直接用原文
        if not text or not text.strip():
            return CustomerServiceResponse(
                intent=IntentType.OTHER, confidence=0.0,
                reply="抱歉，我暂时无法处理您的请求。", requires_human=True,
            )
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "基于以下客服回复内容，提取结构化信息。reply 字段直接使用原文，不要修改。",
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
                response_format=CustomerServiceResponse,
            )
            parsed = response.choices[0].message.parsed
            if not parsed.reply or len(parsed.reply.strip()) < 5:
                parsed.reply = text  # 提取失败时用原文
            return parsed
        except Exception:
            return self._extract_structured_fallback(text)

    def _extract_structured_fallback(self, text: str) -> CustomerServiceResponse:
        intent_values = ", ".join(f'"{e.value}"' for e in IntentType)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "基于以下客服回复内容，提取结构化信息并输出 JSON。\n"
                            "reply 字段直接使用原文，不要修改或缩减。\n\n"
                            "必须严格按照以下 JSON 格式输出（不要加 markdown 代码块）：\n"
                            "{\n"
                            f'  "intent": <从以下选择: {intent_values}>,\n'
                            '  "confidence": <0.0到1.0的浮点数>,\n'
                            '  "reply": <原文回复内容>,\n'
                            '  "requires_human": <true或false>,\n'
                            '  "follow_up_question": <追问问题或null>\n'
                            "}"
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
            )
            raw = (response.choices[0].message.content or "").strip()
            return CustomerServiceResponse.model_validate(extract_json_object(raw))
        except Exception:
            # 最终兜底：LLM 输出不可用（空内容/非法 JSON）时用原文，保证不抛错
            return CustomerServiceResponse(
                intent=IntentType.OTHER, confidence=0.0,
                reply=text, requires_human=False,
            )

    def _maybe_compress(self) -> None:
        """基于 Token 利用率的渐进式压缩。每轮打印当前状态。"""
        tokens = TokenCounter.count_messages(self.raw_messages)
        utilization = tokens / settings.token_budget

        # 始终打印当前 Token 状态（每轮都能看到）
        bar = int(utilization * 20)
        print(f"\n[Token预算] {tokens}/{settings.token_budget} tokens "
              f"({'#' * bar}{'·' * (20 - bar)}) {utilization:.1%}")

        if utilization < settings.compression_light_pct:
            return

        result = self._compressor.compress(self.raw_messages, self.summary)

        self.raw_messages = result.messages
        if result.summary != self.summary:
            self.summary = result.summary

        level_names = {0: "无", 1: "L1-轻量截断", 2: "L2-中度压缩", 3: "L3-LLM摘要"}
        label = level_names.get(result.level.value, "未知")
        print(
            f"[Token预算] 触发压缩: 级别={label} | "
            f"Token: {result.tokens_before}->{result.tokens_after}"
        )