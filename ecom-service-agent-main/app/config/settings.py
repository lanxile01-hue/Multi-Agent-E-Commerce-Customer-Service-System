from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """项目配置，从 .env 文件读取"""

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.7

    # Embedding 独立配置（可与 LLM 使用不同提供商）
    embedding_api_key: str = ""  # 为空则回退到 openai_api_key
    embedding_api_base: str = ""  # 为空则回退到 openai_base_url
    embedding_batch_size: int = 10  # DashScope 限制 batch <= 10

    # ReAct 循环
    max_react_steps: int = 5

    # 可观测性 / LangSmith（可选）。设置 LANGCHAIN_TRACING_V2=true 时，
    # LangGraph 会自动接入 LangSmith；未配置时项目照常运行，不受影响。
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = ""

    # MCP 配置
    mcp_enabled: bool = False
    mcp_server_url: str = "http://127.0.0.1:9123/mcp"

    # RAG 配置（第5期）
    embedding_model: str = "text-embedding-3-small"
    kb_dir: str = "app/agent/rag/knowledge"
    rag_backend: str = "numpy"
    kb_index_path: str = "app/sessions/kb_index.json"
    chroma_persist_dir: str = "app/sessions/chroma"
    chroma_collection: str = "ecom_kb"
    # 检索可靠性阈值：top1 分数低于该值则视为不可靠。
    # Adaptive-RAG 的 RRF 分数与向量余弦分数不在同一量纲，因此分开配置。
    rag_score_threshold: float = 0.45
    rag_adaptive_score_threshold: float = 0.02

    # Adaptive-RAG 配置（魔改：混合检索 + 策略路由 + 重排序）
    adaptive_rag_enabled: bool = False
    bm25_index_path: str = "app/sessions/bm25_index.json"
    cohere_api_key: str = ""  # 可选，填写后启用 Cohere 重排序

    # Milvus 配置（向量数据库，可选替换 ChromaDB）
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_timeout: int = 10000

    # 文档分块配置
    chunk_max_size: int = 800
    chunk_overlap: int = 100

    # Prometheus 配置（告警查询等）
    prometheus_base_url: str = "http://127.0.0.1:9090"
    prometheus_request_timeout: float = 10.0

    # Multi-Agent 配置（第6期）
    multi_agent_enabled: bool = False

    # LangGraph 配置（魔改：替换 ReAct 为状态图）
    langgraph_enabled: bool = False

    # Supervisor 配置（LangGraph 父图 + 三个子图，默认推荐模式）
    supervisor_enabled: bool = True
    # Checkpoint 配置（父图持有）
    # - checkpoint_backend: memory（仅进程内，重启丢失）/ sqlite（磁盘持久化，可跨进程）
    #   / postgres（未来扩展，需额外安装 langgraph-checkpoint-postgres）/ auto
    # - CHECKPOINT_PATH 只在 sqlite（文件路径）与 postgres（连接串）下生效。
    checkpoint_enabled: bool = True
    checkpoint_backend: str = "memory"
    checkpoint_path: str = "app/sessions/checkpoints.db"
    # 人工确认（退款/赔偿/投诉升级等敏感操作）
    human_confirmation_enabled: bool = True
    # 子图最大循环步数（达到后强制生成回复，防死循环）
    subgraph_max_steps: int = 5
    # 工具调用超时（秒），0 表示不设超时
    tool_timeout: float = 30.0
    # 退款幂等与审计存储（SQLite，避免进程重启/多进程重复提交）
    refund_store_path: str = "app/sessions/refunds.db"

    # Memory 配置（第7期）
    memory_enabled: bool = True
    memory_dir: str = "app/sessions/memory"
    memory_user_id: str = "default"
    max_ltm_facts: int = 50

    # Skill 配置（第8期）
    skills_enabled: bool = True
    skills_dir: str = "skills"

    # Evaluation 配置（第9期，离线评估工具，无聊天开关）
    eval_dataset_path: str = "app/evaluation/cases.json"
    eval_use_judge: bool = True  # 是否启用 LLM-as-judge（质量/幻觉/过程合理性）
    eval_pass_threshold: float = 0.6  # 单维度通过阈值（judge 归一化到 0-1 后比较）

    # 多轮对话管理
    session_path: str = "app/sessions/session.json"
    history_threshold: int = 10
    history_keep_recent: int = 3

    # Token 预算配置（魔改：3层渐进式压缩）
    token_budget: int = 128000  # 最大 token 预算
    compression_light_pct: float = 0.50  # L1 触发阈值
    compression_medium_pct: float = 0.70  # L2 触发阈值
    compression_aggressive_pct: float = 0.85  # L3 触发阈值

    # 护栏配置（魔改：幻觉检测 + 自纠错 + 双向 Fallback）
    guardrails_enabled: bool = False

    model_config = {"env_file": ".env"}


settings = Settings()
