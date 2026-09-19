"""知识库访问上下文与最小 RBAC 策略。

当前实现是进程内请求上下文，适合单体 API；生产环境可把身份解析替换为 JWT/OIDC，
但下游检索只依赖 AccessContext，不需要改权限判断逻辑。
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AccessContext:
    tenant_id: str = "default"
    user_id: str = "anonymous"
    role: str = "member"
    # `*` 代表兼容旧版调用的默认可见范围；真实身份应传入明确部门集合。
    departments: tuple[str, ...] = field(default_factory=lambda: ("*",))


_current_context: ContextVar[AccessContext] = ContextVar(
    "knowledge_access_context", default=AccessContext()
)


def get_access_context() -> AccessContext:
    return _current_context.get()


def set_access_context(context: AccessContext):
    return _current_context.set(context)


def reset_access_context(token) -> None:
    _current_context.reset(token)


def can_read(tenant_id: str, access_level: str, department: str) -> bool:
    """判断当前请求是否能读取一个知识 chunk。"""
    context = get_access_context()
    if (tenant_id or "default") != context.tenant_id:
        return False

    level = (access_level or "internal").lower()
    role = context.role.lower()
    allowed_levels = {
        "guest": {"public"},
        "member": {"public", "internal"},
        "manager": {"public", "internal", "confidential"},
        "admin": {"public", "internal", "confidential", "restricted"},
    }.get(role, {"public"})
    if level not in allowed_levels:
        return False

    # 管理角色可跨部门；普通成员只能读取自己所属部门或无部门公共文档。
    if role not in {"manager", "admin"} and department:
        return "*" in context.departments or department in context.departments
    return True
