"""按 Markdown 二级标题切分知识库文档。

切分策略：
- 以 `## ` 二级标题为切分边界，每个二级章节作为一个 chunk。
- 二级章节内的三级/四级小节保留在同一 chunk 中（保持语义完整）。
- 文档开头到第一个二级标题之间的内容（含一级标题和导语）作为 "概览" chunk。
- chunk 长度超过 ~1200 字时按段落进一步切分，避免单个 chunk 过长。

每个 chunk 保留：
- doc：文档名（不含扩展名），如 "退换货政策"
- section：章节标题，如 "二、质量问题退换货"
- text：chunk 全文（含小节结构）
- chunk_id：稳定的字符串 id，便于增量更新
- title/source/version/effective_time/updated_at：用于证据引用和文档生命周期
- department/access_level/tags：为后续权限过滤和知识空间隔离预留
- tenant_id：知识空间/租户标识，默认值为 `default`
"""

import re
from dataclasses import asdict, dataclass
from pathlib import Path

MAX_CHUNK_CHARS = 250


@dataclass
class Chunk:
    chunk_id: str
    doc: str
    section: str
    text: str
    # 证据链元数据（来源/版本/生效时间/更新时间）。
    # 当前知识库为纯 markdown，version/effective_time/updated_at 暂为空，
    # 预留字段便于未来接入带 frontmatter 元数据的文档并做过期过滤。
    source: str = ""
    title: str = ""
    version: str = ""
    effective_time: str = ""
    updated_at: str = ""
    department: str = ""
    access_level: str = "internal"
    tags: list[str] | None = None
    tenant_id: str = "default"

    def to_dict(self) -> dict:
        return asdict(self)


def chunk_markdown_dir(kb_dir: Path) -> list[Chunk]:
    """扫描目录下所有 .md 文件，逐一切分并汇总。"""
    chunks: list[Chunk] = []
    for md_path in sorted(kb_dir.glob("*.md")):
        chunks.extend(_chunk_one_file(md_path))
    return chunks


def _chunk_one_file(md_path: Path) -> list[Chunk]:
    raw = md_path.read_text(encoding="utf-8")
    metadata, raw = _parse_frontmatter(raw)
    doc_name = md_path.stem
    sections = _split_by_h2(raw)

    out: list[Chunk] = []
    for idx, (section_title, section_body) in enumerate(sections):
        text = section_body.strip()
        if not text:
            continue

        if len(text) <= MAX_CHUNK_CHARS:
            out.append(_make_chunk(doc_name, section_title, text, idx, 0, metadata))
            continue

        for sub_idx, piece in enumerate(_split_long(text)):
            out.append(_make_chunk(doc_name, section_title, piece, idx, sub_idx, metadata))
    return out


def _split_by_h2(raw: str) -> list[tuple[str, str]]:
    """返回 [(section_title, section_body), ...]。

    第一个 section 是文档头部（H1 + 导语），title 取 H1 文本。
    """
    lines = raw.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_title = ""
    current_body: list[str] = []

    for line in lines:
        if line.startswith("# ") and not current_body and not sections:
            current_title = line[2:].strip() + " · 概览"
            continue
        if line.startswith("## "):
            if current_body:
                sections.append((current_title or "概览", current_body))
            current_title = line[3:].strip()
            current_body = []
            continue
        current_body.append(line)

    if current_body:
        sections.append((current_title or "概览", current_body))

    return [(t, "\n".join(b).strip()) for t, b in sections]


def _split_long(text: str) -> list[str]:
    """按段落贪心打包到 MAX_CHUNK_CHARS。"""
    paragraphs = re.split(r"\n\s*\n", text)
    pieces: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if buf and buf_len + len(p) + 2 > MAX_CHUNK_CHARS:
            pieces.append("\n\n".join(buf))
            buf = [p]
            buf_len = len(p)
        else:
            buf.append(p)
            buf_len += len(p) + 2
    if buf:
        pieces.append("\n\n".join(buf))
    return pieces


def _parse_frontmatter(raw: str) -> tuple[dict[str, str | list[str]], str]:
    """读取轻量 YAML 子集，避免为知识文档引入新的运行时依赖。

    支持位于文档开头的 ``---`` 块，字段值可以是普通字符串或逗号分隔列表。
    没有 frontmatter 的历史文档保持兼容，并使用安全默认值。
    """
    if not raw.startswith("---"):
        return {}, raw
    lines = raw.splitlines()
    if len(lines) < 3:
        return {}, raw
    try:
        end = next(i for i, line in enumerate(lines[1:], 1) if line.strip() == "---")
    except StopIteration:
        return {}, raw

    metadata: dict[str, str | list[str]] = {}
    for line in lines[1:end]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip().strip('"\'')
        if not key:
            continue
        if key == "tags":
            metadata[key] = [item.strip() for item in value.split(",") if item.strip()]
        else:
            metadata[key] = value
    return metadata, "\n".join(lines[end + 1:])


def _make_chunk(
    doc: str,
    section: str,
    text: str,
    idx: int,
    sub_idx: int,
    metadata: dict[str, str | list[str]] | None = None,
) -> Chunk:
    metadata = metadata or {}
    chunk_id = f"{doc}#{idx:02d}-{sub_idx:02d}"
    body = f"【{doc} · {section}】\n{text}"
    return Chunk(
        chunk_id=chunk_id, doc=doc, section=section, text=body,
        source=str(metadata.get("source") or doc),
        title=str(metadata.get("title") or doc),
        version=str(metadata.get("version") or ""),
        effective_time=str(metadata.get("effective_time") or ""),
        updated_at=str(metadata.get("updated_at") or ""),
        department=str(metadata.get("department") or ""),
        access_level=str(metadata.get("access_level") or "internal"),
        tags=list(metadata.get("tags") or []),
        tenant_id=str(metadata.get("tenant_id") or "default"),
    )
