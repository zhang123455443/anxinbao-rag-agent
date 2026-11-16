from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.rag.models import ChunkLevel, KnowledgeChunk, SectionType


CLAUSE_NO_RE = re.compile(r"^\s*((?:第[一二三四五六七八九十百零〇两]+条)|(?:\d+(?:\.\d+)+))")


@dataclass(frozen=True)
class ChunkingResult:
    """一次文档切分产生的父块集合和子块集合。"""

    parent_chunks: list[KnowledgeChunk]  # 按 Markdown 章节生成的完整上下文块。
    child_chunks: list[KnowledgeChunk]  # 在父块内递归切分的检索块。


@dataclass(frozen=True)
class _Section:
    """Markdown 标题切分后的临时章节结构。"""

    section_path: list[str]  # 当前章节从一级标题到三级标题的路径。
    content: str  # 章节 Markdown 原文。


class MarkdownChunker:
    """先按 Markdown 标题生成父块，再在每个父块内部递归生成子块。"""

    def __init__(
        self,
        *,
        child_target_chars: int = 500,
        child_overlap_chars: int = 80,
    ) -> None:
        # overlap 必须小于目标长度，否则递归切分器无法正常前进。
        if child_target_chars <= 0:
            raise ValueError("child_target_chars must be positive")
        if child_overlap_chars < 0:
            raise ValueError("child_overlap_chars must be non-negative")
        if child_overlap_chars >= child_target_chars:
            raise ValueError("child_overlap_chars must be smaller than child_target_chars")

        self.child_target_chars = child_target_chars
        self.child_overlap_chars = child_overlap_chars
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            # 只识别前三层标题，避免过深标题让父块过于零碎。
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ],
            # 标题由 section_path 统一拼接到每个父块和子块，避免切分后只有首块带标题。
            strip_headers=True,
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_target_chars,
            chunk_overlap=child_overlap_chars,
            separators=["\n\n", "\n", "。", "；", ";", "，", ",", " ", ""],
            keep_separator="end",
        )

    def chunk(
        self,
        *,
        markdown: str,
        document_id: UUID,
        clause_id: int | None = None,
    ) -> ChunkingResult:
        """为一份 Markdown 文档生成一一关联的父块和子块。"""

        sections = self._split_parent_sections(markdown)
        parent_chunks: list[KnowledgeChunk] = []
        child_chunks: list[KnowledgeChunk] = []

        for section in sections:
            parent_content = _prepend_section_headings(
                section.section_path,
                section.content,
            )
            # 父块保存完整章节，用于检索后的上下文扩展和引用展示。
            parent = KnowledgeChunk(
                chunk_id=uuid4(),
                document_id=document_id,
                clause_id=clause_id,
                chunk_level=ChunkLevel.PARENT,
                section_path=section.section_path,
                section_type=_infer_section_type(section.section_path, parent_content),
                clause_no=_infer_clause_no(parent_content),
                content=parent_content,
            )
            parent_chunks.append(parent)

            # 只切分正文，再给每个子块补上完整标题链，标题本身不计入正文切片边界。
            for child_text in self._split_child_text(section.content):
                # 子块继承父块元数据，用更短的文本参与向量检索。
                child_chunks.append(
                    KnowledgeChunk(
                        chunk_id=uuid4(),
                        parent_chunk_id=parent.chunk_id,
                        document_id=document_id,
                        clause_id=clause_id,
                        chunk_level=ChunkLevel.CHILD,
                        section_path=parent.section_path,
                        section_type=parent.section_type,
                        clause_no=parent.clause_no,
                        content=_prepend_section_headings(
                            parent.section_path,
                            child_text,
                        ),
                    )
                )

        return ChunkingResult(parent_chunks=parent_chunks, child_chunks=child_chunks)

    def _split_parent_sections(self, markdown: str) -> list[_Section]:
        """按照 Markdown 标题切分章节，没有标题时将全文作为一个父块。"""

        stripped = markdown.strip()
        if not stripped:
            return []

        documents = self.markdown_splitter.split_text(markdown)
        if not documents:
            return [
                _Section(
                    section_path=["全文"],
                    content=stripped,
                )
            ]

        sections: list[_Section] = []
        for document in documents:
            content = str(document.page_content).strip()
            if not content:
                continue
            sections.append(
                _Section(
                    section_path=_section_path_from_metadata(document.metadata),
                    content=content,
                )
            )

        return sections

    def _split_child_text(self, content: str) -> list[str]:
        """在单个父块内部递归切分，保证子块不会跨章节。"""

        if not content.strip():
            return []

        return [
            text.strip() for text in self.child_splitter.split_text(content) if text.strip()
        ]

def _section_path_from_metadata(metadata: dict[str, Any]) -> list[str]:
    """将 LangChain 的标题元数据转换为从一级到三级的章节路径。"""

    section_path = [
        str(metadata[key]).strip()
        for key in ("h1", "h2", "h3")
        if metadata.get(key) and str(metadata[key]).strip()
    ]
    return section_path or ["全文"]


def _prepend_section_headings(section_path: list[str], content: str) -> str:
    """把一至三级标题链拼接到块首，让每个块都保留完整章节语境。"""

    body = content.strip()
    if not body or section_path == ["全文"]:
        return body

    headings = "\n".join(
        f"{'#' * level} {title}"
        for level, title in enumerate(section_path[:3], start=1)
    )
    return f"{headings}\n\n{body}"


def _infer_section_type(section_path: list[str], content: str) -> SectionType:
    """优先根据标题、再根据正文开头识别保险章节类型。"""

    path_text = " ".join(section_path)
    text = path_text + "\n" + content[:200]
    if "等待期" in path_text:
        return SectionType.WAITING_PERIOD
    if "责任免除" in path_text or "免责" in path_text or "除外" in path_text:
        return SectionType.EXCLUSION
    if "保险责任" in path_text or "保障责任" in path_text:
        return SectionType.COVERAGE
    if "健康告知" in path_text:
        return SectionType.HEALTH_NOTICE
    if "理赔材料" in path_text:
        return SectionType.CLAIM_MATERIAL
    if "理赔流程" in path_text or "申请理赔" in path_text:
        return SectionType.CLAIM_PROCESS
    if "释义" in path_text or "定义" in path_text:
        return SectionType.DEFINITION
    if "等待期" in text:
        return SectionType.WAITING_PERIOD
    if "责任免除" in text or "免责" in text or "除外" in text:
        return SectionType.EXCLUSION
    if "保险责任" in text or "保障责任" in text:
        return SectionType.COVERAGE
    if "健康告知" in text:
        return SectionType.HEALTH_NOTICE
    if "理赔材料" in text:
        return SectionType.CLAIM_MATERIAL
    if "理赔流程" in text or "申请理赔" in text:
        return SectionType.CLAIM_PROCESS
    if "释义" in text or "定义" in text:
        return SectionType.DEFINITION
    return SectionType.OTHER


def _infer_clause_no(content: str) -> str | None:
    """从章节首部识别“第X条”或多级数字形式的条款编号。"""

    for line in content.splitlines():
        match = CLAUSE_NO_RE.match(line)
        if match:
            return match.group(1)
    return None
