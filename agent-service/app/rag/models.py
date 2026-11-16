from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class DocumentType(StrEnum):
    """知识文档的业务分类。"""

    CLAUSE = "clause"
    PRODUCT_MANUAL = "product_manual"
    SERVICE_MANUAL = "service_manual"
    CLAIM_GUIDE = "claim_guide"
    FAQ = "faq"


class DocumentStatus(StrEnum):
    """文档在知识库构建 Pipeline 中的执行状态。"""

    PENDING_PARSE = "pending_parse"
    PARSED = "parsed"
    CHUNKED = "chunked"
    EMBEDDED = "embedded"
    INDEXED = "indexed"
    FAILED = "failed"


class ChunkLevel(StrEnum):
    """区分用于展示的父块和用于检索的子块。"""

    PARENT = "parent"
    CHILD = "child"


class SectionType(StrEnum):
    """保险文档中可识别并用于检索过滤的章节类型。"""

    COVERAGE = "coverage"
    EXCLUSION = "exclusion"
    WAITING_PERIOD = "waiting_period"
    HEALTH_NOTICE = "health_notice"
    CLAIM_MATERIAL = "claim_material"
    CLAIM_PROCESS = "claim_process"
    DEFINITION = "definition"
    OTHER = "other"


class KnowledgeDocument(BaseModel):
    """一份进入知识库构建 Pipeline 的文档记录。"""

    document_id: UUID  # 知识文档唯一标识，也是父子块关联键。
    clause_id: int | None = None  # 关联条款；产品手册等资料可以为空。
    document_type: DocumentType  # 文档业务分类。
    document_name: str  # 原始文档名称，也是引用来源的展示名称。
    source_uri: str  # MinerU 支持的本地文件路径或 HTTP/HTTPS URL。
    version: str  # 业务版本号或发布日期。
    parse_provider: str = "mineru_open_sdk"  # 文档解析服务标识。
    metadata: dict[str, Any] = Field(default_factory=dict)  # 文档级扩展属性。
    status: DocumentStatus = DocumentStatus.PENDING_PARSE  # Pipeline 当前执行状态。
    checksum: str  # 原始文件 SHA-256，用于识别重复文档。
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))  # 创建时间。
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))  # 更新时间。


class KnowledgeChunk(BaseModel):
    """父块和子块共用的数据模型，通过 chunk_level 区分层级。"""

    chunk_id: UUID  # 分块唯一标识；子块 ID 同时作为 Milvus 主键。
    parent_chunk_id: UUID | None = None  # 子块所属父块；父块自身为空。
    document_id: UUID  # 所属知识文档。
    clause_id: int | None = None  # 冗余条款 ID，便于引用和过滤。
    chunk_level: ChunkLevel  # 标识当前内容是父块还是子块。
    section_path: list[str] = Field(default_factory=list)  # Markdown 标题层级路径。
    section_type: SectionType = SectionType.OTHER  # 保险责任、免责、等待期等语义章节类型。
    clause_no: str | None = None  # 从章节内容识别出的条款编号。
    page_start: int | None = None  # 来源起始页；当前 Chunker 暂未填充。
    page_end: int | None = None  # 来源结束页；当前 Chunker 暂未填充。
    content: str  # 分块正文；父块用于展示，子块用于向量化。
    token_count: int | None = None  # token 统计预留字段，当前 Pipeline 不计算。
    metadata: dict[str, Any] = Field(default_factory=dict)  # 坐标、表格等扩展信息。
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))  # 创建时间。
