from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from app.rag.models import (
    DocumentStatus,
    DocumentType,
    KnowledgeChunk,
    KnowledgeDocument,
)
from app.rag.protocols import (
    Chunker,
    Embedding,
    KnowledgeRepository,
    MarkdownOptimizer,
    Parser,
    VectorStore,
)


class KnowledgeBuildRequest(BaseModel):
    """构建一份知识文档所需的输入。"""

    product_ids: list[int] = Field(default_factory=lambda: [-1])  # 可检索该文档的产品 ID。
    clause_id: int | None = None  # 关联的保险条款；非条款文档可以为空。
    document_type: DocumentType  # 条款、产品手册、理赔指南等文档分类。
    document_name: str  # 用于知识库管理和检索结果展示的文档名称。
    source_uri: str  # MinerU 可以访问的原始文档路径或 URL。
    version: str  # 文档版本，用于区分同一资料的不同发布版本。
    content: bytes  # 原始文件内容，仅用于计算 checksum，避免重复构建。
    parse_options: dict[str, Any] = Field(default_factory=dict)  # MinerU 解析参数。
    metadata: dict[str, Any] = Field(default_factory=dict)  # 文档级扩展信息。


@dataclass(frozen=True)
class KnowledgeBuildResult:
    """返回文档状态以及本次生成的父块、子块。"""

    document: KnowledgeDocument  # 入库后的文档记录，包含最终构建状态。
    created: bool  # False 表示 checksum 已存在，本次没有重复构建。
    parent_chunks: list[KnowledgeChunk]  # 用于完整上下文展示和条款溯源。
    child_chunks: list[KnowledgeChunk]  # 用于生成向量并执行语义检索。


class KnowledgeBuildPipeline:
    """按固定顺序编排知识库构建节点。"""

    def __init__(
        self,
        *,
        repository: KnowledgeRepository,
        parser: Parser,
        markdown_optimizer: MarkdownOptimizer,
        chunker: Chunker,
        embedding: Embedding,
        vector_store: VectorStore,
    ) -> None:
        self.repository = repository
        self.parser = parser
        self.markdown_optimizer = markdown_optimizer
        self.chunker = chunker
        self.embedding = embedding
        self.vector_store = vector_store

    async def ingest_document(self, request: KnowledgeBuildRequest) -> KnowledgeBuildResult:
        # 1.校验文档的数字指纹，判断是否存在，避免重复处理
        checksum = hashlib.sha256(request.content).hexdigest()
        existing = await self.repository.find_document_by_checksum(checksum)
        if request.clause_id is None and existing is not None:
            return KnowledgeBuildResult(existing, False, [], [])

        document = self._new_document(request, checksum)

        # 2. 先解析PDF文档为Markdown
        markdown = self.parser.parse(request.source_uri, request.parse_options)

        # 3. 使用 DeepSeek 修复 PDF 表格导致的 Markdown 结构和标题编号错乱。
        markdown = await self.markdown_optimizer.optimize(markdown)

        # 4.对文档切片，分为父块和子块
        chunks = self.chunker.chunk(
            markdown=markdown,
            document_id=document.document_id,
            clause_id=document.clause_id,
        )

        # 5.文本向量化，同时生成稠密和稀疏向量
        vectors = self.embedding.embed_documents(
            [chunk.content for chunk in chunks.child_chunks]
        )
        document.status = DocumentStatus.EMBEDDED

        # 6. 把文档、分片保存到PostgreSQL，如果条款已存在，则替换，否则就新增。
        all_chunks = chunks.parent_chunks + chunks.child_chunks
        if document.clause_id is not None:
            # PostgreSQL 在一个事务中替换旧条款，或保存新的通用资料。
            await self.repository.replace_clause(document, all_chunks)
        else:
            await self.repository.save_document(document)
            await self.repository.save_chunks(all_chunks)

        # 7.更新向量库，先删除旧的，再写入新的
        if document.clause_id is not None:
            self.vector_store.delete_by_clause_id(document.clause_id)
        self.vector_store.store(
            document.document_name,
            chunks.child_chunks,
            vectors,
            request.product_ids,
        )
        document = await self.repository.update_document_status(
            document.document_id,
            DocumentStatus.INDEXED,
        )

        return KnowledgeBuildResult(
            document=document,
            created=True,
            parent_chunks=chunks.parent_chunks,
            child_chunks=chunks.child_chunks,
        )

    def _new_document(
        self,
        request: KnowledgeBuildRequest,
        checksum: str,
    ) -> KnowledgeDocument:
        """创建尚未持久化的文档实体。"""

        return KnowledgeDocument(
            document_id=uuid4(),
            clause_id=request.clause_id,
            document_type=request.document_type,
            document_name=request.document_name,
            source_uri=request.source_uri,
            version=request.version,
            metadata=request.metadata,
            checksum=checksum,
        )
