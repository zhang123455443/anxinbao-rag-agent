from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID

if TYPE_CHECKING:
    from app.rag.chunking import ChunkingResult
    from app.rag.models import DocumentStatus, KnowledgeChunk, KnowledgeDocument

# 一、知识库构建 Pipeline 的节点协议，具体实现类通过方法签名自动满足约束。

# 知识库存储协议，同时服务于知识库构建和检索 Pipeline。
class KnowledgeRepository(Protocol):
    """知识库构建和检索共同依赖的数据访问能力。"""
    async def initialize_schema(self) -> None:
        ...

    async def find_document_by_checksum(self, checksum: str) -> KnowledgeDocument | None:
        ...

    async def save_document(self, document: KnowledgeDocument) -> KnowledgeDocument:
        ...

    async def update_document_status(
        self,
        document_id: UUID,
        status: DocumentStatus,
    ) -> KnowledgeDocument:
        ...

    async def save_chunks(self, chunks: list[KnowledgeChunk]) -> list[KnowledgeChunk]:
        ...

    async def replace_clause(
        self,
        document: KnowledgeDocument,
        chunks: list[KnowledgeChunk],
    ) -> None:
        ...

    async def get_parent_chunks(
        self,
        parent_chunk_ids: list[str],
    ) -> dict[str, KnowledgeChunk]:
        ...


# 文档解析协议
class Parser(Protocol):
    """原始文档解析节点。"""

    def parse(
        self,
        source_uri: str,
        options: dict[str, Any] | None = None,
    ) -> str:
        ...


class MarkdownOptimizer(Protocol):
    """对 Parser 生成的 Markdown 做结构修复。"""

    async def optimize(self, markdown: str) -> str:
        ...


# 文档切分协议
class Chunker(Protocol):
    """Markdown 分块节点。"""

    def chunk(
        self,
        *,
        markdown: str,
        document_id: UUID,
        clause_id: int | None = None,
    ) -> ChunkingResult:
        ...


# 文档向量化协议，同时返回稠密和稀疏向量{'dense': [], 'sparse': []}
class Embedding(Protocol):
    """查询和文档向量化节点。"""

    def embed_query(self, text: str) -> dict[str, Any]:
        ...

    def embed_documents(self, texts: list[str]) -> dict[str, Any]:
        ...


# 向量库协议
class VectorStore(Protocol):
    """子块向量存储节点。"""

    def store(
        self,
        document_name: str,
        chunks: list[KnowledgeChunk],
        vectors: dict[str, Any],
        product_ids: list[int],
    ) -> None:
        ...

    def delete_by_clause_id(self, clause_id: int) -> None:
        ...

# 二、知识检索协议

# Milvus 返回的原始分数和字段。
RetrievalHit = tuple[float, dict[str, Any]]

# 检索和重排后结果
@dataclass
class RetrievedChunk:
    """混合检索和重排节点之间传递的子块。"""

    parent_chunk_id: str  # 用于回查完整父块。
    document_id: str  # 用于标识引用文档。
    document_name: str  # 引用来源展示名称。
    section_path: list[str]  # 子块所在的 Markdown 章节路径。
    clause_no: str | None  # 条款编号，无法识别时为空。
    content: str  # 用于 Rerank 的子块正文。
    score: float  # 初始为混合检索分数，Rerank 后替换为重排分数。

# milvus检索协议
class Retriever(Protocol):
    """向量候选召回节点。"""

    def search(
        self,
        query: str,
        candidate_k: int,
        *,
        filter_expression: str | None = None,
        output_fields: list[str] | None = None,
    ) -> list[RetrievalHit]:
        ...

# 重排序协议
class Reranker(Protocol):
    """候选子块重排节点。"""

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        ...
