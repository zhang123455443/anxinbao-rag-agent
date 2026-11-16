from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from app.core.database import initialize_schema
from app.rag.models import (
    DocumentStatus,
    KnowledgeChunk,
    KnowledgeDocument,
)


class PostgresKnowledgeRepository:
    """统一负责知识文档、父块和子块在 PostgreSQL 中的持久化。"""

    def __init__(self, pool: Any) -> None:
        self.pool = pool

    async def initialize_schema(self) -> None:
        """执行 RAG 模块自带的建表 SQL。"""

        await initialize_schema(self.pool)

    async def find_document_by_checksum(self, checksum: str) -> KnowledgeDocument | None:
        """按内容指纹查询文档，用于避免重复构建知识库。"""

        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                SELECT *
                FROM knowledge_documents
                WHERE checksum = $1
                """,
                checksum,
            )
        return _document_from_row(row) if row is not None else None

    async def save_document(self, document: KnowledgeDocument) -> KnowledgeDocument:
        """新增文档记录，并返回数据库中的完整记录。"""

        async with self.pool.acquire() as connection:
            row = await _insert_document(connection, document)
        return _document_from_row(row)

    async def update_document_status(
        self,
        document_id: UUID,
        status: DocumentStatus,
    ) -> KnowledgeDocument:
        """更新 Pipeline 执行状态及文档更新时间。"""

        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                UPDATE knowledge_documents
                SET
                    status = $2,
                    updated_at = CURRENT_TIMESTAMP
                WHERE document_id = $1
                RETURNING *
                """,
                document_id,
                _enum_value(status),
            )
        return _document_from_row(row)

    async def save_parent_chunks(self, chunks: list[KnowledgeChunk]) -> list[KnowledgeChunk]:
        """保存用于上下文扩展和溯源展示的完整父块。"""

        async with self.pool.acquire() as connection:
            await _insert_parent_chunks(connection, chunks)
        return chunks

    async def save_child_chunks(self, chunks: list[KnowledgeChunk]) -> list[KnowledgeChunk]:
        """保存用于知识库管理和向量检索的子块元数据。"""

        async with self.pool.acquire() as connection:
            await _insert_child_chunks(connection, chunks)
        return chunks

    async def save_chunks(self, chunks: list[KnowledgeChunk]) -> list[KnowledgeChunk]:
        """按块级别分组，并分别写入父块表和子块表。"""

        parent_chunks, child_chunks = _split_chunks(chunks)
        if parent_chunks:
            await self.save_parent_chunks(parent_chunks)
        if child_chunks:
            await self.save_child_chunks(child_chunks)
        return chunks

    async def replace_clause(
        self,
        document: KnowledgeDocument,
        chunks: list[KnowledgeChunk],
    ) -> None:
        """在一个事务中删除同一 clause_id 的旧数据并写入新构建结果。"""

        if document.clause_id is None:
            raise ValueError("clause_id is required when replacing clause knowledge")
        parent_chunks, child_chunks = _split_chunks(chunks)
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                old_document_ids = await connection.fetch(
                    "SELECT document_id FROM knowledge_documents WHERE clause_id = $1",
                    document.clause_id,
                )
                document_ids = [row["document_id"] for row in old_document_ids]
                if document_ids:
                    await connection.execute(
                        "DELETE FROM knowledge_ingestion_jobs WHERE document_id = ANY($1::uuid[])",
                        document_ids,
                    )
                    await connection.execute(
                        "DELETE FROM knowledge_child_chunks WHERE document_id = ANY($1::uuid[])",
                        document_ids,
                    )
                    await connection.execute(
                        "DELETE FROM knowledge_parent_chunks WHERE document_id = ANY($1::uuid[])",
                        document_ids,
                    )
                    await connection.execute(
                        "DELETE FROM knowledge_documents WHERE document_id = ANY($1::uuid[])",
                        document_ids,
                    )
                await _insert_document(connection, document)
                await _insert_parent_chunks(connection, parent_chunks)
                await _insert_child_chunks(connection, child_chunks)

    async def get_parent_chunks(
        self,
        parent_chunk_ids: list[str],
    ) -> dict[str, KnowledgeChunk]:
        """批量读取父块，供 Search Tool 将命中的子块扩展为完整上下文。"""

        if not parent_chunk_ids:
            return {}
        query_ids = [UUID(str(parent_chunk_id)) for parent_chunk_id in parent_chunk_ids]
        async with self.pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT *
                FROM knowledge_parent_chunks
                WHERE parent_chunk_id = ANY($1::uuid[])
                """,
                query_ids,
            )
        chunks = [_parent_chunk_from_row(row) for row in rows]
        return {str(chunk.chunk_id): chunk for chunk in chunks}


async def _insert_document(connection: Any, document: KnowledgeDocument) -> Any:
    """使用已有连接插入文档，便于普通保存和事务替换复用。"""

    return await connection.fetchrow(
        """
        INSERT INTO knowledge_documents (
            document_id, clause_id, document_type, document_name, source_uri, version,
            status, checksum, parse_provider, metadata, created_at, updated_at
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11, $12)
        RETURNING *
        """,
        document.document_id,
        document.clause_id,
        _enum_value(document.document_type),
        document.document_name,
        document.source_uri,
        document.version,
        _enum_value(document.status),
        document.checksum,
        document.parse_provider,
        json.dumps(document.metadata, ensure_ascii=False),
        _pg_timestamp(document.created_at),
        _pg_timestamp(document.updated_at),
    )


async def _insert_parent_chunks(connection: Any, chunks: list[KnowledgeChunk]) -> None:
    """使用已有连接批量插入父块。"""

    for chunk in chunks:
        await connection.execute(
            """
            INSERT INTO knowledge_parent_chunks (
                parent_chunk_id, document_id, clause_id, section_path, section_type,
                clause_no, page_start, page_end, content, metadata, token_count, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11, $12)
            """,
            chunk.chunk_id,
            chunk.document_id,
            chunk.clause_id,
            chunk.section_path,
            _enum_value(chunk.section_type),
            chunk.clause_no,
            chunk.page_start,
            chunk.page_end,
            chunk.content,
            json.dumps(chunk.metadata, ensure_ascii=False),
            chunk.token_count,
            _pg_timestamp(chunk.created_at),
        )


async def _insert_child_chunks(connection: Any, chunks: list[KnowledgeChunk]) -> None:
    """使用已有连接批量插入子块。"""

    for chunk in chunks:
        await connection.execute(
            """
            INSERT INTO knowledge_child_chunks (
                child_chunk_id, parent_chunk_id, document_id, clause_id, section_path,
                section_type, clause_no, page_start, page_end, content, token_count,
                metadata, milvus_collection, vector_status, created_at
            )
            VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
                $12::jsonb, $13, $14, $15
            )
            """,
            chunk.chunk_id,
            chunk.parent_chunk_id,
            chunk.document_id,
            chunk.clause_id,
            chunk.section_path,
            _enum_value(chunk.section_type),
            chunk.clause_no,
            chunk.page_start,
            chunk.page_end,
            chunk.content,
            chunk.token_count,
            json.dumps(chunk.metadata, ensure_ascii=False),
            chunk.metadata.get("milvus_collection"),
            chunk.metadata.get("vector_status", "pending"),
            _pg_timestamp(chunk.created_at),
        )


def _split_chunks(
    chunks: list[KnowledgeChunk],
) -> tuple[list[KnowledgeChunk], list[KnowledgeChunk]]:
    """按父块和子块拆分待保存数据。"""

    parent_chunks = [chunk for chunk in chunks if _enum_value(chunk.chunk_level) == "parent"]
    child_chunks = [chunk for chunk in chunks if _enum_value(chunk.chunk_level) == "child"]
    return parent_chunks, child_chunks


def _document_from_row(row: Any) -> KnowledgeDocument:
    """将 asyncpg 文档记录转换为领域模型。"""

    return KnowledgeDocument(
        document_id=row["document_id"],
        clause_id=row["clause_id"],
        document_type=row["document_type"],
        document_name=row["document_name"],
        source_uri=row["source_uri"],
        version=row["version"],
        status=row["status"],
        checksum=row["checksum"],
        parse_provider=row["parse_provider"],
        metadata=_json_value(row["metadata"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _parent_chunk_from_row(row: Any) -> KnowledgeChunk:
    """将 asyncpg 父块记录转换为领域模型。"""

    return KnowledgeChunk(
        chunk_id=row["parent_chunk_id"],
        document_id=row["document_id"],
        clause_id=row["clause_id"],
        chunk_level="parent",
        section_path=list(row["section_path"]),
        section_type=row["section_type"],
        clause_no=row["clause_no"],
        page_start=row["page_start"],
        page_end=row["page_end"],
        content=row["content"],
        token_count=row["token_count"],
        metadata=_json_value(row["metadata"]),
        created_at=row["created_at"],
    )


def _json_value(value: Any) -> dict[str, Any]:
    """兼容 asyncpg 可能返回的 JSON 字符串或映射对象。"""

    if value is None:
        return {}
    if isinstance(value, str):
        return json.loads(value)
    return dict(value)


def _enum_value(value: Any) -> str:
    """将 StrEnum 转为数据库实际保存的字符串值。"""

    return value.value if hasattr(value, "value") else str(value)


def _pg_timestamp(value: datetime | None) -> datetime | None:
    """将带时区时间统一转为 PostgreSQL 当前使用的无时区 UTC 时间。"""

    if value is None or value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)
