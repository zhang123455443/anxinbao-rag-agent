from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import asyncpg

from app.core.config import Settings
from app.core.database import asyncpg_database_url
from app.rag.models import DocumentType
from app.rag.pipeline import KnowledgeBuildRequest
from scripts.rag_ingest import create_pipeline, create_pool


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest every product clause from insurance_biz into the RAG store."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop only RAG tables and the configured Milvus collection before ingesting.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    settings = Settings()
    if not settings.business_database_url:
        raise RuntimeError("BUSINESS_DATABASE_URL or INFRA_HOST is not configured")

    # 先读取业务数据并校验全部文件，避免清空知识库后才发现源文件缺失。
    requests = await load_requests(settings.business_database_url)
    agent_pool = await create_pool(settings)
    try:
        # 模型和外部客户端先初始化；初始化失败时不会破坏已有知识库。
        pipeline = await create_pipeline(settings, agent_pool)
        if args.reset:
            await reset_rag_stores(settings, agent_pool)
            await pipeline.repository.initialize_schema()

        items: list[dict[str, Any]] = []
        for index, request in enumerate(requests, start=1):
            print(f"[{index}/{len(requests)}] ingesting {request.document_name}", flush=True)
            try:
                result = await pipeline.ingest_document(request)
            except Exception as exc:
                items.append(
                    {
                        "clause_id": request.clause_id,
                        "document_name": request.document_name,
                        "error": str(exc),
                    }
                )
                if not args.continue_on_error:
                    raise
                continue
            items.append(
                {
                    "clause_id": request.clause_id,
                    "document_name": request.document_name,
                    "status": result.document.status.value,
                    "parent_chunks": len(result.parent_chunks),
                    "child_chunks": len(result.child_chunks),
                }
            )
    finally:
        await agent_pool.close()

    return {
        "total": len(items),
        "succeeded": sum("error" not in item for item in items),
        "failed": sum("error" in item for item in items),
        "items": items,
    }


async def load_requests(business_database_url: str) -> list[KnowledgeBuildRequest]:
    """从业务库读取唯一条款，并聚合每份条款关联的全部产品 ID。"""

    connection = await asyncpg.connect(asyncpg_database_url(business_database_url))
    try:
        rows = await connection.fetch(
            """
            SELECT
                c.id AS clause_id,
                c.file_name,
                c.file_path,
                c.clause_type,
                c.updated_at,
                array_agg(pc.product_id ORDER BY pc.product_id) AS product_ids
            FROM clauses c
            JOIN product_clauses pc ON pc.clause_id = c.id
            WHERE c.clause_type = 'main_clause'
            GROUP BY c.id, c.file_name, c.file_path, c.clause_type, c.updated_at
            ORDER BY c.id
            """
        )
    finally:
        await connection.close()

    requests: list[KnowledgeBuildRequest] = []
    for row in rows:
        source_path = resolve_source_path(row["file_path"])
        requests.append(
            KnowledgeBuildRequest(
                product_ids=list(row["product_ids"]),
                clause_id=row["clause_id"],
                document_type=_document_type(row["clause_type"]),
                document_name=row["file_name"],
                source_uri=str(source_path),
                version=row["updated_at"].isoformat(),
                content=source_path.read_bytes(),
                metadata={
                    "clause_type": row["clause_type"],
                    "database_file_path": row["file_path"],
                },
            )
        )
    return requests


def resolve_source_path(database_path: str) -> Path:
    """将业务库中的 /data/... 相对路径安全转换为项目内绝对路径。"""

    relative_path = Path(database_path.lstrip("/\\"))
    source_path = (PROJECT_ROOT / relative_path).resolve()
    try:
        source_path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"clause path escapes project root: {database_path}") from exc
    if not source_path.is_file():
        raise FileNotFoundError(f"clause file does not exist: {source_path}")
    return source_path


async def reset_rag_stores(settings: Settings, pool: Any) -> None:
    """删除 RAG 专属表和 Collection，不影响会话及 LangGraph 数据。"""

    async with pool.acquire() as connection:
        await connection.execute(
            """
            DROP TABLE IF EXISTS knowledge_ingestion_jobs;
            DROP TABLE IF EXISTS knowledge_child_chunks;
            DROP TABLE IF EXISTS knowledge_parent_chunks;
            DROP TABLE IF EXISTS knowledge_documents;
            """
        )

    if not settings.milvus_uri:
        raise RuntimeError("MILVUS_URI or MILVUS_HOST is not configured")
    from pymilvus import MilvusClient

    client = MilvusClient(uri=settings.milvus_uri)
    if client.has_collection(settings.milvus_collection):
        client.drop_collection(settings.milvus_collection)


def _document_type(clause_type: str) -> DocumentType:
    if clause_type == "product_brochure":
        return DocumentType.PRODUCT_MANUAL
    return DocumentType.CLAUSE


def main(argv: list[str] | None = None) -> int:
    try:
        result = asyncio.run(run(parse_args(argv)))
    except Exception as exc:
        print(f"product clause ingest failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
