from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.core.database import asyncpg_database_url
from app.rag.chunking import MarkdownChunker
from app.rag.embedding import BGEM3Embedding
from app.rag.markdown_optimization import DeepSeekMarkdownOptimizer
from app.rag.models import DocumentType
from app.rag.parsing import MinerUParser
from app.rag.pipeline import KnowledgeBuildPipeline, KnowledgeBuildRequest
from app.rag.postgres_repository import PostgresKnowledgeRepository
from app.rag.vector_store import MilvusVectorStore


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest knowledge PDFs into the RAG store.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", type=Path, help="Local PDF/document file path.")
    source.add_argument("--dir", type=Path, help="Directory containing PDF files.")
    parser.add_argument("--document-type", required=True, type=_document_type)
    parser.add_argument("--version", required=True)
    parser.add_argument("--product-id", type=int, default=None)
    parser.add_argument("--clause-id", type=int, default=None)
    parser.add_argument("--document-name", default=None)
    parser.add_argument("--source-uri", default=None)
    parser.add_argument("--parse-options-json", type=_json_object, default={})
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Run parse/chunk/PostgreSQL only; skip embedding and Milvus indexing.",
    )
    args = parser.parse_args(argv)
    if args.file is not None and (not args.file.exists() or not args.file.is_file()):
        parser.error(f"--file does not exist or is not a file: {args.file}")
    if args.dir is not None and (not args.dir.exists() or not args.dir.is_dir()):
        parser.error(f"--dir does not exist or is not a directory: {args.dir}")
    if args.file is not None:
        args.file = args.file.resolve()
    if args.dir is not None:
        args.dir = args.dir.resolve()
    args.parse_options = args.parse_options_json
    return args


def build_ingest_request(args: argparse.Namespace) -> KnowledgeBuildRequest:
    return _request_for_file(args, args.file)


def build_ingest_requests(args: argparse.Namespace) -> list[KnowledgeBuildRequest]:
    if args.file is not None:
        return [build_ingest_request(args)]
    files = sorted(args.dir.glob("*.pdf"), key=lambda path: path.name)
    return [_request_for_file(args, file_path) for file_path in files]


def _request_for_file(args: argparse.Namespace, file_path: Path) -> KnowledgeBuildRequest:
    content = file_path.read_bytes()
    return KnowledgeBuildRequest(
        product_ids=[args.product_id] if args.product_id is not None else [-1],
        clause_id=args.clause_id,
        document_type=args.document_type,
        document_name=args.document_name or file_path.name,
        source_uri=args.source_uri or str(file_path),
        version=args.version,
        content=content,
        parse_options=args.parse_options,
        metadata={"ingest_source": "scripts.rag_ingest"},
    )


async def run_ingest(
    args: argparse.Namespace,
) -> str:
    settings = Settings()
    pool = await create_pool(settings)
    try:
        pipeline = await create_pipeline(settings, pool)
        items = []
        for request in build_ingest_requests(args):
            try:
                result = await pipeline.ingest_document(request)
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                items.append(
                    {
                        "document_name": request.document_name,
                        "source_uri": request.source_uri,
                        "error": str(exc),
                    }
                )
                continue
            items.append(_summary_item(result))

        return json.dumps(
            {
                "total": len(items),
                "succeeded": sum(1 for item in items if "error" not in item),
                "failed": sum(1 for item in items if "error" in item),
                "items": items,
            },
            ensure_ascii=False,
            indent=2,
        )
    finally:
        await pool.close()



async def create_pool(settings: Settings) -> Any:
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is not configured")

    import asyncpg

    return await asyncpg.create_pool(asyncpg_database_url(settings.database_url))


async def create_pipeline(
    settings: Settings,
    pool: Any,
) -> KnowledgeBuildPipeline:
    repository = PostgresKnowledgeRepository(pool)
    await repository.initialize_schema()
    parser = MinerUParser(
        token=settings.mineru_token,
        base_url=settings.mineru_api_base_url,
        flash_base_url=settings.mineru_flash_base_url,
    )
    chunker = MarkdownChunker()

    from pymilvus import MilvusClient

    if not settings.milvus_uri:
        raise RuntimeError("MILVUS_URI or MILVUS_HOST is not configured")

    milvus_client = MilvusClient(uri=settings.milvus_uri)
    embedding = BGEM3Embedding(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
    )
    vector_store = MilvusVectorStore(
        client=milvus_client,
        collection_name=settings.milvus_collection,
        dense_dim=1024,
    )
    return KnowledgeBuildPipeline(
        repository=repository,
        parser=parser,
        markdown_optimizer=DeepSeekMarkdownOptimizer(
            deepseek_api_key=settings.deepseek_api_key or "",
            deepseek_base_url=settings.deepseek_base_url,
            model=settings.markdown_optimizer_model,
            timeout=settings.markdown_optimizer_timeout_seconds,
            max_tokens=settings.markdown_optimizer_max_tokens,
        ),
        chunker=chunker,
        embedding=embedding,
        vector_store=vector_store,
    )


def _document_type(value: str) -> DocumentType:
    try:
        return DocumentType(value)
    except ValueError as exc:
        choices = ", ".join(item.value for item in DocumentType)
        message = f"invalid document type {value!r}; choose from: {choices}"
        raise argparse.ArgumentTypeError(message) from exc


def _json_object(value: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON object: {exc.msg}") from exc
    if not isinstance(decoded, dict):
        raise argparse.ArgumentTypeError("parse options must be a JSON object")
    return decoded


def _enum_value(value: Any) -> str:
    return value.value if hasattr(value, "value") else str(value)


def _summary_item(result: Any) -> dict[str, Any]:
    return {
        "document_id": str(result.document.document_id),
        "document_name": result.document.document_name,
        "status": _enum_value(result.document.status),
        "created": result.created,
        "parent_chunks": len(result.parent_chunks),
        "child_chunks": len(result.child_chunks),
    }


def main(argv: list[str] | None = None) -> int:
    try:
        print(asyncio.run(run_ingest(parse_args(argv))))
    except Exception as exc:
        print(f"rag ingest failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
