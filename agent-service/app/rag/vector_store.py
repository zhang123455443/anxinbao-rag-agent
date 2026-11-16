from __future__ import annotations

import json
from typing import Any

from pymilvus import DataType

from app.rag.models import ChunkLevel, KnowledgeChunk


class MilvusVectorStore:
    """负责初始化 Milvus Collection，并将向量化后的子块写入其中。"""

    def __init__(
        self,
        *,
        client: Any,
        collection_name: str = "insurance_knowledge_chunks",
        dense_dim: int = 1024,
    ) -> None:
        # dense_dim 必须与 Embedding 模型输出维度一致，BGE-M3 默认为 1024。
        self.client = client
        self.collection_name = collection_name
        self.dense_dim = dense_dim

    def store(
        self,
        document_name: str,
        chunks: list[KnowledgeChunk],
        vectors: dict[str, Any],
        product_ids: list[int],
    ) -> None:
        """将子块及其稠密、稀疏向量批量写入 Milvus。"""

        self._initialize_collection()
        # 稀疏向量切片保留二维 batch 结构，符合 pymilvus 的字段格式。
        entities = [
            self._to_entity(
                document_name,
                chunk,
                dense_vector=vectors["dense"][index],
                sparse_vector=vectors["sparse"][index : index + 1],
                product_ids=product_ids,
            )
            for index, chunk in enumerate(chunks)
        ]
        self.client.insert(collection_name=self.collection_name, data=entities)

    def _initialize_collection(self) -> None:
        """首次写入时创建 Collection 及两类向量索引。"""

        if self.client.has_collection(self.collection_name):
            return

        # 关闭动态字段，使 Milvus 中保存的数据结构明确且便于教学。
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        _add_varchar(schema, "child_chunk_id", is_primary=True, max_length=64)
        _add_varchar(schema, "parent_chunk_id", max_length=64)
        _add_varchar(schema, "document_id", max_length=64)
        _add_varchar(schema, "document_name", max_length=512)
        schema.add_field(field_name="clause_id", datatype=DataType.INT64)
        schema.add_field(
            field_name="product_ids",
            datatype=DataType.ARRAY,
            element_type=DataType.INT64,
            max_capacity=64,
        )
        _add_varchar(schema, "section_path", max_length=1024)
        _add_varchar(schema, "section_type", max_length=80)
        _add_varchar(schema, "clause_no", max_length=80)
        _add_varchar(schema, "content", max_length=8192)
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.dense_dim,
        )
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 稠密向量和稀疏向量分别建立索引，检索时再融合两路结果。
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_type="AUTOINDEX",
            metric_type="IP",
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )
        index_params.add_index(field_name="product_ids", index_type="AUTOINDEX")
        index_params.add_index(field_name="section_type", index_type="AUTOINDEX")
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def _to_entity(
        self,
        document_name: str,
        chunk: KnowledgeChunk,
        *,
        dense_vector: Any,
        sparse_vector: Any,
        product_ids: list[int],
    ) -> dict[str, Any]:
        """将一个子块转换为与 Milvus Schema 对应的字典。"""

        if chunk.chunk_level != ChunkLevel.CHILD or chunk.parent_chunk_id is None:
            raise ValueError("only child chunks can be stored in Milvus")

        return {
            "child_chunk_id": str(chunk.chunk_id),
            "parent_chunk_id": str(chunk.parent_chunk_id),
            "document_id": str(chunk.document_id),
            "document_name": document_name,
            "clause_id": _nullable_int(chunk.clause_id),
            "product_ids": product_ids,
            "section_path": json.dumps(chunk.section_path, ensure_ascii=False),
            "section_type": chunk.section_type,
            "clause_no": chunk.clause_no or "",
            "content": chunk.content,
            "dense_vector": dense_vector,
            "sparse_vector": sparse_vector,
        }

    def delete_by_clause_id(self, clause_id: int) -> None:
        """在写入新索引前删除同一业务条款的旧向量。"""

        if not self.client.has_collection(self.collection_name):
            return
        self.client.delete(
            collection_name=self.collection_name,
            filter=f"clause_id == {clause_id}",
        )


def _add_varchar(
    schema: Any,
    field_name: str,
    *,
    max_length: int,
    is_primary: bool = False,
) -> None:
    """统一添加 VARCHAR 字段，减少重复的 Schema 配置代码。"""

    schema.add_field(
        field_name=field_name,
        datatype=DataType.VARCHAR,
        max_length=max_length,
        is_primary=is_primary,
    )


def _nullable_int(value: int | None) -> int:
    """Milvus INT64 不保存 None，使用 -1 表示未关联业务条款。"""

    return value if value is not None else -1
