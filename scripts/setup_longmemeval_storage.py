"""Create or verify isolated LongMemEval memory storage without touching daily data.

Run from the project root with ``uv run python -m scripts.setup_longmemeval_storage``.
The evaluation runner must use the same two environment settings before it
imports ``app.config`` or any memory service.
"""

from __future__ import annotations

import os
from pathlib import Path

EVAL_MEMORY_ROOT = Path(__file__).resolve().parents[1] / "data/memory_eval/longmemeval"
EVAL_COLLECTION = "long_term_memory_eval"
EXPECTED_VECTOR_DIM = 1024


def main() -> None:
    # Set these before importing Settings or a service singleton. Do not edit .env.
    os.environ["MEMORY_ROOT"] = str(EVAL_MEMORY_ROOT)
    os.environ["LONG_TERM_MEMORY_COLLECTION_NAME"] = EVAL_COLLECTION

    from pymilvus import Collection, connections, utility

    from app.config import config

    if config.long_term_memory_collection_name != EVAL_COLLECTION:
        raise RuntimeError("拒绝操作非 LongMemEval 的长期记忆 Collection")
    if Path(config.memory_root).resolve() != EVAL_MEMORY_ROOT.resolve():
        raise RuntimeError("拒绝操作非 LongMemEval 的记忆目录")

    connections.connect(
        alias="default",
        host=config.milvus_host,
        port=str(config.milvus_port),
        timeout=config.milvus_timeout / 1000,
    )
    try:
        if utility.has_collection(EVAL_COLLECTION):
            collection = Collection(EVAL_COLLECTION)
            vector_fields = [field for field in collection.schema.fields if field.name == "vector"]
            user_fields = [field for field in collection.schema.fields if field.name == "user_id"]
            if len(vector_fields) != 1 or len(user_fields) != 1:
                raise RuntimeError("评测 Collection 已存在，但缺少 vector/user_id 字段；不会覆盖")
            if int(vector_fields[0].params.get("dim", 0)) != EXPECTED_VECTOR_DIM:
                raise RuntimeError("评测 Collection 向量维度不匹配；不会删除或重建")
            state = "已存在，结构检查通过"
        else:
            from app.services.long_term_memory_service import LongTermMemoryService

            service = LongTermMemoryService()
            service._create_collection()
            state = "已创建"

        EVAL_MEMORY_ROOT.mkdir(parents=True, exist_ok=True)
        print(f"评测目录：{EVAL_MEMORY_ROOT}")
        print(f"Milvus Collection：{EVAL_COLLECTION}（{state}）")
        print("日常记忆目录和 long_term_memory Collection 未修改。")
    finally:
        connections.disconnect("default")


if __name__ == "__main__":
    main()
