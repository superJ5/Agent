"""Standalone E2E test for LangChain baseline chunks.

Uses a SEPARATE Milvus collection ('biz_baseline') so the main 'biz'
collection is never touched.

Flow:  Index baseline chunks → Embed query → Vector search → LLM answer

Usage:
    # Step 1: Index baseline chunks into the separate collection
    python scripts/baseline_e2e_test.py --index

    # Step 2: Run the 60-question E2E test
    python scripts/baseline_e2e_test.py --test

    # Step 3: Clean up the baseline collection when done
    python scripts/baseline_e2e_test.py --drop

    # Run a single ad-hoc query
    python scripts/baseline_e2e_test.py --query "键盘USB-C接口在哪"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from app.config import config

# ── Constants ──────────────────────────────────────────────────────────────
BASELINE_COLLECTION = "biz_baseline"
BASELINE_DIR = Path("./data/manuals/langchain_baseline")
VECTOR_DIM = 1024
ID_MAX_LENGTH = 100
CONTENT_MAX_LENGTH = 8000
EMBEDDING_BATCH = 10

# ── Test cases (same 60 questions from eval_baseline_recall.py) ───────────
TEST_CASES = [
    {"query": "VR头显怎么连接电脑", "must_contain": ["连接"], "manual": "VR头显"},
    {"query": "VR头显瞳距怎么调节", "must_contain": ["瞳距"], "manual": "VR头显"},
    {"query": "VR头显镜片怎么清洁", "must_contain": ["镜片", "清洁"], "manual": "VR头显"},
    {"query": "VR头显可以戴眼镜使用吗", "must_contain": ["眼镜"], "manual": "VR头显"},
    {"query": "VR头显包装清单有什么", "must_contain": ["包装"], "manual": "VR头显"},
    {"query": "工学椅座椅高度怎么调节", "must_contain": ["高度", "调节"], "manual": "人体工学椅"},
    {"query": "工学椅腰部支撑怎么调整", "must_contain": ["腰", "支撑"], "manual": "人体工学椅"},
    {"query": "工学椅扶手怎么调整", "must_contain": ["扶手"], "manual": "人体工学椅"},
    {"query": "工学椅最大承重多少", "must_contain": ["承重"], "manual": "人体工学椅"},
    {"query": "工学椅头枕怎么安装", "must_contain": ["头枕"], "manual": "人体工学椅"},
    {"query": "健身单车座垫高度怎么调", "must_contain": ["座", "高度"], "manual": "健身单车"},
    {"query": "健身单车阻力怎么调节", "must_contain": ["阻力"], "manual": "健身单车"},
    {"query": "健身单车心率怎么测量", "must_contain": ["心率"], "manual": "健身单车"},
    {"query": "健身单车显示屏显示什么参数", "must_contain": ["显示"], "manual": "健身单车"},
    {"query": "健身单车踏板怎么安装", "must_contain": ["踏板", "安装"], "manual": "健身单车"},
    {"query": "健身追踪器怎么和手机配对", "must_contain": ["配对"], "manual": "健身追踪"},
    {"query": "健身追踪器怎么充电", "must_contain": ["充电"], "manual": "健身追踪"},
    {"query": "健身追踪器防水等级是多少", "must_contain": ["防水"], "manual": "健身追踪"},
    {"query": "健身追踪器怎么设置闹钟", "must_contain": ["闹钟"], "manual": "健身追踪"},
    {"query": "健身追踪器步数怎么查看", "must_contain": ["步数"], "manual": "健身追踪"},
    {"query": "儿童摩托车怎么充电", "must_contain": ["充电"], "manual": "儿童电动摩托车"},
    {"query": "儿童摩托车最大承重多少", "must_contain": ["承重"], "manual": "儿童电动摩托车"},
    {"query": "儿童摩托车适合几岁小孩", "must_contain": ["年龄"], "manual": "儿童电动摩托车"},
    {"query": "儿童摩托车速度怎么调节", "must_contain": ["速度"], "manual": "儿童电动摩托车"},
    {"query": "儿童摩托车电池多久充满", "must_contain": ["电池"], "manual": "儿童电动摩托车"},
    {"query": "冰箱温度怎么调节", "must_contain": ["温度"], "manual": "冰箱"},
    {"query": "冰箱冷冻室结霜怎么处理", "must_contain": ["结霜"], "manual": "冰箱"},
    {"query": "冰箱门关不紧怎么办", "must_contain": ["门"], "manual": "冰箱"},
    {"query": "冰箱不运行怎么排查", "must_contain": ["不运行"], "manual": "冰箱"},
    {"query": "冰箱滤水器怎么更换", "must_contain": ["滤水器"], "manual": "冰箱"},
    {"query": "键盘USB-C接口在哪个位置", "must_contain": ["USB-C", "接口"], "manual": "功能键盘"},
    {"query": "键盘怎么更换轴体", "must_contain": ["轴体"], "manual": "功能键盘"},
    {"query": "键盘保修期限是多久", "must_contain": ["保修"], "manual": "功能键盘"},
    {"query": "键盘RGB灯光怎么自定义", "must_contain": ["RGB", "灯光"], "manual": "功能键盘"},
    {"query": "键盘CAM软件怎么下载安装", "must_contain": ["CAM"], "manual": "功能键盘"},
    {"query": "吹风机冷机启动步骤", "must_contain": ["冷机", "启动"], "manual": "吹风机"},
    {"query": "吹风机燃油混合比例是多少", "must_contain": ["混合", "比例"], "manual": "吹风机"},
    {"query": "吹风机空气滤清器怎么清洁", "must_contain": ["滤清器", "清洁"], "manual": "吹风机"},
    {"query": "吹风机火花塞怎么更换", "must_contain": ["火花塞"], "manual": "吹风机"},
    {"query": "吹风机个人防护装备有哪些要求", "must_contain": ["防护", "装备"], "manual": "吹风机"},
    {"query": "烤箱预热温度怎么设置", "must_contain": ["温度"], "manual": "烤箱"},
    {"query": "烤箱定时器怎么使用", "must_contain": ["定时"], "manual": "烤箱"},
    {"query": "烤箱内部怎么清洁", "must_contain": ["清洁"], "manual": "烤箱"},
    {"query": "烤箱使用时要注意什么安全事项", "must_contain": ["安全"], "manual": "烤箱"},
    {"query": "烤箱烤盘怎么取出", "must_contain": ["烤盘"], "manual": "烤箱"},
    {"query": "电钻怎么更换钻头", "must_contain": ["钻头"], "manual": "电钻"},
    {"query": "电钻电池怎么充电", "must_contain": ["电池", "充电"], "manual": "电钻"},
    {"query": "电钻变速拨杆怎么使用", "must_contain": ["变速"], "manual": "电钻"},
    {"query": "电钻正反转怎么切换", "must_contain": ["正反转"], "manual": "电钻"},
    {"query": "电钻安全操作注意事项", "must_contain": ["安全"], "manual": "电钻"},
    {"query": "空调遥控器怎么使用", "must_contain": ["遥控器"], "manual": "空调"},
    {"query": "空调怎么调节风速", "must_contain": ["风速"], "manual": "空调"},
    {"query": "空调定时功能怎么设置", "must_contain": ["定时"], "manual": "空调"},
    {"query": "空调滤网怎么清洁", "must_contain": ["滤网", "清洁"], "manual": "空调"},
    {"query": "空调自清洁功能怎么使用", "must_contain": ["自清洁"], "manual": "空调"},
    {"query": "蓝牙鼠标怎么安装驱动程序", "must_contain": ["驱动程序", "安装"], "manual": "蓝牙激光鼠标"},
    {"query": "蓝牙鼠标电池怎么更换", "must_contain": ["电池"], "manual": "蓝牙激光鼠标"},
    {"query": "蓝牙鼠标图标颜色代表什么", "must_contain": ["图标"], "manual": "蓝牙激光鼠标"},
    {"query": "蓝牙鼠标连接不上怎么排查", "must_contain": ["故障"], "manual": "蓝牙激光鼠标"},
    {"query": "蓝牙鼠标FCC声明的内容是什么", "must_contain": ["FCC"], "manual": "蓝牙激光鼠标"},
]


# ══════════════════════════════════════════════════════════════════════════
# Milvus helpers (isolated from main app singletons)
# ══════════════════════════════════════════════════════════════════════════

def _connect_milvus():
    """Ensure a Milvus connection exists (reuses 'default' alias)."""
    if not connections.has_connection("default"):
        connections.connect(
            alias="default",
            host=config.milvus_host,
            port=str(config.milvus_port),
            timeout=config.milvus_timeout / 1000,
        )
    logger.info("Milvus connected: {}:{}", config.milvus_host, config.milvus_port)


def _ensure_collection() -> Collection:
    """Create the baseline collection if it doesn't exist, then load it."""
    _connect_milvus()
    if utility.has_collection(BASELINE_COLLECTION):
        col = Collection(BASELINE_COLLECTION)
        col.load()
        return col

    schema = CollectionSchema(
        fields=[
            FieldSchema("id", DataType.VARCHAR, max_length=ID_MAX_LENGTH, is_primary=True),
            FieldSchema("vector", DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
            FieldSchema("content", DataType.VARCHAR, max_length=CONTENT_MAX_LENGTH),
            FieldSchema("metadata", DataType.JSON),
        ],
        description="LangChain baseline chunks (isolated from main biz collection)",
    )
    col = Collection(BASELINE_COLLECTION, schema, num_shards=2)
    col.create_index("vector", {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}})
    col.load()
    logger.info("Created and loaded collection '{}'", BASELINE_COLLECTION)
    return col


def _drop_collection():
    _connect_milvus()
    if utility.has_collection(BASELINE_COLLECTION):
        utility.drop_collection(BASELINE_COLLECTION)
        logger.info("Dropped collection '{}'", BASELINE_COLLECTION)
    else:
        logger.info("Collection '{}' does not exist, nothing to drop", BASELINE_COLLECTION)


# ══════════════════════════════════════════════════════════════════════════
# Indexing
# ══════════════════════════════════════════════════════════════════════════

def _load_baseline_chunks() -> list[dict]:
    """Load all text chunks from the baseline directory."""
    chunks = []
    for jf in sorted(BASELINE_DIR.rglob("chunks.jsonl")):
        for line in jf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("chunk_type") == "metadata_image_path":
                continue
            chunks.append(row)
    return chunks


def _truncate_utf8(text: str, max_bytes: int = CONTENT_MAX_LENGTH) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    budget = max_bytes - 3
    truncated = encoded[:budget]
    while truncated:
        try:
            return truncated.decode("utf-8") + "..."
        except UnicodeDecodeError:
            truncated = truncated[:-1]
    return "..."


def index_baseline():
    """Index all baseline chunks into the isolated Milvus collection."""
    from app.services.vector_embedding_service import vector_embedding_service

    col = _ensure_collection()
    chunks = _load_baseline_chunks()
    if not chunks:
        print("No baseline chunks found. Run build_langchain_baseline.py first.")
        return

    print(f"Indexing {len(chunks)} baseline chunks into '{BASELINE_COLLECTION}'...")

    rows = []
    for batch_start in range(0, len(chunks), EMBEDDING_BATCH):
        batch = chunks[batch_start: batch_start + EMBEDDING_BATCH]
        texts = [c.get("index_text") or c.get("text", "") for c in batch]
        embeddings = vector_embedding_service.embed_documents(texts)

        for chunk, emb in zip(batch, embeddings):
            chunk_id = chunk.get("chunk_id", "")
            if not chunk_id or len(chunk_id) > ID_MAX_LENGTH:
                import hashlib
                chunk_id = "bl_" + hashlib.sha1(
                    (chunk.get("text", ""))[:200].encode()
                ).hexdigest()[:20]

            rows.append({
                "id": chunk_id,
                "vector": emb,
                "content": _truncate_utf8(chunk.get("text", "")),
                "metadata": {
                    "doc_id": chunk.get("doc_id", ""),
                    "doc_name": chunk.get("doc_name", ""),
                    "title": chunk.get("title", ""),
                    "chunk_id": chunk_id,
                    "text": _truncate_utf8(chunk.get("text", "")),
                },
            })
        print(f"  Embedded {min(batch_start + EMBEDDING_BATCH, len(chunks))}/{len(chunks)}")

    col.insert(rows)
    col.flush()
    print(f"Done! Indexed {len(rows)} chunks into '{BASELINE_COLLECTION}'.")


# ══════════════════════════════════════════════════════════════════════════
# Search
# ══════════════════════════════════════════════════════════════════════════

def search_baseline(query: str, top_k: int = 5) -> list[dict]:
    """Search the baseline collection and return top-k results."""
    from app.services.vector_embedding_service import vector_embedding_service

    col = _ensure_collection()
    query_vec = vector_embedding_service.embed_query(query)

    results = col.search(
        data=[query_vec],
        anns_field="vector",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["id", "content", "metadata"],
    )

    hits = []
    for hit_list in results:
        for hit in hit_list:
            meta = hit.entity.get("metadata", {}) or {}
            hits.append({
                "id": hit.entity.get("id", ""),
                "score": hit.distance,
                "title": meta.get("title", ""),
                "text": meta.get("text", hit.entity.get("content", "")),
                "doc_name": meta.get("doc_name", ""),
            })
    return hits


# ══════════════════════════════════════════════════════════════════════════
# LLM answer generation
# ══════════════════════════════════════════════════════════════════════════

def _llm_answer(query: str, context_chunks: list[dict]) -> str:
    """Call DashScope Qwen to generate an answer from retrieved context."""
    from openai import OpenAI

    client = OpenAI(
        api_key=config.dashscope_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    context = "\n\n---\n\n".join(
        f"[来源: {c['title']}]\n{c['text'][:600]}" for c in context_chunks
    )

    messages = [
        {"role": "system", "content": (
            "你是产品说明书助手。根据下方检索到的手册片段回答问题。"
            "只基于给定内容作答，如果信息不足请明确说明。回答简洁。"
        )},
        {"role": "user", "content": f"检索到的手册片段：\n{context}\n\n用户问题：{query}"},
    ]

    resp = client.chat.completions.create(
        model=config.rag_model,
        messages=messages,
        temperature=0.3,
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()


# ══════════════════════════════════════════════════════════════════════════
# E2E test runner
# ══════════════════════════════════════════════════════════════════════════

def _check_hit(text: str, must_contain: list[str]) -> bool:
    t = text.lower()
    return all(kw.lower() in t for kw in must_contain)


def run_e2e_test(use_llm: bool = True, top_k: int = 5):
    """Run 60-question E2E test against the baseline collection."""
    total = len(TEST_CASES)
    recall_hits = {1: 0, 3: 0, 5: 0}
    results = []

    print(f"\n{'='*80}")
    print(f"  Baseline E2E Test — {total} questions, top_k={top_k}")
    print(f"{'='*80}\n")

    for i, tc in enumerate(TEST_CASES, 1):
        query = tc["query"]
        must = tc["must_contain"]
        print(f"[{i:2d}/{total}] {query}")

        hits = search_baseline(query, top_k=top_k)

        # Check recall at different K
        for k in [1, 3, 5]:
            found = any(_check_hit(h["text"], must) for h in hits[:k])
            if found:
                recall_hits[k] += 1

        # Top hit info
        top = hits[0] if hits else {}
        hit_at_1 = _check_hit(top.get("text", ""), must) if top else False
        print(f"       Top1: {'✅' if hit_at_1 else '❌'} [{top.get('score', 0):.4f}] {top.get('title', '')[:40]}")

        # LLM answer (optional)
        answer = ""
        if use_llm and hits:
            try:
                answer = _llm_answer(query, hits[:3])
                print(f"       LLM:  {answer[:100]}...")
            except Exception as e:
                answer = f"[ERROR] {e}"
                print(f"       LLM:  {answer[:100]}")

        results.append({
            "query": query,
            "manual": tc["manual"],
            "must_contain": must,
            "hit_at_1": hit_at_1,
            "top_title": top.get("title", ""),
            "top_score": top.get("score", 0),
            "answer": answer[:500],
        })
        print()

    # Summary
    print(f"\n{'='*80}")
    print(f"  RECALL SUMMARY (Baseline E2E — real embeddings)")
    print(f"{'='*80}")
    for k in [1, 3, 5]:
        pct = recall_hits[k] / total * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  Recall@{k}: {recall_hits[k]:>2d}/{total}  ({pct:5.1f}%)  {bar}")
    print(f"{'='*80}\n")

    # Save results
    out = Path("data/manuals/langchain_baseline/e2e_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Results saved to {out}")


def run_single_query(query: str):
    """Run a single query through the baseline pipeline."""
    print(f"\n[QUERY] {query}")
    hits = search_baseline(query, top_k=5)

    print(f"\n[RETRIEVED {len(hits)} chunks]")
    for i, h in enumerate(hits, 1):
        print(f"  {i}. [{h['score']:.4f}] {h['title'][:40]}  |  {h['text'][:80]}...")

    if hits:
        print(f"\n[LLM ANSWER]")
        answer = _llm_answer(query, hits[:3])
        print(answer)
    print()


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Standalone E2E test for LangChain baseline (uses separate Milvus collection)"
    )
    parser.add_argument("--index", action="store_true", help="Index baseline chunks into biz_baseline collection")
    parser.add_argument("--test", action="store_true", help="Run the 60-question E2E test")
    parser.add_argument("--test-no-llm", action="store_true", help="Run E2E test without LLM (retrieval only)")
    parser.add_argument("--query", type=str, help="Run a single ad-hoc query")
    parser.add_argument("--drop", action="store_true", help="Drop the biz_baseline collection")
    args = parser.parse_args()

    if not any([args.index, args.test, args.test_no_llm, args.query, args.drop]):
        parser.print_help()
        return

    if args.drop:
        _drop_collection()
    if args.index:
        index_baseline()
    if args.test:
        run_e2e_test(use_llm=True)
    if args.test_no_llm:
        run_e2e_test(use_llm=False)
    if args.query:
        run_single_query(args.query)


if __name__ == "__main__":
    main()
