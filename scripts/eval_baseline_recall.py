"""Offline recall evaluation for LangChain baseline chunks.

Uses TF-IDF cosine similarity as a proxy for vector search.
No Milvus or main app dependencies needed — fully standalone.

Usage:
    # Run with default test cases (5 per manual, 60 total)
    python scripts/eval_baseline_recall.py

    # Run with custom test cases from a JSON file
    python scripts/eval_baseline_recall.py --test-cases my_tests.json

    # Evaluate a different set of chunks (e.g. structured chunks)
    python scripts/eval_baseline_recall.py --chunks-dir ./data/manuals/chunks
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Test cases: 5 per manual, 60 total
# Each test has: query, must_contain (keywords in a correct hit), manual filter
# ---------------------------------------------------------------------------

TEST_CASES: list[dict[str, Any]] = [
    # ── VR头显手册 (6 chunks) ──
    {"query": "VR头显怎么连接电脑", "must_contain": ["连接"], "manual": "VR头显"},
    {"query": "VR头显瞳距怎么调节", "must_contain": ["瞳距"], "manual": "VR头显"},
    {"query": "VR头显镜片怎么清洁", "must_contain": ["镜片", "清洁"], "manual": "VR头显"},
    {"query": "VR头显可以戴眼镜使用吗", "must_contain": ["眼镜"], "manual": "VR头显"},
    {"query": "VR头显包装清单有什么", "must_contain": ["包装"], "manual": "VR头显"},

    # ── 人体工学椅手册 (4 chunks) ──
    {"query": "工学椅座椅高度怎么调节", "must_contain": ["高度", "调节"], "manual": "人体工学椅"},
    {"query": "工学椅腰部支撑怎么调整", "must_contain": ["腰", "支撑"], "manual": "人体工学椅"},
    {"query": "工学椅扶手怎么调整", "must_contain": ["扶手"], "manual": "人体工学椅"},
    {"query": "工学椅最大承重多少", "must_contain": ["承重"], "manual": "人体工学椅"},
    {"query": "工学椅头枕怎么安装", "must_contain": ["头枕"], "manual": "人体工学椅"},

    # ── 健身单车手册 (25 chunks) ──
    {"query": "健身单车座垫高度怎么调", "must_contain": ["座", "高度"], "manual": "健身单车"},
    {"query": "健身单车阻力怎么调节", "must_contain": ["阻力"], "manual": "健身单车"},
    {"query": "健身单车心率怎么测量", "must_contain": ["心率"], "manual": "健身单车"},
    {"query": "健身单车显示屏显示什么参数", "must_contain": ["显示"], "manual": "健身单车"},
    {"query": "健身单车踏板怎么安装", "must_contain": ["踏板", "安装"], "manual": "健身单车"},

    # ── 健身追踪手册 (30 chunks) ──
    {"query": "健身追踪器怎么和手机配对", "must_contain": ["配对"], "manual": "健身追踪"},
    {"query": "健身追踪器怎么充电", "must_contain": ["充电"], "manual": "健身追踪"},
    {"query": "健身追踪器防水等级是多少", "must_contain": ["防水"], "manual": "健身追踪"},
    {"query": "健身追踪器怎么设置闹钟", "must_contain": ["闹钟"], "manual": "健身追踪"},
    {"query": "健身追踪器步数怎么查看", "must_contain": ["步数"], "manual": "健身追踪"},

    # ── 儿童电动摩托车手册 (12 chunks) ──
    {"query": "儿童摩托车怎么充电", "must_contain": ["充电"], "manual": "儿童电动摩托车"},
    {"query": "儿童摩托车最大承重多少", "must_contain": ["承重"], "manual": "儿童电动摩托车"},
    {"query": "儿童摩托车适合几岁小孩", "must_contain": ["年龄"], "manual": "儿童电动摩托车"},
    {"query": "儿童摩托车速度怎么调节", "must_contain": ["速度"], "manual": "儿童电动摩托车"},
    {"query": "儿童摩托车电池多久充满", "must_contain": ["电池"], "manual": "儿童电动摩托车"},

    # ── 冰箱手册 (10 chunks) ──
    {"query": "冰箱温度怎么调节", "must_contain": ["温度"], "manual": "冰箱"},
    {"query": "冰箱冷冻室结霜怎么处理", "must_contain": ["结霜"], "manual": "冰箱"},
    {"query": "冰箱门关不紧怎么办", "must_contain": ["门"], "manual": "冰箱"},
    {"query": "冰箱不运行怎么排查", "must_contain": ["不运行"], "manual": "冰箱"},
    {"query": "冰箱滤水器怎么更换", "must_contain": ["滤水器"], "manual": "冰箱"},

    # ── 功能键盘手册 (8 chunks) ──
    {"query": "键盘USB-C接口在哪个位置", "must_contain": ["USB-C", "接口"], "manual": "功能键盘"},
    {"query": "键盘怎么更换轴体", "must_contain": ["轴体"], "manual": "功能键盘"},
    {"query": "键盘保修期限是多久", "must_contain": ["保修"], "manual": "功能键盘"},
    {"query": "键盘RGB灯光怎么自定义", "must_contain": ["RGB", "灯光"], "manual": "功能键盘"},
    {"query": "键盘CAM软件怎么下载安装", "must_contain": ["CAM"], "manual": "功能键盘"},

    # ── 吹风机手册 (20 chunks) ──
    {"query": "吹风机冷机启动步骤", "must_contain": ["冷机", "启动"], "manual": "吹风机"},
    {"query": "吹风机燃油混合比例是多少", "must_contain": ["混合", "比例"], "manual": "吹风机"},
    {"query": "吹风机空气滤清器怎么清洁", "must_contain": ["滤清器", "清洁"], "manual": "吹风机"},
    {"query": "吹风机火花塞怎么更换", "must_contain": ["火花塞"], "manual": "吹风机"},
    {"query": "吹风机个人防护装备有哪些要求", "must_contain": ["防护", "装备"], "manual": "吹风机"},

    # ── 烤箱手册 (7 chunks) ──
    {"query": "烤箱预热温度怎么设置", "must_contain": ["温度"], "manual": "烤箱"},
    {"query": "烤箱定时器怎么使用", "must_contain": ["定时"], "manual": "烤箱"},
    {"query": "烤箱内部怎么清洁", "must_contain": ["清洁"], "manual": "烤箱"},
    {"query": "烤箱使用时要注意什么安全事项", "must_contain": ["安全"], "manual": "烤箱"},
    {"query": "烤箱烤盘怎么取出", "must_contain": ["烤盘"], "manual": "烤箱"},

    # ── 电钻手册 (17 chunks) ──
    {"query": "电钻怎么更换钻头", "must_contain": ["钻头"], "manual": "电钻"},
    {"query": "电钻电池怎么充电", "must_contain": ["电池", "充电"], "manual": "电钻"},
    {"query": "电钻变速拨杆怎么使用", "must_contain": ["变速"], "manual": "电钻"},
    {"query": "电钻正反转怎么切换", "must_contain": ["正反转"], "manual": "电钻"},
    {"query": "电钻安全操作注意事项", "must_contain": ["安全"], "manual": "电钻"},

    # ── 空调手册 (11 chunks) ──
    {"query": "空调遥控器怎么使用", "must_contain": ["遥控器"], "manual": "空调"},
    {"query": "空调怎么调节风速", "must_contain": ["风速"], "manual": "空调"},
    {"query": "空调定时功能怎么设置", "must_contain": ["定时"], "manual": "空调"},
    {"query": "空调滤网怎么清洁", "must_contain": ["滤网", "清洁"], "manual": "空调"},
    {"query": "空调自清洁功能怎么使用", "must_contain": ["自清洁"], "manual": "空调"},

    # ── 蓝牙激光鼠标手册 (8 chunks) ──
    {"query": "蓝牙鼠标怎么安装驱动程序", "must_contain": ["驱动程序", "安装"], "manual": "蓝牙激光鼠标"},
    {"query": "蓝牙鼠标电池怎么更换", "must_contain": ["电池"], "manual": "蓝牙激光鼠标"},
    {"query": "蓝牙鼠标图标颜色代表什么", "must_contain": ["图标"], "manual": "蓝牙激光鼠标"},
    {"query": "蓝牙鼠标连接不上怎么排查", "must_contain": ["故障"], "manual": "蓝牙激光鼠标"},
    {"query": "蓝牙鼠标FCC声明的内容是什么", "must_contain": ["FCC"], "manual": "蓝牙激光鼠标"},
]

BASELINE_DIR = Path("./data/manuals/langchain_baseline")
K_VALUES = [1, 3, 5]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline recall evaluation for baseline chunks. "
            "Uses TF-IDF cosine similarity as proxy for vector search."
        ),
    )
    parser.add_argument(
        "--chunks-dir",
        default=str(BASELINE_DIR),
        help=(
            "Directory containing chunks.jsonl files. "
            "Can be langchain_baseline or structured chunks."
        ),
    )
    parser.add_argument(
        "--test-cases",
        default=None,
        help="Path to a JSON file with custom test cases.",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=K_VALUES,
        help="K values for recall@k (default: 1 3 5).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write detailed JSON results.",
    )
    return parser


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    doc_name: str
    title: str
    text: str
    index_text: str
    chunk_type: str

    @property
    def search_text(self) -> str:
        return f"{self.title} {self.index_text} {self.text}"


def load_all_chunks(base_dir: Path) -> list[ChunkRecord]:
    """Load all text chunks from all subdirectories or directly from .jsonl files."""
    chunks: list[ChunkRecord] = []
    jsonl_files = list(base_dir.rglob("chunks.jsonl"))
    if not jsonl_files:
        # Fallback: scan for any .jsonl files directly in the directory
        jsonl_files = list(base_dir.glob("*.jsonl"))

    for jsonl_file in sorted(jsonl_files):
        for line in jsonl_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("chunk_type") == "metadata_image_path":
                continue  # Skip image metadata
            chunks.append(ChunkRecord(
                chunk_id=row.get("chunk_id", ""),
                doc_id=row.get("doc_id", ""),
                doc_name=row.get("doc_name", ""),
                title=row.get("title", ""),
                text=row.get("text", ""),
                index_text=row.get("index_text", ""),
                chunk_type=row.get("chunk_type", ""),
            ))
    return chunks


# ---------------------------------------------------------------------------
# TF-IDF retrieval
# ---------------------------------------------------------------------------

def build_tfidf_index(chunks: list[ChunkRecord]):
    """Build TF-IDF matrix over all chunk texts."""
    corpus = [c.search_text for c in chunks]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",  # Character n-grams work better for Chinese
        ngram_range=(2, 4),
        max_features=50000,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix


def retrieve_top_k(
    query: str,
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    chunks: list[ChunkRecord],
    k: int = 5,
    manual_filter: str | None = None,
) -> list[tuple[float, ChunkRecord]]:
    """Retrieve top-k chunks for a query using TF-IDF cosine similarity."""
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Apply manual filter if specified
    if manual_filter:
        for i, chunk in enumerate(chunks):
            if manual_filter not in chunk.doc_name and manual_filter not in chunk.doc_id:
                scores[i] = -1.0

    top_indices = np.argsort(scores)[::-1][:k]
    return [(float(scores[i]), chunks[i]) for i in top_indices if scores[i] > 0]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    query: str
    manual: str
    must_contain: list[str]
    hits_at_k: dict[int, bool] = field(default_factory=dict)
    top_hit_title: str = ""
    top_hit_score: float = 0.0
    top_hit_preview: str = ""
    top_hit_chunk_id: str = ""


def check_hit(chunk: ChunkRecord, must_contain: list[str]) -> bool:
    """Check if a chunk contains ALL required keywords."""
    text = f"{chunk.title} {chunk.text}".lower()
    return all(keyword.lower() in text for keyword in must_contain)


def run_evaluation(
    test_cases: list[dict],
    chunks: list[ChunkRecord],
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    k_values: list[int],
) -> list[EvalResult]:
    """Run recall evaluation for all test cases."""
    results: list[EvalResult] = []
    max_k = max(k_values)

    for tc in test_cases:
        query = tc["query"]
        must_contain = tc["must_contain"]
        manual = tc.get("manual", "")

        top_results = retrieve_top_k(
            query, vectorizer, tfidf_matrix, chunks,
            k=max_k, manual_filter=manual if manual else None,
        )

        result = EvalResult(
            query=query,
            manual=manual,
            must_contain=must_contain,
        )

        if top_results:
            result.top_hit_score = top_results[0][0]
            result.top_hit_title = top_results[0][1].title[:50]
            result.top_hit_chunk_id = top_results[0][1].chunk_id
            result.top_hit_preview = top_results[0][1].text[:80].replace("\n", " ")

        for k in k_values:
            found = any(
                check_hit(chunk, must_contain)
                for _, chunk in top_results[:k]
            )
            result.hits_at_k[k] = found

        results.append(result)

    return results


def print_results(results: list[EvalResult], k_values: list[int], label: str) -> dict[str, Any]:
    """Print evaluation results and return summary dict."""
    total = len(results)

    print(f"\n{'='*90}")
    print(f"  {label} — Recall Evaluation ({total} queries)")
    print(f"{'='*90}\n")

    # Per-query results
    for i, r in enumerate(results, 1):
        hits_str = "  ".join(
            f"@{k}={'✅' if r.hits_at_k.get(k) else '❌'}" for k in k_values
        )
        status = "✅" if r.hits_at_k.get(k_values[0]) else "❌"
        print(f"  {i:2d}. [{r.manual:12s}] {status} {r.query}")
        print(f"      keywords: {r.must_contain}  |  {hits_str}")
        print(f"      top hit: [{r.top_hit_score:.3f}] {r.top_hit_chunk_id} — {r.top_hit_title}")
        print()

    # Aggregate metrics
    summary: dict[str, Any] = {"label": label}
    print(f"{'='*90}")
    print(f"  RECALL SUMMARY — {label}")
    print(f"{'='*90}")
    for k in k_values:
        hits = sum(1 for r in results if r.hits_at_k.get(k))
        recall = hits / total * 100 if total else 0
        bar = "█" * int(recall / 5) + "░" * (20 - int(recall / 5))
        print(f"  Recall@{k}: {hits:>2d}/{total}  ({recall:5.1f}%)  {bar}")
        summary[f"recall_at_{k}"] = round(recall, 1)
        summary[f"hits_at_{k}"] = hits

    # Per-manual breakdown
    manuals = sorted(set(r.manual for r in results))
    print(f"\n  Per-manual Recall@{k_values[-1]}:")
    manual_breakdown: dict[str, str] = {}
    for manual in manuals:
        manual_results = [r for r in results if r.manual == manual]
        hits = sum(1 for r in manual_results if r.hits_at_k.get(k_values[-1]))
        total_m = len(manual_results)
        recall = hits / total_m * 100 if total_m else 0
        print(f"    {manual:15s}: {hits}/{total_m} ({recall:.0f}%)")
        manual_breakdown[manual] = f"{hits}/{total_m} ({recall:.0f}%)"

    # Per-manual Recall@1 breakdown
    print(f"\n  Per-manual Recall@{k_values[0]}:")
    for manual in manuals:
        manual_results = [r for r in results if r.manual == manual]
        hits = sum(1 for r in manual_results if r.hits_at_k.get(k_values[0]))
        total_m = len(manual_results)
        recall = hits / total_m * 100 if total_m else 0
        print(f"    {manual:15s}: {hits}/{total_m} ({recall:.0f}%)")

    # Failure analysis
    failures = [r for r in results if not r.hits_at_k.get(k_values[-1])]
    if failures:
        print(f"\n  FAILURES (missed at Recall@{k_values[-1]}):")
        for r in failures:
            print(f"    ❌ [{r.manual}] {r.query}  (need: {r.must_contain})")

    print(f"\n{'='*90}\n")

    summary["total_queries"] = total
    summary["manual_breakdown"] = manual_breakdown
    summary["failures"] = [
        {"query": r.query, "manual": r.manual, "must_contain": r.must_contain}
        for r in failures
    ]
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()
    chunks_dir = Path(args.chunks_dir).resolve()
    k_values = sorted(args.k)

    # Load test cases
    if args.test_cases:
        tc_path = Path(args.test_cases)
        test_cases = json.loads(tc_path.read_text(encoding="utf-8"))
        print(f"Loaded {len(test_cases)} test cases from {tc_path}")
    else:
        test_cases = TEST_CASES
        print(f"Using built-in test cases: {len(test_cases)} queries")

    # Load chunks
    print(f"Loading chunks from {chunks_dir}...")
    chunks = load_all_chunks(chunks_dir)
    print(f"Loaded {len(chunks)} text chunks")

    if not chunks:
        print("No chunks found. Run build_langchain_baseline.py first.")
        return

    # Build index
    print("Building TF-IDF index...")
    vectorizer, tfidf_matrix = build_tfidf_index(chunks)
    print(f"Index built: {tfidf_matrix.shape[0]} docs, {tfidf_matrix.shape[1]} features\n")

    # Label for display
    label = chunks_dir.name
    if "baseline" in label.lower():
        label = "LangChain Baseline"
    elif "chunks" in label.lower():
        label = "Structured Chunks"

    # Run evaluation
    results = run_evaluation(test_cases, chunks, vectorizer, tfidf_matrix, k_values)
    summary = print_results(results, k_values, label)

    # Optional: write detailed JSON
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "summary": summary,
            "results": [
                {
                    "query": r.query,
                    "manual": r.manual,
                    "must_contain": r.must_contain,
                    "hits_at_k": r.hits_at_k,
                    "top_hit_title": r.top_hit_title,
                    "top_hit_score": r.top_hit_score,
                    "top_hit_chunk_id": r.top_hit_chunk_id,
                    "top_hit_preview": r.top_hit_preview,
                }
                for r in results
            ],
        }
        output_path.write_text(
            json.dumps(output_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Detailed results written to {output_path}")


if __name__ == "__main__":
    main()
