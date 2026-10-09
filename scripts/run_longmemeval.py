"""Replay LongMemEval history into isolated memory and answer final questions.

Default mode only inspects the selected cases. Add ``--execute`` to make real
model/embedding calls. The official source JSON is never changed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

from scripts.setup_longmemeval_storage import EVAL_MEMORY_ROOT, main as setup_storage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT.parents[1] / "LongMemEval/data/longmemeval_s_cleaned.json"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def load_cases(path: Path, *, case_id: str | None, limit: int, skip: int = 0) -> list[dict]:
    if limit < 1:
        raise ValueError("--limit 必须大于 0")
    if skip < 0:
        raise ValueError("--skip 不能小于 0")
    if case_id and skip:
        raise ValueError("--case-id 与 --skip 不能同时使用")
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError("LongMemEval 数据顶层必须是数组")
    if case_id:
        entries = [entry for entry in entries if entry.get("question_id") == case_id]
        if not entries:
            raise ValueError(f"找不到 question_id={case_id}")
    selected = entries[skip:skip + limit]
    if not selected:
        raise ValueError(f"跳过前 {skip} 题后没有可运行的题目")
    return selected


async def execute_cases(cases: list[dict], *, run_id: str) -> tuple[Path, Path]:
    from app.evaluation.longmemeval_replay import LongMemEvalReplayer

    output_dir = EVAL_MEMORY_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    hypotheses_path = output_dir / f"{run_id}.jsonl"
    diagnostics_path = output_dir / f"{run_id}.diagnostics.jsonl"
    if hypotheses_path.exists() or diagnostics_path.exists():
        raise FileExistsError("本次 run_id 的评测结果已存在；请使用新的 --run-id")

    replayer = LongMemEvalReplayer()
    with hypotheses_path.open("x", encoding="utf-8") as hypotheses, diagnostics_path.open(
        "x", encoding="utf-8"
    ) as diagnostics:
        for index, entry in enumerate(cases, start=1):
            question_id = str(entry["question_id"])
            print(f"[{index}/{len(cases)}] 回放 {question_id} ...", flush=True)
            try:
                result = await replayer.replay_case(entry, run_id=run_id)
                hypothesis = result.hypothesis
                detail = {
                    "question_id": question_id,
                    "status": "success",
                    "user_id": result.user_id,
                    "session_count": result.session_count,
                    "completed_turn_count": result.completed_turn_count,
                    "compaction_count": result.compaction_count,
                }
            except Exception as exc:
                hypothesis = ""
                detail = {
                    "question_id": question_id,
                    "status": "error",
                    "error_type": type(exc).__name__,
                }
                print(f"  失败：{type(exc).__name__}；本题记为空答案", flush=True)
            hypotheses.write(json.dumps({"question_id": question_id, "hypothesis": hypothesis}, ensure_ascii=False) + "\n")
            diagnostics.write(json.dumps(detail, ensure_ascii=False) + "\n")
            hypotheses.flush()
            diagnostics.flush()

    return hypotheses_path, diagnostics_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--case-id", help="只选指定的 question_id")
    parser.add_argument("--skip", type=int, default=0, help="跳过数据文件开头的 N 题，默认 0")
    parser.add_argument("--limit", type=int, default=1, help="最多评测几题，默认 1")
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    parser.add_argument("--execute", action="store_true", help="真正回放并调用模型；默认只检查数据")
    args = parser.parse_args()
    if not RUN_ID_RE.fullmatch(args.run_id):
        parser.error("--run-id 只能包含英文字母、数字、_、-，长度不超过 32")

    cases = load_cases(args.data, case_id=args.case_id, limit=args.limit, skip=args.skip)
    print(f"数据文件：{args.data}")
    print(f"选中 {len(cases)} 题；每题使用不同的内部 user_id")
    for entry in cases:
        print(
            f"- {entry['question_id']} [{entry['question_type']}] "
            f"{len(entry['haystack_sessions'])} 段历史会话"
        )
    if not args.execute:
        print("仅检查数据，未创建评测结果或调用模型。需要真实回放时加 --execute。")
        return

    setup_storage()
    hypotheses, diagnostics = asyncio.run(execute_cases(cases, run_id=args.run_id))
    print(f"答案文件：{hypotheses}")
    print(f"运行记录：{diagnostics}")
    trace_path = EVAL_MEMORY_ROOT / "traces" / f"{args.run_id}.jsonl"
    if trace_path.exists():
        print(f"结构化响应失败诊断：{trace_path}")


if __name__ == "__main__":
    main()
