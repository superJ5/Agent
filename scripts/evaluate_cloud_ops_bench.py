"""Run and score offline evidence-to-diagnosis evaluation on 30 public cases."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_qwq import ChatQwen
from pydantic import BaseModel, Field

from app.config import config
from app.evaluation.cloud_ops_bench import (
    ROOT_CAUSE_LABELS,
    read_jsonl,
    score_prediction,
    summarize_scores,
    write_jsonl,
)

DEFAULT_CASES = Path("data/aiops_eval/cloud_ops_bench_30/cases.jsonl")
DEFAULT_RESULTS = Path("data/aiops_eval/cloud_ops_bench_30/results.jsonl")


class DiagnosisPrediction(BaseModel):
    root_cause: Literal[
        "pod_cpu_overload",
        "code_memory_leak",
        "node_network_delay",
        "node_network_packet_loss",
        "containerd_unavailable",
        "kubelet_unavailable",
    ] = Field(description="最可能的一个根因标签")
    fault_object: str = Field(description="最可能的故障对象，例如 app/adservice 或 node/worker-01")
    evidence: list[str] = Field(description="直接支持结论的观测证据")
    reasoning: str = Field(description="区分已确认事实和推测的简短分析")
    confidence: float = Field(ge=0, le=1)


SYSTEM_PROMPT = """你是离线运维诊断评测中的诊断 Agent。
你会获得一个公开基准案例的告警、指标摘要和缓存工具观测。请选择最可能的根因和故障对象。
只能使用给出的证据，不得假装执行修复，不得把相关性描述成已确认因果。
候选根因标签：{labels}
"""


def _model_input(case: dict) -> str:
    payload = {"input": case["input"], "evidence": case["evidence"]}
    return json.dumps(payload, ensure_ascii=False, indent=2)


async def diagnose_case(case: dict, model_name: str) -> tuple[dict, float]:
    llm = ChatQwen(
        model=model_name,
        api_key=config.dashscope_api_key,
        base_url=config.dashscope_api_base,
        temperature=0,
    ).with_structured_output(DiagnosisPrediction)
    started = time.perf_counter()
    try:
        response = await llm.ainvoke(
            [
                SystemMessage(content=SYSTEM_PROMPT.format(labels=", ".join(ROOT_CAUSE_LABELS))),
                HumanMessage(content=_model_input(case)),
            ]
        )
        prediction = response.model_dump() if isinstance(response, BaseModel) else dict(response)
    except Exception as exc:
        prediction = {"error": f"{type(exc).__name__}: {exc}"}
    return prediction, time.perf_counter() - started


async def run(args: argparse.Namespace) -> None:
    cases = read_jsonl(args.cases)
    if args.case_id:
        selected = set(args.case_id)
        cases = [case for case in cases if case["case_id"] in selected]
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("没有选中任何案例")

    results: list[dict] = []
    for index, case in enumerate(cases, 1):
        prediction, latency = await diagnose_case(case, args.model)
        result = {
            "case_id": case["case_id"],
            "expected": case["expected"],
            "prediction": prediction,
            "scores": score_prediction(case, prediction),
            "latency_seconds": round(latency, 3),
        }
        results.append(result)
        if result["scores"]["completed"]:
            print(
                f"[{index}/{len(cases)}] {case['case_id']} "
                f"root={result['scores']['root_cause_correct']} "
                f"object={result['scores']['fault_object_correct']}"
            )
        else:
            print(f"[{index}/{len(cases)}] {case['case_id']} ERROR: {prediction['error']}")

    write_jsonl(results, args.output)
    summary = summarize_scores(results)
    summary_path = args.output.with_name(f"{args.output.stem}.summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"results -> {args.output}")
    print(f"summary -> {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--model", default=config.rag_model)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-id", action="append")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
