"""Evaluate the real Plan-Execute-Replan Agent with replayed benchmark tools."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import time
from pathlib import Path
from typing import Any

from app.evaluation.cloud_ops_bench import (
    ROOT_CAUSE_LABELS,
    read_jsonl,
    score_prediction,
    summarize_scores,
    write_jsonl,
)
from app.evaluation.cloud_ops_replay_tools import (
    CloudOpsReplayTools,
    ReplayMCPClient,
    parse_agent_prediction,
)

DEFAULT_DATASET_ROOT = Path("data/aiops_eval/cloud_ops_bench_30")
DEFAULT_CASES = DEFAULT_DATASET_ROOT / "cases.jsonl"
DEFAULT_RESULTS = DEFAULT_DATASET_ROOT / "agent_results.jsonl"


def build_task(case: dict[str, Any]) -> str:
    user_input = case["input"]
    return f"""Diagnose this Kubernetes incident by planning and calling the available tools.

Incident: {user_input.get("query")}
Namespace: {user_input.get("namespace") or "unknown"}

Use only observations returned by tools. Do not assume the expected answer.
The final report must distinguish facts from inference and end with exactly:

EVAL_RESULT
root_cause: <one of {", ".join(ROOT_CAUSE_LABELS)}>
fault_object: <resource type/name, for example app/checkoutservice or node/worker-02>
"""


def inject_replay_client(client: ReplayMCPClient) -> None:
    async def get_client(*args, **kwargs):
        return client

    for module_name in (
        "app.agent.aiops.planner",
        "app.agent.aiops.executor",
        "app.agent.aiops.replanner",
    ):
        module = importlib.import_module(module_name)
        module.get_mcp_client_with_retry = get_client


async def evaluate_case(case: dict[str, Any], dataset_root: Path) -> dict[str, Any]:
    cache_path = dataset_root / "benchmark" / case["source_path"] / "tool_cache.json"
    replay = CloudOpsReplayTools(cache_path)
    inject_replay_client(ReplayMCPClient(replay))

    # Import after tool injection, so the compiled graph uses the real patched nodes.
    from app.services.aiops_service import AIOpsService

    service = AIOpsService()
    events: list[dict[str, Any]] = []
    started = time.perf_counter()
    async for event in service.execute(build_task(case), session_id=f"eval-{case['case_id']}"):
        events.append(event)
    latency = time.perf_counter() - started

    complete = next((event for event in reversed(events) if event.get("type") == "complete"), {})
    report = str(complete.get("response", ""))
    prediction = parse_agent_prediction(report)
    return {
        "case_id": case["case_id"],
        "expected": case["expected"],
        "prediction": prediction,
        "scores": score_prediction(case, prediction),
        "agent_trace": {
            "plans": [event.get("plan") for event in events if event.get("type") == "plan"],
            "executed_steps": [
                event.get("current_step")
                for event in events
                if event.get("type") == "step_complete"
            ],
            "tool_calls": replay.calls,
            "event_types": [event.get("type") for event in events],
        },
        "latency_seconds": round(latency, 3),
    }


async def run(args: argparse.Namespace) -> None:
    cases = read_jsonl(args.cases)
    if args.case_id:
        selected = set(args.case_id)
        cases = [case for case in cases if case["case_id"] in selected]
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("没有选中任何案例")

    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        result = await evaluate_case(case, args.dataset_root)
        results.append(result)
        scores = result["scores"]
        print(
            f"[{index}/{len(cases)}] {case['case_id']} "
            f"completed={scores['completed']} root={scores['root_cause_correct']} "
            f"object={scores['fault_object_correct']} "
            f"tools={len(result['agent_trace']['tool_calls'])}",
            flush=True,
        )
        write_jsonl(results, args.output)

    summary = summarize_scores(results)
    summary["evaluation_target"] = "plan_execute_replan_agent_with_replay_tools"
    summary["total_tool_calls"] = sum(
        len(result["agent_trace"]["tool_calls"]) for result in results
    )
    summary_path = args.output.with_name(f"{args.output.stem}.summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"results -> {args.output}")
    print(f"summary -> {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-id", action="append")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
