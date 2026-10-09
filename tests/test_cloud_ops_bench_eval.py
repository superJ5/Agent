import json
from pathlib import Path

from app.evaluation.cloud_ops_bench import (
    build_case,
    score_prediction,
    summarize_scores,
)
from app.evaluation.cloud_ops_replay_tools import CloudOpsReplayTools, parse_agent_prediction


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_build_case_extracts_expected_label_and_metric_summary(tmp_path):
    benchmark = tmp_path / "benchmark"
    case_dir = benchmark / "boutique" / "performance" / "1"
    _write_json(
        case_dir / "metadata.json",
        {
            "namespace": "boutique",
            "query": "degraded",
            "difficulty": "hard",
            "result": {
                "fault_taxonomy": "Performance_Fault",
                "fault_object": "app/adservice",
                "root_cause": "pod_cpu_overload",
            },
        },
    )
    _write_json(case_dir / "raw_data" / "alert.json", {"status": "has_anomalies"})
    _write_json(
        case_dir / "tool_cache.json",
        {
            "collection_timestamp": "2026-01-01",
            'GetErrorLogs:{"service_name":"adservice"}': "cpu throttled",
        },
    )
    (case_dir / "raw_data" / "metrics.csv").write_text(
        "time,adservice-cpu,other-cpu\n2026-01-01,0.1,0.2\n2026-01-02,0.9,0.3\n",
        encoding="utf-8",
    )

    case = build_case(case_dir / "metadata.json", benchmark)

    assert case["case_id"] == "boutique-performance-1"
    assert case["expected"]["root_cause"] == "pod_cpu_overload"
    assert case["evidence"]["metrics"]["series"]["adservice-cpu"]["max"] == 0.9
    assert len(case["evidence"]["cached_observations"]) == 2


def test_score_prediction_supports_chinese_aliases():
    case = {
        "expected": {
            "root_cause": "code_memory_leak",
            "fault_object": "app/checkoutservice",
        }
    }
    prediction = {"root_cause": "检测到内存泄漏", "fault_object": "checkoutservice"}

    scores = score_prediction(case, prediction)

    assert scores["root_cause_correct"] is True
    assert scores["fault_object_correct"] is True
    assert scores["joint_correct"] is True


def test_summary_rates():
    results = [
        {
            "scores": {
                "completed": True,
                "root_cause_correct": True,
                "fault_object_correct": False,
                "joint_correct": False,
            },
            "latency_seconds": 2.0,
        },
        {
            "scores": {
                "completed": True,
                "root_cause_correct": True,
                "fault_object_correct": True,
                "joint_correct": True,
            },
            "latency_seconds": 4.0,
        },
    ]

    summary = summarize_scores(results)

    assert summary["root_cause_accuracy"] == 1.0
    assert summary["fault_object_accuracy"] == 0.5
    assert summary["joint_accuracy"] == 0.5
    assert summary["average_latency_seconds"] == 3.0


def test_summary_does_not_count_api_errors_as_wrong_diagnoses():
    results = [
        {
            "scores": {
                "completed": True,
                "root_cause_correct": True,
                "fault_object_correct": True,
                "joint_correct": True,
            },
            "latency_seconds": 2.0,
        },
        {
            "scores": {
                "completed": False,
                "root_cause_correct": False,
                "fault_object_correct": False,
                "joint_correct": False,
            },
            "latency_seconds": 1.0,
        },
    ]

    summary = summarize_scores(results)

    assert summary["completed"] == 1
    assert summary["errors"] == 1
    assert summary["completion_rate"] == 0.5
    assert summary["root_cause_accuracy"] == 1.0
    assert summary["fault_object_accuracy"] == 1.0
    assert summary["joint_accuracy"] == 1.0


def test_summary_has_no_accuracy_when_every_call_fails():
    results = [
        {
            "scores": {
                "completed": False,
                "root_cause_correct": False,
                "fault_object_correct": False,
                "joint_correct": False,
            },
            "latency_seconds": 1.0,
        }
    ]

    summary = summarize_scores(results)

    assert summary["completed"] == 0
    assert summary["errors"] == 1
    assert summary["root_cause_accuracy"] is None
    assert summary["fault_object_accuracy"] is None
    assert summary["joint_accuracy"] is None


def test_replay_tools_match_requested_arguments(tmp_path):
    cache_path = tmp_path / "tool_cache.json"
    _write_json(
        cache_path,
        {
            'GetResources:{"resource_type":"pods","name":"","namespace":"boutique"}': ("pod-list"),
            'GetResources:{"resource_type":"nodes","name":"","namespace":""}': "node-list",
        },
    )
    replay = CloudOpsReplayTools(cache_path)
    tools = {tool.name: tool for tool in replay.get_tools()}

    result = tools["GetResources"].invoke({"resource_type": "pod", "namespace": "boutique"})

    assert result == "pod-list"
    assert replay.calls[0]["matched_cache_key"] is not None


def test_parse_agent_prediction_requires_explicit_eval_fields():
    report = """Diagnosis complete.

EVAL_RESULT
root_cause: node_network_delay
fault_object: node/worker-02
"""

    prediction = parse_agent_prediction(report)

    assert prediction["root_cause"] == "node_network_delay"
    assert prediction["fault_object"] == "node/worker-02"
