"""Convert and score the selected Cloud-OpsBench diagnosis cases."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT_CAUSE_LABELS = (
    "pod_cpu_overload",
    "code_memory_leak",
    "node_network_delay",
    "node_network_packet_loss",
    "containerd_unavailable",
    "kubelet_unavailable",
)

ROOT_CAUSE_ALIASES = {
    "pod_cpu_overload": ("pod cpu overload", "cpu overload", "cpu过载", "cpu 使用率过高"),
    "code_memory_leak": ("code memory leak", "memory leak", "内存泄漏"),
    "node_network_delay": ("node network delay", "network delay", "网络延迟"),
    "node_network_packet_loss": (
        "node network packet loss",
        "network packet loss",
        "packet loss",
        "网络丢包",
    ),
    "containerd_unavailable": ("containerd unavailable", "containerd 不可用"),
    "kubelet_unavailable": ("kubelet unavailable", "kubelet 不可用"),
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} 顶层必须是 JSON 对象")
    return value


def _metric_tokens(root_cause: str) -> tuple[str, ...]:
    if "cpu" in root_cause:
        return ("-cpu", "-cpu_cfs", "latency", "success_rate", "-rps")
    if "memory" in root_cause:
        return ("-mem", "latency", "success_rate", "-rps")
    if "network" in root_cause:
        return ("network_receive", "network_transmit", "latency", "success_rate", "-rps")
    return ()


def summarize_metrics(
    path: Path,
    *,
    fault_object: str,
    root_cause: str,
    max_metrics: int = 12,
) -> dict[str, Any]:
    """Summarize target-related numeric columns without copying the whole time series."""
    if not path.exists():
        return {"available": False, "reason": "upstream case has no metrics.csv"}

    target = fault_object.split("/", 1)[-1].lower()
    tokens = _metric_tokens(root_cause)
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        target_columns = [
            name
            for name in fieldnames
            if name != "time"
            and target in name.lower()
            and (not tokens or any(token in name.lower() for token in tokens))
        ][:max_metrics]
        rows = list(reader)

    summaries: dict[str, Any] = {}
    for column in target_columns:
        values: list[float] = []
        for row in rows:
            raw = (row.get(column) or "").strip()
            if not raw:
                continue
            try:
                values.append(float(raw))
            except ValueError:
                continue
        if values:
            summaries[column] = {
                "first": round(values[0], 4),
                "last": round(values[-1], 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "mean": round(fmean(values), 4),
                "samples": len(values),
            }

    return {
        "available": bool(summaries),
        "start_time": rows[0].get("time") if rows else None,
        "end_time": rows[-1].get("time") if rows else None,
        "target": target,
        "series": summaries,
    }


def _cache_key_matches(key: str, *, root_cause: str, fault_object: str) -> bool:
    target = fault_object.split("/", 1)[-1].lower()
    lowered = key.lower()
    if key == "collection_timestamp" or key.startswith("GetAlerts:"):
        return True
    if key.startswith("GetResources:") and any(
        marker in lowered for marker in ('"resource_type":"pods"', '"resource_type":"nodes"')
    ):
        return '"name":""' in lowered
    if key.startswith("GetErrorLogs:") and target in lowered:
        return True
    if root_cause == "containerd_unavailable":
        return (
            key.startswith("CheckNodeServiceStatus:")
            and target in lowered
            and "containerd" in lowered
        )
    if root_cause == "kubelet_unavailable":
        return (
            key.startswith("CheckNodeServiceStatus:") and target in lowered and "kubelet" in lowered
        )
    return False


def extract_cached_observations(
    path: Path,
    *,
    root_cause: str,
    fault_object: str,
    max_items: int = 6,
    max_output_chars: int = 2500,
) -> list[dict[str, str]]:
    """Keep a compact, evidence-focused subset of official cached tool outputs."""
    cache = load_json(path)
    observations: list[dict[str, str]] = []
    for key, value in cache.items():
        if not _cache_key_matches(key, root_cause=root_cause, fault_object=fault_object):
            continue
        output = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        observations.append({"tool_call": key, "output": output[:max_output_chars]})
        if len(observations) >= max_items:
            break
    return observations


def build_case(metadata_path: Path, benchmark_root: Path) -> dict[str, Any]:
    metadata = load_json(metadata_path)
    result = metadata.get("result") or {}
    if not isinstance(result, dict):
        raise ValueError(f"{metadata_path} 的 result 必须是对象")

    root_cause = str(result.get("root_cause", ""))
    if root_cause not in ROOT_CAUSE_LABELS:
        raise ValueError(f"不支持的 root_cause: {root_cause}")

    case_dir = metadata_path.parent
    relative = case_dir.relative_to(benchmark_root)
    case_id = "-".join(relative.parts)
    fault_object = str(result.get("fault_object", ""))

    return {
        "case_id": case_id,
        "source": "Cloud-OpsBench",
        "source_path": relative.as_posix(),
        "input": {
            "query": metadata.get("query", "Diagnose the service incident."),
            "namespace": metadata.get("namespace"),
            "difficulty": metadata.get("difficulty"),
        },
        "evidence": {
            "alerts": load_json(case_dir / "raw_data" / "alert.json"),
            "metrics": summarize_metrics(
                case_dir / "raw_data" / "metrics.csv",
                fault_object=fault_object,
                root_cause=root_cause,
            ),
            "cached_observations": extract_cached_observations(
                case_dir / "tool_cache.json",
                root_cause=root_cause,
                fault_object=fault_object,
            ),
        },
        "expected": {
            "fault_taxonomy": result.get("fault_taxonomy"),
            "fault_object": fault_object,
            "root_cause": root_cause,
        },
    }


def prepare_cases(dataset_root: Path) -> list[dict[str, Any]]:
    benchmark_root = dataset_root / "benchmark"
    metadata_files = sorted(benchmark_root.glob("**/metadata.json"))
    return [build_case(path, benchmark_root) for path in metadata_files]


def write_jsonl(records: Iterable[dict[str, Any]], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} 必须是 JSON 对象")
            records.append(value)
    return records


def _normalize(value: str) -> str:
    return "".join(char.lower() for char in value if char.isalnum())


def root_cause_matches(predicted: str, expected: str) -> bool:
    predicted_normalized = _normalize(predicted)
    candidates = (expected, *ROOT_CAUSE_ALIASES.get(expected, ()))
    return any(_normalize(candidate) in predicted_normalized for candidate in candidates)


def fault_object_matches(predicted: str, expected: str) -> bool:
    target = expected.split("/", 1)[-1]
    return bool(target) and _normalize(target) in _normalize(predicted)


def score_prediction(case: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    expected = case["expected"]
    completed = not bool(prediction.get("error"))
    root_correct = completed and root_cause_matches(
        str(prediction.get("root_cause", "")), str(expected["root_cause"])
    )
    object_correct = completed and fault_object_matches(
        str(prediction.get("fault_object", "")), str(expected["fault_object"])
    )
    return {
        "completed": completed,
        "root_cause_correct": root_correct,
        "fault_object_correct": object_correct,
        "joint_correct": root_correct and object_correct,
    }


def summarize_scores(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "completed": 0,
            "errors": 0,
            "completion_rate": None,
            "root_cause_accuracy": None,
            "fault_object_accuracy": None,
            "joint_accuracy": None,
            "average_latency_seconds": None,
        }

    completed_results = [item for item in results if item["scores"].get("completed")]
    completed = len(completed_results)

    def scored_rate(key: str) -> float | None:
        if not completed_results:
            return None
        correct = sum(bool(item["scores"].get(key)) for item in completed_results)
        return round(correct / completed, 4)

    latencies = [float(item["latency_seconds"]) for item in results if item.get("latency_seconds")]
    return {
        "total": total,
        "completed": completed,
        "errors": total - completed,
        "completion_rate": round(completed / total, 4),
        "root_cause_accuracy": scored_rate("root_cause_correct"),
        "fault_object_accuracy": scored_rate("fault_object_correct"),
        "joint_accuracy": scored_rate("joint_correct"),
        "average_latency_seconds": round(fmean(latencies), 3) if latencies else None,
    }
