"""Read live host metrics for the local AIOps monitor MCP server."""

from __future__ import annotations

import socket
from datetime import datetime
from typing import Any

import psutil

CPU_ALERT_THRESHOLD = 80.0
MEMORY_ALERT_THRESHOLD = 70.0


def _now() -> datetime:
    """Return a timezone-aware local timestamp; split out for deterministic tests."""
    return datetime.now().astimezone()


def read_cpu_snapshot(service_name: str) -> dict[str, Any]:
    """Measure the current host CPU usage without fabricating historical points."""
    value = round(psutil.cpu_percent(interval=0.2), 1)
    timestamp = _now().isoformat(timespec="seconds")
    threshold_exceeded = value > CPU_ALERT_THRESHOLD

    try:
        load_average = [round(item, 2) for item in psutil.getloadavg()]
    except (AttributeError, OSError):
        load_average = []

    return {
        "service_name": service_name,
        "observed_host": socket.gethostname(),
        "metric_name": "cpu_usage_percent",
        "data_source": "local_system",
        "collection_mode": "live_snapshot",
        "historical_available": False,
        "sampled_at": timestamp,
        "data_points": [{"timestamp": timestamp, "value": value}],
        "statistics": {
            "current": value,
            "sample_count": 1,
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "load_average_1m_5m_15m": load_average,
            # Retain the old key for callers while describing snapshot semantics.
            "spike_detected": threshold_exceeded,
        },
        "alert_info": {
            "triggered": threshold_exceeded,
            "threshold": CPU_ALERT_THRESHOLD,
            "message": (
                f"当前 CPU 使用率 {value}% 超过 {CPU_ALERT_THRESHOLD}% 阈值"
                if threshold_exceeded
                else f"当前 CPU 使用率 {value}%，未超过 {CPU_ALERT_THRESHOLD}% 阈值"
            ),
        },
    }


def read_memory_snapshot(service_name: str) -> dict[str, Any]:
    """Read the current host virtual-memory counters."""
    memory = psutil.virtual_memory()
    value = round(float(memory.percent), 1)
    timestamp = _now().isoformat(timespec="seconds")
    threshold_exceeded = value > MEMORY_ALERT_THRESHOLD
    gib = 1024**3

    return {
        "service_name": service_name,
        "observed_host": socket.gethostname(),
        "metric_name": "memory_usage_percent",
        "data_source": "local_system",
        "collection_mode": "live_snapshot",
        "historical_available": False,
        "sampled_at": timestamp,
        "data_points": [
            {
                "timestamp": timestamp,
                "value": value,
                "used_gb": round(memory.used / gib, 2),
                "available_gb": round(memory.available / gib, 2),
                "total_gb": round(memory.total / gib, 2),
            }
        ],
        "statistics": {
            "current": value,
            "sample_count": 1,
            "memory_pressure": threshold_exceeded,
        },
        "alert_info": {
            "triggered": threshold_exceeded,
            "threshold": MEMORY_ALERT_THRESHOLD,
            "message": (
                f"当前内存使用率 {value}% 超过 {MEMORY_ALERT_THRESHOLD}% 阈值"
                if threshold_exceeded
                else f"当前内存使用率 {value}%，未超过 {MEMORY_ALERT_THRESHOLD}% 阈值"
            ),
        },
    }
