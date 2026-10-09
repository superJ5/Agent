"""LangChain tools that replay one Cloud-OpsBench tool cache."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool


def parse_agent_prediction(report: str) -> dict[str, Any]:
    """Read the explicit machine-scoreable footer from the Agent report."""
    root_match = re.search(r"root_cause\s*[:：]\s*`?([a-z_]+)", report, re.IGNORECASE)
    object_match = re.search(r"fault_object\s*[:：]\s*`?([^\s`]+)", report, re.IGNORECASE)
    if not root_match or not object_match:
        return {"error": "Agent final report is missing the EVAL_RESULT fields", "report": report}
    return {
        "root_cause": root_match.group(1).lower(),
        "fault_object": object_match.group(1).rstrip(".,;，。；"),
        "report": report,
    }


class CloudOpsReplayTools:
    """Expose cached benchmark observations through normal Agent tools."""

    def __init__(self, tool_cache_path: Path, max_output_chars: int = 20_000):
        self.cache = json.loads(tool_cache_path.read_text(encoding="utf-8"))
        self.max_output_chars = max_output_chars
        self.calls: list[dict[str, Any]] = []

    @staticmethod
    def _meaningful_args(args: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in args.items()
            if key != "self" and value not in (None, "", False)
        }

    def _lookup(self, tool_name: str, args: dict[str, Any]) -> str:
        requested = self._meaningful_args(args)
        candidates: list[tuple[int, str, Any]] = []
        prefix = f"{tool_name}:"
        for key, value in self.cache.items():
            if not key.startswith(prefix):
                continue
            try:
                cached_args = json.loads(key[len(prefix) :])
            except json.JSONDecodeError:
                continue

            def matches(name: str, expected: Any) -> bool:
                actual = cached_args.get(name)
                if (
                    name == "resource_type"
                    and isinstance(actual, str)
                    and isinstance(expected, str)
                ):
                    return actual.lower().rstrip("s") == expected.lower().rstrip("s")
                return actual == expected

            if all(matches(name, expected) for name, expected in requested.items()):
                candidates.append((len(cached_args) - len(requested), key, value))

        if not candidates:
            result = (
                f"No cached result for {tool_name} with arguments "
                f"{json.dumps(requested, ensure_ascii=False)}"
            )
            matched_key = None
        else:
            _, matched_key, raw = min(candidates, key=lambda item: (item[0], item[1]))
            result = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
            if len(result) > self.max_output_chars:
                result = result[: self.max_output_chars] + "\n[output truncated]"

        self.calls.append(
            {
                "tool": tool_name,
                "arguments": requested,
                "matched_cache_key": matched_key,
                "result_chars": len(result),
            }
        )
        return result

    def get_tools(self) -> list[BaseTool]:
        """Build the tool schemas visible to Planner and Executor."""

        def get_alerts() -> str:
            """Return current cluster alerts and anomaly observations."""
            return self._lookup("GetAlerts", {})

        def get_cluster_configuration() -> str:
            """Return cluster nodes, namespaces and basic configuration."""
            return self._lookup("GetClusterConfiguration", {})

        def get_resources(
            resource_type: str,
            name: str = "",
            namespace: str = "",
            output_wide: bool = False,
            show_labels: bool = False,
            label_selector: str = "",
            output: str = "",
        ) -> str:
            """List Kubernetes resources such as pods or nodes."""
            return self._lookup("GetResources", locals())

        def describe_resource(resource_type: str, name: str, namespace: str = "") -> str:
            """Describe one Kubernetes pod, node or other resource in detail."""
            return self._lookup("DescribeResource", locals())

        def check_node_service_status(node_name: str, service_name: str) -> str:
            """Check whether kubelet, containerd, kube-proxy or another node service is healthy."""
            return self._lookup("CheckNodeServiceStatus", locals())

        def get_error_logs(namespace: str, service_name: str) -> str:
            """Return cached error logs for an application service."""
            return self._lookup("GetErrorLogs", locals())

        def get_service_dependencies(service_name: str) -> str:
            """Return upstream and downstream dependencies of a service."""
            return self._lookup("GetServiceDependencies", locals())

        def check_service_connectivity(namespace: str, service_name: str, port: int) -> str:
            """Check network connectivity to one Kubernetes service and port."""
            return self._lookup("CheckServiceConnectivity", locals())

        def get_app_yaml(app_name: str) -> str:
            """Return the Kubernetes YAML configuration for an application."""
            return self._lookup("GetAppYAML", locals())

        def list_code_files(app_name: str) -> str:
            """List source-code files available for an application."""
            return self._lookup("ListCodeFiles", locals())

        definitions = [
            ("GetAlerts", get_alerts),
            ("GetClusterConfiguration", get_cluster_configuration),
            ("GetResources", get_resources),
            ("DescribeResource", describe_resource),
            ("CheckNodeServiceStatus", check_node_service_status),
            ("GetErrorLogs", get_error_logs),
            ("GetServiceDependencies", get_service_dependencies),
            ("CheckServiceConnectivity", check_service_connectivity),
            ("GetAppYAML", get_app_yaml),
            ("ListCodeFiles", list_code_files),
        ]
        return [
            StructuredTool.from_function(func=func, name=name, description=func.__doc__)
            for name, func in definitions
        ]


class ReplayMCPClient:
    """Small get_tools-compatible adapter injected into the real Agent graph."""

    def __init__(self, replay: CloudOpsReplayTools):
        self.replay = replay

    async def get_tools(self) -> list[BaseTool]:
        return self.replay.get_tools()
