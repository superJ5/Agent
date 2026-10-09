"""智能运维监控 MCP Server

本地实现的监控服务 MCP Server，提供：
- 监控数据查询（CPU、内存、磁盘、网络等）
- 进程信息查询
- 历史工单查询
- 服务信息查询

用于支持运维 Agent 的故障排查场景。
"""

import functools
import json
import logging
from typing import Any

from fastmcp import FastMCP

from app.services.local_monitor_service import read_cpu_snapshot, read_memory_snapshot

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Monitor_MCP_Server")

mcp = FastMCP("Monitor")


def log_tool_call(func):
    """装饰器：记录工具调用的日志，包括方法名、参数和返回状态"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        method_name = func.__name__

        # 记录调用信息
        logger.info("=" * 80)
        logger.info(f"调用方法: {method_name}")

        # 记录参数（排除self等）
        if kwargs:
            # 使用 json.dumps 格式化参数，处理可能的序列化错误
            try:
                params_str = json.dumps(kwargs, ensure_ascii=False, indent=2)
            except (TypeError, ValueError):
                params_str = str(kwargs)
            logger.info(f"参数信息:\n{params_str}")
        else:
            logger.info("参数信息: 无")

        # 执行方法
        try:
            result = func(*args, **kwargs)

            # 记录返回状态
            logger.info("返回状态: SUCCESS")

            # 记录返回结果摘要（避免日志过长）
            if isinstance(result, dict):
                summary = {
                    k: v
                    if not isinstance(v, (list, dict))
                    else f"<{type(v).__name__} with {len(v)} items>"
                    for k, v in list(result.items())[:5]
                }
                logger.info(f"返回结果摘要: {json.dumps(summary, ensure_ascii=False)}")
            else:
                logger.info(f"返回结果: {result}")

            logger.info("=" * 80)
            return result

        except Exception as e:
            # 记录错误状态
            logger.error("返回状态: ERROR")
            logger.error(f"错误信息: {str(e)}")
            logger.error("=" * 80)
            raise

    return wrapper


# ============================================================
# 监控数据查询工具
# ============================================================


@mcp.tool()
@log_tool_call
def query_cpu_metrics(
    service_name: str,
    start_time: str | None = None,
    end_time: str | None = None,
    interval: str = "1m",
) -> dict[str, Any]:
    """读取本机当前真实 CPU 使用率。

    Args:
        service_name: 服务名称（必填）
            示例: "data-sync-service"

        start_time: 兼容旧接口的请求时间，不用于生成历史数据

        end_time: 兼容旧接口的请求时间，不用于生成历史数据

        interval: 兼容旧接口；本地实时模式只返回一个当前采样点

    Returns:
        Dict: 当前 CPU 快照。data_source 为 local_system，
        historical_available 为 false，不代表请求时间段内的历史趋势。
    """
    result = read_cpu_snapshot(service_name)
    result["requested_time_range"] = {"start_time": start_time, "end_time": end_time}
    result["requested_interval"] = interval
    return result


@mcp.tool()
@log_tool_call
def query_memory_metrics(
    service_name: str,
    start_time: str | None = None,
    end_time: str | None = None,
    interval: str = "1m",
) -> dict[str, Any]:
    """读取本机当前真实内存使用率。

    Args:
        service_name: 服务名称（必填）
            示例: "data-sync-service"

        start_time: 兼容旧接口的请求时间，不用于生成历史数据

        end_time: 兼容旧接口的请求时间，不用于生成历史数据

        interval: 兼容旧接口；本地实时模式只返回一个当前采样点

    Returns:
        Dict: 当前内存快照。data_source 为 local_system，
        historical_available 为 false，不代表请求时间段内的历史趋势。
    """
    result = read_memory_snapshot(service_name)
    result["requested_time_range"] = {"start_time": start_time, "end_time": end_time}
    result["requested_interval"] = interval
    return result


if __name__ == "__main__":
    # 使用 streamable-http 模式，运行在 8004 端口
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8004, path="/mcp")
