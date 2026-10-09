# MCP Servers

为 AIOps 智能诊断提供日志查询和监控数据工具。

## 📚 服务列表

### CLS Server (`cls_server.py`)
**日志查询服务** - 端口 8003

**核心工具：**
- `get_current_timestamp` - 获取当前时间戳
- `get_region_code_by_name` - 将地区名称转换成地域代码
- `get_topic_info_by_name` - 查询日志主题
- `search_topic_by_service_name` - 按服务或机器名称查询日志主题
- `search_log` - 日志搜索

### Monitor Server (`monitor_server.py`)
**本机实时监控服务** - 端口 8004

**核心工具：**
- `query_cpu_metrics` - 读取本机当前真实 CPU 使用率
- `query_memory_metrics` - 读取本机当前真实内存使用率
- `query_process_list` - 进程列表
- `search_historical_tickets` - 历史工单查询
- `get_service_info` / `list_all_services` - 服务信息

## 🚀 快速开始

### 安装依赖
```bash
uv pip install fastmcp
```

### 启动服务

**方式一：使用 Makefile（推荐）**
```bash
make start       # 启动所有 MCP 服务和 FastAPI
make stop        # 停止所有服务
make status-mcp  # 查看 MCP 服务状态
```

**方式二：手动启动**
```bash
python mcp_servers/cls_server.py
python mcp_servers/monitor_server.py
```

## 💡 使用示例

### AIOps 诊断场景

```
用户: 检查当前虚拟机最近一小时是否存在异常日志

Agent 自动执行:
1. search_topic_by_service_name("superj-vm") → 获取真实 Topic ID
2. get_current_timestamp() → 确定查询结束时间
3. search_log(..., query="error OR warning") → 查询腾讯云 CLS
4. 综合真实日志证据 → 生成诊断报告和处理建议
```

### 工具参数示例

**查询 CPU 指标：**
```python
query_cpu_metrics(
    service_name="data-sync-service",
    start_time="2024-02-14 02:00:00",
    interval="1m"
)
```

**搜索日志：**
```python
search_log(
    topic_id="DescribeTopics 返回的真实 Topic ID",
    start_time=开始时间戳毫秒,
    end_time=结束时间戳毫秒,
    query="error",
    limit=100,
)
```

**搜索历史工单：**
```python
search_historical_tickets(
    service_name="data-sync-service",
    issue_type="cpu",
    limit=10
)
```

## 🔧 高级配置

### 腾讯云 CLS 数据源

系统支持两种显式模式：

- `CLS_DATA_SOURCE=mock`：本地演示数据（默认）。
- `CLS_DATA_SOURCE=tencent`：腾讯云 CLS 真实日志；配置或 API 失败时直接返回错误，不回退到模拟数据。

使用交互脚本将密钥安全写入 Git 忽略的 `.env.local`：

```bash
uv run python scripts/configure_tencent_cls.py
```

需要的配置项为 `TENCENTCLOUD_SECRET_ID`、`TENCENTCLOUD_SECRET_KEY`、
`TENCENT_CLS_REGION`、`TENCENT_CLS_TOPIC_ID`、`TENCENT_CLS_TOPIC_NAME` 和
`TENCENT_CLS_SERVICE_NAME`。不要提交 `.env.local`，不要在日志或聊天中发送密钥。

**其他监控系统：**
- Prometheus
- Grafana
- 云监控（腾讯云/阿里云/AWS）
- 自建监控平台

### 自定义 Mock 数据

修改各 Server 文件中的数据生成逻辑，模拟实际场景。

## 📚 参考资料

- [FastMCP 文档](https://github.com/jlowin/fastmcp)
- [MCP 协议](https://modelcontextprotocol.io/)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [主项目 README](../README.md)

---

**注意**：Monitor Server 通过 `psutil` 返回调用时的本机真实快照，不提供历史曲线；
CLS Server 已支持 mock 和腾讯云真实日志两种模式。需要历史 CPU/内存趋势时，应接入
Prometheus 等时序监控系统。
